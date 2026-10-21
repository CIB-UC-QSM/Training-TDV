"""Máscaras de cerebro para ldm100k3d: deepbet ∩ HD-BET sin TTA, alineadas con la cache.

Tres pasos por lista, en orden. `deepbet` y `hdbet` corren cada uno en el python de su entorno, no
en el .venv del proyecto, y dejan la máscara de cada sujeto en el espacio de vóxeles del NIfTI,
comprimida, en `<trabajo>/<herramienta>/`. `combina`, en el .venv, interseca las dos, rellena
huecos, recorta con la misma caja que `cachea_uno` de `cache_ldm100k3d.py`, verifica contra el
volumen cacheado y escribe `<salida>/mascaras/<split>/<sujeto>.npy` (bool), que es lo que lee
`train3d.py --mascara-cerebro`. H es la carpeta de herramientas (abajo, cómo armarla):

    CUDA_VISIBLE_DEVICES=0 $H/deepbet/bin/python -I datasets/mascaras_ldm100k3d.py deepbet \\
        --lista lista_train_2000.txt
    CUDA_VISIBLE_DEVICES=0 $H/hdbet/bin/python -I datasets/mascaras_ldm100k3d.py hdbet \\
        --lista lista_train_2000.txt --herramientas $H
    .venv/bin/python datasets/mascaras_ldm100k3d.py combina --lista lista_train_2000.txt \\
        --split train

Por cerebro, 0.16 s deepbet y 0.51 s HD-BET en una 5070 Ti; en CPU (16 hilos), 0.48 y 21.8 s.
Un sujeto ya hecho se saltea: cortar y relanzar sigue donde quedó. `combina` exige que el
volumen ya esté cacheado (`cache_ldm100k3d.py cachea` con la misma lista y el mismo split).

Cómo armar H con uv, con las versiones que se midieron el 2026-10-09. Los pesos de deepbet vienen
en su wheel; los de HD-BET se bajan de Zenodo (14445620) a una carpeta propia, no a
`~/hd-bet_params` como su CLI:

    uv venv --python 3.12 $H/deepbet
    uv pip install --python $H/deepbet/bin/python deepbet==1.0.2 torch==2.14.1 nibabel==5.4.2
    uv venv --python 3.12 $H/hdbet
    uv pip install --python $H/hdbet/bin/python hd-bet==2.0.1 nnunetv2==2.8.1 torch==2.14.1 \\
        SimpleITK==2.5.6 nibabel==5.4.2
    curl -L -o $H/release_v1.5.0.zip \\
        'https://zenodo.org/records/14445620/files/release_v1.5.0.zip?download=1'
    unzip $H/release_v1.5.0.zip -d $H/hdbet_params/release_2.0.0

El torch de PyPI trae CUDA 13; en una tarjeta que pida otra versión, el índice de torch que le
corresponda. Ojo: las máscaras de `combina` sólo valen para volúmenes cacheados con este
`cache_ldm100k3d.py`; si cambia la normalización o MARGEN, cambia la caja y hay que rehacerlas.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

# Lo de la cache, de su única fuente. El directorio va a mano porque con `python -I` el del script
# no entra en `sys.path`; `cache_ldm100k3d` sólo pide nibabel y numpy, que están en los tres
# entornos.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_ldm100k3d import MARGEN, RAW, UMBRAL_TEJIDO, caja_tejido, nifti, normaliza  # noqa: E402

FORMA = (160, 224, 160)          # la de los 100 000 NIfTI de LDM_100k, 1 mm isotrópico
HERRAMIENTAS = ("deepbet", "hdbet_sintta")   # subcarpetas de --trabajo, una por paso


# --- RUTAS -----------------------------------------------------------------------------------

def lee_lista(salida: Path, lista: str) -> list[str]:
    """La lista por ruta, o por nombre relativo a --salida, como en `cachea`."""
    ruta = Path(lista)
    if not ruta.exists():
        ruta = salida / lista
    return [l.strip() for l in ruta.read_text().splitlines() if l.strip()]


def trabajo(args: argparse.Namespace) -> Path:
    return Path(args.trabajo) if args.trabajo else Path(args.salida) / "mascaras-nifti"


def guarda_npz(destino: Path, mascara: np.ndarray) -> None:
    """Comprimida (~100 veces menos que el bool crudo) y a un temporal: un corte no deja una a
    medias con el nombre bueno, que el relanzamiento saltearía."""
    temporal = destino.with_name(destino.stem + ".tmp.npz")
    np.savez_compressed(temporal, mascara=mascara)
    temporal.replace(destino)


def lee_mascara(carpeta: Path, sujeto: str) -> np.ndarray:
    """La de `deepbet`/`hdbet` (`.npz`), o un `.npy` bool suelto en el espacio del NIfTI."""
    if (carpeta / f"{sujeto}.npz").exists():
        with np.load(carpeta / f"{sujeto}.npz") as z:
            return z["mascara"]
    return np.load(carpeta / f"{sujeto}.npy")


# --- PASOS 1 Y 2: LAS DOS HERRAMIENTAS -------------------------------------------------------

def prepara_torch(disp: str, hilos: int):
    import torch
    if disp == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--disp cuda y torch no ve ninguna GPU: elegirla con "
                             "CUDA_VISIBLE_DEVICES, o --disp cpu")
        return torch.device("cuda")
    torch.set_num_threads(hilos)
    return torch.device("cpu")


def segmenta_todos(args: argparse.Namespace, nombre: str, segmenta) -> None:
    """Corre `segmenta(ruta del NIfTI) -> bool [D, H, W]` sobre la lista, salteando lo hecho."""
    sujetos = lee_lista(Path(args.salida), args.lista)
    destino = trabajo(args) / nombre
    destino.mkdir(parents=True, exist_ok=True)
    pendientes = [s for s in sujetos if not (destino / f"{s}.npz").exists()]
    print(f"{nombre}: {len(sujetos) - len(pendientes)} de {len(sujetos)} ya hechos -> {destino}",
          flush=True)
    t0, tiempos, fallidos = time.time(), [], []
    for k, sujeto in enumerate(pendientes, 1):
        t = time.perf_counter()
        try:
            mascara = segmenta(nifti(args.rawdata, sujeto))
        except Exception as e:   # un NIfTI corrupto no justifica matar la tanda
            fallidos.append(sujeto)
            print(f"  {sujeto}: falló ({type(e).__name__}: {e})", flush=True)
            continue
        if mascara.shape != FORMA or mascara.dtype != bool:
            raise ValueError(f"{sujeto}: máscara {mascara.dtype} {mascara.shape}, se esperaba "
                             f"bool {FORMA}")
        guarda_npz(destino / f"{sujeto}.npz", mascara)
        tiempos.append(time.perf_counter() - t)
        if k % 250 == 0:
            print(f"  {k}/{len(pendientes)} ({time.time() - t0:.0f} s)", flush=True)
    if tiempos:
        print(f"{nombre}: {len(tiempos)} en {time.time() - t0:.0f} s, mediana "
              f"{np.median(tiempos):.2f} s por cerebro con la lectura"
              + (f" | fallidos: {fallidos[:10]}" if fallidos else ""), flush=True)


def deepbet(args: argparse.Namespace) -> None:
    from deepbet.bet import BrainExtraction
    dispositivo = prepara_torch(args.disp, args.hilos)
    bet = BrainExtraction(no_gpu=dispositivo.type == "cpu")

    def segmenta(ruta):
        img = nib.load(str(ruta))
        # Los defaults: umbral 0.5, sin dilatar; ya se queda con la componente más grande y
        # rellena huecos. Devuelve la máscara en el espacio de vóxeles de la entrada.
        _, mascara, _ = bet.run(img)
        if not np.allclose(mascara.affine, img.affine):
            raise ValueError("deepbet devolvió otro afín que el del NIfTI")
        return np.asarray(mascara.dataobj).astype(bool)

    segmenta_todos(args, HERRAMIENTAS[0], segmenta)


def hdbet(args: argparse.Namespace) -> None:
    os.environ["nnUNet_compile"] = "F"
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    dispositivo = prepara_torch(args.disp, args.hilos)
    params = Path(args.params) if args.params else (
        Path(args.herramientas) / "hdbet_params" / "release_2.0.0")
    if not (params / "fold_all" / "checkpoint_final.pth").exists():
        raise SystemExit(f"no están los pesos de HD-BET en {params}: ver el docstring")
    # El predictor de `HD_BET.hd_bet_prediction.get_hdbet_predictor`, sin espejado (sin TTA) y
    # con los pesos en `params`. Se llama en memoria, sin escribir NIfTI.
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                perform_everything_on_device=True, device=dispositivo,
                                verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(params), "all")

    def segmenta(ruta):
        datos, props = SimpleITKIO().read_images([str(ruta)])
        seg = predictor.predict_single_npy_array(datos, props)
        # SimpleITK da [z, y, x]; nibabel, [x, y, z].
        return np.ascontiguousarray(seg.transpose(2, 1, 0) == 1)

    segmenta_todos(args, HERRAMIENTAS[1], segmenta)


# --- PASO 3: INTERSECCIÓN Y RECORTE ----------------------------------------------------------

def combina_uno(tarea: tuple) -> tuple[str, dict | str]:
    """Un sujeto: la intersección con huecos rellenos, recortada con la caja de `cachea_uno`."""
    from scipy import ndimage
    sujeto, raw, dir_deepbet, dir_hdbet, cache, destino = tarea
    try:
        a = lee_mascara(Path(dir_deepbet), sujeto)
        b = lee_mascara(Path(dir_hdbet), sujeto)
        vol = normaliza(np.asarray(nib.load(nifti(raw, sujeto)).dataobj, dtype=np.float32))
        cacheado = np.load(Path(cache) / f"{sujeto}.npy")
    except Exception as e:   # uno que falta o está corrupto se informa y no mata a los demás
        return sujeto, f"{type(e).__name__}: {e}"
    if a.shape != vol.shape or b.shape != vol.shape:
        return sujeto, f"máscaras {a.shape} y {b.shape} contra el NIfTI {vol.shape}"
    caja = caja_tejido(vol > UMBRAL_TEJIDO, MARGEN)
    # La alineación vóxel a vóxel no se supone: el volumen cacheado tiene que ser este recorte.
    if caja is None or not np.array_equal(cacheado.view(np.uint16),
                                          vol[caja].astype(np.float16).view(np.uint16)):
        return sujeto, "el .npy cacheado no es normaliza(NIfTI)[caja]: recachear o revisar"
    cruda = np.asarray(a, bool) & np.asarray(b, bool)
    inter = ndimage.binary_fill_holes(cruda)
    recorte = np.ascontiguousarray(inter[caja])
    np.save(Path(destino) / f"{sujeto}.npy", recorte)
    return sujeto, {"forma": recorte.shape, "ml": recorte.sum() / 1e3, "fraccion": recorte.mean(),
                    "huecos": int(inter.sum() - cruda.sum()),
                    "fuera_del_recorte": int(inter.sum() - recorte.sum())}


def combina(args: argparse.Namespace) -> None:
    salida = Path(args.salida)
    sujetos = lee_lista(salida, args.lista)
    dir_deepbet = Path(args.deepbet) if args.deepbet else trabajo(args) / HERRAMIENTAS[0]
    dir_hdbet = Path(args.hdbet) if args.hdbet else trabajo(args) / HERRAMIENTAS[1]
    cache = salida / "images" / args.split
    destino = salida / "mascaras" / args.split
    destino.mkdir(parents=True, exist_ok=True)
    t0, filas, fallidos = time.time(), [], []
    tareas = [(s, args.rawdata, str(dir_deepbet), str(dir_hdbet), str(cache), str(destino))
              for s in sujetos]
    with ProcessPoolExecutor(max_workers=args.procesos) as pool:
        for k, (sujeto, r) in enumerate(pool.map(combina_uno, tareas, chunksize=4), 1):
            if isinstance(r, str):
                fallidos.append(sujeto)
                print(f"  {sujeto}: {r}", flush=True)
            else:
                filas.append(r)
            if k % 500 == 0:
                print(f"  {k}/{len(tareas)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"{args.split}: {len(filas)}/{len(sujetos)} máscaras en {time.time() - t0:.0f} s -> "
          f"{destino}; cada volumen cacheado, idéntico bit a bit a normaliza(NIfTI)[caja]"
          + (f" | fallidos {len(fallidos)}: {fallidos[:10]}" if fallidos else ""), flush=True)
    if filas:
        for clave, formato in (("ml", ".1f"), ("fraccion", ".3f"), ("huecos", ".0f"),
                               ("fuera_del_recorte", ".0f")):
            x = [f[clave] for f in filas]
            print(f"  {clave}: min {min(x):{formato}}  mediana {np.median(x):{formato}}  "
                  f"max {max(x):{formato}}", flush=True)
    if fallidos:
        raise SystemExit(1)


# --- SUBCOMANDOS -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="paso", required=True)

    comunes = argparse.ArgumentParser(add_help=False)
    comunes.add_argument("--lista", required=True, help="ruta, o nombre relativo a --salida")
    comunes.add_argument("--rawdata", default=RAW)
    comunes.add_argument("--salida", default="datasets/ldm100k3d")
    comunes.add_argument("--trabajo", default=None,
                         help="máscaras en el espacio del NIfTI; por defecto, "
                              "<salida>/mascaras-nifti")
    herramienta = argparse.ArgumentParser(add_help=False, parents=[comunes])
    herramienta.add_argument("--disp", choices=("cuda", "cpu"), default="cuda")
    herramienta.add_argument("--hilos", type=int, default=16, help="sólo con --disp cpu")

    d = sub.add_parser("deepbet", parents=[herramienta],
                       help="deepbet 1.0.2, en $H/deepbet/bin/python")
    d.set_defaults(fn=deepbet)

    h = sub.add_parser("hdbet", parents=[herramienta],
                       help="HD-BET 2.0.1 sin TTA, en $H/hdbet/bin/python")
    h.add_argument("--herramientas", default=None,
                   help="H: los pesos van en H/hdbet_params/release_2.0.0")
    h.add_argument("--params", default=None, help="la carpeta de pesos, si no es la de H")
    h.set_defaults(fn=hdbet)

    c = sub.add_parser("combina", parents=[comunes],
                       help="intersección, recorte y verificación, en el .venv del proyecto")
    c.add_argument("--split", required=True)
    c.add_argument("--deepbet", default=None,
                   help="carpeta de máscaras de deepbet (.npz o .npy bool), si no es la de "
                        "--trabajo")
    c.add_argument("--hdbet", default=None, help="lo mismo para HD-BET sin TTA")
    c.add_argument("--procesos", type=int, default=12)
    c.set_defaults(fn=combina)

    args = p.parse_args(argv)
    if args.paso == "hdbet" and not (args.herramientas or args.params):
        p.error("hdbet: falta --herramientas (o --params), donde están los pesos")
    args.fn(args)


if __name__ == "__main__":
    main()

'''
# --- PASOS 1 Y 2: LAS DOS HERRAMIENTAS -------------------------------------------------------
# deepbet 1.0.2 y HD-BET 2.0.1 (sobre nnU-Net 2.8.1), cada una en un entorno de uv con python
# 3.12 y torch 2.14.1: no se instalan en el .venv del proyecto (python 3.13, torch 2.13). deepbet
# se llama con `BrainExtraction.run` y sus defaults; HD-BET, con el `nnUNetPredictor` que arma su
# CLI, en memoria y sin espejado. Verificado el 2026-10-09 sobre 2
# sujetos: las CLI oficiales dan las mismas máscaras vóxel a vóxel y el afín del NIfTI original;
# CPU contra GPU difieren en 20–36 vóxeles por cerebro (Dice 0.99999). Las dos máscaras salen en
# el espacio del NIfTI, [160, 224, 160] LAS a 1 mm.
# Sin TTA, HD-BET cuesta 5x menos (0.51 contra 2.53 s en GPU) y su Dice contra la versión con TTA
# es 0.9972 [0.9958–0.9978] en 24 sujetos: con el margen de 1 vóxel de `ddr.data3d`, el 0.0 %
# [0.0–0.3] de los cubos 48^3 que salen de esta intersección toca algo que la de con TTA deja
# afuera (docs/bitacora.md, «Datos: LDM_100k en 3D»). Costo derivado de esos tiempos: 0.67 s por
# cerebro en GPU, ~0.4 h para 2 000 y ~1.5 h para 8 000; en CPU, 12 h y 50 h.

---------------------------------------------------------------------------------------------------

# --- PASO 3: INTERSECCIÓN Y RECORTE ----------------------------------------------------------
# deepbet es una máscara intracraneal: llega a la tabla interna y mete 69–178 ml que HD-BET deja
# afuera (LCR externo, la zona de los senos, la cisterna magna, un foco brillante de la base);
# HD-BET sigue la superficie pial. La intersección mide casi lo mismo que HD-BET (1 387 contra
# 1 389 ml de mediana) y, con margen 1, quedó sobre corteza en todas las vistas de borde miradas
# (medido el 2026-10-09). Los huecos se rellenan como en la medición
# (`binary_fill_holes`); en los 24 medidos no había ninguno, y `combina` imprime cuántos vóxeles
# rellenó. El recorte es `caja_tejido(normaliza(v) > UMBRAL_TEJIDO, MARGEN)`, la caja con que
# `cachea_uno` escribió el volumen: `combina` lo vuelve a calcular y exige que el `.npy` cacheado
# sea, bit a bit, `normaliza(v)[caja]` en float16, así que máscara y volumen quedan alineados vóxel
# a vóxel por construcción y verificados. Ningún vóxel de cerebro cae fuera del recorte en los 24
# (la columna `fuera_del_recorte` lo vigila). La salida es bool sin comprimir, ~5.2 MB por sujeto:
# `train3d.py` las lee una por vez al arrancar y se queda sólo con las esquinas válidas.
'''
