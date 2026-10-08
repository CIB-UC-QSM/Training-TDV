"""Cache de slices 2D a partir de LDM_100k (T1w sintéticas, 100 000 sujetos en BIDS).

Escribe `<salida>/images/<split>/*.npy` en el layout que `ddr.data.list_images` exige, más la
lista de evaluación que `train.py --eval-list` consume. Los `.npy` salen en float32 y en [0, 1],
que es la escala en la que `T` y `lambda` están aprendidos: entrenar en unidades arbitrarias de
T1 y después aplicar en ppm es exactamente el problema que obliga a calibrar `alpha` en `../tdv`.

Un corte por plano y por sujeto, y los sujetos muestreados al azar entre los 100 000: la variedad
está en los sujetos, no en los cortes vecinos de un mismo cerebro, que son casi la misma imagen.
Los sujetos de evaluación son disjuntos de los de entrenamiento por construcción.

    python datasets/cache_ldm100k.py --rawdata <bids>/rawdata --salida datasets/ldm100k
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

# Los tres planos de un volumen 1 mm isotrópico de 160x224x160. El valor es el eje que se colapsa.
PLANOS = {"sag": 0, "cor": 1, "axi": 2}
PATCH_MIN = 93          # PATCH_SIZE de train.py: un corte más chico que esto no sirve
P_NORM = 99.5           # percentil de los voxels positivos que se manda a 1.0
UMBRAL_TEJIDO = 0.05    # por encima de esta fracción de la norma, un píxel cuenta como tejido


def normaliza(vol: np.ndarray) -> np.ndarray:
    """A [0, 1] dividiendo por el percentil P_NORM de los voxels positivos.

    Por el máximo sería frágil: hay voxels calientes sueltos en 6.55e4, el techo del uint16.
    """
    pos = vol[vol > 0]
    escala = np.percentile(pos, P_NORM) if pos.size else 1.0
    return np.clip(vol / max(escala, 1e-6), 0.0, 1.0)


def elige_corte(vol: np.ndarray, eje: int, rng: np.random.Generator,
                min_cerebro: float, intentos: int = 12) -> np.ndarray | None:
    """Un corte al azar del 60 % central del eje, con al menos `min_cerebro` de tejido."""
    n = vol.shape[eje]
    lo, hi = int(0.20 * n), int(0.80 * n)
    for _ in range(intentos):
        sl = np.take(vol, int(rng.integers(lo, hi)), axis=eje)
        if min(sl.shape) >= PATCH_MIN and (sl > UMBRAL_TEJIDO).mean() >= min_cerebro:
            return np.ascontiguousarray(sl, dtype=np.float32)
    return None


def procesa(tarea: tuple) -> tuple[int, list[str]]:
    """Un sujeto: carga, normaliza y escribe un corte por cada plano pedido."""
    sujeto, destino, planos, semilla, min_cerebro = tarea
    nii = next(Path(sujeto).glob("anat/*_T1w.nii.gz"), None)
    if nii is None:
        return 0, []
    try:
        vol = normaliza(np.asarray(nib.load(nii).dataobj, dtype=np.float32))
    except Exception:
        return 0, []   # un volumen corrupto entre 100 000 no justifica matar la extracción
    rng = np.random.default_rng(semilla)
    nombres = []
    for plano in planos:
        sl = elige_corte(vol, PLANOS[plano], rng, min_cerebro)
        if sl is None:
            continue
        nombre = f"{Path(sujeto).name}_{plano}.npy"
        np.save(Path(destino) / nombre, sl)
        nombres.append(nombre)
    return 1, nombres


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rawdata", required=True)
    p.add_argument("--salida", required=True)
    p.add_argument("--sujetos-train", type=int, default=4000)
    p.add_argument("--sujetos-eval", type=int, default=68,
                   help="68 iguala el cardinal de CBSD68, así el costo del eval no cambia")
    p.add_argument("--planos", default="sag,cor,axi")
    p.add_argument("--planos-eval", default="axi",
                   help="un solo plano deja el benchmark homogéneo")
    p.add_argument("--min-cerebro", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=20261007)
    p.add_argument("--procesos", type=int, default=12)
    p.add_argument("--solo-medir", type=int, default=0,
                   help="procesa N sujetos de train y sale, para cronometrar")
    args = p.parse_args()

    planos = args.planos.split(",")
    raiz, salida = Path(args.rawdata), Path(args.salida)
    sujetos = sorted(d.path for d in os.scandir(raiz) if d.is_dir() and d.name.startswith("sub-"))
    print(f"{len(sujetos):,} sujetos en {raiz}", flush=True)

    # Muestreo sin reemplazo con semilla fija: los dos splits son disjuntos por construcción y la
    # elección es reproducible sin guardar la lista de sujetos.
    rng = np.random.default_rng(args.seed)
    n = min(args.sujetos_train + args.sujetos_eval, len(sujetos))
    elegidos = rng.choice(len(sujetos), size=n, replace=False)
    val_idx, train_idx = elegidos[:args.sujetos_eval], elegidos[args.sujetos_eval:]
    if args.solo_medir:
        train_idx, val_idx = train_idx[:args.solo_medir], val_idx[:0]

    for split in ("train", "val"):
        (salida / "images" / split).mkdir(parents=True, exist_ok=True)

    hechos = {}
    for split, idx, pl in (("train", train_idx, planos), ("val", val_idx, [args.planos_eval])):
        destino = salida / "images" / split
        tareas = [(sujetos[i], str(destino), pl, args.seed + int(i), args.min_cerebro) for i in idx]
        vivos, nombres = 0, []
        with ProcessPoolExecutor(max_workers=args.procesos) as pool:
            for k, (ok, ns) in enumerate(pool.map(procesa, tareas, chunksize=8), 1):
                vivos += ok
                nombres.extend(ns)
                if k % 500 == 0:
                    print(f"  {split}: {k}/{len(tareas)} sujetos, {len(nombres)} cortes", flush=True)
        hechos[split] = nombres
        print(f"{split}: {vivos}/{len(tareas)} sujetos leídos, {len(nombres)} cortes", flush=True)

    if hechos.get("val"):
        lista = salida / "lista_eval.txt"
        lista.write_text("\n".join(sorted(hechos["val"])) + "\n")
        print(f"lista de evaluación -> {lista}", flush=True)


if __name__ == "__main__":
    main()
