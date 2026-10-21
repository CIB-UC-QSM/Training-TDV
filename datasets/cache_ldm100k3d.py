"""Cache de volúmenes 3D a partir de LDM_100k (T1w sintéticas, 100 000 sujetos en BIDS).

Tres pasos, en orden. `escanea` describe los 100 000 con una miniatura y escalares de control;
`selecciona` elige sobre esa descripción un conjunto de evaluación representativo y listas de
entrenamiento de máxima variedad; `cachea` normaliza y recorta los sujetos de una lista y los
escribe en el layout `<salida>/images/<split>/*.npy` que leen `ddr.data3d` y `train3d.py`.

    python datasets/cache_ldm100k3d.py escanea --salida datasets/ldm100k3d/escaneo
    python datasets/cache_ldm100k3d.py selecciona --escaneo datasets/ldm100k3d/escaneo \\
        --n 2000 8000 --n-eval 12
    python datasets/cache_ldm100k3d.py cachea --lista lista_eval.txt --split val
    python datasets/cache_ldm100k3d.py cachea --lista lista_train_2000.txt --split train

Los `.npy` salen en float16 `[D, H, W]` y en [0, 1], con `normaliza` (percentil 99.5 de los
vóxeles positivos a 1.0). Ojo: el volumen conserva el cráneo, que LDM_100k no saca; esto es T1, no
QSM.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

# --- NORMALIZACIÓN ---------------------------------------------------------------------------
P_NORM = 99.5           # percentil de los vóxeles positivos que se manda a 1.0
UMBRAL_TEJIDO = 0.05    # por encima de esta fracción de la norma, un vóxel cuenta como tejido


def normaliza(vol: np.ndarray) -> np.ndarray:
    """A [0, 1] dividiendo por el percentil P_NORM de los vóxeles positivos.

    Por el máximo sería frágil: hay vóxeles calientes sueltos en 6.55e4, el techo del uint16.
    """
    pos = vol[vol > 0]
    escala = np.percentile(pos, P_NORM) if pos.size else 1.0
    return np.clip(vol / max(escala, 1e-6), 0.0, 1.0)


# --- ESCANEO ---------------------------------------------------------------------------------
RAW = "/home/santicien/Public/LDM-data/LDM_100k/LDM_100k/rawdata"   # en la máquina del autor
N_SUJETOS = 100_000              # sub-000000 a sub-099999
FORMA = (160, 224, 160)          # 1 mm isotrópico, los 100 000 alineados entre sí
BLOQUE = 8                       # miniatura 20x28x20: promedios de 8 mm, comparables entre sujetos
SATURADO = 65_000                # bajo el techo del uint16 (65 535), donde hay vóxeles calientes
OSCURO = (0.02, 0.3)             # LCR en T1 normalizado; el piso deja afuera el aire (p99 0.0098)
CUANTILES_CAJA = (10, 50, 90)
ESCALARES = ("p995_crudo", "frac_tejido", "frac_oscura_caja", "q10_caja", "q50_caja", "q90_caja",
             "saturados")

# --- SELECCIÓN -------------------------------------------------------------------------------
# Los de `scripts/selecciona_ldm100k.py`, con que se midió la selección, salvo Z_CALIDAD y N_TRAIN
# (allá `--n 500 2000 8000`): cambiar uno cambia las listas.
SEMILLA = 20261008
N_TRAIN = (2000, 8000)           # 2 000 entran en RAM (~19 GB); 8 000 (~78 GB), no
N_EVAL = 12                      # 12 recortes de 128^3: ≤ 1 min de evaluación a 2.26 µs/vóxel
COMPONENTES = 32
# Ningún volumen de los 100 000 pasa de z = 8.1; a 6 salían 31 con anatomía plausible, la cola que
# se busca (medido el 2026-10-08). A 10 no sale ninguno: queda de salvaguarda.
Z_CALIDAD = 10.0
MUESTRA = 20_000                 # filas para ajustar la PCA y para el k-means
ITER_KMEANS = 50
MAD_SD = 1.4826                  # 1/Φ⁻¹(3/4): lleva la MAD de una normal a su desviación estándar

# --- CACHE -----------------------------------------------------------------------------------
MARGEN = 4      # vóxeles de aire alrededor de la caja del tejido; medido, en el bloque final


# --- CAJAS Y RUTAS ---------------------------------------------------------------------------

def caja_central(forma: tuple[int, ...]) -> tuple[slice, ...]:
    """La mitad central de cada eje: (40:120, 56:168, 40:120) en FORMA, el núcleo del cerebro."""
    return tuple(slice(n // 4, 3 * n // 4) for n in forma)


def caja_tejido(tejido: np.ndarray, margen: int) -> tuple[slice, ...] | None:
    """Caja mínima que contiene a `tejido`, agrandada `margen` vóxeles y recortada al volumen."""
    caja = []
    for eje, n in enumerate(tejido.shape):
        idx = np.flatnonzero(tejido.any(axis=tuple(a for a in range(tejido.ndim) if a != eje)))
        if not idx.size:
            return None
        caja.append(slice(max(int(idx[0]) - margen, 0), min(int(idx[-1]) + 1 + margen, n)))
    return tuple(caja)


def nifti(raw: str | Path, sujeto: str) -> Path:
    return Path(raw) / sujeto / "anat" / f"{sujeto}_T1w.nii.gz"


# --- PASO 1: ESCANEO -------------------------------------------------------------------------

def describe(tarea: tuple) -> tuple[int, np.ndarray | None, np.ndarray | None]:
    """Un sujeto: miniatura float16 y escalares. `None` si no se lee o no tiene la forma común."""
    i, raw, forma = tarea
    try:
        crudo = np.asarray(nib.load(nifti(raw, f"sub-{i:06d}")).dataobj)
    except Exception:
        return i, None, None
    if crudo.shape != forma:
        return i, None, None
    v = crudo.astype(np.float32)
    pos = v[v > 0]
    p995 = float(np.percentile(pos, P_NORM)) if pos.size else 1.0
    v = np.clip(v / max(p995, 1e-6), 0.0, 1.0)
    d, h, w = (n // BLOQUE for n in forma)
    mini = v.reshape(d, BLOQUE, h, BLOQUE, w, BLOQUE).mean((1, 3, 5)).astype(np.float16)
    caja = v[caja_central(forma)]
    q10, q50, q90 = np.percentile(caja, CUANTILES_CAJA)
    oscura = ((caja > OSCURO[0]) & (caja < OSCURO[1])).mean()
    esc = np.array([p995, (v > UMBRAL_TEJIDO).mean(), oscura, q10, q50, q90,
                    int((crudo >= SATURADO).sum())], dtype=np.float32)
    return i, mini, esc


def escanea(args: argparse.Namespace) -> None:
    out = Path(args.salida)
    out.mkdir(parents=True, exist_ok=True)
    n, forma = args.n, tuple(args.forma)
    # A disco a medida que llegan: las 100 000 miniaturas son 2.2 GB.
    minis = np.lib.format.open_memmap(out / "miniaturas.npy", mode="w+", dtype=np.float16,
                                      shape=(n, *(k // BLOQUE for k in forma)))
    esc = np.full((n, len(ESCALARES)), np.nan, dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    t0 = time.time()
    tareas = ((i, args.rawdata, forma) for i in range(n))
    with ProcessPoolExecutor(max_workers=args.procesos) as pool:
        for k, (i, mini, e) in enumerate(pool.map(describe, tareas, chunksize=64), 1):
            if mini is not None:
                minis[i], esc[i], ok[i] = mini, e, True
            if k % 5000 == 0:
                print(f"  {k}/{n} ({time.time() - t0:.0f} s, {(~ok[:k]).sum()} fallidos)",
                      flush=True)
    minis.flush()
    np.savez(out / "escalares.npz", escalares=esc, ok=ok, nombres=np.array(ESCALARES))
    print(f"{ok.sum()}/{n} volúmenes en {time.time() - t0:.0f} s -> {out}", flush=True)


# --- PASO 2: SELECCIÓN -----------------------------------------------------------------------

def mad_z(x: np.ndarray) -> np.ndarray:
    """|x - mediana| en desviaciones robustas, por columna. Una columna sin dispersión da 0."""
    med = np.median(x, 0)
    mad = np.median(np.abs(x - med), 0) * MAD_SD
    return np.abs(x - med) / np.where(mad > 0, mad, np.inf)


def fps(X: np.ndarray, n: int, inicio: int) -> np.ndarray:
    """k-center voraz (farthest point sampling): el siguiente es el más lejano de los elegidos."""
    elegidos = [inicio]
    d = np.linalg.norm(X - X[inicio], axis=1)
    for _ in range(n - 1):
        k = int(np.argmax(d))
        elegidos.append(k)
        d = np.minimum(d, np.linalg.norm(X - X[k], axis=1))
    return np.array(elegidos)


def cobertura(X: np.ndarray, idx: np.ndarray, bloque: int = 4096) -> np.ndarray:
    """Distancia de cada fila de X al elemento más cercano de X[idx]."""
    C = X[idx]
    c2 = (C ** 2).sum(1)
    out = np.empty(len(X), dtype=np.float32)
    for a in range(0, len(X), bloque):
        x = X[a:a + bloque]
        d2 = (x ** 2).sum(1)[:, None] - 2 * x @ C.T + c2[None]
        out[a:a + bloque] = np.sqrt(np.maximum(d2.min(1), 0))
    return out


def kmeans(X: np.ndarray, k: int, rng: np.random.Generator,
           iters: int = ITER_KMEANS) -> np.ndarray:
    """Centroides de Lloyd desde k filas al azar. Un grupo que se vacía conserva su centroide."""
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        etiq = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), 1)
        C = np.stack([X[etiq == j].mean(0) if (etiq == j).any() else C[j] for j in range(k)])
    return C


def selecciona(args: argparse.Namespace) -> None:
    raiz, out = Path(args.escaneo), Path(args.salida)
    out.mkdir(parents=True, exist_ok=True)
    # Un solo generador para todo, en este orden: muestra de la PCA, k-means y un sorteo por N.
    rng = np.random.default_rng(args.seed)
    e = np.load(raiz / "escalares.npz")
    ok = e["ok"]
    mini = np.load(raiz / "miniaturas.npy", mmap_mode="r")
    ids = np.flatnonzero(ok)
    print(f"{ok.sum()} volúmenes leídos de {len(ok)}")

    # --- PCA ---------------------------------------------------------------------------------
    # Sin blanquear: la distancia en el espacio PCA aproxima la L2 entre volúmenes de 8 mm.
    muestra = rng.choice(ids, size=min(MUESTRA, len(ids)), replace=False)
    M = mini[np.sort(muestra)].reshape(len(muestra), -1).astype(np.float32)
    media = M.mean(0)
    _, sv, Vt = np.linalg.svd(M - media, full_matrices=False)
    V = Vt[:args.componentes].T
    var = sv ** 2
    print(f"PCA: {args.componentes} componentes explican "
          f"{var[:args.componentes].sum() / var.sum():.1%} de la varianza de las miniaturas "
          f"(1a: {var[0] / var.sum():.1%})")
    X = np.empty((len(ids), args.componentes), dtype=np.float32)
    for a in range(0, len(ids), 8192):
        bloque = mini[ids[a:a + 8192]].reshape(-1, M.shape[1]).astype(np.float32)
        X[a:a + 8192] = (bloque - media) @ V

    # --- CONTROL DE CALIDAD ------------------------------------------------------------------
    # Un volumen generado mal es lo primero que elegiría un muestreo por distancia máxima.
    esc = e["escalares"][ids].copy()
    # el conteo de saturados tiene cola pesada sana (p1 4, p99 688 en 2 000): se juzga en log
    j_sat = list(e["nombres"]).index("saturados")
    z_cols = mad_z(np.column_stack([esc[:, :j_sat], np.log1p(esc[:, j_sat:])]))
    z_esc = z_cols.max(1)
    z_pca = np.linalg.norm(X - np.median(X, 0), axis=1)
    z_pca = (z_pca - np.median(z_pca)) / (np.median(np.abs(z_pca - np.median(z_pca))) * MAD_SD)
    fuera = (z_esc > args.z) | (z_pca > args.z)
    print(f"control de calidad: {fuera.sum()} fuera ({(z_esc > args.z).sum()} por escalares, "
          f"{(z_pca > args.z).sum()} por PCA). Los diez peores por PCA: "
          f"{[f'sub-{i:06d}' for i in ids[np.argsort(-z_pca)[:10]]]}")
    for j, nombre in enumerate(e["nombres"]):
        q = np.percentile(esc[:, j], (1, 50, 99))
        print(f"  {nombre:>17}: p1 {q[0]:.4g}  p50 {q[1]:.4g}  p99 {q[2]:.4g}  "
              f"fuera por este: {(z_cols[:, j] > args.z).sum()}")
    vivos = np.flatnonzero(~fuera)
    Xv = X[vivos]

    # --- EVALUACIÓN --------------------------------------------------------------------------
    # El sujeto más cercano a cada centroide: representativos de los modos, no extremos.
    C = kmeans(Xv[rng.choice(len(Xv), min(MUESTRA, len(Xv)), replace=False)], args.n_eval, rng)
    eval_loc = np.unique([int(np.argmin(((Xv - c) ** 2).sum(1))) for c in C])
    resto = np.setdiff1d(np.arange(len(vivos)), eval_loc)
    print(f"evaluación: {len(eval_loc)} sujetos")
    (out / "lista_eval.txt").write_text("".join(f"sub-{ids[vivos[i]]:06d}\n" for i in eval_loc))

    # --- ENTRENAMIENTO -----------------------------------------------------------------------
    Xr = Xv[resto]
    inicio = int(np.argmin(((Xr - Xr.mean(0)) ** 2).sum(1)))
    orden = fps(Xr, max(args.n), inicio)
    # FPS cubre las colas y el sorteo sigue la densidad; la mezcla toma la mitad de cada uno. Las
    # otras dos reglas se sortean igual aunque no se escriban: consumen el generador.
    print("\ncobertura = distancia de cada sujeto del corpus a su elegido más cercano (PCA)")
    print(f"{'N':>6} {'regla':>8} {'mediana':>9} {'p99':>9} {'max':>9}")
    for n in args.n:
        sorteo = rng.permutation(np.setdiff1d(np.arange(len(Xr)), orden[:n // 2]))
        reglas = {"fps": orden[:n], "sorteo": rng.choice(len(Xr), n, replace=False),
                  "mezcla": np.concatenate([orden[:n // 2], sorteo[:n - n // 2]])}
        for regla, idx in reglas.items():
            c = cobertura(Xv, resto[idx])
            print(f"{n:>6} {regla:>8} {np.median(c):>9.3f} {np.percentile(c, 99):>9.3f} "
                  f"{c.max():>9.3f}")
        (out / f"lista_train_{n}.txt").write_text(
            "".join(f"sub-{ids[vivos[resto[i]]]:06d}\n" for i in reglas["mezcla"]))
    np.savez(out / "seleccion.npz", ids=ids, X=X, fuera=fuera, vivos=vivos, eval_loc=eval_loc,
             resto=resto, orden_fps=orden, media=media, V=V)
    print(f"listas -> {out}", flush=True)


# --- PASO 3: CACHE ---------------------------------------------------------------------------

def cachea_uno(tarea: tuple) -> tuple[str, tuple[int, ...] | None]:
    """Un sujeto: normaliza, recorta a la caja del tejido con MARGEN y escribe float16."""
    sujeto, raw, destino = tarea
    try:
        vol = normaliza(np.asarray(nib.load(nifti(raw, sujeto)).dataobj, dtype=np.float32))
    except Exception:
        return sujeto, None   # un volumen corrupto no justifica matar la extracción
    caja = caja_tejido(vol > UMBRAL_TEJIDO, MARGEN)
    if caja is None:
        return sujeto, None
    recorte = vol[caja].astype(np.float16)
    np.save(Path(destino) / f"{sujeto}.npy", recorte)
    return sujeto, recorte.shape


def cachea(args: argparse.Namespace) -> None:
    salida = Path(args.salida)
    lista = Path(args.lista)
    if not lista.exists():
        lista = salida / lista
    sujetos = [l.strip() for l in lista.read_text().splitlines() if l.strip()]
    destino = salida / "images" / args.split
    destino.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    formas, fallidos = [], []
    tareas = [(s, args.rawdata, str(destino)) for s in sujetos]
    with ProcessPoolExecutor(max_workers=args.procesos) as pool:
        for k, (sujeto, forma) in enumerate(pool.map(cachea_uno, tareas, chunksize=8), 1):
            if forma is None:
                fallidos.append(sujeto)
            else:
                formas.append(forma)
            if k % 500 == 0:
                print(f"  {k}/{len(tareas)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"{args.split}: {len(formas)}/{len(sujetos)} volúmenes en {time.time() - t0:.0f} s -> "
          f"{destino}" + (f" | fallidos: {fallidos[:10]}" if fallidos else ""), flush=True)
    if formas:
        F = np.array(formas)
        print(f"  caja con margen {MARGEN}: min {F.min(0).tolist()} mediana "
              f"{np.median(F, 0).astype(int).tolist()} max {F.max(0).tolist()} | "
              f"{F.prod(1).sum() * 2 / 2**30:.2f} GB en float16", flush=True)


# --- SUBCOMANDOS -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="paso", required=True)

    e = sub.add_parser("escanea", help="miniaturas 8^3 y escalares de control de cada sujeto")
    e.add_argument("--rawdata", default=RAW)
    e.add_argument("--salida", default="datasets/ldm100k3d/escaneo")
    e.add_argument("--n", type=int, default=N_SUJETOS, help="primeros N sujetos, para cronometrar")
    e.add_argument("--forma", type=int, nargs=3, default=FORMA,
                   help="forma común de los volúmenes; otra, sólo para tests sintéticos")
    e.add_argument("--procesos", type=int, default=8)
    e.set_defaults(fn=escanea)

    s = sub.add_parser("selecciona", help="evaluación por k-means y listas de entrenamiento")
    s.add_argument("--escaneo", default="datasets/ldm100k3d/escaneo",
                   help="donde están miniaturas.npy y escalares.npz")
    s.add_argument("--salida", default="datasets/ldm100k3d")
    s.add_argument("--n", type=int, nargs="+", default=list(N_TRAIN))
    s.add_argument("--n-eval", type=int, default=N_EVAL)
    s.add_argument("--componentes", type=int, default=COMPONENTES)
    s.add_argument("--z", type=float, default=Z_CALIDAD)
    s.add_argument("--seed", type=int, default=SEMILLA)
    s.set_defaults(fn=selecciona)

    c = sub.add_parser("cachea", help="normaliza, recorta y escribe los sujetos de una lista")
    c.add_argument("--lista", required=True, help="ruta, o nombre relativo a --salida")
    c.add_argument("--split", required=True)
    c.add_argument("--rawdata", default=RAW)
    c.add_argument("--salida", default="datasets/ldm100k3d")
    c.add_argument("--procesos", type=int, default=12)
    c.set_defaults(fn=cachea)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

'''
# --- ESCANEO ---------------------------------------------------------------------------------
# `participants.tsv` viene vacío (0 bytes): edad, sexo y volúmenes, las variables con que se
# condicionó el modelo generativo, no están, así que la variedad se mide en las imágenes. Por
# sujeto, sobre el volumen normalizado: una miniatura de 20x28x20 por promedio en bloques de 8^3
# (los volúmenes están alineados, así que es comparable vóxel a vóxel) y siete escalares:
# percentil 99.5 crudo, fracción de tejido, fracción oscura (LCR) y cuantiles 10/50/90 en la caja
# central (proxy de ventrículos y atrofia), y vóxeles saturados. Un sujeto que no se lee o no mide
# FORMA queda con `ok = False` y fuera de todo lo que sigue. Salida: `miniaturas.npy` (float16,
# [N, 20, 28, 20]) y `escalares.npz` (escalares, ok, nombres). Leer un volumen cuesta ~0.025 s.
# La normalización del escaneo es la de `normaliza`, escrita de nuevo sólo porque además necesita
# `p995` como escalar. Las dos dividen en float32 (`np.percentile` de un float32 devuelve float32,
# y pasarlo por un float de Python es exacto) y dan los mismos bits: verificado el 2026-10-08 sobre
# 20 volúmenes uint16 sintéticos, 0 valores distintos.

---------------------------------------------------------------------------------------------------

# --- SELECCIÓN -------------------------------------------------------------------------------
#   1. Control de calidad: fuera los escalares a más de Z_CALIDAD desviaciones robustas (MAD) de la
#      mediana, y las miniaturas a más de Z_CALIDAD del centro en el espacio PCA.
#   2. PCA de las miniaturas, ajustada sobre MUESTRA sujetos y proyectada sobre todos.
#   3. Evaluación: el sujeto más cercano a cada centroide de un k-means de `--n-eval` grupos. Si
#      dos centroides comparten el más cercano, salen menos que `--n-eval`.
#   4. Entrenamiento, regla `mezcla`: la mitad de cada N por k-center voraz sobre el resto (cubre
#      las colas) y la otra mitad sorteada sin repetir entre los que el voraz no tomó (sigue la
#      densidad). Se imprime la cobertura —distancia de cada sujeto a su elegido más cercano— de
#      las tres reglas, `fps`, `sorteo` y `mezcla`, y se escribe sólo la mezcla, como
#      `lista_train_<N>.txt` (`scripts/selecciona_ldm100k.py` la llamaba `lista_train_mezcla_<N>`).
# Evaluación y entrenamiento son disjuntos por construcción. Todo sale de un generador con
# `--seed`, consumido en orden: la lista de un N depende de la lista `--n` entera, porque los
# sorteos de los N anteriores van antes: `--n 2000 8000` y `--n 500 2000 8000` dan listas de 2 000
# distintas. Las listas de 2 000 y de 8 000 no están anidadas: comparten la mitad voraz, no la
# sorteada.
# Verificado el 2026-10-08, bit a bit: sobre el escaneo completo, `selecciona --z 6 --n 500 2000
# 8000` da las listas mezcla y el `seleccion.npz` de `scripts/selecciona_ldm100k.py`, que se corrió
# así; sobre 600 sujetos, `escanea` da su `miniaturas.npy` y su `escalares.npz`. Con los 100 000,
# todos legibles, a z = 6 quedaban 31 fuera (14 por escalares, 22 por PCA); a Z_CALIDAD = 10,
# ninguno. Costo: el escaneo, 25.5 min con 8 procesos; la selección, 9 min con BLAS en 32 hilos.
# Con los defaults, la cobertura de las listas está en `datasets/ldm100k3d/README.md`.
# Repetida el 2026-10-08 con BLAS en 16 y en 32 hilos, la lista de evaluación y la de 2 000 salen
# idénticas; la de 8 000 tiene los mismos sujetos, pero con dos pares de vecinos del voraz en otro
# orden (X difiere en ≤ 2.9e-6 y el voraz desempata distinto). El orden importa, porque el dataset
# sortea el volumen por posición: la que vale es la lista versionada, no el comando.

---------------------------------------------------------------------------------------------------

# --- CACHE -----------------------------------------------------------------------------------
# Por sujeto de la lista: `normaliza` (percentil 99.5 de los positivos a 1.0, recorte a [0, 1]),
# caja del tejido (> UMBRAL_TEJIDO) agrandada MARGEN vóxeles y recortada al campo, y `.npy` en
# float16. float16 tiene 11 bits de mantisa: el error de redondeo en [0.5, 1] es ≤ 2.4e-4, 400
# veces menor que el ruido de sigma 25 (0.098), y el limpio ES el valor guardado, así que no sesga.
# Medido el 2026-10-08 sobre 200 volúmenes sorteados con `scripts/mide_tejido_ldm100k3d.py`: la
# caja mide 144–160 x 195–223 x 144–154 y en el eje 2 toca el borde del campo en los 200 (con
# cráneo, la cabeza llena el volumen). Fuera de la caja todo vale ≤ 0.05 por construcción, y el
# p99 es 0.0098: el margen agrega aire, no tejido.
# MARGEN = 4 deja aire entre la cabeza y el borde del parche donde el campo lo permite, por 9.9 MB
# de mediana por volumen (10.9 MB el volumen entero, 9.1 MB sin margen); un parche 48^3 sorteado
# pasa el MIN_TEJIDO de `ddr/data3d.py` el 93.5 % de las veces (96.3 % sin margen).
'''