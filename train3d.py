"""
Entrenamiento de TDV 3D sobre volúmenes T1 de LDM_100k: la receta de `train.py` llevada a 3D.

Una iteración toma un lote de 32 parches de 48^3, cada uno con una de las 48 simetrías del cubo, y
lo pasa en los micro-lotes que entran en la memoria (16 en 96 GiB): ruido gaussiano nuevo
(sigma = 25) -> los S = 10 pasos del esquema desenrollado -> ||x_S - y||^2 -> backward -> el
disyuntor decide si el lote se acepta -> Adam con el lr del calendario -> re-proyección. Cada
--eval-every valida el denoising y, con la cadencia de --eval-qsm-cada, pasa la red por el ADMM de
QSM sobre COSMOS: ese NRMSE elige `best_qsm.pth` y, con --meseta, decide cuándo bajar el lr.

Cómo leer este archivo: empieza por `main`, al final, que arma la corrida en pasos con nombre y
llama a `entrena`, el bucle, y a `cierra`. Lo que se arma una vez va en `piezas`; las cuentas que
el bucle cambia y que viajan en `estado-<it>.pth`, en `corrida` (`corrida_nueva`). Cada función
está antes de quien la llama: subiendo desde `main` aparecen los pasos del bucle, los de la
preparación, `corrida_nueva`, las banderas (`opciones`; las principales, también en el README), la
sonda de memoria, la evaluación, los archivos, las funciones chicas y las constantes. El porqué de
cada número, en docs/porque-del-codigo.md, «## train3d.py».

    uv run train3d.py --out runs/sonda --sonda-vram                  # una vez por tarjeta: el pico
    uv run train3d.py --out runs/tronco3d --hitos                    # tronco a lr constante
    uv run train3d.py --out runs/rama3d-10k --reanudar runs/tronco3d/estado-010000.pth \\
        --iterations 15000 --hitos 10000                             # rama: lr/2 desde la 10 001
    uv run train3d.py --out runs/tronco3d --reanudar runs/tronco3d/estado-020000.pth \\
        --hitos                                                      # sigue tras un corte
    uv run train3d.py --out runs/ciego3d --sigma-max 55              # ruido ciego U(0, 55)
    uv run train3d.py --out runs/cerebro3d --hitos --mascara-cerebro # cubos 100 % cerebro
    uv run train3d.py --out runs/tronco3d --hitos --eval-qsm-cada 10000   # QSM cada 10 000
    uv run train3d.py --out runs/tronco3d --hitos --meseta 2         # lr/2 si el QSM se estanca
    uv run train3d.py --out runs/tronco3d --hitos --eval-every 100 --eval-qsm-cada 500 \\
        --meseta 10 --meseta-desde 10000 --meseta-psnr               # ... y el PSNR, desde la 10 000
    uv run train3d.py --out runs/sobreajuste --sobreajuste --lote 4 --hitos --iterations 2000 \\
        --eval-every 250 --qsm-datos '' --estabilidad-lado 0         # prueba de sobreajuste
    wandb sync runs/tronco3d/wandb/offline-run-*                     # subir una corrida offline

Ojo con `--hitos`: sin la bandera el calendario es el del paper (lr/2 cada 25 000); `--hitos` sola
deja el lr constante, y `--hitos 10000` lo divide a la mitad desde la 10 001. Y con `--vram-gb`:
va en GiB, como los reporta torch, que da algo menos que la etiqueta (la 5070 Ti de «16 GB»
reporta 15.45), y el default se recorta a eso; `--vram-gb 44.7` sólo si la porción de una tarjeta
compartida se pactó en GB decimales.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn.utils import get_total_norm
from torch.utils.data import DataLoader, Subset

from ddr import seguimiento
from ddr.data3d import EVAL_RECORTE, PatchDataset3D, VolumeDataset3D, add_noise, add_noise_ciego
from ddr.model import VNet3D
from ddr.qsm3d import (LADO_ESTABILIDAD, ORIENTACION_T1, SIM2_ARCHIVO, TARGET_STD,
                       UMBRAL_ESTABILIDAD, admm_wtdv, carga_cosmos, carga_sim2, denoiser_3d,
                       estabilidad, lamina_denoising, lamina_qsm, metricas_chi, punto_qsm3d,
                       residuo_dato_pct)
from evaluate import evaluate_denoising3d
from train import (BETAS, DATA_RANGE, DELTA_ADAM, FEATURES, GUARD_LIMPIAS,
                   GUARD_SEGUIDOS, GUARD_VENTANA, LAMBDA_INIT, LR, LR_HALVE_EVERY,
                   MACRO_BLOCKS, MODES, NOISE_SCALE, SCALES, SIGMA, STEPS, T_INIT, rebobina,
                   rechaza_lote, respalda, vnet_config)

# --- RECETA 3D -------------------------------------------------------------------------------
# Lo del paper (lr, betas, sigma, S, TDV33, inicializaciones, disyuntor y modos) se importa de
# `train.py`, que es la única fuente; también lo que acá no se usa (SCALES, MACRO_BLOCKS,
# DELTA_ADAM), porque los tests verifican que la receta sea esa. Lo que la tercera dimensión
# obliga a cambiar:
PATCH_SIZE = 48              # el menor costo por vóxel medido en la 5070 Ti: 2.26 µs a 48^3 x 4
LOTE = 32                    # los parches del paper (sec. 5.1), no su contenido: 4.3x los valores
MICRO_LOTE = "auto"          # es hardware, no receta: el que parte LOTE en menos pasadas (MEMORIA)
ITERACIONES_TANDA = 40_000   # el tronco del piloto
GUARDAR_CADA = 10_000        # los puntos de rama del piloto: 10 000, 20 000 y 30 000

# --- MEMORIA ---------------------------------------------------------------------------------
# El pico de `torch.cuda.max_memory_allocated` del entrenamiento, sin la evaluación, en GiB:
# FIJA + micro-lote · P^3 · POR_VOXEL. Ajustado a lo medido en la 5070 Ti (bitácora, «Costo y
# memoria», 2026-10-08): en otra tarjeta es una extrapolación, y lo que de verdad entra lo mide
# `--sonda-vram`.
VRAM_GB = 96.0               # la tarjeta de destino, RTX PRO 6000 Blackwell Max-Q de 96 GB, en GiB
FIJA = 0.61                  # la recta de 48^3 x 1 y x 4 en `rapido` (4.14 y 14.72 GB) en n = 0
POR_VOXEL = {                # GiB por vóxel del micro-lote, por (modo, checkpointing)
    ("rapido", False): 3.19e-5,        # la pendiente de esa recta: 3.527 GiB por parche de 48^3
    ("intermedio", False): 7.0e-5,     # sobre los 4 medidos de los dos modos: 2.48 a 7.60 GB
    ("referencia", False): 7.0e-5,
    # Medidos con los S pasos checkpointeados. Sin --guard-amp el primero va sin checkpoint, y
    # `vram_predicha` le suma lo suyo.
    ("rapido", True): 3.4e-6,          # 64^3 x 4: 4.11 GB
    ("intermedio", True): 5.6e-6,      # sin medir: el de `referencia`
    ("referencia", True): 5.6e-6,      # 64^3 x 2: 3.56 GB
}
MARGEN = 0.10                # supuesto: evaluación, fragmentación y cuDNN de otra tarjeta
# `--sonda-vram`, por micro-lote: la primera iteración compila y deja a cuDNN elegir algoritmos, y
# no se cronometra; las otras tres dan los segundos por parche. Ordena micro-lotes; no reemplaza
# medir s/it en la corrida.
SONDA_ITERACIONES = 4
SONDA_DESCARTADAS = 1

# --- DISYUNTOR EN 3D -------------------------------------------------------------------------
# El de train.py con la razón, la cota y el respaldo medidos en 3D (docs/porque-del-codigo.md,
# DISYUNTOR EN 3D): razón 10, sin cota absoluta (en 3D sólo decidía en el arranque, y lo trababa),
# y el respaldo cada 50 iteraciones sanas, sólo con el relativo armado.
GUARD_3D = 10.0
GUARD_RESPALDO_3D = 50
# La alerta de W&B por la validación: el PSNR que elige `best.pth` al menos esto bajo el mejor
# visto. Supuesto: con la semilla fija de la validación, una corrida sana no cae tanto.
CAIDA_PSNR = 3.0

# --- DATOS -----------------------------------------------------------------------------------
DATA_ROOT = "datasets/ldm100k3d"
TRAIN_SPLIT, EVAL_SPLIT = "train", "val"
TRAIN_LIST = "datasets/ldm100k3d/lista_train_2000.txt"   # 2 000 entran en RAM (~19 GB); 8 000 no
EVAL_LIST = "datasets/ldm100k3d/lista_eval.txt"          # 12, uno por centroide de un k-means
CHANNELS = 1                 # T1 en gris
SIGMA_MIN = 0.0              # ruido ciego U(0, 55): la convención de DnCNN-B (Zhang et al., 2017)
# Un parche cuesta 0.31 ms con la CPU libre y 0.81 ms cargada: los 32 de una iteración, 26 ms en
# el peor caso medido, contra ~8 s de GPU. Uno alcanzaría; dos, porque la tarjeta de destino está
# en otra máquina, con otra CPU, y el segundo cuesta poco.
WORKERS = 2

# --- EVALUACIÓN QSM --------------------------------------------------------------------------
# El protocolo y sus constantes son los de `ddr/qsm3d.py`, salvo la parada: un techo de 25
# iteraciones, sin el corte por umbral del autor, y el ADMM para cuando el NRMSE sube dos
# iteraciones seguidas, con la mejor reconstrucción que vio (decisiones del 2026-10-19; bitácora,
# «La evaluación QSM»).
QSM_MAX_ITER = 25
QSM_SUBIDAS = 2
QSM_DATOS = "datasets/qsm_cosmos"
QSM_ARCHIVOS = ("params.mat", "chi_cosmos.mat")   # los que lee `carga_cosmos`
QSM_ORIGEN = "../tdv/data_mat"                    # de dónde se copian: el repo del pipeline QSM
# Sim2, el segundo sujeto de la evaluación QSM (`--qsm-sim2`): lo arma `datasets/prepara_sim2.py`.
QSM_SIM2 = "datasets/qsm_sim2"
# --val-snr: SNR = pico/σ, con el pico = DATA_RANGE = 1, el percentil 99.5 al que se normaliza
# cada T1: σ = 1/SNR. Es la convención del PSNR (PSNR de entrada = 20·log10 SNR), y σ = 25/255, la
# de la validación, es SNR 10.2 (AGENTS.md, «Entrenamiento 3D»).
# Lo de `best_qsm.pth` que va a `estado.json`: la iteración, lo que lo eligió (el NRMSE mínimo del
# ADMM) y lo que se registra de diagnóstico.
RESUMEN_BEST_QSM = ("iteration", "nrmse_min_pct", "nrmse_final_pct", "iteraciones", "motivo",
                    "estabilidad")

# --- ESTABILIDAD -----------------------------------------------------------------------------
# El número del paso explícito, (T/S)·λ_max(∇²R), de `ddr/qsm3d.estabilidad`. La semilla es la de
# la validación (`evaluate_denoising3d`): el punto de validación lleva el mismo ruido que ella.
SEMILLA_VAL = 1234

# --- MESETA ----------------------------------------------------------------------------------
# Con --meseta N decide el NRMSE QSM (el mínimo del ADMM): N evaluaciones seguidas sin bajar del
# mejor visto dan lr/2 desde la iteración siguiente, y agotadas las mitades la meseta siguiente
# corta la corrida. Con --meseta-psnr, además sin récord del PSNR de la validación; con
# --meseta-desde, sólo desde esa iteración. El tope de 5 es el que se pidió el 2026-10-16.
MESETA_MITADES = 5

# --- REANUDACIÓN -----------------------------------------------------------------------------
# LIBRES: las banderas que se pueden cambiar al reanudar. Ninguna toca la aritmética (las del
# seguimiento, la evaluación QSM y la estabilidad, con test), salvo `micro_lote` y `vram_gb`, que
# lo elige con `auto`: cambian el orden de suma del gradiente y, a veces, el ruido o la
# amplificación (el AVISO de `lee_estado`), no la receta. Se permiten para seguir en otra tarjeta,
# y desde ahí la corrida no es bit a bit. `guard`, `guard_abs` y `guard_respaldo` tampoco la tocan
# mientras el disyuntor no rechaza, y se permiten para reanudar protegida una corrida lanzada sin
# él. CALENDARIO: lo que una rama (otro `--out`) puede cambiar además.
LIBRES = frozenset({"out", "reanudar", "iterations", "log_every", "eval_every", "guardar_cada",
                    "workers", "micro_lote", "vram_gb", "sonda_vram", "seguimiento",
                    "wandb_proyecto", "qsm_datos", "eval_qsm_cada", "qsm_max_iter",
                    "qsm_target_std", "estabilidad_lado", "qsm_sim2", "val_snr",
                    "guard", "guard_abs", "guard_respaldo"})
CALENDARIO = frozenset({"hitos", "meseta", "meseta_mitades", "meseta_desde", "meseta_psnr"})

# --- TRAZA -----------------------------------------------------------------------------------
# Una fila por iteración en `traza.npy` (sus nombres en W&B, en `seguimiento.COLUMNAS_TRAZA`):
#   0 pérdida | 1 T | 2 ∂L/∂T | 3 λ | 4 ∂L/∂λ | 5 |g| | 6 la mediana de |g| del disyuntor |
#   7 la amplificación adjunta | 8 rechazado | 9 rebobinado | 10 |g| del primer micro-lote, como
#   si fuera el lote | 11 lr | 12 iteración efectiva | 13 sigma medio del lote
# Las diez primeras son las de `train.py`, en su orden: los `scripts/` las leen por posición, y una
# columna nueva va al final.
COLUMNAS = 14


# --- MODELO Y DISYUNTOR ----------------------------------------------------------------------

def vnet_config3d(t_init: float, features: int = FEATURES) -> dict:
    """La config de `train.vnet_config` en gris, con el regularizador 3D. `features` sólo cambia
    para los tests en CPU; el `type` le dice a un cargador qué VNet construir."""
    config = vnet_config(t_init, CHANNELS)
    config["R"]["type"] = "tdv3d"
    config["R"]["config"]["num_features"] = features
    return config


# --- MEMORIA Y MICRO-LOTES -------------------------------------------------------------------

def vram_predicha(micro: int, patch: int, modo: str, checkpointing: bool,
                  entrada_con_grad: bool = False) -> float:
    """El pico en GiB que predice el modelo de MEMORIA para un micro-lote: 28.83 a 48^3 x 8 en
    `rapido`. Es el del entrenamiento; la evaluación va por el MARGEN. `entrada_con_grad` es lo
    que hace --guard-amp finito; el default, el de la corrida sin la bandera."""
    por_voxel = POR_VOXEL[modo, checkpointing]
    # `VNet2D.forward` checkpointea un paso sólo si su x lleva gradiente, y el x_0 lo lleva sólo
    # con --guard-amp: sin él, el paso 1 guarda sus activaciones, ~1/S de las de un backward
    # sin checkpointing.
    if checkpointing and not entrada_con_grad:
        por_voxel += POR_VOXEL[modo, False] / STEPS
    return FIJA + micro * patch ** 3 * por_voxel


def particion(lote: int, micro: int) -> list[int]:
    """Los micro-lotes de un lote: ceil(lote / micro) trozos que difieren a lo sumo en uno, los
    mayores primero. 32 con 12 da 11 + 11 + 10; con un divisor, todos iguales a `micro`."""
    trozos = -(-lote // micro)
    base, resto = divmod(lote, trozos)
    return [base + 1] * resto + [base] * (trozos - resto)


def tamanos_utiles(lote: int) -> list[int]:
    """El mayor trozo de cada partición posible, de menor a mayor: para 32, de 1 a 8, 11, 16 y 32.
    Un micro-lote entre dos de estos da los mismos trozos que el menor y pide más memoria."""
    return sorted({particion(lote, m)[0] for m in range(1, lote + 1)})


def micro_lote_auto(lote: int, patch: int, modo: str, checkpointing: bool, presupuesto: float,
                    entrada_con_grad: bool = False) -> int:
    """El micro-lote que parte `lote` en menos trozos con un pico predicho que no pasa de
    `presupuesto · (1 − MARGEN)` GiB: el mayor de `tamanos_utiles` que entra.

    A 48^3 en 48 GiB entran 12 y da 11 (32 = 11 + 11 + 10): las mismas tres pasadas que 12, con
    menos memoria. `ValueError` si no entra ni un parche.
    """
    util = presupuesto * (1 - MARGEN)
    entran = [n for n in tamanos_utiles(lote)
              if vram_predicha(n, patch, modo, checkpointing, entrada_con_grad) <= util]
    if not entran:
        raise ValueError(
            f"ni un parche de {patch}^3 entra en {presupuesto:.2f} GiB: en `{modo}` predice "
            f"{vram_predicha(1, patch, modo, checkpointing, entrada_con_grad):.2f} GiB contra "
            f"{util:.2f} útiles. Probar --checkpointing, un --patch menor o un --micro-lote "
            "explícito")
    return entran[-1]


def entero_o_auto(texto: str) -> int | str:
    """El tipo de `--micro-lote`: `auto` o un entero >= 1."""
    if texto == "auto":
        return texto
    try:
        n = int(texto)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{texto!r}: un entero o `auto`") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"{n}: tiene que ser al menos 1")
    return n


# --- CALENDARIO DEL LR Y MESETA --------------------------------------------------------------

def lr_en(it: int, hitos) -> float:
    """El lr que USA la iteración `it` (1-indexada): LR a la mitad por cada hito <= it - 1.

    Es el de `train.py`, donde `MultiStepLR` avanza con `sched.step()` DESPUÉS del paso: con un
    hito en 25 000, la 25 000 todavía usa 4e-4 y la 25 001 usa 2e-4. Un hito repetido cuenta dos
    veces, como en `MultiStepLR`.
    """
    return LR * 0.5 ** sum(h <= it - 1 for h in hitos)


def meseta_nueva() -> dict:
    """El estado de --meseta al arrancar: sin hitos propios, sin evaluaciones y sin mejor."""
    return {"hitos": [], "sin_mejora": 0, "mejor": None, "it_mejor": None}


def paso_meseta(meseta: dict, nrmse: float, it: int, paciencia: int, mitades: int,
                psnr: float | None = None) -> str:
    """Anota en `meseta` el NRMSE QSM de la evaluación de la iteración `it` y devuelve qué hacer:
    "mejora", "espera", "mitad" (lr/2 desde la `it + 1`: `it` pasa a sus hitos), "corte" o "nula":
    un NRMSE que no es finito no cuenta, ni mejora ni suma a la paciencia. Tras una mitad, la
    cuenta vuelve a cero y el mejor se conserva. Con `psnr` (--meseta-psnr), el PSNR de la
    validación lleva su propio mejor y su cuenta, y la mitad o el corte piden las dos cuentas en
    `paciencia`: N evaluaciones seguidas sin récord en ninguna de las dos. "mejora" es que alguna
    hizo récord."""
    if not math.isfinite(nrmse) or (psnr is not None and not math.isfinite(psnr)):
        return "nula"
    mejora = meseta["mejor"] is None or nrmse < meseta["mejor"]
    if mejora:
        meseta.update(mejor=nrmse, it_mejor=it, sin_mejora=0)
    else:
        meseta["sin_mejora"] += 1
    if psnr is not None:
        if meseta.get("mejor_psnr") is None or psnr > meseta["mejor_psnr"]:
            meseta.update(mejor_psnr=psnr, it_mejor_psnr=it, sin_mejora_psnr=0)
            mejora = True
        else:
            meseta["sin_mejora_psnr"] = meseta.get("sin_mejora_psnr", 0) + 1
    if meseta["sin_mejora"] < paciencia or (psnr is not None
                                            and meseta["sin_mejora_psnr"] < paciencia):
        return "mejora" if mejora else "espera"
    meseta["sin_mejora"] = 0
    if psnr is not None:
        meseta["sin_mejora_psnr"] = 0
    if len(meseta["hitos"]) < mitades:
        meseta["hitos"].append(it)
        return "mitad"
    return "corte"


# --- ARCHIVOS --------------------------------------------------------------------------------

def lee_lista(path) -> list[str]:
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


def guarda_traza(destino: Path, filas: np.ndarray) -> None:
    """A un temporal y después `replace`: quien la lee en vuelo, o tras un corte, ve una entera."""
    temporal = destino.with_name(destino.stem + ".tmp.npy")
    np.save(temporal, filas)
    temporal.replace(destino)


def guarda_pth(objeto, destino: Path) -> None:
    """Lo mismo para un `.pth`: un corte a mitad de `torch.save` no deja un archivo roto con el
    nombre bueno, ni un estado del que no se reanuda ni un `last.pth` que `aparta` no lee."""
    temporal = destino.with_suffix(".tmp")
    torch.save(objeto, temporal)
    temporal.replace(destino)


def aparta(path: Path, vigente: dict | None) -> None:
    """Renombra `path` a `<nombre>-descartada-<it>.pth` si no es el checkpoint `vigente` (misma
    iteración y mismo PSNR): es de una corrida o de un linaje que no sigue, y no se borra. Uno que
    no se puede leer se aparta como `<nombre>-descartada-ilegible.pth`, y un nombre ya tomado no se
    pisa: se le agrega `-2`, `-3`..."""
    if not path.exists():
        return
    try:
        viejo = torch.load(path, map_location="cpu", weights_only=False)
        marca, etiqueta = [viejo["iteration"], viejo["psnr"]], f"{viejo['iteration']:06d}"
    except Exception:
        marca, etiqueta = None, "ilegible"
    # `equal_nan`: con `!=`, un PSNR NaN apartaría el checkpoint vigente.
    if marca is not None and vigente is not None and np.array_equal(
            marca, [vigente["iteration"], vigente["psnr"]], equal_nan=True):
        return
    # `rename` pisa sin avisar, y dos linajes descartados dejan checkpoints de la misma iteración.
    destino, n = path.with_name(f"{path.stem}-descartada-{etiqueta}.pth"), 1
    while destino.exists():
        n += 1
        destino = path.with_name(f"{path.stem}-descartada-{etiqueta}-{n}.pth")
    path.rename(destino)


def anota_jsonl(destino: Path, fila: dict) -> None:
    """Agrega `fila` como una línea de JSON. Un error (disco lleno) se avisa y la corrida sigue:
    el estado lleva la misma fila."""
    try:
        with destino.open("a") as f:
            f.write(json.dumps(fila) + "\n")
    except Exception as e:
        print(f"AVISO: no se pudo escribir {destino} ({type(e).__name__}: {e})", flush=True)


def commit_git() -> str | None:
    """El commit de este archivo, con `+cambios` si hay cambios sin commitear en lo versionado;
    None si no hay git o no es un repositorio (una copia sin `.git` en otra máquina). Ojo: desde
    una copia sin `.git` dentro de otro repositorio (un `git archive` en `runs/`) da el HEAD de
    ese repositorio, no el del código copiado."""
    raiz = Path(__file__).resolve().parent
    try:
        commit, cambios = (subprocess.run(["git", "-C", str(raiz), *orden], capture_output=True,
                                          text=True, timeout=10, check=True).stdout.strip()
                           for orden in (["rev-parse", "HEAD"],
                                         ["status", "--porcelain", "--untracked-files=no"]))
        return commit + ("+cambios" if cambios else "")
    except Exception:
        return None


# --- EVALUACIÓN QSM, ESTABILIDAD Y LÁMINAS ---------------------------------------------------

def carga_qsm(carpeta: str) -> dict | None:
    """COSMOS preparado por `carga_cosmos`, en numpy y en CPU; None, con un AVISO, si falta un
    archivo o no se puede leer: la evaluación QSM queda apagada y el entrenamiento sigue. Con
    `carpeta` vacía, None sin aviso."""
    if not carpeta:
        return None
    faltan = [a for a in QSM_ARCHIVOS if not (Path(carpeta) / a).is_file()]
    if faltan:
        print(f"AVISO: evaluación QSM apagada: falta{'n' if len(faltan) > 1 else ''} "
              f"{' y '.join(faltan)} en {carpeta}. Se copian "
              f"{' y '.join(QSM_ARCHIVOS)} de {QSM_ORIGEN} (el data_mat/ del pipeline QSM) a esa "
              "carpeta, o se pasa otra con --qsm-datos; el entrenamiento sigue sin ella",
              flush=True)
        return None
    try:
        return carga_cosmos(carpeta)
    except Exception as e:
        print(f"AVISO: evaluación QSM apagada: {carpeta} no se pudo leer ({type(e).__name__}: "
              f"{e}); el entrenamiento sigue sin ella", flush=True)
        return None


def carga_sim2_o_aviso(carpeta: str) -> dict | None:
    """Sim2 en numpy y en CPU; None, con un AVISO, si no se puede leer (la evaluación en Sim2
    queda apagada y el entrenamiento sigue). Con `carpeta` vacía, None sin aviso."""
    if not carpeta:
        return None
    try:
        return carga_sim2(carpeta)
    except Exception as e:
        print(f"AVISO: evaluación en Sim2 apagada: {Path(carpeta) / SIM2_ARCHIVO} no se pudo leer "
              f"({type(e).__name__}: {e}); se arma con datasets/prepara_sim2.py", flush=True)
        return None


def metricas_qsm(r: dict, datos: dict) -> dict:
    """Las métricas del panel (`metricas_chi`) de la reconstrucción y de χ₀, la primera, sin la
    red (`chi0_*`), y la consistencia de χ_GT con el dato, la referencia de la del mapa. Sin el
    NRMSE de la reconstrucción, que ya es `nrmse_final_pct`. {} con un aviso si fallan: como la
    evaluación, no pueden matar el entrenamiento."""
    try:
        m = {k: v for k, v in metricas_chi(r["chi"], datos).items() if k != "nrmse_pct"}
        if r.get("chi0") is not None:
            m |= {f"chi0_{k}": v for k, v in metricas_chi(r["chi0"], datos).items()
                  if k != "factor_cono"}
        m["gt_residuo_dato_pct"] = residuo_dato_pct(datos["chi"], datos)
        return m
    except Exception as e:
        print(f"    métricas QSM FALLARON ({type(e).__name__}: {e}); el entrenamiento sigue",
              flush=True)
        return {}


def linea_metricas(m: dict) -> str:
    if not m:
        return ""
    return (f" | HFEN {m['hfen_pct']:.2f} %, XSIM {m['xsim']:.4f}, residuo {m['residuo_dato_pct']:.2f}"
            f" % (GT {m['gt_residuo_dato_pct']:.2f}), cono ×{m['factor_cono']:.2f} | χ₀ sin red: "
            f"NRMSE {m['chi0_nrmse_pct']:.2f} %, XSIM {m['chi0_xsim']:.4f}")


def evalua_qsm(model, cosmos: dict, device, amp, max_iter: int,
               target_std: float = TARGET_STD) -> dict:
    """El ADMM de `ddr/qsm3d.py` con el modelo como denoiser sobre el volumen entero, con α tal
    que α·χ₀ tenga desviación `target_std` en la máscara: hasta
    `max_iter` iteraciones, sin el corte por umbral (`tol` = -inf), y para cuando el NRMSE
    contra COSMOS sube QSM_SUBIDAS iteraciones seguidas (o si el update no es finito), con la
    mejor reconstrucción que vio. Devuelve las claves de `qsm/*`, el NRMSE y el update de cada
    iteración, y `chi`, la reconstrucción en numpy (para la lámina, que se arma afuera), o
    `{"fallo": <tipo: mensaje>}`. No levanta nunca, ni con un OOM: la evaluación QSM no puede
    matar el entrenamiento. Deja el modelo en modo train."""
    fallo = None
    model.eval()
    try:
        with torch.no_grad():
            r = admm_wtdv([denoiser_3d(model, autocast_dtype=amp)], cosmos, device=device,
                          target_std=target_std, max_iter=max_iter, tol=-math.inf,
                          subidas_para_cortar=QSM_SUBIDAS)
        return {"nrmse_final_pct": r["nrmse_final"], "nrmse_min_pct": r["nrmse_min"],
                "iter_min": r["it_min"], "iteraciones": r["iteraciones"], "alpha": r["alpha"],
                "segundos": r["segundos"], "motivo": r["motivo"],
                "nrmse_trayectoria_pct": r["nrmse"], "update_trayectoria": r["x_update"],
                "chi": r["chi"], "chi0": r.get("chi0")}
    except Exception as e:
        fallo = f"{type(e).__name__}: {e}"
    finally:
        model.train()
    # Fuera del `except`: adentro, la traza del error retiene los tensores del ADMM y la caché no
    # puede devolverlos.
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return {"fallo": fallo}


def resumen_qsm(registros: list[dict]) -> dict:
    """Lo de la evaluación QSM que va a `estado.json` y al resumen de W&B: la última que terminó,
    la de menor NRMSE final y cuántas fallaron. None donde no hubo ninguna."""
    buenas = [r for r in registros if "fallo" not in r]
    finitas = [r for r in buenas if math.isfinite(r["nrmse_final_pct"])]
    ultima = buenas[-1] if buenas else {}
    mejor = min(finitas, key=lambda r: r["nrmse_final_pct"], default={})
    return {"qsm_evaluaciones": len(registros), "qsm_fallos": len(registros) - len(buenas),
            "qsm_nrmse_ultimo_pct": ultima.get("nrmse_final_pct"),
            "qsm_iteracion_ultima": ultima.get("iteration"),
            "qsm_motivo_ultimo": ultima.get("motivo"),
            "qsm_nrmse_mejor_pct": mejor.get("nrmse_final_pct"),
            "qsm_iteracion_mejor": mejor.get("iteration")}


def punto_validacion(eval_set, lado: int, sigma: float) -> torch.Tensor:
    """El cubo `lado`^3 central del primer recorte de evaluación con el ruido que le pone la
    validación (`evaluate_denoising3d` con SEMILLA_VAL, volumen 0), `[1, 1, lado, lado, lado]`
    en CPU: el ruido se sortea sobre el recorte entero y después se corta."""
    limpio = eval_set[0][0][None]
    ruidoso = add_noise(limpio, sigma, torch.Generator(device="cpu").manual_seed(SEMILLA_VAL),
                        NOISE_SCALE)
    ventana = tuple(slice((n - lado) // 2, (n - lado) // 2 + lado) for n in ruidoso.shape[2:])
    return ruidoso[(..., *ventana)].contiguous()


def mide_estabilidad(model, punto: torch.Tensor, device) -> dict:
    """`estabilidad` del modelo en `punto`, o `{"fallo": <tipo: mensaje>}`: como `evalua_qsm`, no
    levanta nunca, ni con un OOM."""
    fallo = None
    try:
        return estabilidad(model, punto)
    except Exception as e:
        fallo = f"{type(e).__name__}: {e}"
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return {"fallo": fallo}


def claves_estabilidad(r: dict) -> dict:
    """Lo de `mide_estabilidad` que va a `history.json` y a `qsm.jsonl`: sin el reloj, que
    cambiaría la fila entre una corrida y su reanudación."""
    if "fallo" in r:
        return {"estabilidad_fallo": r["fallo"]}
    return {"estabilidad": r["paso"], "lambda_max": r["lambda_max"],
            "estabilidad_cambio": r["cambio_relativo"]}


def linea_estabilidad(punto: str, r: dict) -> str:
    if "fallo" in r:
        return f"    estabilidad {punto} FALLÓ ({r['fallo']}); el entrenamiento sigue"
    return (f"    estabilidad {punto}: (T/S)·λ_max = {r['paso']:.4f}"
            + ("  *INESTABLE*" if r["paso"] >= UMBRAL_ESTABILIDAD else "")
            + f" | λ_max {r['lambda_max']:.2f} (Rayleigh {r['rayleigh']:+.2f}), T {r['T']:.5f}"
            f" | cambio relativo {r['cambio_relativo']:.1e} en {r['iteraciones']} iteraciones | "
            f"{r['segundos']:.1f} s")


def registra_lamina(run, clave: str, hacer, it: int, leyenda: str) -> None:
    """La imagen que devuelve `hacer()`, en W&B. Si armarla falla se pierde la imagen, no la
    corrida; sin seguimiento ni se arma."""
    if run is None:
        return
    try:
        imagen = hacer()
    except Exception as e:
        print(f"seguimiento: la lámina {clave} falló ({type(e).__name__}: {e})", flush=True)
        return
    seguimiento.registra_imagenes(run, {clave: imagen}, it, leyendas={clave: leyenda})


# --- SONDA DE MEMORIA ------------------------------------------------------------------------

def sonda_vram(model, args, mode: dict, presupuesto: float, out: Path) -> None:
    """SONDA_ITERACIONES de forward + backward con cada micro-lote de `tamanos_utiles(--lote)`, de
    menor a mayor, hasta el primero que no entra: el pico medido contra el predicho, y los segundos
    por parche de las iteraciones que no compilan. El modelo llega compilado como para entrenar.
    Mide el entrenamiento, no la evaluación. En `--out` deja sólo `sonda-vram.json`.
    """
    util = presupuesto * (1 - MARGEN)
    candidatos = tamanos_utiles(args.lote)
    con_grad = math.isfinite(args.guard_amp)
    # Cada micro-lote son dos entradas de `R.grad` en la caché de dynamo (el primer paso con x sin
    # gradiente y los demás con; medido en CPU el 2026-10-09), y pasado `recompile_limit` (8)
    # sigue en eager: mediría otra memoria, 61 % más a 64^3 x 1.
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit,
                                               2 * len(candidatos) + 2)
    print(f"sonda: lote {args.lote} de {args.patch}^3 en `{args.mode}`"
          + (" con checkpointing" if args.checkpointing else "")
          + f"; entra si el pico medido no pasa de {util:.2f} GiB", flush=True)
    cabecera = (f"{'micro-lote':>10} | {'predicho GiB':>12} | {'medido GiB':>10} | "
                f"{'s/parche':>8} | entra")
    print(cabecera + "\n" + "-" * len(cabecera), flush=True)
    filas = []
    for n in candidatos:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            for i in range(SONDA_ITERACIONES):
                if i == SONDA_DESCARTADAS:
                    torch.cuda.synchronize()
                    inicio = time.perf_counter()
                model.zero_grad(set_to_none=True)
                # Ruido uniforme: la memoria no depende del contenido.
                clean = torch.rand(n, CHANNELS, *(3 * [args.patch]), device="cuda")
                noisy = clean + args.sigma / NOISE_SCALE * torch.randn_like(clean)
                noisy.requires_grad_(con_grad)
                with torch.autocast("cuda", dtype=mode["amp"], enabled=mode["amp"] is not None):
                    states = model(noisy, noisy)
                    loss = (states[-1].float() - clean.float()).pow(2).flatten(1).sum(1).mean()
                loss.backward()
            torch.cuda.synchronize()
            por_parche = (time.perf_counter() - inicio) / (
                (SONDA_ITERACIONES - SONDA_DESCARTADAS) * n)
            medido = torch.cuda.max_memory_allocated() / 2 ** 30
        except torch.OutOfMemoryError:
            medido = por_parche = None
        clean = noisy = states = loss = None
        model.zero_grad(set_to_none=True)
        predicho = vram_predicha(n, args.patch, args.mode, args.checkpointing, con_grad)
        entra = medido is not None and medido <= util
        filas.append({"micro_lote": n, "predicho_gib": round(predicho, 2),
                      "medido_gib": None if medido is None else round(medido, 2),
                      "s_por_parche": None if por_parche is None else round(por_parche, 4),
                      "entra": entra})
        print(f"{n:>10} | {predicho:>12.2f} | "
              f"{'OOM' if medido is None else format(medido, '.2f'):>10} | "
              f"{'-' if por_parche is None else format(por_parche, '.4f'):>8} | "
              + ("sí" if entra else "no"), flush=True)
        if not entra:
            break

    mayor = max((f["micro_lote"] for f in filas if f["entra"]), default=None)
    veloz = min((f for f in filas if f["entra"]), key=lambda f: f["s_por_parche"], default=None)
    try:
        auto = micro_lote_auto(args.lote, args.patch, args.mode, args.checkpointing, presupuesto,
                               con_grad)
    except ValueError:
        auto = None
    out.mkdir(parents=True, exist_ok=True)
    (out / "sonda-vram.json").write_text(json.dumps({
        "fecha": time.strftime("%Y-%m-%d %H:%M"), "dispositivo": torch.cuda.get_device_name(),
        "memoria_total_gib": round(torch.cuda.get_device_properties().total_memory / 2 ** 30, 2),
        "presupuesto_gib": round(presupuesto, 2), "util_gib": round(util, 2), "margen": MARGEN,
        "torch": torch.__version__, "modo": args.mode, "parche": args.patch, "lote": args.lote,
        "checkpointing": args.checkpointing, "entrada_con_grad": con_grad,
        "compilado": not args.sin_compilar,
        "iteraciones": SONDA_ITERACIONES, "descartadas": SONDA_DESCARTADAS,
        "filas": filas, "mayor_que_entra_medido": mayor, "auto_predicho": auto,
        "mas_rapido_por_parche": None if veloz is None else veloz["micro_lote"]}, indent=2))
    print(f"\nmedido, entra hasta micro-lote {mayor}; `--micro-lote auto` elige {auto}"
          + ("" if mayor == auto else ": para usar el medido, pasarlo explícito")
          + ("" if veloz is None else f" | el más rápido por parche, {veloz['micro_lote']} "
             f"({veloz['s_por_parche']:.4f} s)")
          + f" | {out / 'sonda-vram.json'}", flush=True)


# --- BANDERAS --------------------------------------------------------------------------------

def opciones() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    p.add_argument("--out", required=True, help="directorio de salida")
    # `referencia` cuesta ~21x más en 3D y a 48^3 x 4 no entra: sirve para tests cortos.
    p.add_argument("--mode", choices=MODES, default="rapido")
    p.add_argument("--iterations", type=int, default=ITERACIONES_TANDA)
    p.add_argument("--t-init", type=float, default=T_INIT)
    p.add_argument("--sigma", type=float, default=SIGMA,
                   help="sigma del ruido de entrenamiento y, siempre, el de la evaluación")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--train-list", default=TRAIN_LIST)
    p.add_argument("--eval-list", default=EVAL_LIST)
    p.add_argument("--patch", type=int, default=PATCH_SIZE)
    p.add_argument("--lote", type=int, default=LOTE, help="parches por iteración: es receta")
    p.add_argument("--epocas", action="store_true",
                   help="épocas: cada una es una pasada por los volúmenes de entrenamiento, un "
                        "parche de cada uno en orden al azar (sin reemplazo), con un DataLoader "
                        "que se agota al terminarla; iteraciones por época = volúmenes // lote. "
                        "Sin ella, cada parche sale de un volumen al azar, con reemplazo")
    p.add_argument("--micro-lote", type=entero_o_auto, default=MICRO_LOTE,
                   help="máximo de parches por backward: el lote se parte en ceil(lote/m) trozos "
                        "casi iguales (32 con 12: 11 + 11 + 10). `auto`: el que deja menos "
                        "trozos y entra en --vram-gb según el modelo de memoria")
    p.add_argument("--vram-gb", type=float, default=VRAM_GB,
                   help="presupuesto en GiB, como los reporta torch: algo menos que la etiqueta "
                        "(la 5070 Ti de «16 GB» da 15.45); 44.7 si se pactaron 48 GB decimales. "
                        "Se recorta a la memoria de la tarjeta; si es menor, PyTorch no pasa de "
                        "ahí")
    p.add_argument("--sonda-vram", action="store_true",
                   help="mide en la GPU el pico y los s por parche de cada micro-lote "
                        "posible, escribe sonda-vram.json en --out y sale sin entrenar")
    p.add_argument("--hitos", type=int, nargs="*", default=None,
                   help="iteraciones tras las que el lr se divide por 2. Sin la bandera, cada "
                        "25 000 como en el paper; sola, ninguno")
    p.add_argument("--meseta", type=int, default=0,
                   help="N > 0: si el NRMSE QSM mínimo del ADMM (el de su mejor iteración) no "
                        "baja del mejor visto en N evaluaciones QSM seguidas, lr/2 desde la "
                        "iteración siguiente; agotadas --meseta-mitades, la meseta siguiente corta "
                        "la corrida. 0 lo apaga")
    p.add_argument("--meseta-mitades", type=int, default=MESETA_MITADES,
                   help="cuántas veces puede partir el lr --meseta antes de cortar")
    p.add_argument("--meseta-desde", type=int, default=0,
                   help="la meseta mira sólo las evaluaciones QSM desde esta iteración: las de "
                        "antes no cuentan ni dejan un mejor, así que la primera desde aquí lo fija")
    p.add_argument("--meseta-psnr", action="store_true",
                   help="lr/2 (o el corte) sólo si en esas N evaluaciones tampoco hizo récord el "
                        "PSNR de la validación, el que elige best.pth (en el cerebro con "
                        "--mascara-cerebro)")
    p.add_argument("--sobreajuste", action="store_true",
                   help="prueba de sobreajuste: cada iteración ve el mismo lote (los primeros "
                        "--lote parches, de los primeros --lote volúmenes de --train-list) con el "
                        "mismo ruido, sin las cotas del disyuntor; la validación agrega el PSNR de "
                        "ese lote")
    # Candidatas de la fase 2, apagadas: con los defaults la corrida es la receta.
    p.add_argument("--sigma-max", type=float, default=0.0,
                   help="> 0 sortea sigma ~ U(--sigma-min, --sigma-max) por parche (ruido ciego)")
    p.add_argument("--sigma-min", type=float, default=SIGMA_MIN)
    p.add_argument("--sigma-log", action="store_true",
                   help="con --sigma-max y --sigma-min > 0: sigma log-uniforme en [--sigma-min, "
                        "--sigma-max] en vez de uniforme")
    p.add_argument("--fraccion-limpias", type=float, default=0.0,
                   help="p en [0, 1): cada parche va sin ruido (sigma = 0) con probabilidad p")
    p.add_argument("--aug-signo", action="store_true",
                   help="multiplica el parche por -1 con probabilidad 1/2")
    p.add_argument("--aug-ganancia", type=float, default=1.0,
                   help="g > 1 multiplica el parche por un factor log-uniforme en [1/g, g]")
    p.add_argument("--mascara-cerebro", action="store_true",
                   help="cada parche es un cubo entero dentro del cerebro, según "
                        "<data-root>/mascaras/train/ (datasets/mascaras_ldm100k3d.py)")
    p.add_argument("--clip", type=float, default=float("inf"), help="norma máxima del gradiente")
    p.add_argument("--guard", type=float, default=GUARD_3D,
                   help="rechaza el lote si |g| supera esta razón sobre su mediana móvil. `inf` "
                        "lo apaga, y entonces el respaldo no se refresca")
    p.add_argument("--guard-abs", type=float, default=math.inf,
                   help="cota absoluta de |g|; sin la bandera, ninguna (en 3D traba el arranque)")
    p.add_argument("--guard-ventana", type=int, default=GUARD_VENTANA)
    p.add_argument("--guard-seguidos", type=int, default=GUARD_SEGUIDOS,
                   help="rechazos seguidos que disparan el rebobinado al respaldo. 0 lo apaga")
    p.add_argument("--guard-respaldo", type=int, default=GUARD_RESPALDO_3D,
                   help="cada cuántas iteraciones sanas se refresca el respaldo, con el relativo "
                        "armado")
    p.add_argument("--guard-limpias", type=int, default=GUARD_LIMPIAS,
                   help="iteraciones sanas seguidas que exige el respaldo antes de refrescarse")
    p.add_argument("--guard-amp", type=float, default=float("inf"),
                   help="rechaza también si la amplificación adjunta ||p_0||/||p_S|| supera esto")
    p.add_argument("--checkpointing", action="store_true",
                   help="recomputa el forward en el backward: ~5x menos memoria (~9x con "
                        "--guard-amp, que checkpointea también el paso 1), +27%% de tiempo")
    p.add_argument("--guardar-cada", type=int, default=GUARDAR_CADA,
                   help="escribe estado-<it>.pth con todo lo que hace falta para reanudar. 0 no")
    p.add_argument("--reanudar", default=None,
                   help="estado-<it>.pth del que seguir; con otro --out es una rama")
    p.add_argument("--sin-compilar", action="store_true", help="sin torch.compile (tests en CPU)")
    p.add_argument("--workers", type=int, default=WORKERS)
    # Seguimiento, evaluación QSM y estabilidad: no tocan la aritmética, y están en LIBRES.
    p.add_argument("--seguimiento", choices=seguimiento.MODOS, default="auto",
                   help="W&B. `auto`: online si hay credencial (`wandb login` o WANDB_API_KEY) y "
                        "offline si no; `online` sale con error antes de entrenar si W&B no "
                        "abre; `no` ni importa wandb")
    p.add_argument("--wandb-proyecto", default=seguimiento.PROYECTO)
    p.add_argument("--qsm-datos", default=QSM_DATOS,
                   help="carpeta con params.mat y chi_cosmos.mat (los de ../tdv/data_mat). Si "
                        "falta alguno, la evaluación QSM queda apagada con un aviso; '' la apaga "
                        "sin aviso")
    p.add_argument("--qsm-sim2", default="",
                   help=f"carpeta con {SIM2_ARCHIVO} (`datasets/prepara_sim2.py`, por defecto en "
                        f"{QSM_SIM2}): la evaluación QSM corre también en Sim2, a W&B como "
                        "qsm_sim2/*, sin elegir checkpoints. '' la apaga")
    p.add_argument("--val-snr", type=float, nargs="*", default=[],
                   help="SNR (pico/σ = 1/σ) donde se valida el denoising además de σ = --sigma, "
                        "a W&B como val_snr<SNR>/*, sin elegir checkpoints; p. ej. 25 50 100")
    p.add_argument("--eval-qsm-cada", type=int, default=0,
                   help="0: la evaluación QSM va en cada validación; N > 0: en las validaciones "
                        "de iteración múltiplo de N y en la última")
    p.add_argument("--qsm-max-iter", type=int, default=QSM_MAX_ITER,
                   help="techo de iteraciones del ADMM, que para antes si el NRMSE sube dos "
                        "iteraciones seguidas; sin el corte por umbral del autor (su script llega "
                        "hasta 100 con TOL/CORTE)")
    p.add_argument("--qsm-target-std", type=float, default=TARGET_STD,
                   help="desviación de α·χ₀ en la máscara con que la evaluación QSM escala la "
                        "entrada de la red (y el punto de la estabilidad en QSM); 0.06 por "
                        "defecto, el del oficial 2D")
    p.add_argument("--estabilidad-lado", type=int, default=LADO_ESTABILIDAD,
                   help="lado del cubo donde se mide (T/S)·λ_max(∇²R), con la cadencia de la "
                        "evaluación QSM: en la validación y, con COSMOS, en la entrada del primer "
                        "denoiser del ADMM. Se recorta a lo que entra. 0 no la mide")
    # Sólo para los tests en CPU: con los defaults la red y la evaluación son las de la receta.
    p.add_argument("--features", type=int, default=FEATURES)
    p.add_argument("--eval-recorte", type=int, default=EVAL_RECORTE)
    return p


# --- LO QUE ARMA `main` Y LO QUE CAMBIA EN EL BUCLE ------------------------------------------
# Sin clases (AGENTS.md, «El código»): `piezas` y `corrida` son `SimpleNamespace`, como `args`.
# `piezas` lo arma `main` una vez: las banderas, el modelo y Adam, los datos y dónde se escribe.
# Los pesos, Adam y el generador del ruido cambian por dentro y viajan en el estado. `corrida`
# son las cuentas que el bucle va cambiando.

def corrida_nueva(iteraciones: int) -> SimpleNamespace:
    """Las cuentas de una corrida que empieza. `guarda_estado` las escribe en `estado-<it>.pth` y
    `corrida_desde_estado` las lee: un campo nuevo va en las tres funciones. `corte` no viaja: al
    reanudar vuelve a None."""
    return SimpleNamespace(
        traza=np.zeros((iteraciones, COLUMNAS), dtype=np.float64),   # una fila por iteración
        history=[],                     # una entrada por validación
        best_psnr=0.0,                  # el PSNR que eligió `best.pth`
        mejor=None,                     # `best.pth`, para reponerlo al reanudar
        max_grad=0.0,                   # el mayor |g| desde el último log
        horas_previas=0.0,              # las de los tramos anteriores del linaje
        # El disyuntor (`rechaza_lote`, `respalda` y `rebobina` de train.py).
        escala=[],                      # los |g| aceptados que dan la mediana
        rechazos=0,
        seguidos=0,                     # rechazos seguidos
        max_seguidos=0,
        respaldo=None,                  # pesos, Adam y ventana (`respalda`)
        it_respaldo=0,                  # la iteración del respaldo
        rebobinados=0,
        limpias=0,                      # iteraciones aceptadas seguidas
        descartadas=0,                  # iteraciones perdidas por rebobinar
        it_rebobinado=0,                # la del último rebobinado
        # La evaluación QSM y la meseta.
        qsm_hist=[],                    # las filas de `qsm.jsonl`
        mejor_qsm=None,                 # `best_qsm.pth`
        meseta=meseta_nueva(),
        corte=None,                     # la iteración en que cortó la meseta
    )


def corrida_desde_estado(estado: dict, iteraciones: int) -> SimpleNamespace:
    """La corrida tal como quedó en un `estado-<it>.pth`. Las claves de la QSM, `best_qsm.pth` y
    la meseta pueden faltar en un estado anterior a ellas."""
    corrida = corrida_nueva(iteraciones)
    corrida.history, corrida.best_psnr = estado["history"], estado["best_psnr"]
    corrida.mejor, corrida.max_grad = estado["mejor"], estado["max_grad"]
    corrida.horas_previas, corrida.escala = estado["horas"], estado["escala"]
    corrida.rechazos, corrida.seguidos = estado["rechazos"], estado["seguidos"]
    corrida.max_seguidos, corrida.limpias = estado["max_seguidos"], estado["limpias"]
    corrida.respaldo, corrida.it_respaldo = estado["respaldo"], estado["it_respaldo"]
    corrida.rebobinados, corrida.descartadas = estado["rebobinados"], estado["descartadas"]
    corrida.it_rebobinado = estado["it_rebobinado"]
    corrida.traza[:estado["iteration"]] = estado["traza"]
    corrida.qsm_hist = estado.get("qsm", [])
    corrida.mejor_qsm = estado.get("mejor_qsm")
    corrida.meseta = estado.get("meseta") or meseta_nueva()
    return corrida


# --- PREPARACIÓN: LOS PASOS DE `main` --------------------------------------------------------

def revisa_banderas(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Completa lo que depende de otras banderas y rechaza lo imposible antes de tocar nada: lo
    que haría fallar la corrida tiene que fallar al arrancar."""
    # Con el mismo lote, un rechazo deja la iteración siguiente igual y se repetiría siempre, y el
    # rebobinado volvería al mismo estallido: el sobreajuste va sin las dos cotas del disyuntor.
    # Siguen el rechazo de un gradiente no finito y el rebobinado.
    if args.sobreajuste:
        args.guard = args.guard_abs = math.inf
    if args.lote < 1:
        parser.error(f"--lote {args.lote}: tiene que ser al menos 1")
    if args.epocas and args.sobreajuste:
        parser.error("--epocas no va con --sobreajuste: el lote fijo no recorre los volúmenes")
    if args.sigma_log and not (args.sigma_max > 0 and 0 < args.sigma_min <= args.sigma_max):
        parser.error(f"--sigma-log pide --sigma-max > 0 y --sigma-min en (0, --sigma-max]: hay "
                     f"{args.sigma_min:g} y {args.sigma_max:g}")
    if not 0 <= args.fraccion_limpias < 1:
        parser.error(f"--fraccion-limpias {args.fraccion_limpias:g}: tiene que estar en [0, 1)")
    if args.meseta < 0 or args.meseta_mitades < 0 or args.meseta_desde < 0:
        parser.error("--meseta, --meseta-mitades y --meseta-desde no pueden ser negativos")
    if (args.meseta_desde or args.meseta_psnr) and not args.meseta:
        parser.error("--meseta-desde y --meseta-psnr ajustan la meseta: necesitan --meseta N")
    if args.qsm_max_iter < 1:
        parser.error(f"--qsm-max-iter {args.qsm_max_iter}: tiene que ser al menos 1")
    if any(not (math.isfinite(s) and s > 0) for s in args.val_snr):
        parser.error(f"--val-snr {args.val_snr}: cada SNR tiene que ser positivo y finito")
    if not (math.isfinite(args.qsm_target_std) and args.qsm_target_std > 0):
        parser.error(f"--qsm-target-std {args.qsm_target_std}: tiene que ser positivo y finito")
    if args.meseta and not (args.qsm_datos and all((Path(args.qsm_datos) / a).is_file()
                                                   for a in QSM_ARCHIVOS)):
        parser.error(f"--meseta {args.meseta} decide con el NRMSE QSM, y la evaluación QSM no "
                     f"tiene datos: faltan {' o '.join(QSM_ARCHIVOS)} en --qsm-datos "
                     f"{args.qsm_datos!r}")
    if args.seguimiento == "online" and (falta := seguimiento.falta_para_online()):
        parser.error(f"--seguimiento online: {falta}. Con --seguimiento auto la corrida sigue "
                     "offline")


def elige_micro_lote(parser: argparse.ArgumentParser, args: argparse.Namespace,
                     device: torch.device, mide_amp: bool) -> tuple[float, list[int], float]:
    """El presupuesto de memoria y en qué micro-lotes se parte el lote. Con `--micro-lote auto` lo
    elige el modelo de MEMORIA. Deja en `args.micro_lote` el efectivo y devuelve (presupuesto en
    GiB, los micro-lotes, el pico predicho en GiB)."""
    # En CPU no hay tarjeta y el presupuesto es --vram-gb: `auto` elige lo mismo que con una
    # tarjeta de ese tamaño, y los tests no dependen de la máquina.
    total = (torch.cuda.get_device_properties(device).total_memory / 2 ** 30
             if device.type == "cuda" else None)
    presupuesto = args.vram_gb if total is None else min(args.vram_gb, total)
    if total is not None and presupuesto < total:
        # La tarjeta puede estar compartida: mejor un OOM propio que pisar a otro proceso.
        torch.cuda.set_per_process_memory_fraction(presupuesto / total)
    pedido = args.micro_lote
    if pedido == "auto":
        try:
            args.micro_lote = micro_lote_auto(args.lote, args.patch, args.mode,
                                              args.checkpointing, presupuesto, mide_amp)
        except ValueError as e:
            parser.error(f"--micro-lote auto: {e}")
    elif args.micro_lote > args.lote:
        parser.error(f"--micro-lote {args.micro_lote} es mayor que --lote {args.lote}")
    # El efectivo es el mayor trozo, que es lo que ocupa memoria: 12 con 32 queda en 11.
    tamanos = particion(args.lote, args.micro_lote)
    args.micro_lote = tamanos[0]
    predicha = vram_predicha(args.micro_lote, args.patch, args.mode, args.checkpointing, mide_amp)
    print("dispositivo: " + (f"{torch.cuda.get_device_name(device)}, {total:.2f} GiB"
                             if total is not None else "cpu, sin memoria que medir")
          + f" | presupuesto {presupuesto:.2f} GiB, {presupuesto * (1 - MARGEN):.2f} útiles"
          + f" | micro-lote {args.micro_lote}" + (" (auto)" if pedido == "auto" else
                                                   f" (pedido {pedido})"
                                                   if pedido != args.micro_lote else "")
          + f", pico predicho {predicha:.2f} GiB", flush=True)
    return presupuesto, tamanos, predicha


def lee_estado(parser: argparse.ArgumentParser, args: argparse.Namespace,
               out: Path) -> tuple[dict | None, int]:
    """Con --reanudar, el estado del que se sigue y su iteración; sin la bandera, (None, 0).
    Rechaza lo que no se puede cambiar al reanudar: la receta nunca, y el calendario sólo en una
    rama (otro --out). Otro micro-lote pasa con un AVISO: la corrida deja de ser bit a bit."""
    if not args.reanudar:
        return None, 0
    # Sin `map_location` cada tensor vuelve a su dispositivo.
    estado = torch.load(args.reanudar, weights_only=False,
                        map_location=None if torch.cuda.is_available() else "cpu")
    rama = Path(args.reanudar).resolve().parent != out.resolve()
    libres = LIBRES | (CALENDARIO if rama else set())
    ahora = vars(args)
    # Una bandera de receta posterior al estado vale ahí su default: los de runs/humo3d no tienen
    # `mascara_cerebro`, y sin esto no se podían reanudar.
    distintas = sorted(k for k in estado["args"].keys() | ahora.keys()
                       if k not in libres
                       and estado["args"].get(k, parser.get_default(k)) != ahora.get(k))
    if distintas:
        parser.error(f"--reanudar: cambian {distintas}; "
                     + ("una rama sólo puede cambiar el calendario: --hitos, --meseta, "
                        "--meseta-mitades, --meseta-desde y --meseta-psnr" if rama else
                        "seguir en el mismo --out no puede cambiar la receta (otro --micro-lote "
                        "sí, con un AVISO)"))
    it0 = estado["iteration"]
    if args.iterations < it0:
        parser.error(f"--iterations {args.iterations} es menor que la del estado, {it0}")
    if estado["args"]["micro_lote"] != args.micro_lote:
        # Qué más cambia, y por qué, en docs/porque-del-codigo.md, REANUDACIÓN.
        cambia = ["el orden de suma del gradiente"]
        if args.sigma_max > 0 or args.fraccion_limpias > 0 or args.patch ** 3 % 16:
            cambia.append("el ruido de cada parche")
        if math.isfinite(args.guard_amp):
            cambia.append("la amplificación adjunta que mira --guard-amp")
        print(f"AVISO: el micro-lote pasa de {estado['args']['micro_lote']} a "
              f"{args.micro_lote}. Cambia " + ", ".join(cambia[:-1])
              + (" y " if len(cambia) > 1 else "") + cambia[-1] + ": desde acá la corrida "
              "no es bit a bit la que hubiera seguido, y la columna 10 de la traza mide "
              f"micro-lotes de {args.micro_lote}", flush=True)
    return estado, it0


def prepara_backend(seed: int, mode: dict) -> None:
    """La semilla global, el determinismo y la precisión de las operaciones fp32 del modo."""
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = mode["benchmark"]
    if mode["deterministic"]:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    precision = "tf32" if mode["tf32"] else "ieee"
    torch.backends.cudnn.conv.fp32_precision = precision
    torch.backends.cuda.matmul.fp32_precision = precision


def arma_modelo(parser: argparse.ArgumentParser, args: argparse.Namespace, config: dict,
                device: torch.device, estado: dict | None) -> VNet3D:
    """La VNet3D en `device`, con los pesos del estado si se reanuda y `R.grad` compilado (salvo
    con --sin-compilar)."""
    model = VNet3D(config, efficient=args.checkpointing).to(device)
    if estado is not None:
        model.load_state_dict(estado["model"])
    if not args.sin_compilar:
        # Compilado en CPU tardaría semanas: casi siempre es un driver que falta.
        if device.type != "cuda":
            parser.error("torch no ve ninguna GPU (torch.cuda.is_available() es False): revisar "
                         "el driver con nvidia-smi (CUDA 13 pide 580 o más). En CPU, sólo para "
                         "probar, --sin-compilar")
        # El método y no el módulo, como en `train.py`: el state_dict queda sin `_orig_mod.`.
        model.R.grad = torch.compile(model.R.grad, dynamic=False)
        # Trozos desiguales son dos formas más en la caché de dynamo.
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 16)
    return model


def lotes_por_epoca(train_set: PatchDataset3D, lote: int, workers: int, device):
    """--epocas: un DataLoader por época, que se agota al terminarla (cada volumen de
    entrenamiento dio un parche) y se rehace para la siguiente. Rinde lotes, como `iter(loader)`,
    y se acaba con la corrida. Una reanudación a mitad de época empieza por lo que le falta. El
    generador propio de cada DataLoader no toca el global, que restaura la reanudación."""
    P = train_set.parches_por_epoca
    epoca = train_set.offset // P
    while True:
        desde = max(epoca * P - train_set.offset, 0)
        hasta = min((epoca + 1) * P - train_set.offset, len(train_set))
        if desde >= hasta:
            return
        yield from DataLoader(Subset(train_set, range(desde, hasta)), batch_size=lote,
                              shuffle=False, drop_last=True, num_workers=workers,
                              pin_memory=(device.type == "cuda"),
                              generator=torch.Generator().manual_seed(epoca))
        epoca += 1


def arma_entrenamiento(parser: argparse.ArgumentParser, args: argparse.Namespace, it0: int,
                       device: torch.device):
    """El dataset de parches y cómo se piden: un DataLoader de lotes enteros o, con --sobreajuste,
    el único lote, armado una vez. Devuelve (dataset, loader, lote fijo), None donde no va.

    El parche `i` depende sólo de `(seed, offset + i)`: reanudar en `it0` es empezar en el parche
    `it0 · lote`, y el DataLoader no guarda nada."""
    archivos = [Path(args.data_root) / "images" / TRAIN_SPLIT / f"{Path(n).stem}.npy"
                for n in lee_lista(args.train_list)]
    # La prueba de sobreajuste ve un solo lote: con los primeros --lote volúmenes alcanza.
    if args.sobreajuste:
        archivos = archivos[:args.lote]
    mascaras = None
    if args.mascara_cerebro:
        rutas = [Path(args.data_root) / "mascaras" / TRAIN_SPLIT / f.name for f in archivos]
        faltan = [r.stem for r in rutas if not r.exists()]
        if faltan:
            herramienta = "datasets/mascaras_ldm100k3d.py"
            lista = f"--lista {args.train_list} --salida {args.data_root}"
            parser.error(f"--mascara-cerebro: faltan {len(faltan)} de {len(rutas)} máscaras "
                         f"en {rutas[0].parent} (la primera, {faltan[0]}). Se generan en tres "
                         f"pasos, cada herramienta en su entorno (cómo armarlos, en el docstring "
                         f"de {herramienta}):\n"
                         f"  $H/deepbet/bin/python -I {herramienta} deepbet {lista}\n"
                         f"  $H/hdbet/bin/python -I {herramienta} hdbet {lista} "
                         f"--herramientas $H\n"
                         f"  .venv/bin/python {herramienta} combina {lista} "
                         f"--split {TRAIN_SPLIT}")
        # Una por vez, con un generador: 2 000 juntas serían ~10 GB, y el dataset se queda sólo
        # con las esquinas válidas.
        mascaras = (np.load(r) for r in rutas)
    train_set = PatchDataset3D(
        archivos, patch_size=args.patch,
        length=args.lote if args.sobreajuste else (args.iterations - it0) * args.lote,
        seed=args.seed * 100, offset=0 if args.sobreajuste else it0 * args.lote,
        signo=args.aug_signo, ganancia=args.aug_ganancia, mascaras=mascaras,
        lote_epoca=args.lote if args.epocas else None,
    )
    if mascaras is not None:
        esquinas = [e.size for e in train_set.esquinas]
        print(f"máscara de cerebro: {len(train_set.con_esquinas)} de {len(esquinas)} volúmenes "
              f"con cubos de {args.patch}^3 a {train_set.margen_cerebro} vóxel o más del "
              f"borde; mediana {np.median(esquinas):.0f} esquinas por volumen", flush=True)
    # El sobreajuste no tiene loader: su único lote se arma una vez.
    if args.sobreajuste:
        return train_set, None, torch.stack([train_set[i] for i in range(args.lote)])
    if args.epocas:
        por_epoca = train_set.parches_por_epoca // args.lote
        print(f"épocas: {por_epoca} iteraciones por época ({train_set.parches_por_epoca} "
              f"volúmenes, un parche de cada uno); {args.iterations / por_epoca:.2f} épocas en "
              f"{args.iterations} iteraciones", flush=True)
        return train_set, lotes_por_epoca(train_set, args.lote, args.workers, device), None
    # Un lote entero por pedido; los micro-lotes se cortan en el bucle. Con prefetch 2 y los dos
    # workers quedan en vuelo 2 x 2 x 32 parches, los mismos que con el loader de antes, que daba
    # micro-lotes de 16 con prefetch 4 (4 x 2 x 16).
    persistentes = ({"persistent_workers": True, "prefetch_factor": 2} if args.workers else {})
    loader = DataLoader(train_set, batch_size=args.lote, shuffle=False, drop_last=True,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"),
                        **persistentes)
    return train_set, loader, None


def arma_validacion(parser: argparse.ArgumentParser, args: argparse.Namespace):
    """Los volúmenes de validación. Si están las máscaras de cerebro de todos, la validación mide
    también el PSNR ahí, que es lo que ve el entrenamiento con --mascara-cerebro. Si falta alguna,
    no lo mide y el log lo dice; con --mascara-cerebro es un error, porque `best.pth` se elige con
    ese PSNR. Devuelve (volúmenes, si mide en el cerebro, dónde busca las máscaras)."""
    mascaras_eval = Path(args.data_root) / "mascaras" / EVAL_SPLIT
    sujetos_eval = [Path(n).stem for n in lee_lista(args.eval_list)]
    faltan_eval = [s for s in sujetos_eval if not (mascaras_eval / f"{s}.npy").is_file()]
    con_cerebro = not faltan_eval
    if args.mascara_cerebro and faltan_eval:
        parser.error(f"--mascara-cerebro: faltan {len(faltan_eval)} de {len(sujetos_eval)} "
                     f"máscaras de validación en {mascaras_eval} (la primera, {faltan_eval[0]}), "
                     "y `best.pth` se elige con el PSNR en el cerebro. Se generan como las de "
                     f"entrenamiento, con --lista {args.eval_list} y --split {EVAL_SPLIT} "
                     "(docstring de datasets/mascaras_ldm100k3d.py)")
    eval_set = VolumeDataset3D.from_root(args.data_root, EVAL_SPLIT, recorte=args.eval_recorte,
                                         names=sujetos_eval,
                                         mascaras=mascaras_eval if con_cerebro else None)
    return eval_set, con_cerebro, mascaras_eval


def arma_puntos_de_estabilidad(args: argparse.Namespace, eval_set, cosmos: dict | None,
                               device: torch.device):
    """Dónde se mide la estabilidad: un cubo de validación y, con COSMOS, la entrada del primer
    denoiser del ADMM. Se arman una vez, porque no dependen del modelo, y un lado que no entra se
    recorta. Devuelve (punto de validación, punto de COSMOS), None donde no se mide."""
    punto_val = punto_qsm = None
    if args.estabilidad_lado > 0 and len(eval_set):
        punto_val = punto_validacion(eval_set, min(args.estabilidad_lado, args.eval_recorte),
                                     args.sigma)
        if cosmos is not None:
            try:
                punto_qsm = torch.from_numpy(punto_qsm3d(
                    cosmos, min(args.estabilidad_lado, *cosmos["mascara"].shape), device=device,
                    target_std=args.qsm_target_std))
            except Exception as e:
                print(f"AVISO: estabilidad en QSM apagada: el punto no se pudo armar "
                      f"({type(e).__name__}: {e}); el entrenamiento sigue", flush=True)
    return punto_val, punto_qsm


def prepara_salida(piezas, corrida) -> str | None:
    """Deja `--out` como lo describe el estado del que se sigue (sin --reanudar, como una corrida
    nueva): lo que haya de otra corrida se reescribe, y un checkpoint ajeno se aparta con otro
    nombre en vez de borrarse. Borra `estado.json`, que sólo escribe `cierra`. Devuelve el commit
    del código."""
    out, args = piezas.out, piezas.args
    out.mkdir(parents=True, exist_ok=True)
    (out / "estado.json").unlink(missing_ok=True)
    commit = commit_git()
    (out / "config.json").write_text(json.dumps({**vars(args), "commit": commit}, indent=2))
    (out / "history.json").write_text(json.dumps(corrida.history, indent=2))
    (out / "qsm.jsonl").unlink(missing_ok=True)
    if corrida.qsm_hist:
        (out / "qsm.jsonl").write_text("".join(json.dumps(fila) + "\n"
                                               for fila in corrida.qsm_hist))
    guarda_traza(out / "traza.npy", corrida.traza[:piezas.it0])
    aparta(out / "last.pth", corrida.history[-1] if corrida.history else None)
    for nombre, vigente in (("best.pth", corrida.mejor), ("best_qsm.pth", corrida.mejor_qsm)):
        aparta(out / nombre, vigente)
        if vigente is not None and not (out / nombre).exists():
            guarda_pth(vigente, out / nombre)
    return commit


def imprime_arranque(piezas, corrida) -> None:
    """Las líneas del log que dicen qué va a correr: la receta, la validación, QSM y la meseta."""
    args = piezas.args
    ruido = ((f"sigma~logU({args.sigma_min:g}, {args.sigma_max:g})" if args.sigma_log
              else f"sigma~U({args.sigma_min:g}, {args.sigma_max:g})") if args.sigma_max > 0
             else f"sigma={args.sigma:g}")
    if args.fraccion_limpias > 0:
        ruido += f", {100 * args.fraccion_limpias:g} % de parches limpios"
    trozos = (f"{len(piezas.tamanos)}x{args.micro_lote}" if len(set(piezas.tamanos)) == 1
              else "+".join(map(str, piezas.tamanos)))
    print(f"parámetros: {sum(q.numel() for q in piezas.model.parameters()):,} | "
          f"lote {args.lote} = {trozos} de {args.patch}^3 | "
          f"modo {args.mode} | datos {args.data_root} ({len(piezas.train_set.volumes)} vols, "
          f"{len(piezas.eval_set)} de eval a {args.eval_recorte}^3) | hitos lr {piezas.hitos}"
          + (f" y de meseta {corrida.meseta['hitos']}" if corrida.meseta["hitos"] else "") + " | "
          f"T_init={args.t_init:g} | lambda_init={LAMBDA_INIT:g} | {ruido} | "
          f"signo={args.aug_signo} | ganancia={args.aug_ganancia:g} | "
          f"mascara_cerebro={args.mascara_cerebro} | clip={args.clip:g} | "
          f"guard={args.guard:g}x mediana({args.guard_ventana}) | abs={args.guard_abs:g} | "
          f"rebobina a los {args.guard_seguidos} seguidos | respaldo tras "
          f"{args.guard_limpias} limpias, cada {args.guard_respaldo} con el relativo armado | "
          f"guard_amp={args.guard_amp:g}"
          + ("  [checkpointing]" if args.checkpointing else "")
          + (f"  [reanuda {args.reanudar}]" if args.reanudar else ""), flush=True)
    qsm_cuando = ("en cada validación" if args.eval_qsm_cada <= 0 else
                  f"en las validaciones múltiplo de {args.eval_qsm_cada} y en la última")
    print(f"validación: cada {args.eval_every} iteraciones y en la última | PSNR en cerebro "
          + ("sí" if piezas.con_cerebro else f"no (faltan máscaras en {piezas.mascaras_eval})")
          + " | QSM " + (f"{qsm_cuando}, COSMOS "
                         f"{'x'.join(map(str, piezas.cosmos['chi'].shape))} de "
                         f"{args.qsm_datos}, hasta {args.qsm_max_iter} iteraciones de ADMM "
                         f"con target_std {args.qsm_target_std:g}"
                         if piezas.cosmos is not None else "apagada")
          + " | estabilidad " + ("apagada (--estabilidad-lado 0)" if piezas.punto_val is None else
                                 f"con la cadencia de QSM, en {piezas.punto_val.shape[-1]}^3 de "
                                 "validación" + (f" y {piezas.punto_qsm.shape[-1]}^3 de COSMOS"
                                                 if piezas.punto_qsm is not None else "")),
          flush=True)
    # `best_qsm.pth` pide sólo la evaluación QSM: la estabilidad se registra, no elige.
    print("best_qsm.pth: " + ("el de menor NRMSE QSM mínimo del ADMM, el de su mejor iteración"
                              if piezas.cosmos is not None else "no se busca (sin COSMOS)")
          + (f" | hereda el del estado, de la {corrida.mejor_qsm['iteration']}"
             if corrida.mejor_qsm else ""), flush=True)
    if args.meseta:
        m = corrida.meseta
        print(f"meseta: lr/2 si el NRMSE QSM mínimo del ADMM no baja del mejor"
              + (" ni el PSNR de la validación sube del suyo" if args.meseta_psnr else "")
              + f" en {args.meseta} evaluaciones QSM seguidas, hasta {args.meseta_mitades} veces; "
              "la meseta siguiente corta"
              + (f"; mira desde la {args.meseta_desde}" if args.meseta_desde else "")
              + (f" | hereda {len(m['hitos'])} mitades y el mejor, {m['mejor']:.2f} % "
                 f"de la {m['it_mejor']}" if m["mejor"] is not None else ""), flush=True)
    if args.sobreajuste:
        print(f"sobreajuste: el mismo lote de {args.lote} parches, de "
              f"{len(piezas.train_set.volumes)} volúmenes, con el mismo ruido en cada iteración, "
              "y sin las cotas del disyuntor: un rechazo se repetiría igual en cada iteración",
              flush=True)


def abre_seguimiento(parser: argparse.ArgumentParser, piezas, commit: str | None):
    """La corrida de W&B, o None. Va después de `iter(loader)`: los workers ya existen y no se
    forkea con los hilos de wandb vivos. `abre` no levanta ni toca los generadores."""
    args = piezas.args
    run = seguimiento.abre(piezas.out, {**vars(args), "commit": commit}, modo=args.seguimiento,
                           proyecto=args.wandb_proyecto, nombre=piezas.out.resolve().name,
                           padre=args.reanudar, desde=piezas.it0)
    if args.seguimiento == "no":
        print("seguimiento: sin W&B (--seguimiento no)", flush=True)
    elif args.seguimiento == "online" and run is None:
        # Antes de la primera iteración: con el supervisor mirando W&B, mejor no arrancar a ciegas.
        parser.error("--seguimiento online: W&B no abrió (el motivo, en la línea «seguimiento:» "
                     "de arriba). Con --seguimiento auto la corrida sigue offline")
    return run


def restaura_generadores(estado: dict) -> None:
    """Los generadores global y de CUDA del estado. Va después de `iter(loader)`, que sortea una
    semilla del generador global: la corrida de origen lo hizo una sola vez, al arrancar."""
    torch.set_rng_state(estado["rng_cpu"])
    if torch.cuda.is_available() and len(estado["rng_cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(estado["rng_cuda"])


# --- EL BUCLE Y SUS PASOS --------------------------------------------------------------------

def gradiente_del_lote(piezas, mide_corr: bool) -> tuple:
    """Forward y backward de un lote, micro-lote por micro-lote. El gradiente queda acumulado en
    los parámetros, cada trozo con su peso en la media; el paso de Adam lo da el bucle.

    Devuelve (la pérdida del lote, |g| del primer micro-lote como si fuera el lote, la suma de los
    sigma, ‖p_0‖ y ‖p_S‖ con --guard-amp, cuyo cociente es la amplificación adjunta, y `corr`)."""
    args, model, device = piezas.args, piezas.model, piezas.device
    acumulacion = len(piezas.tamanos)
    piezas.opt.zero_grad(set_to_none=True)
    perdida = torch.zeros((), device=device)
    p0, pS = torch.zeros((), device=device), torch.zeros((), device=device)
    suma_sigma = 0.0
    lote = piezas.lote_fijo if args.sobreajuste else next(piezas.lotes)
    for i, trozo in enumerate(torch.split(lote, piezas.tamanos)):
        # Cuántas veces cabe el trozo en el lote: su peso en la media. Con trozos iguales es la
        # cantidad de trozos, entera y exacta, y la aritmética es la de siempre.
        veces = args.lote / len(trozo)
        if args.sigma_max > 0 or args.fraccion_limpias > 0:
            # Con σ fijo y una fracción de limpias, el σ fijo es el rango entero.
            bajo, alto = ((args.sigma_min, args.sigma_max) if args.sigma_max > 0
                          else (args.sigma, args.sigma))
            noisy, sigmas = add_noise_ciego(trozo, bajo, alto, piezas.noise_gen, NOISE_SCALE,
                                            log=args.sigma_log,
                                            fraccion_limpias=args.fraccion_limpias)
            suma_sigma += float(sigmas.sum())
        else:
            noisy = add_noise(trozo, args.sigma, piezas.noise_gen, NOISE_SCALE)
            suma_sigma += args.sigma * len(trozo)
        noisy = noisy.to(device, non_blocking=True)
        clean = trozo.to(device, non_blocking=True)
        if piezas.mide_amp:
            noisy.requires_grad_(True)

        with torch.autocast(device.type, dtype=piezas.mode["amp"],
                            enabled=piezas.mode["amp"] is not None):
            states = model(noisy, noisy)
            loss = (states[-1].float() - clean.float()).pow(2).flatten(1).sum(1).mean()

        (loss / veces).backward()
        perdida += loss.detach() / veces
        # Sólo LEE `.grad`: el backward siguiente acumula sobre el mismo tensor y da los mismos
        # bits. Por `veces`, es la norma de este micro-lote como si fuera el lote.
        if i == 0:
            g_micro = veces * get_total_norm(
                [q.grad for q in model.parameters() if q.grad is not None]).item()
        if piezas.mide_amp:
            p0 += noisy.grad.norm() / acumulacion
            pS += 2 * (states[-1].detach().float() - clean.float()).norm() / (
                args.lote * acumulacion)

    # `corr`, en las iteraciones de log: cuánto movieron la imagen los S pasos, relativo al ruido,
    # en el último micro-lote. ~1 es denoising sano.
    corr = math.nan
    if mide_corr:
        with torch.no_grad():
            corr = ((states[-1] - states[0]).norm() / (noisy - clean).norm()).item()
    return perdida, g_micro, suma_sigma, p0, pS, corr


def alerta(piezas, titulo: str, texto: str) -> None:
    """Lo que el autor tiene que saber sin mirar la corrida: al log y, si hay seguimiento, como
    alerta de W&B (AGENTS.md, «Entrenamiento 3D»)."""
    print(f"    ALERTA: {titulo}: {texto}", flush=True)
    seguimiento.alerta(piezas.run, titulo, texto)


def aplica_disyuntor(piezas, corrida, it: int, rechazado: bool, gnorm: float, centro: float,
                     amp: float, perdida: torch.Tensor) -> bool:
    """Lo que hace el disyuntor después del paso de Adam: cuenta el rechazo y, tras
    --guard-seguidos seguidos, rebobina pesos, Adam y ventana al respaldo. Tras --guard-limpias
    aceptadas seguidas, toma el primer respaldo y, con el relativo armado, lo refresca cada
    --guard-respaldo iteraciones: sin él, una aceptada es sólo una finita, y el respaldo podría ser
    una red ya rota. Dos rebobinados seguidos al mismo respaldo son una corrida TRABADA, que se
    avisa y no se corta. Devuelve si rebobinó."""
    args = piezas.args
    if not rechazado:
        corrida.seguidos = 0
        corrida.limpias += 1
        armado = math.isfinite(args.guard) and len(corrida.escala) >= args.guard_ventana
        if corrida.limpias >= args.guard_limpias and (
                corrida.respaldo is None
                or (armado and it - corrida.it_respaldo >= args.guard_respaldo)):
            corrida.respaldo = respalda(piezas.model, piezas.opt, corrida.escala)
            corrida.it_respaldo = it
        return False
    corrida.rechazos += 1
    corrida.seguidos, corrida.limpias = corrida.seguidos + 1, 0
    corrida.max_seguidos = max(corrida.max_seguidos, corrida.seguidos)
    if corrida.respaldo is not None and 0 < args.guard_seguidos <= corrida.seguidos:
        rebobina(piezas.model, piezas.opt, corrida.respaldo, corrida.escala)
        # El rebobinado anterior fue después de este respaldo: se vuelve por segunda vez seguida
        # al mismo estado, y la red volvió a entrar en el estallido desde ahí.
        trabada = corrida.it_rebobinado > corrida.it_respaldo
        # Se pierde lo hecho desde el respaldo, salvo lo que un rebobinado anterior al mismo
        # respaldo ya había descartado.
        corrida.descartadas += min(it - corrida.it_respaldo, it - corrida.it_rebobinado)
        corrida.it_rebobinado = it
        corrida.rebobinados, corrida.seguidos = corrida.rebobinados + 1, 0
        print(f"[{it:6d}] REBOBINADO a la it {corrida.it_respaldo} "
              f"({corrida.rebobinados} en lo que va) |g|={gnorm:.3g}", flush=True)
        alerta(piezas, "rebobinado", f"{args.guard_seguidos} rechazos seguidos en la it {it}; "
               f"vuelve a la it {corrida.it_respaldo}")
        if trabada:
            alerta(piezas, "TRABADA", f"dos rebobinados seguidos al respaldo de la it "
                   f"{corrida.it_respaldo} (en la {it}). Si sigue, se relanza con otra semilla")
        return True
    if corrida.seguidos <= 3 or corrida.seguidos % 1000 == 0:
        print(f"[{it:6d}] LOTE RECHAZADO |g|={gnorm:.1f} mediana={centro:.1f} "
              f"amp={amp:.3g} loss={perdida.item():.3f} (seguidos {corrida.seguidos})", flush=True)
    return False


def registra_progreso(piezas, corrida, it: int, perdida: torch.Tensor, corr: float, gnorm: float,
                      lr: float, t0: float, reloj: tuple[float, int]) -> None:
    """La línea de log de cada --log-every y sus métricas en W&B."""
    args, model = piezas.args, piezas.model
    eta = (time.time() - t0) / (it - piezas.it0) * (args.iterations - it) / 3600
    print(f"[{it:6d}/{args.iterations}] loss={perdida.item():8.3f} lr={lr:.2e} "
          f"T={model.T.item():.5f} lambda={model.lmbda.item():.5f} "
          f"|g|={gnorm:.1f} max|g|={corrida.max_grad:.3g} corr={corr:.3f} "
          f"rech={corrida.rechazos} reb={corrida.rebobinados} ef={it - corrida.descartadas} "
          f"({eta:.1f}h restantes)", flush=True)
    # El max|g| del intervalo va también: un estallido entre dos muestras no se pierde.
    extra = {"train/grad_norm_max": corrida.max_grad,
             "train/s_por_it": (time.time() - reloj[0]) / (it - reloj[1])}
    if piezas.device.type == "cuda":
        extra["train/vram_pico_gib"] = torch.cuda.max_memory_allocated(piezas.device) / 2 ** 30
    if args.epocas:   # épocas completadas, con fracción: el eje x alternativo de W&B
        extra["train/epoca"] = it * args.lote / piezas.train_set.parches_por_epoca
    seguimiento.registra(piezas.run, seguimiento.metricas_traza(corrida.traza, it) | extra, it)


def valida_denoising(piezas, corrida, it: int, perdida: torch.Tensor):
    """La validación de denoising: una entrada en `history.json`, `traza.npy` y `last.pth` y, si
    es la mejor, `best.pth`. Devuelve (la entrada, el PSNR, el PSNR que elige `best.pth`)."""
    args, model, out = piezas.args, piezas.model, piezas.out
    model.eval()
    report = evaluate_denoising3d(model, piezas.eval_set, args.sigma, piezas.device,
                                  seed=SEMILLA_VAL, data_range=DATA_RANGE,
                                  noise_scale=NOISE_SCALE, amp=piezas.mode["amp"],
                                  devuelve_primero=piezas.run is not None)
    model.train()
    psnr = report["psnr_denoised"]
    entrada = {"iteration": it, "efectiva": it - corrida.descartadas, "psnr": psnr,
               "psnr_tejido": report["psnr_denoised_tejido"],
               "nrmse_pct": report["nrmse_denoised"],
               "nrmse_pct_tejido": report["nrmse_denoised_tejido"],
               "T": model.T.item(), "lambda": model.lmbda.item()}
    if piezas.con_cerebro:
        entrada["psnr_cerebro"] = report["psnr_denoised_cerebro"]
        entrada["nrmse_pct_cerebro"] = report["nrmse_denoised_cerebro"]
    if args.epocas:
        entrada["epoca"] = it * args.lote / piezas.train_set.parches_por_epoca
    if args.sobreajuste:
        # La pérdida suma C·P^3 valores por parche: dividida, es el MSE del lote fijo.
        mse = perdida.item() / (CHANNELS * args.patch ** 3)
        entrada["psnr_lote"] = 10 * math.log10(DATA_RANGE ** 2 / mse) if mse > 0 else math.inf
    corrida.history.append(entrada)
    (out / "history.json").write_text(json.dumps(corrida.history, indent=2))

    guarda_traza(out / "traza.npy", corrida.traza[:it])
    payload = {"config": piezas.config, "model": model.state_dict(), "iteration": it,
               "psnr": psnr, "optimizer": piezas.opt.state_dict()}
    guarda_pth(payload, out / "last.pth")
    # Con --mascara-cerebro la red no ve fondo ni cráneo: elige el PSNR en el cerebro.
    psnr_criterio = entrada["psnr_cerebro"] if args.mascara_cerebro else psnr
    psnr_ruidosa = report["psnr_noisy_cerebro" if args.mascara_cerebro else "psnr_noisy"]
    caida = corrida.best_psnr - psnr_criterio if corrida.best_psnr else 0.0
    if psnr_criterio > corrida.best_psnr:
        corrida.best_psnr = psnr_criterio
        guarda_pth(payload, out / "best.pth")
        # Una copia: `payload` apunta a los pesos vivos. El estado la lleva para reponer
        # `best.pth` al reanudar.
        corrida.mejor = copy.deepcopy(payload)
    print(f"    eval PSNR={psnr:.3f} dB, en tejido {report['psnr_denoised_tejido']:.3f} dB"
          + (f", en cerebro {entrada['psnr_cerebro']:.3f} dB" if piezas.con_cerebro else "")
          + f" | NRMSE {entrada['nrmse_pct']:.2f} %, en tejido "
          f"{entrada['nrmse_pct_tejido']:.2f} % ({report['num_volumenes']} vols)"
          + (f" | lote fijo {entrada['psnr_lote']:.3f} dB" if args.sobreajuste else "")
          + ("  *mejor*" if psnr_criterio == corrida.best_psnr else ""), flush=True)
    # Un estallido hunde el PSNR decenas de dB (`ciego-epocas-l32`: −32 dB en su primera
    # validación). Peor que la entrada ruidosa lo avisa aunque sea la primera.
    if psnr_criterio < psnr_ruidosa:
        alerta(piezas, "la red empeora la entrada", f"PSNR {psnr_criterio:.2f} dB en la it {it}, "
               f"contra {psnr_ruidosa:.2f} dB de la ruidosa")
    elif caida >= CAIDA_PSNR:
        alerta(piezas, "PSNR de validación", f"{psnr_criterio:.2f} dB en la it {it}, "
               f"{caida:.1f} dB bajo el mejor")
    seguimiento.registra(piezas.run, {f"val/{k}": v for k, v in entrada.items()
                                      if k not in ("iteration", "efectiva", "T", "lambda")}, it)
    registra_lamina(piezas.run, "val/denoising", lambda: lamina_denoising(*report["primero"]),
                    it, f"{piezas.eval_set.files[0].stem}: limpia | ruidosa | salida | "
                        "5·|salida − limpia|, en [0, 1]")
    return entrada, psnr, psnr_criterio


def valida_niveles(piezas, corrida, it: int, entrada: dict) -> None:
    """--val-snr: el denoising de validación a otros SNR, con las mismas semillas; a su entrada de
    `history.json`, al log y a W&B (val_snr<SNR>/*, con su lámina). No elige checkpoints."""
    args, model = piezas.args, piezas.model
    for snr in args.val_snr:
        model.eval()
        rep = evaluate_denoising3d(model, piezas.eval_set, DATA_RANGE / snr * NOISE_SCALE,
                                   piezas.device, seed=SEMILLA_VAL, data_range=DATA_RANGE,
                                   noise_scale=NOISE_SCALE, amp=piezas.mode["amp"],
                                   devuelve_primero=piezas.run is not None)
        model.train()
        sufijo = "_cerebro" if piezas.con_cerebro else ""
        m = {"psnr": rep[f"psnr_denoised{sufijo}"], "psnr_entrada": rep[f"psnr_noisy{sufijo}"]}
        if piezas.con_cerebro:
            m |= {"nrmse_pct": rep["nrmse_denoised_cerebro"],
                  "nrmse_entrada_pct": rep["nrmse_noisy_cerebro"]}
        clave = f"val_snr{snr:g}"
        entrada |= {f"{clave}_{k}": v for k, v in m.items()}
        print(f"    eval SNR {snr:g}{' en cerebro' if sufijo else ''}: PSNR "
              f"{m['psnr_entrada']:.3f} → {m['psnr']:.3f} dB"
              + (f" | NRMSE {m['nrmse_entrada_pct']:.2f} → {m['nrmse_pct']:.2f} %"
                 if "nrmse_pct" in m else ""), flush=True)
        seguimiento.registra(piezas.run, {f"{clave}/{k}": v for k, v in m.items()}, it)
        registra_lamina(piezas.run, f"{clave}/denoising",
                        lambda: lamina_denoising(*rep["primero"]), it,
                        f"{piezas.eval_set.files[0].stem}, SNR {snr:g}: limpia | ruidosa | "
                        "salida | 5·|salida − limpia|, en [0, 1]")
    (piezas.out / "history.json").write_text(json.dumps(corrida.history, indent=2))


def valida_sim2(piezas, corrida, it: int) -> None:
    """La evaluación QSM en Sim2: el mismo ADMM que en COSMOS, a `qsm_sim2.jsonl` y a W&B
    (qsm_sim2/*, con su lámina). No elige checkpoints ni toca la meseta."""
    args = piezas.args
    r = evalua_qsm(piezas.model, piezas.sim2, piezas.device, piezas.mode["amp"],
                   args.qsm_max_iter, args.qsm_target_std)
    met = metricas_qsm(r, piezas.sim2) if "fallo" not in r else {}
    fila = {"iteration": it, "efectiva": it - corrida.descartadas,
            "target_std": args.qsm_target_std,
            **{k: v for k, v in r.items() if k not in ("chi", "chi0")}, **met}
    anota_jsonl(piezas.out / "qsm_sim2.jsonl", fila)
    if "fallo" in r:
        print(f"    qsm Sim2 FALLÓ ({r['fallo']}); el entrenamiento sigue", flush=True)
        seguimiento.registra(piezas.run, {"qsm_sim2/fallo": 1}, it)
        return
    print(f"    qsm Sim2 NRMSE={r['nrmse_final_pct']:.2f} % (mínimo {r['nrmse_min_pct']:.2f} % en "
          f"la iteración {r['iter_min']}) | alpha={r['alpha']:.4f} | {r['segundos']:.1f} s"
          + linea_metricas(met), flush=True)
    seguimiento.registra(piezas.run, {"qsm_sim2/fallo": 0, **{
        f"qsm_sim2/{k}": r[k] for k in ("nrmse_final_pct", "nrmse_min_pct", "iter_min",
                                        "iteraciones", "alpha", "segundos")},
        **{f"qsm_sim2/{k}": v for k, v in met.items()}}, it)
    registra_lamina(piezas.run, "qsm_sim2/reconstruccion",
                    lambda: lamina_qsm(r["chi"], piezas.sim2["chi"], piezas.sim2["mascara_eval"],
                                       orientacion=ORIENTACION_T1, chi0=r["chi0"]),
                    it, "Sim2: GT | χ₀ sin red | reconstrucción | error, en la máscara, ±0.1 ppm")


def valida_estabilidad(piezas, corrida, it: int, entrada: dict) -> None:
    """El número de estabilidad en el punto de validación, en su entrada de `history.json`."""
    est = mide_estabilidad(piezas.model, piezas.punto_val, piezas.device)
    entrada.update(claves_estabilidad(est))
    (piezas.out / "history.json").write_text(json.dumps(corrida.history, indent=2))
    print(linea_estabilidad("val", est), flush=True)
    if "fallo" not in est:
        seguimiento.registra(piezas.run, {"val/estabilidad": est["paso"],
                                          "val/lambda_max": est["lambda_max"]}, it)


def aplica_meseta(piezas, corrida, it: int, nrmse: float, psnr: float) -> None:
    """--meseta: anota la evaluación y, si el NRMSE (y con --meseta-psnr, también el PSNR) lleva
    N evaluaciones sin récord, parte el lr o, agotadas las mitades, marca el corte."""
    args, m = piezas.args, corrida.meseta
    accion = paso_meseta(m, nrmse, it, args.meseta, args.meseta_mitades,
                         psnr=psnr if args.meseta_psnr else None)
    visto = ("ningún NRMSE finito" if m["mejor"] is None else
             f"{m['mejor']:.2f} % (la {m['it_mejor']})")
    if m.get("mejor_psnr") is not None:
        visto += f" ni subir de {m['mejor_psnr']:.3f} dB (la {m['it_mejor_psnr']})"
    if accion == "nula":
        que = "NRMSE mínimo" if not math.isfinite(nrmse) else "PSNR"
        print(f"    meseta: sin un {que} finito, esta evaluación no cuenta", flush=True)
    elif accion == "mitad":
        print(f"    meseta: {args.meseta} evaluaciones QSM sin bajar de {visto}: lr/2 desde la "
              f"{it + 1}, mitad {len(m['hitos'])} de {args.meseta_mitades}", flush=True)
    elif accion == "corte":
        corrida.corte = it
        print(f"    meseta: {args.meseta} evaluaciones QSM sin bajar de {visto} tras "
              f"{args.meseta_mitades} mitades del lr: la corrida se corta en la {it}",
              flush=True)
    seguimiento.registra(piezas.run, {"meseta/mitades": len(m["hitos"]),
                                      "meseta/sin_mejora": m["sin_mejora"]}
                         | ({} if m["mejor"] is None else
                            {"meseta/mejor_nrmse_min_pct": m["mejor"]})
                         | ({} if m.get("mejor_psnr") is None else
                            {"meseta/sin_mejora_psnr": m["sin_mejora_psnr"],
                             "meseta/mejor_psnr": m["mejor_psnr"]}), it)


def valida_qsm(piezas, corrida, it: int, psnr: float,
               psnr_criterio: float) -> None:
    """Una evaluación QSM: la estabilidad en la entrada del ADMM, el ADMM con la red como denoiser
    sobre COSMOS, una fila en `qsm.jsonl`, `best_qsm.pth` si su NRMSE mínimo es el menor visto, la
    lámina en W&B y, con --meseta, la decisión sobre el lr."""
    args = piezas.args
    est = (mide_estabilidad(piezas.model, piezas.punto_qsm, piezas.device)
           if piezas.punto_qsm is not None else None)
    r = evalua_qsm(piezas.model, piezas.cosmos, piezas.device, piezas.mode["amp"],
                   args.qsm_max_iter, args.qsm_target_std)
    met = metricas_qsm(r, piezas.cosmos) if "fallo" not in r else {}
    fila = {"iteration": it, "efectiva": it - corrida.descartadas,
            "target_std": args.qsm_target_std,
            **{k: v for k, v in r.items() if k not in ("chi", "chi0")}, **met,
            **(claves_estabilidad(est) if est is not None else {})}
    corrida.qsm_hist.append(fila)
    anota_jsonl(piezas.out / "qsm.jsonl", fila)
    if est is not None:
        print(linea_estabilidad("qsm", est), flush=True)
        if "fallo" not in est:
            seguimiento.registra(piezas.run, {"qsm/estabilidad": est["paso"],
                                              "qsm/lambda_max": est["lambda_max"]}, it)
    if "fallo" in r:
        print(f"    qsm FALLÓ ({r['fallo']}); el entrenamiento sigue", flush=True)
        seguimiento.registra(piezas.run, {"qsm/fallo": 1}, it)
        return
    print(f"    qsm NRMSE={r['nrmse_final_pct']:.2f} % (mínimo "
          f"{r['nrmse_min_pct']:.2f} % en la iteración {r['iter_min']} del ADMM)"
          f" | {r['iteraciones']} iteraciones, motivo {r['motivo']} | "
          f"alpha={r['alpha']:.4f} | {r['segundos']:.1f} s" + linea_metricas(met), flush=True)
    if r["iter_min"] == 0:
        alerta(piezas, "el ADMM devuelve χ0", f"en la it {it}, el mínimo del ADMM en COSMOS "
               f"({r['nrmse_min_pct']:.2f} %) es su iteración 0: la red no mejora la "
               "reconstrucción sin red")
    # `best_qsm.pth`: la evaluación de menor NRMSE mínimo del ADMM, lo mismo que mira la meseta.
    # El final, el motivo y la estabilidad viajan con él, pero no eligen.
    if r["nrmse_min_pct"] < (corrida.mejor_qsm["nrmse_min_pct"] if corrida.mejor_qsm
                             else math.inf):
        # Una copia, como `mejor`: el estado la lleva para reponer el archivo.
        corrida.mejor_qsm = copy.deepcopy({
            "config": piezas.config, "model": piezas.model.state_dict(), "iteration": it,
            "psnr": psnr, **{k: r[k] for k in ("nrmse_min_pct", "iter_min", "nrmse_final_pct",
                                               "iteraciones", "motivo")},
            "estabilidad": (est["paso"] if est is not None and "fallo" not in est else None)})
        guarda_pth(corrida.mejor_qsm, piezas.out / "best_qsm.pth")
        print(f"    best_qsm.pth: NRMSE QSM mínimo {r['nrmse_min_pct']:.2f} % "
              f"(final {r['nrmse_final_pct']:.2f} %, motivo {r['motivo']}), el "
              "menor hasta ahora", flush=True)
    seguimiento.registra(piezas.run, {"qsm/fallo": 0, **{
        f"qsm/{k}": r[k] for k in ("nrmse_final_pct", "nrmse_min_pct", "iter_min",
                                   "iteraciones", "alpha", "segundos")},
        **{f"qsm/{k}": v for k, v in met.items()}}, it)
    registra_lamina(piezas.run, "qsm/reconstruccion",
                    lambda: lamina_qsm(r["chi"], piezas.cosmos["chi"],
                                       piezas.cosmos["mascara_eval"], chi0=r["chi0"]),
                    it, "COSMOS: GT | χ₀ sin red | reconstrucción | error, en la máscara, "
                        "±0.1 ppm")
    # La meseta mira las evaluaciones sin fallo desde --meseta-desde, y nunca la última: con la
    # corrida terminando, ya no hay lr que partir.
    if args.meseta and args.meseta_desde <= it < args.iterations:
        aplica_meseta(piezas, corrida, it, r["nrmse_min_pct"], psnr_criterio)


def valida(piezas, corrida, it: int, perdida: torch.Tensor) -> None:
    """Una validación: denoising siempre; estabilidad y QSM, con la cadencia de --eval-qsm-cada."""
    args = piezas.args
    entrada, psnr, psnr_criterio = valida_denoising(piezas, corrida, it, perdida)
    if args.val_snr:
        valida_niveles(piezas, corrida, it, entrada)
    # Después de `valida_denoising`, que ya escribió `history.json` y los `.pth`: un corte duro a
    # mitad de la estabilidad o del ADMM no los pierde.
    toca_qsm = args.eval_qsm_cada <= 0 or it % args.eval_qsm_cada == 0 or it == args.iterations
    if piezas.punto_val is not None and toca_qsm:
        valida_estabilidad(piezas, corrida, it, entrada)
    if piezas.cosmos is not None and toca_qsm:
        valida_qsm(piezas, corrida, it, psnr, psnr_criterio)
    if piezas.sim2 is not None and toca_qsm:
        valida_sim2(piezas, corrida, it)


def guarda_estado(piezas, corrida, it: int, t0: float) -> None:
    """`estado-<it>.pth`: todo lo que hace falta para seguir, bit a bit, desde esta iteración. Al
    reanudar lo leen `lee_estado` (las banderas), `arma_modelo` (los pesos), `main` (Adam y el
    ruido), `corrida_desde_estado` (las cuentas) y `restaura_generadores`."""
    guarda_traza(piezas.out / "traza.npy", corrida.traza[:it])
    guarda_pth({
        "iteration": it, "config": piezas.config, "args": vars(piezas.args),
        "model": piezas.model.state_dict(), "optimizer": piezas.opt.state_dict(),
        "escala": list(corrida.escala), "respaldo": corrida.respaldo,
        "it_respaldo": corrida.it_respaldo, "rechazos": corrida.rechazos,
        "seguidos": corrida.seguidos, "max_seguidos": corrida.max_seguidos,
        "rebobinados": corrida.rebobinados, "limpias": corrida.limpias,
        "descartadas": corrida.descartadas, "it_rebobinado": corrida.it_rebobinado,
        "history": corrida.history, "best_psnr": corrida.best_psnr, "mejor": corrida.mejor,
        "traza": corrida.traza[:it].copy(), "qsm": corrida.qsm_hist,
        "mejor_qsm": corrida.mejor_qsm, "meseta": corrida.meseta, "max_grad": corrida.max_grad,
        "noise_gen": piezas.noise_gen.get_state(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "horas": corrida.horas_previas + (time.time() - t0) / 3600,
    }, piezas.out / f"estado-{it:06d}.pth")


def entrena(piezas, corrida, t0: float) -> None:
    """El bucle: una iteración por vuelta, hasta --iterations o hasta que corte la meseta. `t0` es
    la hora del arranque: de ahí salen la ETA del log y las horas del estado."""
    args, model, opt = piezas.args, piezas.model, piezas.opt
    # Desde dónde mide `train/s_por_it`: se reinicia después de cada log, validación y estado,
    # para contar sólo iteraciones de entrenamiento.
    reloj = (t0, piezas.it0)
    for it in range(piezas.it0 + 1, args.iterations + 1):
        # El lr sale del calendario en CADA iteración: así una rama reanuda con otro calendario,
        # y ningún estado restaurado puede traer un lr viejo.
        for grupo in opt.param_groups:
            grupo["lr"] = lr_en(it, piezas.hitos + corrida.meseta["hitos"])
        # El sobreajuste repite también el ruido: el generador vuelve a su semilla.
        if args.sobreajuste:
            piezas.noise_gen.manual_seed(args.seed * 100 + 1)
        perdida, g_micro, suma_sigma, p0, pS, corr = gradiente_del_lote(
            piezas, mide_corr=it % args.log_every == 0)

        # La norma ANTES de recortar, que es la que sirve de diagnóstico.
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
        amp = (p0 / pS).item() if piezas.mide_amp else float("nan")
        rechazado, centro = rechaza_lote(gnorm, amp, corrida.escala, args.guard, args.guard_abs,
                                         args.guard_amp, args.guard_ventana)
        # loss T dT lambda dlambda |g| centro amp rechazado rebobinado |g|micro lr efectiva sigma
        # Las cinco primeras, antes del paso de Adam: los valores que produjeron esta pérdida.
        corrida.traza[it - 1, :5] = torch.stack((perdida, model.T.detach(), model.T.grad,
                                                 model.lmbda.detach(),
                                                 model.lmbda.grad)).cpu().numpy()
        # Se lee del optimizador y no del calendario: es el que de verdad usa `opt.step()`.
        lr = opt.param_groups[0]["lr"]
        if not rechazado:
            opt.step()
        model.project_()   # media cero y norma <= 1 en K1, positividad de T y lambda
        rebobinado = aplica_disyuntor(piezas, corrida, it, rechazado, gnorm, centro, amp, perdida)
        corrida.traza[it - 1, 5:] = (gnorm, centro, amp, rechazado, rebobinado, g_micro, lr,
                                     it - corrida.descartadas, suma_sigma / args.lote)
        corrida.max_grad = max(corrida.max_grad, gnorm)

        if it % args.log_every == 0:
            registra_progreso(piezas, corrida, it, perdida, corr, gnorm, lr, t0, reloj)
            reloj = (time.time(), it)
            corrida.max_grad = 0.0
        if it % args.eval_every == 0 or it == args.iterations:
            valida(piezas, corrida, it, perdida)
            reloj = (time.time(), it)
        # Después de la validación, para que el estado lleve su entrada de `history`. Siempre en
        # la última: una corrida terminada se puede extender sin rehacer nada.
        if args.guardar_cada and (it % args.guardar_cada == 0 or it == args.iterations
                                  or corrida.corte):
            guarda_estado(piezas, corrida, it, t0)
            reloj = (time.time(), it)
        # Que el forward siguiente no cargue con los tensores de este lote.
        del perdida, p0, pS
        # El único corte del bucle, y queda escrito: `estado.json` lleva `corte` y las hechas.
        if corrida.corte:
            break


def cierra(piezas, corrida, t0: float) -> None:
    """`estado.json`, que marca la corrida como terminada, la última línea del log y el cierre de
    W&B."""
    args = piezas.args
    hechas = corrida.corte or args.iterations
    peak = torch.cuda.max_memory_allocated() / 2**30 if piezas.device.type == "cuda" else 0.0
    horas = corrida.horas_previas + (time.time() - t0) / 3600
    final = {
        "terminada": True, "iteraciones_pedidas": args.iterations,
        "iteraciones_hechas": hechas, "iteracion_efectiva": hechas - corrida.descartadas,
        "corte": corrida.corte and {"motivo": "meseta del NRMSE QSM"
                                    + (" y del PSNR" if args.meseta_psnr else ""),
                                    "iteracion": corrida.corte,
                                    "mitades": len(corrida.meseta["hitos"])},
        "meseta": corrida.meseta if args.meseta or corrida.meseta["hitos"] else None,
        "mejor_psnr": corrida.best_psnr,
        "psnr_final": corrida.history[-1]["psnr"] if corrida.history else None,
        "rechazos": corrida.rechazos, "max_seguidos": corrida.max_seguidos,
        "rebobinados": corrida.rebobinados,
        "reanudada_de": args.reanudar, "horas": round(horas, 3),
        "vram_pico_gb": round(peak, 2), "vram_predicha_gb": round(piezas.predicha, 2),
        **resumen_qsm(corrida.qsm_hist),
        "best_qsm": corrida.mejor_qsm and {k: corrida.mejor_qsm.get(k) for k in RESUMEN_BEST_QSM}}
    (piezas.out / "estado.json").write_text(json.dumps(final, indent=2))
    print(f"\nmejor PSNR{' en el cerebro' if args.mascara_cerebro else ''}: "
          f"{corrida.best_psnr:.3f} dB | {horas:.2f} h | VRAM pico {peak:.2f} GiB "
          f"(predicho {piezas.predicha:.2f}) | lotes rechazados {corrida.rechazos} de {hechas} "
          f"(máx. seguidos {corrida.max_seguidos}) | rebobinados {corrida.rebobinados} | "
          f"iteración efectiva {hechas - corrida.descartadas}"
          + (f" | cortada por meseta en la {corrida.corte}, tras "
             f"{len(corrida.meseta['hitos'])} mitades del lr" if corrida.corte else "")
          + (f" | QSM: mejor NRMSE {final['qsm_nrmse_mejor_pct']:.2f} % en la "
             f"{final['qsm_iteracion_mejor']}, {final['qsm_fallos']} fallos"
             if final["qsm_nrmse_mejor_pct"] is not None else ""), flush=True)
    # Después de `estado.json`: online, cerrar puede tardar en subir lo pendiente.
    seguimiento.resumen(piezas.run, final)
    seguimiento.cierra(piezas.run)


# --- MAIN ------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Arma la corrida paso a paso, entrena y cierra. El orden importa: la semilla va antes del
    modelo, y el generador global se restaura después de arrancar el loader."""
    parser = opciones()
    args = parser.parse_args(argv)
    revisa_banderas(parser, args)
    mide_amp = math.isfinite(args.guard_amp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.sonda_vram and device.type != "cuda":
        parser.error("--sonda-vram mide el pico de memoria de una GPU y torch no ve ninguna "
                     "(torch.cuda.is_available() es False): correrla en la tarjeta de destino")
    presupuesto, tamanos, predicha = elige_micro_lote(parser, args, device, mide_amp)
    # Los del calendario: los de --hitos o, sin la bandera, los del paper. Los de la meseta van
    # aparte, en `corrida.meseta`.
    hitos = args.hitos if args.hitos is not None else list(
        range(LR_HALVE_EVERY, args.iterations + 1, LR_HALVE_EVERY))
    mode, out = MODES[args.mode], Path(args.out)

    # Después de completar las banderas: el estado se compara con las efectivas.
    estado, it0 = lee_estado(parser, args, out)
    prepara_backend(args.seed, mode)
    config = vnet_config3d(args.t_init, args.features)
    model = arma_modelo(parser, args, config, device, estado)
    # Antes de los datos: la sonda no los usa, y en `--out` no escribe más que su json.
    if args.sonda_vram:
        sonda_vram(model, args, mode, presupuesto, out)
        return

    train_set, loader, lote_fijo = arma_entrenamiento(parser, args, it0, device)
    eval_set, con_cerebro, mascaras_eval = arma_validacion(parser, args)
    # COSMOS una sola vez y en CPU: `admm_wtdv` lo pasa al device en cada evaluación.
    cosmos = carga_qsm(args.qsm_datos)
    if args.meseta and cosmos is None:
        parser.error(f"--meseta {args.meseta}: COSMOS no se pudo cargar de {args.qsm_datos} "
                     "(el AVISO de arriba), y sin evaluación QSM no hay NRMSE con qué decidir")
    sim2 = carga_sim2_o_aviso(args.qsm_sim2)
    punto_val, punto_qsm = arma_puntos_de_estabilidad(args, eval_set, cosmos, device)

    # Sin scheduler: el lr lo fija `lr_en` al principio de cada iteración.
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS)
    noise_gen = torch.Generator(device="cpu").manual_seed(args.seed * 100 + 1)
    if estado is not None:
        opt.load_state_dict(estado["optimizer"])
        noise_gen.set_state(estado["noise_gen"])
        corrida = corrida_desde_estado(estado, args.iterations)
    else:
        corrida = corrida_nueva(args.iterations)

    # `lotes` (el loader en marcha) y `run` (W&B) se llenan más abajo; `None` con --sobreajuste y
    # sin seguimiento.
    piezas = SimpleNamespace(
        args=args, device=device, mode=mode, out=out, it0=it0, hitos=hitos, tamanos=tamanos,
        predicha=predicha, mide_amp=mide_amp, config=config, model=model, opt=opt,
        noise_gen=noise_gen, train_set=train_set, lote_fijo=lote_fijo, eval_set=eval_set,
        con_cerebro=con_cerebro, mascaras_eval=mascaras_eval, cosmos=cosmos, sim2=sim2,
        punto_val=punto_val, punto_qsm=punto_qsm, lotes=None, run=None)
    commit = prepara_salida(piezas, corrida)
    imprime_arranque(piezas, corrida)
    # Arranca los workers y sortea una semilla del generador global: antes de W&B y de restaurar
    # los generadores.
    piezas.lotes = iter(loader) if loader is not None else None
    piezas.run = abre_seguimiento(parser, piezas, commit)
    if estado is not None:
        restaura_generadores(estado)
        del estado

    t0 = time.time()
    entrena(piezas, corrida, t0)
    cierra(piezas, corrida, t0)


if __name__ == "__main__":
    main()

'''
# --- BUCLE -----------------------------------------------------------------------------------
# El de `train.py`, repartido en funciones con nombre: `main` arma `piezas` y `corrida` y llama a
# `entrena`, que en cada iteración
#   - fija el lr con `lr_en`;
#   - hace el forward y el backward del lote, micro-lote por micro-lote (`gradiente_del_lote`);
#   - deja que `rechaza_lote` decida si hay paso de Adam, re-proyecta y, con `aplica_disyuntor`,
#     rebobina o refresca el respaldo;
#   - escribe una fila de traza y, cada --log-every, la línea de log (`registra_progreso`); cada
#     --eval-every valida (`valida`: denoising y, con la cadencia de --eval-qsm-cada, estabilidad
#     y QSM), y cada --guardar-cada escribe el estado (`guarda_estado`).
# Al final, `cierra` escribe `estado.json`. Respecto de `train.py`, cuatro diferencias:
#   1. Parches de PATCH_SIZE^3 de `ddr.data3d.PatchDataset3D`, con al menos 35 % de tejido (o,
#      con --mascara-cerebro, enteros dentro del cerebro) y una de las 48 simetrías del cubo, en
#      ceil(--lote / --micro-lote) micro-lotes casi iguales; el micro-lote lo elige la memoria
#      (MEMORIA y MICRO-LOTES).
#   2. El lr no lo lleva un scheduler: `lr_en(it, hitos)` lo fija al principio de cada iteración,
#      con los hitos de --meseta sumados (MESETA).
#   3. La cota absoluta del disyuntor escala con el parche (`cota_absoluta`).
#   4. `--guardar-cada` escribe el estado completo, y `--reanudar` sigue desde ahí bit a bit (con
#      el mismo micro-lote).
# Y tres agregados que no tocan la aritmética (test): el seguimiento en W&B (SEGUIMIENTO), la
# evaluación QSM en las validaciones (EVALUACIÓN QSM) y el número de estabilidad (ESTABILIDAD).
# Como en `train.py`, el bucle no corta ni por NaN, ni por rechazos, ni por un fallo de W&B, de la
# evaluación QSM o de la estabilidad. El único corte es el de --meseta, apagada por defecto, y
# queda escrito en `estado.json` (MESETA). Una corrida terminó si su `estado.json` dice
# `"terminada": true`. --sobreajuste cambia los datos para la prueba de sobreajuste (SOBREAJUSTE).
# Las secciones entre paréntesis son las de «## train3d.py» en docs/porque-del-codigo.md, donde
# está el resto de las notas de este archivo.
'''