"""
Entrenamiento de TDV 3D sobre volúmenes T1 de LDM_100k: la receta de `train.py` llevada a 3D.
Una iteración: LOTE = 32 parches de 48^3 con una de las 48 simetrías del cubo, en los micro-lotes
que entran en la memoria (16 en 96 GiB) -> ruido gaussiano fresco (sigma=25) -> S=10 pasos del
esquema desenrollado -> ||x_S - y||^2 sobre el estado final -> backward -> disyuntor de lotes ->
Adam con el lr del calendario -> re-proyección.

    uv run train3d.py --out runs/sonda --sonda-vram                  # una vez por tarjeta: el pico
    uv run train3d.py --out runs/tronco3d --hitos                    # tronco a lr constante
    uv run train3d.py --out runs/rama3d-10k --reanudar runs/tronco3d/estado-010000.pth \\
        --iterations 15000 --hitos 10000                             # rama: lr/2 desde la 10 001
    uv run train3d.py --out runs/tronco3d --reanudar runs/tronco3d/estado-020000.pth \\
        --hitos                                                      # sigue tras un corte
    uv run train3d.py --out runs/ciego3d --sigma-max 55              # ruido ciego U(0, 55)
    uv run train3d.py --out runs/cerebro3d --hitos --mascara-cerebro # cubos 100 % cerebro
    uv run train3d.py --out runs/tronco3d --hitos --eval-qsm-cada 10000   # QSM cada 10 000
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

import numpy as np
import torch
from torch.nn.utils import get_total_norm
from torch.utils.data import DataLoader

from ddr import seguimiento
from ddr.data3d import EVAL_RECORTE, PatchDataset3D, VolumeDataset3D, add_noise, add_noise_ciego
from ddr.model import VNet3D
from ddr.qsm3d import MAX_ITER as QSM_MAX_ITER
from ddr.qsm3d import (LADO_ESTABILIDAD, UMBRAL_ESTABILIDAD, admm_wtdv, carga_cosmos,
                       denoiser_3d, estabilidad, lamina_denoising, lamina_qsm, punto_qsm3d)
from evaluate import evaluate_denoising3d
from train import CHANNELS as CHANNELS_BSDS
from train import PATCH_SIZE as PATCH_SIZE_BSDS
from train import (BETAS, DATA_RANGE, DELTA_ADAM, FEATURES, GUARD_ABS, GUARD_LIMPIAS, GUARD_RATIO,
                   GUARD_RESPALDO, GUARD_SEGUIDOS, GUARD_VENTANA, LAMBDA_INIT, LR, LR_HALVE_EVERY,
                   MACRO_BLOCKS, MODES, NOISE_SCALE, SCALES, SIGMA, STEPS, T_INIT, rebobina,
                   rechaza_lote, respalda, vnet_config)

# --- RECETA 3D -------------------------------------------------------------------------------
# Lo del paper (lr, betas, sigma, S, TDV33, inits, disyuntor, modos) se importa de `train.py`, que
# es una sola fuente; también lo que acá no se usa (SCALES, MACRO_BLOCKS, DELTA_ADAM), porque los
# tests verifican que la receta sea esa. Lo que la tercera dimensión obliga a cambiar va abajo, y
# cada número tiene su porqué en el bloque final.
PATCH_SIZE = 48              # el menor costo por vóxel medido en la 5070 Ti: 2.26 µs a 48^3 x 4
LOTE = 32                    # los parches del paper (sec. 5.1), no su contenido: 4.3x los valores
MICRO_LOTE = "auto"          # es hardware, no receta: el mayor divisor de LOTE que entra (MEMORIA)
ITERACIONES_TANDA = 40_000   # el tronco del piloto; su costo, en s/it, en el bloque final
GUARDAR_CADA = 10_000        # los puntos de rama del piloto: 10 000, 20 000 y 30 000

# --- MEMORIA ---------------------------------------------------------------------------------
# Pico de `torch.cuda.max_memory_allocated`, en GiB, del entrenamiento sin la evaluación:
# FIJA + micro-lote · P^3 · POR_VOXEL. Ajustado a la grilla de la 5070 Ti
# (docs/mediciones/costo-tdv3d.txt, 2026-10-08): en otra tarjeta es una extrapolación, y lo que
# de verdad entra lo mide `--sonda-vram`. Cada número, en el bloque final.
VRAM_GB = 96.0               # la tarjeta de destino, RTX PRO 6000 Blackwell Max-Q de 96 GB, en GiB
FIJA = 0.61                  # la recta de 48^3 x 1 y x 4 en `rapido` (4.14 y 14.72 GB) en n = 0
POR_VOXEL = {                # GiB por vóxel del micro-lote, por (modo, checkpointing)
    ("rapido", False): 3.19e-5,        # la pendiente de esa recta: 3.527 GiB por parche de 48^3
    ("intermedio", False): 7.0e-5,     # sobre los 4 medidos de los dos modos: 2.48 a 7.60 GB
    ("referencia", False): 7.0e-5,
    # Medidos con los S pasos checkpointeados: en `train3d`, sin --guard-amp, el primero va sin
    # checkpoint, y `vram_predicha` le suma lo suyo (bloque final, MEMORIA).
    ("rapido", True): 3.4e-6,          # 64^3 x 4: 4.11 GB
    ("intermedio", True): 5.6e-6,      # sin medir: el de `referencia`
    ("referencia", True): 5.6e-6,      # 64^3 x 2: 3.56 GB
}
MARGEN = 0.10                # supuesto: evaluación, fragmentación y cuDNN de otra tarjeta
# `--sonda-vram`, por micro-lote: la primera iteración compila `R.grad` para esa forma y deja a
# cuDNN elegir algoritmos, y no se cronometra; las otras tres dan los s por parche. Supuesto, sin
# medir su dispersión: ordena micro-lotes, no reemplaza a medir s/it en la corrida.
SONDA_ITERACIONES = 4
SONDA_DESCARTADAS = 1

# --- DISYUNTOR EN 3D -------------------------------------------------------------------------
# La pérdida es una SUMA por parche, así que ‖g‖ escala con los valores sumados: la cota absoluta
# de BSDS se lleva a 3D por ese cociente. Derivado, no calibrado: ver el bloque final.
VALORES_BSDS = PATCH_SIZE_BSDS ** 2 * CHANNELS_BSDS   # 93^2 x 3 = 25 947 valores por parche

# --- DATASET ---------------------------------------------------------------------------------
DATA_ROOT = "datasets/ldm100k3d"
TRAIN_SPLIT, EVAL_SPLIT = "train", "val"
TRAIN_LIST = "datasets/ldm100k3d/lista_train_2000.txt"   # 2 000 entran en RAM (~19 GB); 8 000 no
EVAL_LIST = "datasets/ldm100k3d/lista_eval.txt"          # 12, uno por centroide de un k-means
CHANNELS = 1                 # T1 en gris
SIGMA_MIN = 0.0              # ruido ciego U(0, 55): la convención de DnCNN-B (Zhang et al., 2017)
# Un parche cuesta 0.31 ms de media con la CPU libre y 0.81 ms cargada (2026-10-08, crudo en
# docs/mediciones/tejido-ldm100k3d.txt): los 32 de una iteración, 26 ms en el peor caso medido,
# contra ~8 s de GPU en la 5070 Ti. Uno alcanzaría; dos, porque la tarjeta de destino es otra
# máquina con otra CPU, sin medir, y el segundo cuesta poco. Supuesto.
WORKERS = 2

# --- EVALUACIÓN QSM --------------------------------------------------------------------------
# El protocolo y sus constantes son los de `ddr/qsm3d.py`; el porqué de cada decisión de la
# integración, en el bloque final.
QSM_DATOS = "datasets/qsm_cosmos"
QSM_ARCHIVOS = ("params.mat", "chi_cosmos.mat")   # los que lee `carga_cosmos`
QSM_ORIGEN = "../tdv/data_mat"                    # de dónde se copian: el repo del pipeline QSM

# --- ESTABILIDAD -----------------------------------------------------------------------------
# El número del paso explícito, (T/S)·λ_max(∇²R), de `ddr/qsm3d.estabilidad`; el porqué, en el
# bloque final. La semilla es la `seed` por defecto de `evaluate_denoising3d`, que la validación
# pasa explícita: el punto de validación lleva el mismo ruido que ve la validación.
SEMILLA_VAL = 1234

# Banderas que pueden cambiar al reanudar. Las primeras no tocan la aritmética, tampoco las del
# seguimiento, la evaluación QSM y la estabilidad (test). `micro_lote` y `vram_gb` (que lo elige
# con `auto`) cambian el orden de suma del gradiente, no la receta (con --sigma-max o --guard-amp,
# algo más: bloque final): seguir en otra tarjeta está permitido, y `--reanudar` avisa que desde
# ahí no es bit a bit. Una rama (otro `--out`) puede cambiar además `--hitos`; nada más.
LIBRES = frozenset({"out", "reanudar", "iterations", "log_every", "eval_every", "guardar_cada",
                    "workers", "micro_lote", "vram_gb", "sonda_vram", "seguimiento",
                    "wandb_proyecto", "qsm_datos", "eval_qsm_cada", "qsm_max_iter",
                    "estabilidad_lado"})
COLUMNAS = 14


def vnet_config3d(t_init: float, features: int = FEATURES) -> dict:
    """La config de `train.vnet_config` en gris, con el regularizador 3D. `features` sólo cambia
    para los tests en CPU; el `type` le dice a un cargador qué VNet construir."""
    config = vnet_config(t_init, CHANNELS)
    config["R"]["type"] = "tdv3d"
    config["R"]["config"]["num_features"] = features
    return config


def cota_absoluta(patch: int) -> float:
    """`GUARD_ABS` de BSDS escalada por los valores sumados por parche: 4.26e4 a 48^3."""
    return GUARD_ABS * patch ** 3 / VALORES_BSDS


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


def divisores(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def micro_lote_auto(lote: int, patch: int, modo: str, checkpointing: bool, presupuesto: float,
                    entrada_con_grad: bool = False) -> int:
    """El mayor divisor de `lote` cuyo pico predicho no pasa de `presupuesto · (1 − MARGEN)` GiB.

    Un divisor y no el mayor entero que entra: el lote es la receta y se parte entero (a 48^3 en
    48 GiB entrarían 12, y da 8). `ValueError` si no entra ni un parche.
    """
    util = presupuesto * (1 - MARGEN)
    entran = [n for n in divisores(lote)
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


def lr_en(it: int, hitos) -> float:
    """El lr que USA la iteración `it` (1-indexada): LR a la mitad por cada hito <= it - 1.

    Es el de `train.py`, donde `MultiStepLR` avanza con `sched.step()` DESPUÉS del paso: con un
    hito en 25 000, la 25 000 todavía usa 4e-4 y la 25 001 usa 2e-4. Un hito repetido cuenta dos
    veces, como en `MultiStepLR`.
    """
    return LR * 0.5 ** sum(h <= it - 1 for h in hitos)


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


def evalua_qsm(model, cosmos: dict, device, amp, max_iter: int) -> dict:
    """El ADMM de `ddr/qsm3d.py` con el modelo como denoiser sobre el volumen entero: las claves
    de `qsm/*` y `chi`, la reconstrucción en numpy (para la lámina, que se arma afuera), o
    `{"fallo": <tipo: mensaje>}`. No levanta nunca, ni con un OOM: la evaluación QSM no puede
    matar el entrenamiento. Deja el modelo en modo train."""
    fallo = None
    model.eval()
    try:
        with torch.no_grad():
            r = admm_wtdv([denoiser_3d(model, autocast_dtype=amp)], cosmos, device=device,
                          max_iter=max_iter)
        return {"nrmse_final_pct": r["nrmse_final"], "nrmse_min_pct": r["nrmse_min"],
                "iter_min": r["it_min"], "iteraciones": r["iteraciones"], "alpha": r["alpha"],
                "segundos": r["segundos"], "motivo": r["motivo"], "chi": r["chi"]}
    except Exception as e:
        fallo = f"{type(e).__name__}: {e}"
    finally:
        model.train()
    # Fuera del `except`: adentro, la traza del error retiene los tensores del ADMM y la caché no
    # puede devolverlos.
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return {"fallo": fallo}


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


def anota_jsonl(destino: Path, fila: dict) -> None:
    """Agrega `fila` como una línea de JSON. Un error (disco lleno) se avisa y la corrida sigue:
    el estado lleva la misma fila."""
    try:
        with destino.open("a") as f:
            f.write(json.dumps(fila) + "\n")
    except Exception as e:
        print(f"AVISO: no se pudo escribir {destino} ({type(e).__name__}: {e})", flush=True)


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


def commit_git() -> str | None:
    """El commit de este archivo, con `+cambios` si hay cambios sin commitear en lo versionado;
    None si no hay git o no es un repositorio (una copia sin `.git` en otra máquina)."""
    raiz = Path(__file__).resolve().parent
    try:
        commit, cambios = (subprocess.run(["git", "-C", str(raiz), *orden], capture_output=True,
                                          text=True, timeout=10, check=True).stdout.strip()
                           for orden in (["rev-parse", "HEAD"],
                                         ["status", "--porcelain", "--untracked-files=no"]))
        return commit + ("+cambios" if cambios else "")
    except Exception:
        return None


def sonda_vram(model, args, mode: dict, presupuesto: float, out: Path) -> None:
    """SONDA_ITERACIONES de forward + backward con cada divisor de `--lote`, de menor a mayor,
    hasta el primero que no entra: el pico medido contra el predicho, y los segundos por parche de
    las iteraciones que no compilan. El modelo llega compilado como para entrenar. Mide el
    entrenamiento, no la evaluación. En `--out` deja sólo `sonda-vram.json`.
    """
    util = presupuesto * (1 - MARGEN)
    candidatos = divisores(args.lote)
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
                # Ruido uniforme, como en costo-tdv3d.txt: la memoria no depende del contenido.
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
    p.add_argument("--micro-lote", type=entero_o_auto, default=MICRO_LOTE,
                   help="parches por backward, un divisor de --lote. `auto`: el mayor que entra "
                        "en --vram-gb según el modelo de memoria")
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
    # Candidatas de la fase 2, apagadas: con los defaults la corrida es la receta.
    p.add_argument("--sigma-max", type=float, default=0.0,
                   help="> 0 sortea sigma ~ U(--sigma-min, --sigma-max) por parche (ruido ciego)")
    p.add_argument("--sigma-min", type=float, default=SIGMA_MIN)
    p.add_argument("--aug-signo", action="store_true",
                   help="multiplica el parche por -1 con probabilidad 1/2")
    p.add_argument("--aug-ganancia", type=float, default=1.0,
                   help="g > 1 multiplica el parche por un factor log-uniforme en [1/g, g]")
    p.add_argument("--mascara-cerebro", action="store_true",
                   help="cada parche es un cubo entero dentro del cerebro, según "
                        "<data-root>/mascaras/train/ (datasets/mascaras_ldm100k3d.py)")
    p.add_argument("--clip", type=float, default=float("inf"), help="norma máxima del gradiente")
    p.add_argument("--guard", type=float, default=GUARD_RATIO,
                   help="rechaza el lote si |g| supera esta razón sobre su mediana móvil")
    p.add_argument("--guard-abs", type=float, default=None,
                   help="cota absoluta de |g|; sin la bandera, GUARD_ABS escalada al parche")
    p.add_argument("--guard-ventana", type=int, default=GUARD_VENTANA)
    p.add_argument("--guard-seguidos", type=int, default=GUARD_SEGUIDOS,
                   help="rechazos seguidos que disparan el rebobinado al respaldo. 0 lo apaga")
    p.add_argument("--guard-respaldo", type=int, default=GUARD_RESPALDO,
                   help="cada cuántas iteraciones sanas se refresca el respaldo")
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
                        "offline si no; `no` ni importa wandb")
    p.add_argument("--wandb-proyecto", default=seguimiento.PROYECTO)
    p.add_argument("--qsm-datos", default=QSM_DATOS,
                   help="carpeta con params.mat y chi_cosmos.mat (los de ../tdv/data_mat). Si "
                        "falta alguno, la evaluación QSM queda apagada con un aviso; '' la apaga "
                        "sin aviso")
    p.add_argument("--eval-qsm-cada", type=int, default=0,
                   help="0: la evaluación QSM va en cada validación; N > 0: en las validaciones "
                        "de iteración múltiplo de N y en la última")
    p.add_argument("--qsm-max-iter", type=int, default=QSM_MAX_ITER,
                   help="tope de iteraciones del ADMM; el del autor es 100")
    p.add_argument("--estabilidad-lado", type=int, default=LADO_ESTABILIDAD,
                   help="lado del cubo donde se mide (T/S)·λ_max(∇²R), con la cadencia de la "
                        "evaluación QSM: en la validación y, con COSMOS, en la entrada del primer "
                        "denoiser del ADMM. Se recorta a lo que entra. 0 no la mide")
    # Sólo para los tests en CPU: con los defaults la red y la evaluación son las de la receta.
    p.add_argument("--features", type=int, default=FEATURES)
    p.add_argument("--eval-recorte", type=int, default=EVAL_RECORTE)
    return p


def main(argv: list[str] | None = None) -> None:
    p = opciones()
    args = p.parse_args(argv)
    if args.guard_abs is None:
        args.guard_abs = cota_absoluta(args.patch)
    mide_amp = math.isfinite(args.guard_amp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.sonda_vram and device.type != "cuda":
        p.error("--sonda-vram mide el pico de memoria de una GPU y torch no ve ninguna "
                "(torch.cuda.is_available() es False): correrla en la tarjeta de destino")

    # --- MEMORIA -----------------------------------------------------------------------------
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
            p.error(f"--micro-lote auto: {e}")
    elif args.lote % args.micro_lote:
        p.error(f"--lote {args.lote} no es múltiplo de --micro-lote {args.micro_lote}")
    acumulacion = args.lote // args.micro_lote
    predicha = vram_predicha(args.micro_lote, args.patch, args.mode, args.checkpointing, mide_amp)
    print("dispositivo: " + (f"{torch.cuda.get_device_name(device)}, {total:.2f} GiB"
                             if total is not None else "cpu, sin memoria que medir")
          + f" | presupuesto {presupuesto:.2f} GiB, {presupuesto * (1 - MARGEN):.2f} útiles"
          + f" | micro-lote {args.micro_lote}" + (" (auto)" if pedido == "auto" else "")
          + f", pico predicho {predicha:.2f} GiB", flush=True)
    hitos = args.hitos if args.hitos is not None else list(
        range(LR_HALVE_EVERY, args.iterations + 1, LR_HALVE_EVERY))

    mode = MODES[args.mode]
    out = Path(args.out)

    # --- REANUDACIÓN -------------------------------------------------------------------------
    estado, it0 = None, 0
    if args.reanudar:
        # Sin `map_location` cada tensor vuelve a su dispositivo. Desde el arreglo de `rebobina()`
        # (copia del respaldo, 2026-10-08), el estado de Adam no
        # comparte memoria con el respaldo tras un rebobinado.
        estado = torch.load(args.reanudar, weights_only=False,
                            map_location=None if torch.cuda.is_available() else "cpu")
        rama = Path(args.reanudar).resolve().parent != out.resolve()
        libres = LIBRES | ({"hitos"} if rama else set())
        ahora = vars(args)
        # Una bandera de receta posterior al estado vale ahí su default: los de runs/humo3d no
        # tienen `mascara_cerebro`, y sin esto no se podían reanudar.
        distintas = sorted(k for k in estado["args"].keys() | ahora.keys()
                           if k not in libres
                           and estado["args"].get(k, p.get_default(k)) != ahora.get(k))
        if distintas:
            p.error(f"--reanudar: cambian {distintas}; "
                    + ("una rama sólo puede cambiar --hitos" if rama else
                       "seguir en el mismo --out no puede cambiar la receta (otro --micro-lote "
                       "sí, con un AVISO)"))
        it0 = estado["iteration"]
        if args.iterations < it0:
            p.error(f"--iterations {args.iterations} es menor que la del estado, {it0}")
        if estado["args"]["micro_lote"] != args.micro_lote:
            # Qué cambia además del orden de suma, y por qué, en el bloque final (REANUDACIÓN).
            cambia = ["el orden de suma del gradiente"]
            if args.sigma_max > 0 or args.patch ** 3 % 16:
                cambia.append("el ruido de cada parche")
            if math.isfinite(args.guard_amp):
                cambia.append("la amplificación adjunta que mira --guard-amp")
            print(f"AVISO: el micro-lote pasa de {estado['args']['micro_lote']} a "
                  f"{args.micro_lote}. Cambia " + ", ".join(cambia[:-1])
                  + (" y " if len(cambia) > 1 else "") + cambia[-1] + ": desde acá la corrida "
                  "no es bit a bit la que hubiera seguido, y la columna 10 de la traza mide "
                  f"micro-lotes de {args.micro_lote}", flush=True)

    # --- BACKEND: semillas, determinismo y precisión de las operaciones fp32 -----------------
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = mode["benchmark"]
    if mode["deterministic"]:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    precision = "tf32" if mode["tf32"] else "ieee"
    torch.backends.cudnn.conv.fp32_precision = precision
    torch.backends.cuda.matmul.fp32_precision = precision

    # --- MODELO ------------------------------------------------------------------------------
    config = vnet_config3d(args.t_init, args.features)
    model = VNet3D(config, efficient=args.checkpointing).to(device)
    if estado is not None:
        model.load_state_dict(estado["model"])
    # El método, no el módulo, como en `train.py`: el state_dict queda sin prefijo `_orig_mod.`.
    if not args.sin_compilar:
        model.R.grad = torch.compile(model.R.grad, dynamic=False)
    # Antes de los datos: la sonda no los usa, y en `--out` no escribe más que su json.
    if args.sonda_vram:
        sonda_vram(model, args, mode, presupuesto, out)
        return

    # --- DATOS -------------------------------------------------------------------------------
    # El parche `i` depende sólo de `(seed, offset + i)`: reanudar en `it0` es empezar en el
    # parche `it0 * lote`, y el DataLoader no guarda nada.
    archivos = [Path(args.data_root) / "images" / TRAIN_SPLIT / f"{Path(n).stem}.npy"
                for n in lee_lista(args.train_list)]
    mascaras = None
    if args.mascara_cerebro:
        rutas = [Path(args.data_root) / "mascaras" / TRAIN_SPLIT / f.name for f in archivos]
        faltan = [r.stem for r in rutas if not r.exists()]
        if faltan:
            herramienta = "datasets/mascaras_ldm100k3d.py"
            lista = f"--lista {args.train_list} --salida {args.data_root}"
            p.error(f"--mascara-cerebro: faltan {len(faltan)} de {len(rutas)} máscaras en "
                    f"{rutas[0].parent} (la primera, {faltan[0]}). Se generan en tres pasos, "
                    f"cada herramienta en su entorno (cómo armarlos, en el docstring de "
                    f"{herramienta}):\n"
                    f"  $H/deepbet/bin/python -I {herramienta} deepbet {lista}\n"
                    f"  $H/hdbet/bin/python -I {herramienta} hdbet {lista} --herramientas $H\n"
                    f"  .venv/bin/python {herramienta} combina {lista} --split {TRAIN_SPLIT}")
        # Una por vez, con un generador: 2 000 juntas serían ~10 GB, y el dataset se queda sólo
        # con las esquinas válidas.
        mascaras = (np.load(r) for r in rutas)
    train_set = PatchDataset3D(
        archivos, patch_size=args.patch, length=(args.iterations - it0) * args.lote,
        seed=args.seed * 100, offset=it0 * args.lote, signo=args.aug_signo,
        ganancia=args.aug_ganancia, mascaras=mascaras,
    )
    if mascaras is not None:
        esquinas = [e.size for e in train_set.esquinas]
        print(f"máscara de cerebro: {len(train_set.con_esquinas)} de {len(esquinas)} volúmenes "
              f"con cubos de {args.patch}^3 a {train_set.margen_cerebro} vóxel o más del "
              f"borde; mediana {np.median(esquinas):.0f} esquinas por volumen", flush=True)
    persistentes = {"persistent_workers": True, "prefetch_factor": 4} if args.workers else {}
    loader = DataLoader(train_set, batch_size=args.micro_lote, shuffle=False, drop_last=True,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"),
                        **persistentes)
    # Con las máscaras de cerebro de todos los de evaluación, la validación mide también ahí: lo
    # que ve el entrenamiento con --mascara-cerebro. Sin alguna, no, y el log lo dice.
    mascaras_eval = Path(args.data_root) / "mascaras" / EVAL_SPLIT
    sujetos_eval = [Path(n).stem for n in lee_lista(args.eval_list)]
    con_cerebro = all((mascaras_eval / f"{s}.npy").is_file() for s in sujetos_eval)
    eval_set = VolumeDataset3D.from_root(args.data_root, EVAL_SPLIT, recorte=args.eval_recorte,
                                         names=sujetos_eval,
                                         mascaras=mascaras_eval if con_cerebro else None)

    # --- EVALUACIÓN QSM ----------------------------------------------------------------------
    # Una sola vez, en CPU: `admm_wtdv` lo pasa al device en cada evaluación.
    cosmos = carga_qsm(args.qsm_datos)

    # --- ESTABILIDAD -------------------------------------------------------------------------
    # Los dos puntos, una sola vez: no dependen del modelo. Un lado que no entra se recorta.
    punto_val = punto_qsm = None
    if args.estabilidad_lado > 0 and len(eval_set):
        punto_val = punto_validacion(eval_set, min(args.estabilidad_lado, args.eval_recorte),
                                     args.sigma)
        if cosmos is not None:
            try:
                punto_qsm = torch.from_numpy(punto_qsm3d(
                    cosmos, min(args.estabilidad_lado, *cosmos["mascara"].shape), device=device))
            except Exception as e:
                print(f"AVISO: estabilidad en QSM apagada: el punto no se pudo armar "
                      f"({type(e).__name__}: {e}); el entrenamiento sigue", flush=True)

    # --- OPTIMIZADOR -------------------------------------------------------------------------
    # Sin scheduler: el lr se fija desde `lr_en` al principio de cada iteración.
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS)
    noise_gen = torch.Generator(device="cpu").manual_seed(args.seed * 100 + 1)

    # --- ESTADO DE LA CORRIDA ----------------------------------------------------------------
    history, best_psnr, mejor, max_grad, horas_previas = [], 0.0, None, 0.0, 0.0
    escala, rechazos, seguidos, max_seguidos = [], 0, 0, 0
    respaldo, it_respaldo, rebobinados, limpias = None, 0, 0, 0
    descartadas, it_rebobinado, qsm_hist, mejor_qsm = 0, 0, [], None
    traza = np.zeros((args.iterations, COLUMNAS), dtype=np.float64)
    # loss T dT lambda dlambda |g| centro amp rechazado rebobinado |g|micro lr efectiva sigma
    if estado is not None:
        opt.load_state_dict(estado["optimizer"])
        noise_gen.set_state(estado["noise_gen"])
        history, best_psnr, mejor = estado["history"], estado["best_psnr"], estado["mejor"]
        max_grad, horas_previas = estado["max_grad"], estado["horas"]
        escala, rechazos, seguidos = estado["escala"], estado["rechazos"], estado["seguidos"]
        max_seguidos, limpias = estado["max_seguidos"], estado["limpias"]
        respaldo, it_respaldo = estado["respaldo"], estado["it_respaldo"]
        rebobinados, descartadas = estado["rebobinados"], estado["descartadas"]
        it_rebobinado = estado["it_rebobinado"]
        traza[:it0] = estado["traza"]
        qsm_hist = estado.get("qsm", [])          # un estado anterior a la evaluación QSM
        mejor_qsm = estado.get("mejor_qsm")       # o a `best_qsm.pth`

    # --- DIRECTORIO DE SALIDA ----------------------------------------------------------------
    # Lo que `--out` tenga de antes y no esté en el estado es de una corrida o de un linaje que no
    # sigue: se reescribe, o se aparta si es un checkpoint. `estado.json` lo escribe sólo el final.
    out.mkdir(parents=True, exist_ok=True)
    (out / "estado.json").unlink(missing_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "qsm.jsonl").unlink(missing_ok=True)
    if qsm_hist:
        (out / "qsm.jsonl").write_text("".join(json.dumps(fila) + "\n" for fila in qsm_hist))
    guarda_traza(out / "traza.npy", traza[:it0])
    aparta(out / "last.pth", history[-1] if history else None)
    for nombre, vigente in (("best.pth", mejor), ("best_qsm.pth", mejor_qsm)):
        aparta(out / nombre, vigente)
        if vigente is not None and not (out / nombre).exists():
            guarda_pth(vigente, out / nombre)
    ruido = (f"sigma~U({args.sigma_min:g}, {args.sigma_max:g})" if args.sigma_max > 0
             else f"sigma={args.sigma:g}")
    print(f"parámetros: {sum(p.numel() for p in model.parameters()):,} | "
          f"lote {args.lote} = {acumulacion}x{args.micro_lote} de {args.patch}^3 | "
          f"modo {args.mode} | datos {args.data_root} ({len(train_set.volumes)} vols, "
          f"{len(eval_set)} de eval a {args.eval_recorte}^3) | hitos lr {hitos} | "
          f"T_init={args.t_init:g} | lambda_init={LAMBDA_INIT:g} | {ruido} | "
          f"signo={args.aug_signo} | ganancia={args.aug_ganancia:g} | "
          f"mascara_cerebro={args.mascara_cerebro} | clip={args.clip:g} | "
          f"guard={args.guard:g}x mediana({args.guard_ventana}) | abs={args.guard_abs:g} | "
          f"rebobina a los {args.guard_seguidos} seguidos | respaldo tras "
          f"{args.guard_limpias} limpias | guard_amp={args.guard_amp:g}"
          + ("  [checkpointing]" if args.checkpointing else "")
          + (f"  [reanuda {args.reanudar}]" if args.reanudar else ""), flush=True)
    qsm_cuando = ("en cada validación" if args.eval_qsm_cada <= 0 else
                  f"en las validaciones múltiplo de {args.eval_qsm_cada} y en la última")
    print(f"validación: cada {args.eval_every} iteraciones y en la última | PSNR en cerebro "
          + ("sí" if con_cerebro else f"no (faltan máscaras en {mascaras_eval})")
          + " | QSM " + (f"{qsm_cuando}, COSMOS {'x'.join(map(str, cosmos['chi'].shape))} de "
                         f"{args.qsm_datos}, hasta {args.qsm_max_iter} iteraciones de ADMM"
                         if cosmos is not None else "apagada")
          + " | estabilidad " + ("apagada (--estabilidad-lado 0)" if punto_val is None else
                                 f"con la cadencia de QSM, en {punto_val.shape[-1]}^3 de "
                                 "validación" + (f" y {punto_qsm.shape[-1]}^3 de COSMOS"
                                                 if punto_qsm is not None else "")), flush=True)
    # `best_qsm.pth` pide las dos: la evaluación QSM y la estabilidad en su punto.
    sin_best_qsm = ("sin COSMOS" if cosmos is None else
                    "--estabilidad-lado 0" if args.estabilidad_lado <= 0 else
                    "sin el punto de estabilidad en COSMOS" if punto_qsm is None else None)
    print("best_qsm.pth: " + (
        "el de menor NRMSE QSM entre las validaciones con estabilidad en COSMOS < "
        f"{UMBRAL_ESTABILIDAD:g}" if sin_best_qsm is None else f"no se busca ({sin_best_qsm})")
          + (f" | hereda el del estado, de la {mejor_qsm['iteration']}" if mejor_qsm else ""),
          flush=True)

    lotes = iter(loader)

    # --- SEGUIMIENTO -------------------------------------------------------------------------
    # Después de `iter(loader)`: los workers ya existen, y no se forkea con los hilos de wandb
    # vivos. `abre` no levanta ni toca los generadores (tests/test_seguimiento.py).
    run = seguimiento.abre(out, {**vars(args), "commit": commit_git()}, modo=args.seguimiento,
                           proyecto=args.wandb_proyecto, nombre=out.resolve().name,
                           padre=args.reanudar, desde=it0)
    if args.seguimiento == "no":
        print("seguimiento: sin W&B (--seguimiento no)", flush=True)

    # Después de `iter(loader)`, que sortea una semilla del generador global: la corrida de origen
    # lo hizo una sola vez, al arrancar.
    if estado is not None:
        torch.set_rng_state(estado["rng_cpu"])
        if torch.cuda.is_available() and len(estado["rng_cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(estado["rng_cuda"])
        del estado

    # --- BUCLE -------------------------------------------------------------------------------
    t0 = time.time()
    # Desde dónde mide `train/s_por_it`: se corre tras cada registro, evaluación y estado, así que
    # cuenta sólo iteraciones de entrenamiento.
    reloj = (t0, it0)
    for it in range(it0 + 1, args.iterations + 1):
        # El lr sale del calendario en CADA iteración: así una rama reanuda con otro calendario, y
        # ningún estado restaurado puede traer un lr viejo.
        for grupo in opt.param_groups:
            grupo["lr"] = lr_en(it, hitos)
        opt.zero_grad(set_to_none=True)
        perdida = torch.zeros((), device=device)
        p0, pS = torch.zeros((), device=device), torch.zeros((), device=device)
        suma_sigma = 0.0
        for micro in range(acumulacion):
            lote = next(lotes)
            if args.sigma_max > 0:
                noisy, sigmas = add_noise_ciego(lote, args.sigma_min, args.sigma_max, noise_gen,
                                                NOISE_SCALE)
                suma_sigma += float(sigmas.sum())
            else:
                noisy = add_noise(lote, args.sigma, noise_gen, NOISE_SCALE)
                suma_sigma += args.sigma * len(lote)
            noisy = noisy.to(device, non_blocking=True)
            clean = lote.to(device, non_blocking=True)
            if mide_amp:
                noisy.requires_grad_(True)

            with torch.autocast(device.type, dtype=mode["amp"], enabled=mode["amp"] is not None):
                states = model(noisy, noisy)
                loss = (states[-1].float() - clean.float()).pow(2).flatten(1).sum(1).mean()

            (loss / acumulacion).backward()
            perdida += loss.detach() / acumulacion
            # Sólo LEE `.grad`: el backward siguiente acumula sobre el mismo tensor y da los
            # mismos bits. Por `acumulacion`, es la norma de este micro-lote como si fuera el lote.
            if micro == 0:
                g_micro = acumulacion * get_total_norm(
                    [q.grad for q in model.parameters() if q.grad is not None]).item()
            if mide_amp:
                p0 += noisy.grad.norm() / acumulacion
                pS += 2 * (states[-1].detach().float() - clean.float()).norm() / (
                    args.micro_lote * acumulacion ** 2)

        # Devuelve la norma ANTES de recortar, que es la que sirve de diagnóstico.
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()

        amp = (p0 / pS).item() if mide_amp else float("nan")
        rechazado, centro = rechaza_lote(gnorm, amp, escala, args.guard, args.guard_abs,
                                         args.guard_amp, args.guard_ventana)

        # antes del paso: los valores que produjeron esta pérdida
        traza[it - 1, :5] = torch.stack((perdida, model.T.detach(), model.T.grad,
                                         model.lmbda.detach(), model.lmbda.grad)).cpu().numpy()

        # Se lee del optimizador y no del calendario: es el que de verdad usa el paso.
        lr = opt.param_groups[0]["lr"]
        if not rechazado:
            opt.step()
        model.project_()   # media cero y norma <=1 en K1, positividad de T y lambda

        rebobinado = False
        if rechazado:
            rechazos += 1
            seguidos, limpias = seguidos + 1, 0
            max_seguidos = max(max_seguidos, seguidos)
            if respaldo is not None and 0 < args.guard_seguidos <= seguidos:
                rebobina(model, opt, respaldo, escala)
                # Se pierde lo hecho desde el respaldo, salvo lo que un rebobinado anterior al
                # mismo respaldo ya había descartado.
                descartadas += min(it - it_respaldo, it - it_rebobinado)
                it_rebobinado = it
                rebobinados, seguidos, rebobinado = rebobinados + 1, 0, True
                print(f"[{it:6d}] REBOBINADO a la it {it_respaldo} "
                      f"({rebobinados} en lo que va) |g|={gnorm:.3g}", flush=True)
            elif seguidos <= 3 or seguidos % 1000 == 0:
                print(f"[{it:6d}] LOTE RECHAZADO |g|={gnorm:.1f} mediana={centro:.1f} "
                      f"amp={amp:.3g} loss={perdida.item():.3f} (seguidos {seguidos})", flush=True)
        else:
            seguidos = 0
            limpias += 1
            if limpias >= args.guard_limpias and (
                    respaldo is None or it - it_respaldo >= args.guard_respaldo):
                respaldo, it_respaldo = respalda(model, opt, escala), it

        traza[it - 1, 5:] = (gnorm, centro, amp, rechazado, rebobinado,
                             g_micro, lr, it - descartadas, suma_sigma / args.lote)
        max_grad = max(max_grad, gnorm)

        if it % args.log_every == 0:
            with torch.no_grad():
                corr = ((states[-1] - states[0]).norm() / (noisy - clean).norm()).item()
            eta = (time.time() - t0) / (it - it0) * (args.iterations - it) / 3600
            print(f"[{it:6d}/{args.iterations}] loss={perdida.item():8.3f} lr={lr:.2e} "
                  f"T={model.T.item():.5f} lambda={model.lmbda.item():.5f} "
                  f"|g|={gnorm:.1f} max|g|={max_grad:.3g} corr={corr:.3f} rech={rechazos} "
                  f"reb={rebobinados} ef={it - descartadas} ({eta:.1f}h restantes)", flush=True)
            # El max|g| del log va también: un estallido entre dos muestras no se pierde.
            extra = {"train/grad_norm_max": max_grad,
                     "train/s_por_it": (time.time() - reloj[0]) / (it - reloj[1])}
            if device.type == "cuda":
                extra["train/vram_pico_gib"] = torch.cuda.max_memory_allocated(device) / 2 ** 30
            seguimiento.registra(run, seguimiento.metricas_traza(traza, it) | extra, it)
            reloj = (time.time(), it)
            max_grad = 0.0

        if it % args.eval_every == 0 or it == args.iterations:
            model.eval()
            report = evaluate_denoising3d(model, eval_set, args.sigma, device, seed=SEMILLA_VAL,
                                          data_range=DATA_RANGE, noise_scale=NOISE_SCALE,
                                          amp=mode["amp"], devuelve_primero=run is not None)
            model.train()
            dB = report["psnr_denoised"]
            entrada = {"iteration": it, "efectiva": it - descartadas, "psnr": dB,
                       "psnr_tejido": report["psnr_denoised_tejido"],
                       "nrmse_pct": report["nrmse_denoised"],
                       "nrmse_pct_tejido": report["nrmse_denoised_tejido"],
                       "T": model.T.item(), "lambda": model.lmbda.item()}
            if con_cerebro:
                entrada["psnr_cerebro"] = report["psnr_denoised_cerebro"]
            history.append(entrada)
            (out / "history.json").write_text(json.dumps(history, indent=2))

            guarda_traza(out / "traza.npy", traza[:it])
            payload = {"config": config, "model": model.state_dict(), "iteration": it,
                       "psnr": dB, "optimizer": opt.state_dict()}
            guarda_pth(payload, out / "last.pth")
            if dB > best_psnr:
                best_psnr = dB
                guarda_pth(payload, out / "best.pth")
                # Una copia: `payload` apunta a los pesos vivos. El estado la lleva para reponer
                # `best.pth` al reanudar.
                mejor = copy.deepcopy(payload)
            print(f"    eval PSNR={dB:.3f} dB, en tejido {report['psnr_denoised_tejido']:.3f} dB"
                  + (f", en cerebro {entrada['psnr_cerebro']:.3f} dB" if con_cerebro else "")
                  + f" | NRMSE {entrada['nrmse_pct']:.2f} %, en tejido "
                  f"{entrada['nrmse_pct_tejido']:.2f} % ({report['num_volumenes']} vols)"
                  + ("  *mejor*" if dB == best_psnr else ""), flush=True)
            seguimiento.registra(run, {f"val/{k}": v for k, v in entrada.items()
                                       if k not in ("iteration", "efectiva", "T", "lambda")}, it)
            registra_lamina(run, "val/denoising", lambda: lamina_denoising(*report["primero"]),
                            it, f"{eval_set.files[0].stem}: limpia | ruidosa | salida | "
                                "5·|salida − limpia|, en [0, 1]")

            # Después de guardar lo de arriba: un corte duro a mitad del ADMM o de la estabilidad
            # no lo pierde. La estabilidad va con la cadencia de la evaluación QSM.
            toca_qsm = (args.eval_qsm_cada <= 0 or it % args.eval_qsm_cada == 0
                        or it == args.iterations)
            if punto_val is not None and toca_qsm:
                est = mide_estabilidad(model, punto_val, device)
                entrada.update(claves_estabilidad(est))
                (out / "history.json").write_text(json.dumps(history, indent=2))
                print(linea_estabilidad("val", est), flush=True)
                if "fallo" not in est:
                    seguimiento.registra(run, {"val/estabilidad": est["paso"],
                                               "val/lambda_max": est["lambda_max"]}, it)

            if cosmos is not None and toca_qsm:
                est = (mide_estabilidad(model, punto_qsm, device) if punto_qsm is not None
                       else None)
                r = evalua_qsm(model, cosmos, device, mode["amp"], args.qsm_max_iter)
                fila = {"iteration": it, "efectiva": it - descartadas,
                        **{k: v for k, v in r.items() if k != "chi"},
                        **(claves_estabilidad(est) if est is not None else {})}
                qsm_hist.append(fila)
                anota_jsonl(out / "qsm.jsonl", fila)
                if est is not None:
                    print(linea_estabilidad("qsm", est), flush=True)
                    if "fallo" not in est:
                        seguimiento.registra(run, {"qsm/estabilidad": est["paso"],
                                                   "qsm/lambda_max": est["lambda_max"]}, it)
                if "fallo" in r:
                    print(f"    qsm FALLÓ ({r['fallo']}); el entrenamiento sigue", flush=True)
                    seguimiento.registra(run, {"qsm/fallo": 1}, it)
                else:
                    print(f"    qsm NRMSE={r['nrmse_final_pct']:.2f} % (mínimo "
                          f"{r['nrmse_min_pct']:.2f} % en la iteración {r['iter_min']} del ADMM)"
                          f" | {r['iteraciones']} iteraciones, motivo {r['motivo']} | "
                          f"alpha={r['alpha']:.4f} | {r['segundos']:.1f} s", flush=True)
                    # El checkpoint para QSM: el menor NRMSE entre los estados estables en COSMOS.
                    if (est is not None and "fallo" not in est
                            and est["paso"] < UMBRAL_ESTABILIDAD and r["nrmse_final_pct"] < (
                                mejor_qsm["nrmse_final_pct"] if mejor_qsm else math.inf)):
                        # Una copia, como `mejor`: el estado la lleva para reponer el archivo.
                        mejor_qsm = copy.deepcopy({
                            "config": config, "model": model.state_dict(), "iteration": it,
                            "psnr": dB, **{k: r[k] for k in ("nrmse_final_pct", "nrmse_min_pct",
                                                             "iteraciones", "motivo")},
                            "estabilidad": est["paso"]})
                        guarda_pth(mejor_qsm, out / "best_qsm.pth")
                        print(f"    best_qsm.pth: NRMSE QSM {r['nrmse_final_pct']:.2f} % con "
                              f"estabilidad {est['paso']:.4f}, el menor entre los estables",
                              flush=True)
                    seguimiento.registra(run, {"qsm/fallo": 0, **{
                        f"qsm/{k}": r[k] for k in ("nrmse_final_pct", "nrmse_min_pct",
                                                   "iter_min", "iteraciones", "alpha",
                                                   "segundos")}}, it)
                    registra_lamina(run, "qsm/reconstruccion",
                                    lambda: lamina_qsm(r["chi"], cosmos["chi"],
                                                       cosmos["mascara_eval"]), it,
                                    "COSMOS: GT | reconstrucción | error, en la máscara, "
                                    "±0.1 ppm")
            reloj = (time.time(), it)

        # Después de la evaluación, para que el estado lleve su entrada de `history`. Siempre en la
        # última: una corrida terminada se puede extender sin rehacer nada.
        if args.guardar_cada and (it % args.guardar_cada == 0 or it == args.iterations):
            guarda_traza(out / "traza.npy", traza[:it])
            guarda_pth({
                "iteration": it, "config": config, "args": vars(args),
                "model": model.state_dict(), "optimizer": opt.state_dict(),
                "escala": list(escala), "respaldo": respaldo, "it_respaldo": it_respaldo,
                "rechazos": rechazos, "seguidos": seguidos, "max_seguidos": max_seguidos,
                "rebobinados": rebobinados, "limpias": limpias, "descartadas": descartadas,
                "it_rebobinado": it_rebobinado, "history": history, "best_psnr": best_psnr,
                "mejor": mejor, "traza": traza[:it].copy(), "qsm": qsm_hist,
                "mejor_qsm": mejor_qsm, "max_grad": max_grad, "noise_gen": noise_gen.get_state(),
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "horas": horas_previas + (time.time() - t0) / 3600,
            }, out / f"estado-{it:06d}.pth")
            reloj = (time.time(), it)

    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    horas = horas_previas + (time.time() - t0) / 3600
    final = {
        "terminada": True, "iteraciones_pedidas": args.iterations,
        "iteraciones_hechas": args.iterations, "iteracion_efectiva": args.iterations - descartadas,
        "mejor_psnr": best_psnr, "psnr_final": history[-1]["psnr"] if history else None,
        "rechazos": rechazos, "max_seguidos": max_seguidos, "rebobinados": rebobinados,
        "reanudada_de": args.reanudar, "horas": round(horas, 3),
        "vram_pico_gb": round(peak, 2), "vram_predicha_gb": round(predicha, 2),
        **resumen_qsm(qsm_hist),
        "best_qsm": mejor_qsm and {k: mejor_qsm[k] for k in ("iteration", "nrmse_final_pct",
                                                             "estabilidad")}}
    (out / "estado.json").write_text(json.dumps(final, indent=2))
    print(f"\nmejor PSNR: {best_psnr:.3f} dB | {horas:.2f} h | VRAM pico {peak:.2f} GiB "
          f"(predicho {predicha:.2f}) | lotes rechazados {rechazos} de {args.iterations} "
          f"(máx. seguidos {max_seguidos}) | rebobinados {rebobinados} | "
          f"iteración efectiva {args.iterations - descartadas}"
          + (f" | QSM: mejor NRMSE {final['qsm_nrmse_mejor_pct']:.2f} % en la "
             f"{final['qsm_iteracion_mejor']}, {final['qsm_fallos']} fallos"
             if final["qsm_nrmse_mejor_pct"] is not None else ""), flush=True)
    # Después de `estado.json`: online, cerrar puede tardar en subir lo pendiente.
    seguimiento.resumen(run, final)
    seguimiento.cierra(run)


if __name__ == "__main__":
    main()

'''
# --- BUCLE -----------------------------------------------------------------------------------
# El de `train.py`, paso por paso, con cuatro diferencias:
#   1. Parches de PATCH_SIZE^3 de `ddr.data3d.PatchDataset3D`, con al menos 35 % de tejido (o,
#      con --mascara-cerebro, enteros dentro del cerebro) y una de las 48 simetrías del cubo, en
#      `--lote / --micro-lote` micro-lotes; el micro-lote lo elige la memoria (MEMORIA).
#   2. El lr no lo lleva un scheduler: `lr_en(it, hitos)` lo fija al principio de cada iteración.
#   3. La cota absoluta del disyuntor escala con el parche (`cota_absoluta`).
#   4. `--guardar-cada` escribe el estado completo, y `--reanudar` sigue desde ahí bit a bit (con
#      el mismo micro-lote).
# Y tres agregados que no tocan la aritmética (test): el seguimiento en W&B (SEGUIMIENTO), la
# evaluación QSM en las validaciones (EVALUACIÓN QSM) y el número de estabilidad (ESTABILIDAD).
# Como en `train.py`, el bucle no corta nunca: ni por NaN, ni por rechazos, ni por early stopping,
# ni por un fallo de W&B, de la evaluación QSM o de la estabilidad. Una corrida terminó si su
# `estado.json` dice `"terminada": true`.

---------------------------------------------------------------------------------------------------

# --- RECETA 3D -------------------------------------------------------------------------------
# Lo que no cambia se importa de `train.py`: LR, BETAS, SIGMA, STEPS, TDV33, T_INIT = 8·δ_Adam,
# LAMBDA_INIT, el disyuntor y los modos. Lo que cambia, cada uno con su procedencia:
#   PATCH_SIZE = 48     48^3 x 4 es el menor costo por vóxel de la grilla medida en `rapido`:
#                       2.26 µs, contra 2.77 a 32^3 x 4 y 2.36 a 64^3 x 1; 64^3 x 2 no entra en
#                       16 GB (docs/mediciones/costo-tdv3d.txt, 2026-10-08).
#   LOTE = 32           el «mini batch size of 32» del paper (sec. 5.1), decisión del autor del
#                       2026-10-09. Es el mismo NÚMERO de parches, no el mismo contenido: 32 x
#                       48^3 = 3.54 M vóxeles por paso, 4.3x los 830 k valores de 32 x 93^2 x 3
#                       en BSDS color; el paper no tiene versión 3D, así que ningún lote 3D es «el
#                       del paper». El de antes era 8 (885 k, del orden de BSDS en valores). El
#                       cambio no está medido: lo mide el piloto, y las columnas 5 y 10 de la traza
#                       dicen si 32 queda por encima de la escala de ruido del gradiente, donde
#                       cada parche de más compra menos. 32 se parte en 16, 8, 4, 2 o 1.
#   MICRO_LOTE = auto   el mayor divisor de LOTE que entra en --vram-gb (MEMORIA): 16 en 96 GiB, 8
#                       en 48, 2 en la 5070 Ti. Con acumulación el gradiente es el mismo para cualquier
#                       partición, porque ninguna capa acopla muestras: el micro-lote es hardware,
#                       no receta. En `referencia` 48^3 x 2 no entró en 16 GB: los tests y las
#                       corridas exactas usan --micro-lote 1.
#   ITERACIONES_TANDA   40 000, el tronco del piloto. El costo se lee en s/it, no en horas: en la
#                       5070 Ti, con el micro-lote 2 de `auto`, la sonda del 2026-10-13 dio
#                       0.280–0.291 s por parche (8.96–9.30 s/it; docs/bitacora.md, pre-registro
#                       de `ventana-3d-12h`), y esa tanda corrió a 9.17–9.24 s/it sin contar el
#                       ADMM ni la estabilidad (docs/mediciones/ventana-3d-12h.txt). La grilla del
#                       2026-10-08, sin el resto del bucle, daba 0.250 s por parche a micro-lote 4
#                       y 0.265 a 2 (docs/mediciones/costo-tdv3d.txt). En otra tarjeta lo mide la
#                       sonda, y N ≈ horas · 3 600 / (s/it), menos las validaciones.
#   GUARDAR_CADA        10 000: las ramas del piloto salen de la 10 000, 20 000 y 30 000; también
#                       se guarda en la última. Un estado pesa ~51 MB en la 40 000 (derivado:
#                       pesos, m y v de Adam tres veces —la corrida, el respaldo y el mejor—,
#                       los pesos de `best_qsm.pth`, 4.7 MB, y 4.5 MB de traza).
# `--mode` vale `rapido` por defecto, no `referencia` como en `train.py`: en 3D `referencia` cuesta
# ~21x (47.9 contra 2.26 µs por vóxel) y a 48^3 x 2 no entró en 16 GB. Una corrida en `rapido` no
# repite bit a bit, así que de ella se lee la tasa sobre semillas, nunca una contra otra.

---------------------------------------------------------------------------------------------------

# --- MEMORIA ---------------------------------------------------------------------------------
# El modelo predice el pico de `torch.cuda.max_memory_allocated` del entrenamiento como
# FIJA + n · P^3 · POR_VOXEL, con n el micro-lote. Se ajustó a la grilla de la 5070 Ti
# (docs/mediciones/costo-tdv3d.txt, 2026-10-08: forward + backward, `R.grad` compilado):
#   FIJA = 0.61         la recta de 48^3 en `rapido` por n = 1 y n = 4 (4.14 y 14.72 GB), en
#                       n = 0. Por mínimos cuadrados, con n = 2 (7.69), sería 0.625 y 3.525: no
#                       cambia ninguna elección.
#   `rapido`            3.19e-5 GiB por vóxel, la pendiente de esa recta (3.527 por parche de
#                       48^3). Predice 8.97 a 64^3 x 1 contra 8.78 medidos, y a 32^3 x 4 subestima
#                       8 % (4.79 contra 5.21): por eso el MARGEN.
#   `intermedio` y      7.0e-5, por encima de los cuatro medidos (`referencia` 32^3 x 1 y x 2,
#   `referencia`        2.48 y 4.77; 48^3 x 1, 7.60; `intermedio` 48^3 x 1, 7.55), que sobreestima
#                       9 a 17 %: la pendiente desde FIJA da 5.7e-5 a 6.3e-5.
#   checkpointing       3.4e-6 en `rapido` (64^3 x 4: 4.11; 96^3 x 1: 3.33, que sobreestima 9 %) y
#                       5.6e-6 en `referencia` (64^3 x 2: 3.56). `intermedio` no se midió: usa el
#                       de `referencia`. La grilla checkpointeaba los S pasos, y `VNet2D.forward`
#                       checkpointea uno sólo si su x lleva gradiente: x_0 lo lleva con
#                       --guard-amp, y sin la bandera el paso 1 guarda sus activaciones hasta el
#                       backward. `vram_predicha` le suma entonces POR_VOXEL[modo, False] / S, un
#                       paso de los S sin checkpointing (medido en CPU: 1.04 pasos medios), 6.6e-6
#                       en total en `rapido` y 1.26e-5 en `referencia`. Por eso --checkpointing
#                       baja la memoria ~5x, y ~9x sólo con --guard-amp.
#   MARGEN = 0.10       del presupuesto, supuesto: la evaluación (recortes de 128^3 sin gradiente),
#                       la fragmentación del allocator y los workspace de cuDNN, que en otra
#                       tarjeta pueden ser otros.
# El presupuesto es min(--vram-gb, memoria de la tarjeta), en GiB como los reporta torch, que es
# algo menos que la etiqueta: la 5070 Ti de «16 GB» da 15.45 (16 303 MiB en nvidia-smi). Si es
# menor que la tarjeta, `set_per_process_memory_fraction` no deja a PyTorch pasar de ahí: la
# tarjeta puede estar compartida, y mejor un OOM propio que pisar a otro proceso. En CPU no hay
# tarjeta: el presupuesto es --vram-gb y `auto` elige lo mismo que en una de ese tamaño.
# VRAM_GB = 96 desde el 2026-10-15: la tarjeta de destino es una RTX PRO 6000 Blackwell Max-Q de
# 96 GB, entera para el autor (antes eran 48 GB de una compartida). En ~95 GiB entrarían 24 y
# `auto` da 16: 57.1 GiB predichos, extrapolados de la 5070 Ti; 32 pediría 113.5. Si 16 es más
# rápido por parche que 8 no está medido: lo dice la sonda en esa tarjeta, y si no lo es,
# `--micro-lote 8` (28.8 GiB) rinde igual.
# Con 48 GiB a 48^3 en `rapido` entrarían 12 (42.9 de 43.2 útiles) y `auto` da 8, el mayor
# divisor de 32: 28.8 GiB predichos, ~60 % de la tarjeta. Está bien: lo que cuenta es saturar el
# cómputo, no llenar la memoria, y en la 5070 Ti de 48^3 x 2 a x 4 el costo por vóxel ya bajaba
# sólo 6 % (2.39 a 2.26 µs). Con --checkpointing `auto` da 32 (23.9 GiB predichos), un solo
# micro-lote: la columna 10 de la traza es entonces la 5 y no estima nada; para conservarla,
# --micro-lote 16.
# Todo esto se ajustó en 16 GB y se extrapola a 48: es derivado, no medido en la tarjeta de
# destino. `--sonda-vram` lo mide: SONDA_ITERACIONES = 4 de forward + backward con cada divisor
# de --lote, de menor a mayor y compilado como para entrenar, hasta el primero cuyo pico pasa de
# los útiles o da OOM; imprime predicho contra medido y lo deja en `sonda-vram.json`. Si el medido
# admite más que `auto`, se pasa --micro-lote explícito. Imprime además los segundos por parche
# de las tres últimas, entre dos `torch.cuda.synchronize` (la primera compila `R.grad` para esa
# forma, porque va con `dynamic=False`, y deja a cuDNN elegir algoritmos: no se cronometra), y
# cuál micro-lote es el más rápido: es lo que dice, en la tarjeta de destino, si llenar más
# memoria acelera. En la 5070 Ti, de 48^3 x 2 a x 4 el costo por vóxel bajaba 6 %. Tres
# iteraciones ordenan micro-lotes; el costo de la corrida se mide en la corrida, con la tarjeta
# sola (regla 3 de AGENTS.md). La sonda no corre la evaluación. En la
# corrida de humo (48^3 x 4, --checkpointing, sin --guard-amp) el pico total, evaluación incluida,
# fue 3.59 GiB, y el modelo predice 3.53 para el entrenamiento solo: no hace falta la evaluación
# para explicarlo, que no se midió por separado. Al terminar, el log y `estado.json` dan el pico de
# la corrida, evaluación incluida, y el predicho.
# La evaluación QSM no entra en el modelo ni en la sonda: es sin gradiente sobre los 160^3 de
# COSMOS y pide 5.19 GiB de pico en `rapido` (6.92 en fp32; docs/mediciones/qsm-eval-gpu.txt), por
# debajo del entrenamiento sin --checkpointing desde 48^3 x 2 (7.69 medidos) y por encima de él con
# --checkpointing a 48^3 x 4 (3.53 predichos): ahí el pico de `estado.json` puede ser el suyo. Si no
# entra, su OOM lo ataja `evalua_qsm` y el entrenamiento sigue (EVALUACIÓN QSM). La estabilidad
# tampoco entra: guarda el grafo de `R.grad` sobre un cubo de 96^3 y pide ~9 GB en float32
# (derivado de la RSS en CPU, sin medir en GPU), más que la evaluación QSM; su OOM lo ataja
# `mide_estabilidad` (ESTABILIDAD).

---------------------------------------------------------------------------------------------------

# --- CALENDARIO DEL LR -----------------------------------------------------------------------
# En `train.py` el lr vive en `opt.param_groups` y lo mueve `MultiStepLR` sólo en los hitos; el
# respaldo lo copia con el resto del estado de Adam, y por eso `rebobina()` lo conserva a mano desde
# su arreglo del 2026-10-08. Acá el lr es una función pura de la iteración, `lr_en(it, hitos)`, y se
# escribe en el optimizador al principio de cada iteración: ni el respaldo ni un estado reanudado
# pueden traer un lr viejo, y una rama puede reanudar con otro calendario. `tests/test_train3d.py`
# lo compara con `MultiStepLR` iteración por iteración.
# `--hitos` acepta cualquier lista: sin la bandera son los múltiplos de 25 000 hasta
# `--iterations`, como en el paper; sola, ninguno, que es el tronco a lr constante.

---------------------------------------------------------------------------------------------------

# --- REANUDACIÓN -----------------------------------------------------------------------------
# `estado-<it>.pth` guarda todo lo que el bucle lee: pesos, `opt.state_dict()`, la ventana
# `escala`, el respaldo completo con `it_respaldo`, los contadores (rechazos, seguidos,
# max_seguidos, rebobinados, limpias, descartadas, it_rebobinado), `history`, `best_psnr`,
# `max_grad`, el generador del ruido, los de torch (CPU y CUDA), la iteración, la config de la VNet
# y las banderas. Además, lo que el directorio tenía hasta `it`: la traza, `mejor` (el payload de
# `best.pth`), `qsm`, las filas de `qsm.jsonl`, y `mejor_qsm`, el payload de `best_qsm.pth` (o
# None). Así el estado se basta solo: no depende de que `traza.npy`, `best.pth`, `best_qsm.pth` o
# `qsm.jsonl` de su directorio sigan enteros ni de que sean de su linaje, y una rama puede arrancar
# mientras el tronco los reescribe.
# Un estado anterior a una bandera de receta la tiene en su default: `--reanudar` compara con
# `p.get_default` la que falta. Los de runs/humo3d no tienen `mascara_cerebro` y se reanudan sin
# ella (test con un estado sin la clave); con la bandera prendida, se rechaza. Uno sin `qsm`
# arranca sin evaluaciones QSM previas, y uno sin `mejor_qsm`, sin `best_qsm.pth`.
# Los parches no necesitan estado: el `i` sale de `(seed, offset + i)`, así que se reanuda con
# `offset = it * lote`. El generador global de torch no lo usa nada del bucle; se restaura igual,
# después de `iter(loader)`, que saca de él una semilla para los workers.
# Seguir en el mismo `--out` sólo admite cambiar LIBRES; una rama, además, `--hitos`. Lo demás
# cambiaría la aritmética, y `--reanudar` lo rechaza en vez de ignorarlo.
# Seguir en otra tarjeta cambia `--micro-lote` (o `--vram-gb`, que lo elige con `auto`), y está
# permitido: el lote, los parches y el gradiente en aritmética exacta son los mismos, pero el orden
# de suma no, así que desde el corte la corrida no es bit a bit la que hubiera seguido, y el log lo
# dice con un AVISO. `config.json` y el estado guardan el micro-lote efectivo, no `auto`. El ruido
# sale de `torch.randn` con el generador en CPU, micro-lote por micro-lote: con --sigma fijo y P^3
# múltiplo de 16 da los mismos valores partido de cualquier manera, porque se generan de a 16
# (verificado el 2026-10-09 con torch 2.13: 4 parches de 48^3 de una vez, 2 + 2 y 1 + 1 + 1 + 1,
# idénticos bit a bit; a 15^3 no). Con --sigma-max no: cada micro-lote sortea sus sigmas antes que
# su ruido, y otra partición le da a cada parche otro ruido. Y con --guard-amp la amplificación
# adjunta suma normas micro-lote por micro-lote: su valor depende de la partición, y con él lo que
# el disyuntor rechaza. El AVISO nombra lo que cambia en cada caso.
# Verificado en `tests/test_train3d.py`: en `referencia` y CPU, 12 iteraciones seguidas contra
# 6 + 6 y 9 + 3 reanudadas dan `traza.npy`, `history.json`, log y estado final idénticos bit a bit,
# con ruido ciego, aumentos y rebobinados a los dos lados del corte. `rebobina()` restaura una copia
# del respaldo (arreglo del 2026-10-08), así que el respaldo guardado en el estado es el del momento
# en que se tomó, y el test que lo exige ya no está marcado. Y seguir desde la 6 con --micro-lote 2
# un estado de micro-lote 1 hereda la traza bit a bit y queda a menos de 1e-6 relativo de la
# corrida seguida en las 6 iteraciones siguientes, sin ser bit a bit; con otro --lote se rechaza.

---------------------------------------------------------------------------------------------------

# --- DIRECTORIO DE SALIDA --------------------------------------------------------------------
# Al arrancar, `--out` queda como lo describe el estado (vacío, sin `--reanudar`):
#   - `estado.json` se borra: mientras falta, la corrida está en vuelo o la mataron, como en
#     `train.py`, y los análisis de supervivencia cuentan con eso.
#   - `history.json`, `traza.npy` y `qsm.jsonl` se reescriben hasta `it0` (`qsm.jsonl` se borra si
#     el estado no trae evaluaciones QSM).
#   - un `last.pth`, `best.pth` o `best_qsm.pth` que no sea el del estado (otra iteración u otro
#     PSNR) se aparta como `<nombre>-descartada-<it>.pth`, sin borrarlo, y `best.pth` y
#     `best_qsm.pth` se reponen desde el estado. Uno que no se puede leer no traba el arranque: se
#     aparta como `<nombre>-descartada-ilegible.pth`. Si el nombre ya está tomado, por otro linaje
#     descartado que llegó a la misma evaluación, se le agrega `-2`, `-3`...: nada apartado se
#     pisa.
# Así, seguir en el mismo `--out` desde un estado anterior al último no deja un `estado.json`
# «terminada» de la corrida corta, ni un `best.pth` de iteraciones que no están en la traza ni en
# `history.json`, y `mejor_psnr` de `estado.json` es siempre el de `best.pth` (y `best_qsm`, el de
# `best_qsm.pth`). Una rama hereda `history`, `best_psnr`, `best.pth` y `best_qsm.pth` del tronco.
# Los `estado-<it>.pth` de un linaje descartado se quedan: cada uno se basta solo, y se pisan
# cuando la corrida nueva llega a su iteración.

---------------------------------------------------------------------------------------------------

# --- TRAZA -----------------------------------------------------------------------------------
# float64, una fila por iteración. Las diez primeras columnas son las de `train.py`, en su orden,
# porque los `scripts/` las leen por posición:
#   0 loss   1 T   2 ∂L/∂T   3 λ   4 ∂L/∂λ   5 ‖g‖ sin recortar   6 mediana de la ventana
#   7 amp (NaN sin `--guard-amp`)   8 rechazado (0/1)   9 rebobinado (0/1)
# y cuatro más al final:
#   10 ‖g‖ del primer micro-lote por la acumulación: la norma de ese micro-lote como si fuera el
#      lote. Con la 5 estima offline la escala de ruido del gradiente, B_simple = tr(Σ)/|G|²
#      (McCandlish et al., 2018), con los lotes --micro-lote y --lote: el efectivo, que imprime el
#      log al arrancar; si cambia al reanudar, la columna cambia de tamaño desde el AVISO. Sólo lee
#      `.grad` entre dos backward, que acumulan sobre el mismo tensor: no cambia ningún bit
#      (test). Con --micro-lote = --lote es la 5 y no estima nada, y en la iteración 1 es bit a bit
#      la ‖g‖ de una corrida con lote de --micro-lote (mismo parche, ruido y pesos; los factores 2
#      son exactos en binario).
#   11 lr que usó el paso, leído del optimizador.
#   12 iteración efectiva: `it` menos lo descartado por los rebobinados, min(it − it_respaldo,
#      it − rebobinado anterior) por evento. Es la abscisa del PSNR a iteración pareada.
#   13 sigma media del lote: `--sigma`, o la media de los sorteados con `--sigma-max`.

---------------------------------------------------------------------------------------------------

# --- DISYUNTOR EN 3D -------------------------------------------------------------------------
# GUARD_RATIO, GUARD_VENTANA, GUARD_SEGUIDOS, GUARD_RESPALDO y GUARD_LIMPIAS son los de BSDS: el
# relativo es invariante de escala (docs/mediciones/umbral-guard-escala.txt). La cota absoluta no:
# la pérdida suma valores por parche, y en 3D el arranque sano la cruza (48^3: máximo 8 367 en 80
# iteraciones; 64^3: 15 985, docs/mediciones/arranque-g-3d.txt). `cota_absoluta(P)` la lleva por
# el cociente de valores sumados, GUARD_ABS · P^3 / (93^2 · 3): 4.26e4 a 48^3 y 1.01e5 a 64^3,
# 5.1x y 6.3x sobre esos máximos. Es derivada, no calibrada: los máximos son de una semilla y 80
# iteraciones, y que el hueco [6 113, 1.97e5] de BSDS escale igual es un supuesto. GUARD_SEGUIDOS
# = 25 viene de la racha sana más larga en BSDS (18); en 3D la tiene que medir el tronco.

---------------------------------------------------------------------------------------------------

# --- RUIDO CIEGO Y AUMENTOS ------------------------------------------------------------------
# Apagados por defecto: con los defaults la corrida es la receta del paper en 3D. `--sigma-max` > 0
# sortea un sigma por parche en U(--sigma-min, --sigma-max) con `add_noise_ciego`, del mismo
# generador del ruido. `--aug-signo` y `--aug-ganancia` los aplica el dataset después de elegir el
# parche y su orientación, así que un brazo y su control ven los mismos parches. La evaluación es
# siempre a `--sigma` fijo.

---------------------------------------------------------------------------------------------------

# --- MÁSCARA DE CEREBRO ----------------------------------------------------------------------
# Apagada por defecto, y receta: no está en LIBRES, así que reanudar con otro valor se rechaza.
# `--mascara-cerebro` lee `<data-root>/mascaras/train/<sujeto>.npy` para cada volumen de
# `--train-list` (las escribe `datasets/mascaras_ldm100k3d.py combina`, alineadas vóxel a vóxel
# con `images/train`), una por vez, y cada parche pasa a ser un cubo entero dentro del cerebro, a
# MARGEN_CEREBRO = 1 vóxel o más de su borde. Con la máscara el parche no ve cráneo, cuero
# cabelludo ni aire, y casi nada del LCR externo (HD-BET deja 1–3 vóxeles sobre la superficie
# pial); sin ella, con el filtro de tejido, sólo el 4.0 % de los parches 48^3 cae entero en el
# cerebro. El precio es la periferia: a 48^3 los cubos cubren el 66 % del
# cerebro y el 18.5 % de su franja cortical (≤ 6 mm del borde); a 32^3, el 80 % y el 38 %
# (docs/bitacora.md, «Cubos 100 % cerebro»; por qué deepbet ∩ HD-BET sin TTA y margen 1, en el
# bloque final de `ddr/data3d.py`). Si falta alguna máscara, `train3d` se niega antes de tocar
# `--out` y dice cuántas faltan y los tres comandos que las generan; un volumen sin cubos queda
# fuera del sorteo con un aviso, y el log de arranque dice cuántos tienen y la mediana de esquinas
# por volumen. La evaluación no depende de la bandera (los mismos recortes de 128^3 y los mismos
# PSNR, en todo el recorte, en tejido y, si están las máscaras de `mascaras/val`, en el cerebro),
# así que una corrida con la bandera y una sin ella se miden sobre lo mismo.

---------------------------------------------------------------------------------------------------

# --- EVALUACIÓN Y ARCHIVOS DE SALIDA ---------------------------------------------------------
# Cada `--eval-every` iteraciones y en la última, `evaluate_denoising3d` mide el PSNR y el NRMSE
# medios sobre los recortes fijos de EVAL_RECORTE^3 de la lista de evaluación, con ruido de
# semilla fija por volumen, en todo el recorte y sólo en tejido (vóxeles limpios > 0.05), sin
# gradiente y en el modo de la corrida (bf16 en `rapido`). `train3d` no tiene épocas (sortea
# parches): esta validación es la «época» del seguimiento. Si están las máscaras de cerebro de
# todos los de evaluación (`<data-root>/mascaras/val/<sujeto>.npy`), mide además el PSNR en el
# cerebro, con la máscara recortada con la misma ventana: es la región que ve el entrenamiento con
# --mascara-cerebro, que no ve cráneo ni aire, mientras el recorte entero los tiene. Agregarla no
# cambia los PSNR de antes (test). El NRMSE es el de `ddr/qsm3d.nrmse_pct`, 100·‖x̂ − x‖/‖x‖ por
# volumen y promediado como el PSNR: en un volumen es función del PSNR, y deja hablar en las
# unidades de QSM (bloque final de `ddr/seguimiento.py`, MÉTRICAS). Después de guardar todo eso
# vienen, si toca, la estabilidad (ESTABILIDAD) y la evaluación QSM (EVALUACIÓN QSM). En `--out`
# quedan:
#   config.json    las banderas de la última invocación, defaults incluidos, con el micro-lote
#                  efectivo en vez de `auto`.
#   history.json   una entrada por evaluación: iteration, efectiva, psnr, psnr_tejido, nrmse_pct,
#                  nrmse_pct_tejido, psnr_cerebro (sólo con las máscaras), T, lambda y, en las
#                  que toca la estabilidad, estabilidad, lambda_max y estabilidad_cambio (o
#                  estabilidad_fallo).
#   qsm.jsonl      una línea por evaluación QSM: iteration, efectiva y lo de `evalua_qsm` sin el
#                  volumen `chi`, o `fallo`, más las tres de la estabilidad en COSMOS (o
#                  estabilidad_fallo). No existe si la corrida no hizo ninguna.
#   traza.npy      float64, COLUMNAS = 14 por iteración. Se reescribe al arrancar, en cada
#                  evaluación y en cada estado.
#   last.pth       la última evaluación; best.pth, la de mejor PSNR en todo el recorte. Claves:
#                  config (la de `vnet_config3d`, con R de tipo `tdv3d`), model, iteration, psnr,
#                  optimizer.
#   best_qsm.pth   el checkpoint para QSM (EVALUACIÓN QSM): config, model, iteration, psnr,
#                  nrmse_final_pct, nrmse_min_pct, iteraciones, motivo y estabilidad, sin
#                  optimizer. No existe si ningún estado estable tuvo evaluación QSM.
#   estado-<it>.pth  cada `--guardar-cada` y en la última, todo lo que hace falta para reanudar.
# La traza y los `.pth` se escriben a un temporal y después `replace` (`guarda_traza`,
# `guarda_pth`): un corte no deja ninguno a medio escribir con el nombre bueno. `qsm.jsonl` se
# agrega línea por línea, y un error al escribirla se avisa sin cortar: el estado lleva las filas.
#   estado.json    sólo al terminar, y se borra al arrancar: lo de `train.py` más
#                  iteracion_efectiva, reanudada_de y vram_predicha_gb, el pico del modelo de
#                  MEMORIA junto al medido (vram_pico_gb, que incluye las evaluaciones), y las
#                  qsm_* de `resumen_qsm`: evaluaciones, fallos, el último NRMSE final con su
#                  iteración y su motivo, y el menor con la suya (None sin evaluaciones QSM),
#                  estable o no. Y best_qsm: iteration, nrmse_final_pct y estabilidad de
#                  `best_qsm.pth`, o None si no hay.
#   seguimiento.json, wandb/  con W&B (SEGUIMIENTO): el id de la corrida y, offline, la corrida.
#   sonda-vram.json  sólo con `--sonda-vram`, que no escribe nada más y no entrena.

---------------------------------------------------------------------------------------------------

# --- SEGUIMIENTO -----------------------------------------------------------------------------
# W&B es una vista de `--out`, no otra fuente: todo lo que registra está también en la traza,
# `history.json`, `qsm.jsonl` o `estado.json`. Lo hace `ddr/seguimiento.py`, donde está el porqué
# de los modos, la identidad de la corrida y la protección de cada llamada; acá van dónde se llama
# y qué se registra.
#   - `abre`, después de `iter(loader)`: los workers ya existen y no se forkea con los hilos de
#     wandb vivos. Con `padre` = --reanudar y `desde` = it0, `abre` sigue la misma corrida sólo si
#     W&B no registró nada después de `desde`: extender una terminada, o un corte que cayó justo
#     en un estado. Tras un corte común (registra cada --log-every y los estados van cada
#     --guardar-cada) lo registrado pasa del estado: abre otra corrida, con el mismo nombre, y el
#     id viejo queda en `descartadas` de `seguimiento.json`. Una rama es otra corrida, en el grupo
#     del tronco (los tres casos, test offline). El grupo lo elige `abre` (el del tronco, o su
#     nombre) y no se pasa: así una rama de una rama queda en el grupo del tronco original.
#     config: `vars(args)`, la misma de `config.json` con el micro-lote efectivo, más `commit`
#     (`commit_git`: el de HEAD, con `+cambios` si lo versionado tiene cambios, o None sin git).
#   - cada --log-every, en la fila de la iteración: `metricas_traza` (las 14 columnas, con los
#     nombres de `seguimiento.COLUMNAS_TRAZA`), train/grad_norm_max (el max|g| del log en el
#     intervalo, que no pierde un estallido entre dos muestras), train/s_por_it (sin evaluaciones
#     ni estados: el reloj se corre tras cada uno) y, con GPU, train/vram_pico_gib (el pico desde
#     el arranque, evaluaciones incluidas). Cada --log-every y no por iteración: W&B muestra lo que
#     muestra el log, y la traza tiene todas las filas. Por iteración también entraba (0.185 ms
#     por llamada, medido; 53 MB por corrida de 40 000, derivado; docs/mediciones/
#     costo-seguimiento.txt), pero no agregaba nada que la traza no tenga.
#   - en cada validación: val/psnr, val/psnr_tejido, val/psnr_cerebro (con las máscaras),
#     val/nrmse_pct, val/nrmse_pct_tejido y la lámina val/denoising (`lamina_denoising` del primer
#     volumen de evaluación, siempre el mismo: limpia | ruidosa | salida | 5·|error|).
#   - en cada evaluación QSM: qsm/nrmse_final_pct, qsm/nrmse_min_pct, qsm/iter_min,
#     qsm/iteraciones, qsm/alpha, qsm/segundos, qsm/fallo (0 o 1) y la lámina qsm/reconstruccion
#     (`lamina_qsm`: GT | reconstrucción | error en ±0.1 ppm). `motivo` es texto: va al log, a
#     `qsm.jsonl` y, el de la última, al resumen.
#   - con la cadencia de la evaluación QSM, haya COSMOS o no: val/estabilidad y val/lambda_max;
#     con COSMOS, también qsm/estabilidad y qsm/lambda_max (ESTABILIDAD). Una estabilidad que
#     falla no registra nada en W&B: queda en el log y en `estabilidad_fallo`.
#   - al final, después de `estado.json`: `resumen` con lo mismo que `estado.json`, y `cierra`.
# Las láminas se arman dentro de `registra_lamina`: un error al armarla cuesta la imagen, no la
# corrida ni las métricas QSM de esa evaluación (test), y sin seguimiento no se arma. Con
# `--seguimiento no`, wandb ni se importa (test).
# No toca la aritmética: con W&B offline y la evaluación QSM, la traza es bit a bit la de la
# corrida sin ninguna de las dos (test, con wandb instalado), y `abre` y `registra` no tocan los
# generadores (tests/test_seguimiento.py).

---------------------------------------------------------------------------------------------------

# --- EVALUACIÓN QSM --------------------------------------------------------------------------
# Lo que el autor pidió: medir el modelo como denoiser dentro de la reconstrucción QSM, que es su
# destino. Es el ADMM plug-and-play de `../tdv/wTDV_QSM_torch_3d.py` en `ddr/qsm3d.admm_wtdv`,
# con k = 1 (la red 3D sobre los 160^3 de COSMOS de una vez) y su protocolo: μ = 0.008, μ₂ = 1.0,
# target_std = 0.06 y hasta MAX_ITER = 100 iteraciones con su criterio de parada (`ddr/qsm3d.py`
# reproduce su 26.89 % con la red 2D del autor). La métrica es la suya, el NRMSE contra
# `chi_cosmos` en su máscara.
#   - Cuándo: en las validaciones, después de guardar `history.json` y los `.pth` (un corte duro a
#     mitad del ADMM no pierde la validación) y antes del estado, que lleva la fila. Con
#     --eval-qsm-cada 0, en todas; con N > 0, en las de iteración múltiplo de N y en la última.
#   - Los datos: `--qsm-datos` (QSM_DATOS = datasets/qsm_cosmos) con `params.mat` y
#     `chi_cosmos.mat` de `../tdv/data_mat`; se cargan una vez al arrancar, en numpy y en CPU
#     (102 MB y 0.5 s, medido el 2026-10-12). Si falta alguno, un AVISO dice cuál y de dónde
#     copiarlo, la evaluación queda apagada y el entrenamiento sigue: no es receta, y una corrida
#     sin ella entrena igual.
#     `--qsm-datos ''` la apaga sin aviso (los tests que no la miran).
#   - En el modo de la corrida: `denoiser_3d` con el autocast de `--mode` (bf16 en `rapido`) y el
#     `R.grad` compilado, que en 160^3 compila una forma más (dynamic=False). Con las del
#     entrenamiento y la validación son 4 entradas de dynamo, la mitad de `recompile_limit`, y
#     volver a entrenar no recompila (medido en CPU con formas chicas,
#     docs/mediciones/dynamo-entradas-qsm.txt).
#   - No puede matar el entrenamiento: `evalua_qsm` ataja toda `Exception`, incluido
#     `torch.cuda.OutOfMemoryError`, libera la caché de CUDA fuera del `except` (adentro la traza
#     del error retiene los tensores) y devuelve `fallo`, que el bucle manda al log, a
#     `qsm.jsonl` y a W&B como qsm/fallo = 1. Test con un ADMM que levanta un OOM. La lámina va
#     fuera de `evalua_qsm`: un error al armarla no tira las métricas (SEGUIMIENTO).
#   - No perturba el entrenamiento: corre en modo eval y sin gradiente, deja el modelo en modo
#     train, no usa ningún generador (ni el del ruido ni el de los parches) y no escribe nada que
#     el bucle lea. Test: en `referencia` y CPU, con COSMOS de juguete de 24^3, la traza, la
#     historia, el log y los pesos finales salen bit a bit iguales con y sin ella.
#   - Costo, medido en la 5070 Ti (docs/mediciones/qsm-eval-gpu.txt, 2026-10-12): en `rapido`
#     3.40 s por iteración de ADMM (44 s la primera, que compila) y 5.19 GiB de pico; en fp32 sin
#     compilar, 25.95 s y 6.92 GiB. Una evaluación típica, ~27 iteraciones, son ~1.5 min: 0.5–2 %
#     de un intervalo de 2 000 iteraciones a 9.2 s/it (derivado; el s/it, de la 5070 Ti en
#     `ventana-3d-12h`). Con una red que no corta, las
#     100 iteraciones son ~5.7 min. En la tarjeta de destino no está medido.
#   - `best_qsm.pth` es el checkpoint que se pasa a QSM. En la bitácora (2026-10-12) el PSNR de
#     denoising no servía para elegir un prior PnP; el criterio es el NRMSE QSM entre los estados
#     estables. Tras una evaluación QSM sin fallo, con la estabilidad en COSMOS medida sin fallo y
#     menor que UMBRAL_ESTABILIDAD, si su NRMSE final es el menor hasta ahora se escribe con
#     `guarda_pth` y una línea en el log. Un NRMSE o una estabilidad NaN no lo escriben. Lleva lo
#     de `best.pth` salvo el optimizer (es para usar; para seguir entrenando está el estado) y
#     nrmse_final_pct, nrmse_min_pct, iteraciones, motivo y estabilidad. Viaja en el estado como
#     `mejor_qsm`, y al reanudar se aparta o se repone como `best.pth` (DIRECTORIO DE SALIDA). Sin
#     COSMOS, con --estabilidad-lado 0 o sin el punto de COSMOS no se busca, y el arranque lo dice
#     en una línea. Ojo: `qsm_nrmse_mejor_pct` de `estado.json` es el menor de todas, estables o
#     no; el de `best_qsm.pth` está en `best_qsm`. Comparar sólo vale con los mismos COSMOS,
#     --qsm-max-iter y --estabilidad-lado, que son libres al reanudar: cambiarlos a mitad mezcla
#     criterios en la misma elección, y nada lo verifica. Tests: con la estabilidad forzada, se
#     escribe en la iteración del menor NRMSE entre las estables (2 justo ya no lo es), con todas
#     inestables no hay archivo, reanudar repone el del estado y aparta uno ajeno, y la traza es la
#     de la corrida sin QSM.
#   - Abierto: TARGET_STD = 0.06 se calibró para la red 2D color del autor, no para una 3D
#     entrenada en T1, y un modelo poco entrenado puede no cortar nunca (sin medir).

---------------------------------------------------------------------------------------------------

# --- ESTABILIDAD -----------------------------------------------------------------------------
# El número del paso explícito del esquema, (T/S)·λ_max(∇²R(x)), que `ddr/qsm3d.estabilidad` mide
# en un punto x: el paso amplifica si pasa de UMBRAL_ESTABILIDAD = 2 (el porqué y el método, en el
# bloque final de `ddr/qsm3d.py`). Viene de la 2D: el 2026-10-12 explicó por qué los checkpoints
# 2D propios rinden peor que el oficial como prior del ADMM de QSM y no convergen. Medido a la
# amplitud de entrenamiento (σ = 25) los cuatro dan 1.38–2.01 y el PSNR no los distingue; a la de
# uso (la estimación inicial del ADMM llevada a std 0.06) el oficial da 0.99 y los propios
# 2.8–23.2 (docs/bitacora.md, «POR QUÉ los checkpoints propios rinden peor como prior PnP
# de QSM»). El PSNR de la validación es ciego a esto; este número no.
#   - Dónde: en dos puntos, armados una vez al arrancar. `val`, el punto de entrenamiento: el cubo
#     central de --estabilidad-lado^3 del primer recorte de evaluación, con el ruido que le pone la
#     validación (σ = --sigma, semilla SEMILLA_VAL, la de `evaluate_denoising3d`, que ahora se pasa
#     explícita), así que es lo que la validación ve en ese cubo y no usa el generador del ruido.
#     `qsm`, sólo con COSMOS: `punto_qsm3d`, la entrada del primer denoiser del ADMM (alpha·chi_0)
#     recortada a un cubo centrado en la máscara, en el device de la corrida, que es donde corre el
#     ADMM. Un lado que no entra se recorta al volumen (en los tests, el 96 por defecto queda en
#     los 16^3 del recorte de evaluación).
#   - Cuándo: con la cadencia de la evaluación QSM (`--eval-qsm-cada`), haya COSMOS o no: `val`
#     después de guardar la validación, y `qsm` antes del ADMM, en la misma fila de `qsm.jsonl`.
#     `history.json` se reescribe con la entrada completada.
#   - Cómo: en float32 IEEE (sin autocast ni TF32, aunque la corrida sea `rapido`) y con el
#     `R.grad` de la clase, no el compilado de la instancia: no suma entradas de dynamo (medido en
#     CPU con formas chicas, docs/mediciones/dynamo-estabilidad.txt). No toca pesos, `.grad` ni
#     generadores, y una excepción, también un OOM, la ataja `mide_estabilidad` como
#     `evalua_qsm`: va al log y a `estabilidad_fallo`, y el bucle sigue (test con una
#     `estabilidad` que levanta). Con y sin la medición (--estabilidad-lado 0), en `referencia` y
#     CPU, la traza, el log del entrenamiento, los pesos, Adam y los generadores salen bit a bit
#     iguales (test).
#   - Qué se guarda: estabilidad (el paso), lambda_max (‖Hv‖, el módulo del autovalor dominante,
#     como en el script 2D) y estabilidad_cambio, el cambio relativo de la última iteración de la
#     potencia; no el reloj, que cambiaría la fila al reanudar. El log da además el cociente de
#     Rayleigh (el signo del dominante: ≈ −λ_max si es negativo), T y los segundos. Un cambio
#     grande dice que las 40 iteraciones no alcanzaron: el número queda debajo del verdadero
#     (docs/mediciones/convergencia-estabilidad.txt).
#   - --estabilidad-lado = LADO_ESTABILIDAD = 96, el lado del script 2D. Costo, medido en CPU
#     con 8 hilos y la CPU compartida (docs/mediciones/costo-estabilidad.txt, 2026-10-12): 13.9 s
#     por validación a 32^3 y 208 s a 64^3, los dos puntos. En GPU a 96^3, derivado del fp32 de
#     qsm-eval-gpu.txt por dos caminos: 19–36 s por punto y 39–73 s por validación, debajo del tope
#     de 60 s por punto que fijó la especificación; 0.2–0.5 % de un intervalo de 2 000 iteraciones.
#     Medido en la 5070 Ti (docs/mediciones/ventana-3d-12h.txt): 133–134 s por corrida, en dos
#     validaciones de dos puntos, ~33 s por punto. Memoria, derivada de la RSS: ~9 GB a 96^3
#     (~2.7 a 64^3); en GPU no se midió aparte, y el pico de esas corridas, con ella incluida,
#     fue 8.16 GiB. Si no entra, queda
#     `estabilidad_fallo`; --estabilidad-lado 64 la baja ~3.4x. El número depende del lado (en el
#     modelo de humo, 0.85 a 32^3 y 1.27 a 64^3 en COSMOS): se compara al mismo lado.
#   - Qué decide: sólo cuál estado puede ser `best_qsm.pth` (EVALUACIÓN QSM). No corta ni cambia
#     el entrenamiento.
'''