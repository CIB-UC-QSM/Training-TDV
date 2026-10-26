"""
Entrenamiento de TDV 2D sobre BSDS400. Réplica de Kobler et al. (2020), sec. 5.1.
Una iteración: 32 parches de 93x93 con flips y rotaciones -> ruido gaussiano fresco (sigma=25,
re-muestreado cada vez) -> S=10 pasos del esquema desenrollado -> ||x_S - y||^2 sobre el estado
final -> backward -> disyuntor de lotes -> Adam -> re-proyección de las restricciones.

    uv run train.py --out runs/replica
    uv run train.py --out runs/prueba --iterations 2000 --mode rapido
    uv run train.py --out runs/control --guard inf        # brazo de control, sin disyuntor
    CUDA_VISIBLE_DEVICES=1 uv run train.py --out runs/otra --seed 1     # la otra tarjeta
"""

from __future__ import annotations

import argparse
import json
import copy
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ddr.data import ImageDataset2D, PatchDataset2D, add_noise
from ddr.data3d import add_noise_ciego
from ddr.model import VNet2D
from evaluate import evaluate_denoising

# --- RECETA DEL PAPER (sec. 5.1) -------------------------------------------------------------
# Citado textual: "The ADAM optimizer is employed with a mini batch size of 32 using 1e5
# iterations, beta_1 = 0.9 and beta_2 = 0.999, where the initial learning rate 4e-4 is halved
# every 25000 iterations." No tocar: cambiarlos rompe la comparabilidad con el 31.53 dB.
PATCH_SIZE = 93
BATCH_SIZE = 32              # del paper; se acumula en micro-lotes, ver MICRO_BATCH
ITERATIONS = 100_000
LR = 4e-4
LR_HALVE_EVERY = 25_000
BETAS = (0.9, 0.999)
SIGMA = 25.0                 # en escala 0-255
STEPS = 10                   # S, pasos desenrollados
FEATURES, SCALES, MACRO_BLOCKS = 32, 3, 3   # TDV33

# El batch de 32 se acumula en lotes de 16: mismo gradiente, la mitad de memoria activa. Es lo que
# deja entrar `intermedio` y `referencia` sin checkpointing en los 16 GB de una 5070 Ti.
MICRO_BATCH = 16
ACUMULACION = BATCH_SIZE // MICRO_BATCH

# --- CONTROLES APRENDIDOS --------------------------------------------------------------------
DELTA_ADAM = LR / math.sqrt(1 - BETAS[1])   # = 0.01265, el paso de Adam ante un pico
T_INIT = 8 * DELTA_ADAM                     # ≈ 0.1012
# `ITERATIONS` es la receta publicada y no se toca. Las tandas de trabajo corren 10k: con T en
# 10 delta la PSNR llega a su mejor valor antes de la it 5000 y los eventos caen antes de la 4500.
ITERACIONES_TANDA = 10_000
LAMBDA_INIT = 0.1

# --- DISYUNTOR DE LOTES ----------------------------------------------------------------------
# Dos pruebas con trabajos distintos y un rebobinado para cuando las dos llegan tarde. Cada
# número salió de una medición; cuál, en docs/porque-del-codigo.md, DISYUNTOR DE LOTES.
GUARD_RATIO = 5.0       # relativo: debajo de la banda 7.70-8.97x que metió a `disyuntor-100k`
GUARD_ABS = 1e4         # absoluto: cubre los tramos donde la mediana todavía no es estimable
GUARD_VENTANA = 100     # largo de la mediana móvil; el relativo se arma cuando se llena
GUARD_SEGUIDOS = 25     # rechazos seguidos que disparan el rebobinado; el máximo sano medido es 18
GUARD_RESPALDO = 250    # cada cuántas iteraciones sanas se refresca el respaldo en RAM
GUARD_LIMPIAS = 3       # sanas seguidas que exige el respaldo; el hueco dentro de un evento es 1

# --- DATASET ---------------------------------------------------------------------------------
DATA_ROOT = "datasets/bsds500"
TRAIN_SPLITS = ("train", "test")   # 400 imágenes; CBSD68 vive en `val` y queda fuera
EVAL_SPLIT = "val"
EVAL_LIST = "datasets/bsds500/cbsd68.txt"
CHANNELS = 3
NOISE_SCALE = 255.0                # sigma en niveles de 8 bits
DATA_RANGE = 1.0

# --- MODOS DE EJECUCIÓN ----------------------------------------------------------------------
MODES = {
    "rapido":     {"amp": torch.bfloat16, "tf32": False, "deterministic": False, "benchmark": True},
    "intermedio": {"amp": None,           "tf32": True,  "deterministic": False, "benchmark": True},
    "referencia": {"amp": None,           "tf32": False, "deterministic": True,  "benchmark": False},
}


def vnet_config(t_init: float, channels: int = CHANNELS) -> dict:
    """Config de la VNet, verificada contra los .pth oficiales."""
    return {
        "S": STEPS,
        "R": {"type": "tdv", "config": {
            "in_channels": channels, "out_channels": 1,
            "num_features": FEATURES, "num_scales": SCALES, "num_mb": MACRO_BLOCKS,
            "multiplier": 1,   # los features NO se duplican al bajar de escala
        }},
        "D": {"type": "denoise", "config": {"use_prox": True}},
        "T_mode": "learned", "T": {"init": t_init, "min": 0.0, "max": 1000.0},
        "lambda_mode": "learned", "lambda": {"init": LAMBDA_INIT, "min": 0.0, "max": 1000.0},
    }


def rechaza_lote(gnorm: float, amp: float, escala: list[float], ratio: float, abs_max: float,
                 amp_max: float, ventana: int) -> tuple[bool, float]:
    """`True` si hay que saltarse `opt.step()`, y el centro contra el que se comparó.

    `escala` son las últimas `ventana` normas ACEPTADAS, y esta función la muta: el pico rechazado
    no entra, así que no puede subir su propio umbral. Es lo que lo distingue del clip, que
    reescala la magnitud y a eso Adam ya es invariante (AGENTS.md, «Lo que no sirve»).
    """
    centro = float(np.median(escala)) if escala else float("nan")
    # El relativo sigue la escala de la corrida, que se mueve dos órdenes a lo largo del
    # entrenamiento; se arma recién cuando la ventana se llena, porque antes no es estimable.
    fuera = len(escala) >= ventana and gnorm > ratio * centro
    # El absoluto queda para las primeras 100 iteraciones, el único tramo sin relativo porque la
    # ventana viaja con el respaldo, y no se puede quedar viejo porque no se estima de nada.
    # Un solo paso con NaN o inf vuelve NaN los 388k pesos, así que no pasan nunca.
    if not math.isfinite(gnorm) or gnorm > abs_max or fuera or amp > amp_max:
        return True, centro
    escala.append(gnorm)
    del escala[:-ventana]
    return False, centro


def respalda(model: torch.nn.Module, opt: torch.optim.Optimizer,
             escala: list[float]) -> tuple[dict, dict, list[float]]:
    """Copia de los 388k pesos, de `m`/`v` y de la ventana: ~4.5 MB más 800 bytes."""
    # La ventana entra acá porque es estado de la corrida igual que los momentos, y el rebobinado
    # la tiene que poder devolver.
    return copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()), list(escala)


def rebobina(model: torch.nn.Module, opt: torch.optim.Optimizer,
             respaldo: tuple[dict, dict, list[float]], escala: list[float]) -> None:
    """Vuelve pesos, momentos y ventana al respaldo: los tres son el mismo estado. El lr no."""
    model.load_state_dict(respaldo[0])
    # `load_state_dict` adopta los tensores que recibe: sin la copia, los pasos siguientes mutarían
    # el respaldo y el segundo rebobinado al mismo ya no devolvería Adam.
    lr = [g["lr"] for g in opt.param_groups]
    opt.load_state_dict(copy.deepcopy(respaldo[1]))
    # El lr es del calendario, no de la corrida: restaurarlo desharía un halving cruzado.
    for g, valor in zip(opt.param_groups, lr):
        g["lr"] = valor
    # Borrarla dejaría el relativo desarmado 100 iteraciones justo al salir del evento.
    escala[:] = respaldo[2]


def main() -> None:
    """Una corrida 2D con la receta del paper; las banderas, en el README."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="directorio de salida")
    p.add_argument("--mode", choices=MODES, default="referencia")
    p.add_argument("--iterations", type=int, default=ITERACIONES_TANDA)
    p.add_argument("--t-init", type=float, default=T_INIT)
    p.add_argument("--sigma", type=float, default=SIGMA,
                   help="sigma del ruido de entrenamiento y, siempre, el de la evaluación")
    # Ruido ciego, apagado por defecto: con el default la corrida es la receta.
    p.add_argument("--sigma-max", type=float, default=0.0,
                   help="> 0 sortea sigma ~ U(--sigma-min, --sigma-max) por parche (ruido ciego)")
    p.add_argument("--sigma-min", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--eval-every", type=int, default=2000)
    # El default es BSDS400 en color, que es la receta del paper. `--data-root` sirve para
    # cualquier árbol `<raíz>/images/<split>/` cuya extensión esté en `ddr.data.IMAGE_LOADERS`.
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--train-splits", default=",".join(TRAIN_SPLITS))
    p.add_argument("--eval-split", default=EVAL_SPLIT)
    p.add_argument("--eval-list", default=EVAL_LIST)
    p.add_argument("--channels", type=int, default=CHANNELS,
                   help="1 para datos en gris (T1, QSM): cambia la primera conv y el conteo")
    # Calendario del lr. Con los defaults la lista de hitos es la del StepLR del paper; adelantar
    # el primero y acortar el paso sirve para comprimir el tramo final, que rinde 19 h por 0.01 dB.
    p.add_argument("--lr-halve-every", type=int, default=LR_HALVE_EVERY)
    p.add_argument("--lr-halve-start", type=int, default=0,
                   help="iteración del primer halving; 0 la pone en --lr-halve-every")
    p.add_argument("--lr-halvings", type=int, default=0,
                   help="cuántos halvings como máximo; 0 no pone tope")
    # `inf` no recorta y devuelve la norma igual, así que el diagnóstico sale gratis. Un umbral
    # finito es una desviación de la receta: sólo para experimentos, y anotado en la bitácora.
    p.add_argument("--clip", type=float, default=float("inf"), help="norma máxima del gradiente")
    # El disyuntor no toca la aritmética: mientras no rechace, la corrida es bit a bit la misma
    # que con `--guard inf`, y por eso el control es pareado y basta cambiar este número.
    p.add_argument("--guard", type=float, default=GUARD_RATIO,
                   help="rechaza el lote si |g| supera esta razón sobre su mediana móvil")
    p.add_argument("--guard-abs", type=float, default=GUARD_ABS,
                   help="rechaza el lote si |g| supera esta cota absoluta. `inf` la apaga")
    p.add_argument("--guard-ventana", type=int, default=GUARD_VENTANA)
    # Rebobinar no corta el bucle: la traza sigue teniendo una fila por iteración, que es lo que
    # cuentan los análisis de supervivencia.
    p.add_argument("--guard-seguidos", type=int, default=GUARD_SEGUIDOS,
                   help="rechazos seguidos que disparan el rebobinado al respaldo. 0 lo apaga")
    p.add_argument("--guard-respaldo", type=int, default=GUARD_RESPALDO,
                   help="cada cuántas iteraciones sanas se refresca el respaldo")
    p.add_argument("--guard-limpias", type=int, default=GUARD_LIMPIAS,
                   help="iteraciones sanas seguidas que exige el respaldo antes de refrescarse")
    # Éste instrumenta el forward y cuesta un vjp más, así que viene apagado.
    p.add_argument("--guard-amp", type=float, default=float("inf"),
                   help="rechaza también si la amplificación adjunta ||p_0||/||p_S|| supera esto")
    p.add_argument("--checkpointing", action="store_true",
                   help="recomputa el forward en el backward: -78%% de memoria, +31%% de tiempo")
    args = p.parse_args()
    if args.sigma_max > 0 and not 0 <= args.sigma_min <= args.sigma_max:
        p.error(f"--sigma-min {args.sigma_min:g} tiene que estar en [0, --sigma-max "
                f"{args.sigma_max:g}]")

    mode = MODES[args.mode]
    out = Path(args.out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- BACKEND: semillas, determinismo y precisión de las operaciones fp32 -------------------
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = mode["benchmark"]
    if mode["deterministic"]:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # cuDNN trae TF32 encendido de fábrica: una corrida "fp32" no es IEEE salvo que se apague a
    # mano, y en esta red las convoluciones son casi todo.
    precision = "tf32" if mode["tf32"] else "ieee"
    torch.backends.cudnn.conv.fp32_precision = precision
    torch.backends.cuda.matmul.fp32_precision = precision

    # --- MODELO -------------------------------------------------------------------------------
    config = vnet_config(args.t_init, args.channels)
    model = VNet2D(config, efficient=args.checkpointing).to(device)
    # Se compila el MÉTODO R.grad, no el módulo: `forward` no lo invoca nadie en el bucle, así que
    # compilar el módulo es un no-op silencioso. Compilar el método además deja el state_dict sin
    # prefijo `_orig_mod.`, de modo que el checkpoint sigue siendo intercambiable con los oficiales.
    model.R.grad = torch.compile(model.R.grad, dynamic=False)

    # --- DATOS --------------------------------------------------------------------------------
    train_set = PatchDataset2D.from_root(
        args.data_root, tuple(args.train_splits.split(",")), patch_size=PATCH_SIZE,
        channels=args.channels, length=args.iterations * BATCH_SIZE, seed=args.seed * 100,
    )
    loader = DataLoader(train_set, batch_size=MICRO_BATCH, shuffle=False, drop_last=True,
                        num_workers=16, pin_memory=(device.type == "cuda"),
                        persistent_workers=True, prefetch_factor=4)
    names = [l.strip() for l in Path(args.eval_list).read_text().splitlines() if l.strip()]
    eval_set = ImageDataset2D.from_root(args.data_root, args.eval_split, args.channels, names)

    # --- OPTIMIZADOR --------------------------------------------------------------------------
    # Un solo grupo y un solo lr, que es la receta literal. Bajar el lr de T y lambda por separado
    # se probó y creó un modo de fallo nuevo (AGENTS.md, regla 1): no repetirlo.
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS)
    # Hitos explícitos en vez de StepLR: con los defaults la lista es 25k/50k/75k/100k y el lr
    # vale lo mismo iteración por iteración, pero así el primer halving se puede adelantar.
    primero = args.lr_halve_start or args.lr_halve_every
    hitos = list(range(primero, args.iterations + 1, args.lr_halve_every))
    if args.lr_halvings:
        hitos = hitos[:args.lr_halvings]
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=hitos, gamma=0.5)
    noise_gen = torch.Generator(device="cpu").manual_seed(args.seed * 100 + 1)
    mide_amp = math.isfinite(args.guard_amp)

    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"parámetros: {sum(p.numel() for p in model.parameters()):,} | "
          f"batch {BATCH_SIZE} = {ACUMULACION}x{MICRO_BATCH} | modo {args.mode} | "
          f"datos {args.data_root} ({args.channels}ch, {len(train_set.images)} imgs, "
          f"{len(eval_set)} de eval) | hitos lr {hitos} | "
          f"T_init={args.t_init:g} | lambda_init={LAMBDA_INIT:g} | "
          + (f"sigma~U({args.sigma_min:g}, {args.sigma_max:g}) | " if args.sigma_max > 0 else "")
          + f"clip={args.clip:g} | "
          f"guard={args.guard:g}x mediana({args.guard_ventana}) | abs={args.guard_abs:g} | "
          f"rebobina a los {args.guard_seguidos} seguidos | respaldo tras "
          f"{args.guard_limpias} limpias | guard_amp={args.guard_amp:g}"
          + ("  [checkpointing]" if args.checkpointing else ""), flush=True)

    # --- BUCLE --------------------------------------------------------------------------------
    # `traza` guarda por iteración lo que el log cada N no alcanza a ver: las divergencias pasan
    # dentro de una sola iteración y el post-mortem necesita esa resolución.
    history, best_psnr, max_grad, t0 = [], 0.0, 0.0, time.time()
    escala, rechazos, seguidos, max_seguidos = [], 0, 0, 0
    respaldo, it_respaldo, rebobinados, limpias = None, 0, 0, 0
    # Las seis primeras columnas no se mueven de lugar: hay análisis que leen la traza por posición.
    traza = np.zeros((args.iterations, 10), dtype=np.float64)
    # loss T dT lambda dlambda |g| centro amp rechazado rebobinado
    # La 7 es `amp`, como en las trazas de 30 columnas de la medición del adjunto, así que
    # `traza[:, 7]` significa lo mismo en los dos formatos.
    lotes = iter(loader)

    for it in range(1, args.iterations + 1):
        # Los micro-lotes se acumulan antes del paso: el batch efectivo es BATCH_SIZE y el
        # gradiente es el mismo que procesándolo entero, porque ninguna capa acopla muestras.
        opt.zero_grad(set_to_none=True)
        perdida = torch.zeros((), device=device)
        p0, pS = torch.zeros((), device=device), torch.zeros((), device=device)
        for _ in range(ACUMULACION):
            # El ruido se suma en CPU antes de subir: una transferencia por tensor en vez de tres.
            lote = next(lotes)
            if args.sigma_max > 0:
                noisy = add_noise_ciego(lote, args.sigma_min, args.sigma_max, noise_gen,
                                        NOISE_SCALE)[0].to(device, non_blocking=True)
            else:
                noisy = add_noise(lote, args.sigma, noise_gen, NOISE_SCALE).to(device,
                                                                              non_blocking=True)
            clean = lote.to(device, non_blocking=True)
            # Materializar `dL/dx_0` cuesta un vjp más por el paso 0. Medido el 2026-10-03
            # sobre 120 iteraciones en `referencia`: no mueve la trayectoria ni un bit.
            if mide_amp:
                noisy.requires_grad_(True)

            with torch.autocast(device.type, dtype=mode["amp"], enabled=mode["amp"] is not None):
                states = model(noisy, noisy)
                loss = (states[-1].float() - clean.float()).pow(2).flatten(1).sum(1).mean()

            (loss / ACUMULACION).backward()
            perdida += loss.detach() / ACUMULACION
            # `forward` devuelve los estados APILADOS, así que `states[-1]` es una vista nueva en
            # cada indexado y `retain_grad` se le pegaría a un temporal. No hace falta: de la
            # pérdida sale `p_S = 2(x_S − clean)/(MICRO_BATCH·ACUMULACION)` en forma cerrada.
            if mide_amp:
                p0 += noisy.grad.norm() / ACUMULACION
                pS += 2 * (states[-1].detach().float() - clean.float()).norm() / (
                    MICRO_BATCH * ACUMULACION ** 2)

        # Devuelve la norma ANTES de recortar, que es la que sirve de diagnóstico.
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()

        amp = (p0 / pS).item() if mide_amp else float("nan")
        rechazado, centro = rechaza_lote(gnorm, amp, escala, args.guard, args.guard_abs,
                                         args.guard_amp, args.guard_ventana)

        # antes del paso: los valores que produjeron esta pérdida
        traza[it - 1, :5] = torch.stack((perdida, model.T.detach(), model.T.grad,
                                         model.lmbda.detach(), model.lmbda.grad)).cpu().numpy()

        # El lote rechazado no da el paso, y por eso el pico no entra en `m` ni en `v`. El resto
        # sigue: el lote ya se consumió, así que el calendario del lr no se detiene.
        if not rechazado:
            opt.step()
        sched.step()
        model.project_()   # media cero y norma <=1 en K1, positividad de T y lambda

        # Rechazar llega un paso tarde, y el rebobinado es la única salida que no corta el bucle
        # (docs/porque-del-codigo.md, DISYUNTOR DE LOTES).
        rebobinado = False
        if rechazado:
            rechazos += 1
            seguidos, limpias = seguidos + 1, 0
            max_seguidos = max(max_seguidos, seguidos)
            if respaldo is not None and 0 < args.guard_seguidos <= seguidos:
                rebobina(model, opt, respaldo, escala)
                rebobinados, seguidos, rebobinado = rebobinados + 1, 0, True
                print(f"[{it:6d}] REBOBINADO a la it {it_respaldo} "
                      f"({rebobinados} en lo que va) |g|={gnorm:.3g}", flush=True)
            elif seguidos <= 3 or seguidos % 1000 == 0:
                print(f"[{it:6d}] LOTE RECHAZADO |g|={gnorm:.1f} mediana={centro:.1f} "
                      f"amp={amp:.3g} loss={perdida.item():.3f} (seguidos {seguidos})", flush=True)
        else:
            seguidos = 0
            limpias += 1
            # El respaldo pide `guard_limpias` sanas seguidas y cuenta desde el último, no por
            # múltiplos de `it` (docs/porque-del-codigo.md, DISYUNTOR DE LOTES).
            if limpias >= args.guard_limpias and (
                    respaldo is None or it - it_respaldo >= args.guard_respaldo):
                respaldo, it_respaldo = respalda(model, opt, escala), it

        traza[it - 1, 5:] = (gnorm, centro, amp, rechazado, rebobinado)
        max_grad = max(max_grad, gnorm)

        if it % args.log_every == 0:
            # `corr` = corrección hecha en unidades del ruido, sobre el último micro-lote. ~1 es
            # denoising sano; >1 es sobre-corregir, y muy por encima es que la red amplifica.
            with torch.no_grad():
                corr = ((states[-1] - states[0]).norm() / (noisy - clean).norm()).item()
            eta = (time.time() - t0) / it * (args.iterations - it) / 3600
            print(f"[{it:6d}/{args.iterations}] loss={perdida.item():8.3f} "
                  f"lr={sched.get_last_lr()[0]:.2e} "
                  f"T={model.T.item():.5f} lambda={model.lmbda.item():.5f} "
                  f"|g|={gnorm:.1f} max|g|={max_grad:.3g} corr={corr:.3f} rech={rechazos} "
                  f"reb={rebobinados} ({eta:.1f}h restantes)", flush=True)
            max_grad = 0.0

        if it % args.eval_every == 0 or it == args.iterations:
            model.eval()
            # El sigma de referencia es args.sigma, no SIGMA: se evalúa donde se entrena.
            report = evaluate_denoising(model, eval_set, args.sigma, args.sigma, device,
                                        data_range=DATA_RANGE, noise_scale=NOISE_SCALE)
            model.train()
            dB = report["psnr_denoised"]
            history.append({"iteration": it, "psnr": dB,
                            "T": model.T.item(), "lambda": model.lmbda.item()})
            (out / "history.json").write_text(json.dumps(history, indent=2))

            np.save(out / "traza.npy", traza[:it])
            payload = {"config": config, "model": model.state_dict(), "iteration": it,
                       "psnr": dB, "optimizer": opt.state_dict()}
            torch.save(payload, out / "last.pth")
            if dB > best_psnr:
                best_psnr = dB
                torch.save(payload, out / "best.pth")
            print(f"    eval PSNR={dB:.3f} dB ({report['num_images']} imgs)"
                  + ("  *mejor*" if dB == best_psnr else ""), flush=True)

    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    horas = (time.time() - t0) / 3600
    # El estado se escribe, no se deduce: contar filas de traza confunde «terminó» con «la
    # mataron», y un análisis de supervivencia leería una corrida muerta como censurada viva.
    (out / "estado.json").write_text(json.dumps({
        "terminada": True, "iteraciones_pedidas": args.iterations,
        "iteraciones_hechas": args.iterations, "mejor_psnr": best_psnr,
        "psnr_final": history[-1]["psnr"] if history else None,
        "rechazos": rechazos, "max_seguidos": max_seguidos, "rebobinados": rebobinados,
        "horas": round(horas, 3), "vram_pico_gb": round(peak, 2)}, indent=2))
    print(f"\nmejor PSNR: {best_psnr:.3f} dB | {horas:.2f} h | "
          f"VRAM pico {peak:.2f} GB | lotes rechazados {rechazos} de {args.iterations} "
          f"(máx. seguidos {max_seguidos}) | rebobinados {rebobinados}", flush=True)


if __name__ == "__main__":
    main()

# El bucle y el resto de las notas de este archivo: docs/porque-del-codigo.md, «train.py (la
# receta que hereda el 3D)».
