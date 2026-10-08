"""Paleta, rcParams y cargadores de corridas comunes a todas las figuras de la presentación.

Cada figura importa de acá, así que el color de una corrida es el mismo en todas: el oficial es
siempre gris punteado, las finales siempre azul, v4 siempre aqua, las fallidas siempre naranja.

Ojo con la CPU: hay dos entrenamientos en las GPUs. Nada de acá toca CUDA, y los scripts que
cargan modelos fijan `CUDA_VISIBLE_DEVICES=""` antes de importar torch.
"""
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
RUNS = RAIZ / "runs"
SALIDA = Path(__file__).resolve().parent / "salida"
SALIDA.mkdir(exist_ok=True)

# --- CONSTANTES DEL PAPER Y DEL ORÁCULO ------------------------------------------------------
LR, BETA2, LR_HALVE_EVERY = 4e-4, 0.999, 25_000          # train.py:38-40
DELTA_ADAM = LR / math.sqrt(1 - BETA2)                   # = 0.01265, train.py:51
ORAC_PSNR = 31.534                                       # runs/oracle-cbsd68.json, el paper dice 31.53
T_STAR, L_STAR = 0.024727, 0.104522                      # T y λ del checkpoint oficial
UMBRAL_G = 1e4                                           # evento, scripts/resumen.py:27
OFICIAL = RAIZ / "checkpoints" / "Oficiales" / "tdv3-3-25-f32-color.pth"

# --- PALETA ----------------------------------------------------------------------------------
# Los tres primeros son los slots 1-3 de la paleta de referencia (azul, naranja, aqua), que pasan
# la prueba de daltonismo en todos los pares; violeta va siempre con línea discontinua y etiqueta
# directa, que es la codificación secundaria que exige el cuarto color.
C = dict(final="#2a78d6", fallida="#eb6834", v4="#1baf7a", cercana="#4a3aa7",
         oficial="#3a3936", evento="#c0392b", tenue="#b9b7b0", texto="#52514e", fondo="#fcfcfb")

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13, "legend.fontsize": 11.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.7, "axes.edgecolor": "#8a8984",
    "axes.labelcolor": "#2b2a27", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.titlelocation": "left", "axes.titleweight": "semibold",
    "lines.linewidth": 2.0, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "figure.facecolor": C["fondo"], "axes.facecolor": C["fondo"], "savefig.facecolor": C["fondo"],
})


def delta_adam(it):
    """δ_Adam en la iteración `it`: el lr se divide por 2 cada 25 000 (StepLR, train.py:215)."""
    return DELTA_ADAM * 0.5 ** (np.asarray(it) // LR_HALVE_EVERY)


def miles(ax, eje="x"):
    """Ticks en miles («20k») para no leer ceros."""
    f = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:g}k" if v else "0")
    (ax.xaxis if eje == "x" else ax.yaxis).set_major_formatter(f)


def guardar(fig, nombre):
    fig.savefig(SALIDA / f"{nombre}.png")
    plt.close(fig)
    print("  ", nombre)


# --- CARGADORES ------------------------------------------------------------------------------
PATRON_LOG = re.compile(r"\[\s*(\d+)/\d+\] loss=\s*(\S+) lr=(\S+) T=(\S+) lambda=(\S+)"
                        r"(?: \|g\|=(\S+))?")


def lee_log(path):
    """train.log (viejas) o salida.log (nuevas): una fila cada 250 it y las evaluaciones."""
    it, T, lam, g, lr, ev = [], [], [], [], [], []
    ultima = 0
    for linea in open(path, errors="replace"):
        m = PATRON_LOG.search(linea)
        if m:
            ultima = int(m[1])
            it.append(ultima); lr.append(float(m[3])); T.append(float(m[4]))
            lam.append(float(m[5])); g.append(float(m[6]) if m[6] else np.nan)
        m = re.search(r"eval PSNR=\s*(\S+) dB", linea)
        if m:
            ev.append((ultima, float(m[1])))
    return dict(it=np.array(it), T=np.array(T), lam=np.array(lam), g=np.array(g),
                lr=np.array(lr), psnr=np.array(ev).reshape(-1, 2))


def lee_corrida(nombre):
    """Lo que haya de una corrida: traza por iteración si existe, si no el log cada 250."""
    d = RUNS / nombre
    r = dict(nombre=nombre)
    if (d / "history.json").exists():
        h = json.load(open(d / "history.json"))
        r["psnr"] = np.array([[x["iteration"], x["psnr"]] for x in h], dtype=float)
    if (d / "traza.npy").exists():
        tr = np.load(d / "traza.npy")
        r["traza"] = tr
        r["it"] = np.arange(1, len(tr) + 1)
        r["T"], r["dT"], r["lam"], r["dlam"], r["g"] = tr[:, 1], tr[:, 2], tr[:, 3], tr[:, 4], tr[:, 5]
        r["loss"] = tr[:, 0]
        if tr.shape[1] >= 10:
            r["rech"], r["reb"] = tr[:, 8] > 0, tr[:, 9] > 0
    else:
        log = next((d / n for n in ("train.log", "salida.log") if (d / n).exists()), None)
        if log:
            L = lee_log(log)
            r.update(it=L["it"], T=L["T"], lam=L["lam"], g=L["g"])
            if "psnr" not in r:
                r["psnr"] = L["psnr"]
    if (d / "estado.json").exists():
        r["estado"] = json.load(open(d / "estado.json"))
    if (d / "config.json").exists():
        r["config"] = json.load(open(d / "config.json"))
    return r


def primer_evento(r):
    """Primera iteración con ‖g‖ > 1e4, o None. Sobre la traza si hay, si no sobre el log."""
    g = np.nan_to_num(np.asarray(r.get("g", [])), nan=np.inf)
    i = np.flatnonzero(g > UMBRAL_G)
    return int(r["it"][i[0]]) if len(i) else None


def terminada(r):
    return bool(r.get("estado", {}).get("terminada"))


def cortada(r):
    """Se paró a mano (estado.json con «cortada»): no va a llegar a la 100k. La s0 final, en la 66k."""
    return bool(r.get("estado", {}).get("cortada")) and not terminada(r)


def estado_txt(r):
    """«terminada», «cortada a mano en la 66k» o «en curso, it 67k»: leído de estado.json."""
    if terminada(r):
        return "terminada"
    if cortada(r):
        motivo = r["estado"]["cortada"].split(",")[0].split(":")[0]   # «a mano 2026-10-06», «corte de luz»
        return f"cortada en la {r['estado']['iteraciones_hechas'] // 1000}k ({motivo})"
    return f"en curso, it {int(r['it'][-1]) // 1000}k"


def hasta(it, iters):
    """Recorta una serie (it, ...) a las iteraciones ≤ `iters`: para cortar curvas en el evento."""
    return np.asarray(it) <= iters


def envolvente(it, y, ventana=250):
    """Mediana y máximo por bloques de `ventana` iteraciones: legible donde la serie cruda es ruido."""
    n = len(y) // ventana
    if n == 0:
        return np.asarray(it), np.asarray(y), np.asarray(y)
    yy = np.asarray(y)[: n * ventana].reshape(n, ventana)
    x = np.asarray(it)[: n * ventana].reshape(n, ventana)[:, -1]
    return x, np.nanmedian(yy, axis=1), np.nanmax(yy, axis=1)


def env_cpu():
    """Para los scripts que cargan modelos: sin GPU, y pocos hilos para no frenar los dataloaders."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(int(os.environ.get("HILOS", "6")))
    return torch


def eje_capas(ax, claves):
    """Eje X de «una marca por tensor de pesos»: K1, los tres macro-bloques sombreados y KN.

    El rótulo va centrado en el rango de cada macro-bloque (R.mb.k.*), no en una capa suelta:
    ponerlo en conv_down —casi al final del bloque— hacía leer las capas del bloque 1 como del 0.
    """
    ax.set_xticks([0, len(claves) - 1], ["K1", "KN"])
    for k in range(3):
        idx = [i for i, c in enumerate(claves) if c.startswith(f"R.mb.{k}.")]
        if idx:
            ax.axvspan(idx[0] - 0.4, idx[-1] + 0.4, color="#e9e7e0" if k % 2 == 0 else "#f3f1ec",
                       zorder=0, lw=0)
            ax.text((idx[0] + idx[-1]) / 2, -0.07, f"macro-bloque {k + 1}", ha="center", va="top",
                    transform=ax.get_xaxis_transform(), fontsize=11, color=C["texto"])
    ax.set_xlim(-0.8, len(claves) - 0.2)
