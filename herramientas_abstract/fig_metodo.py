"""La Fig. 1 del abstract (28-10): qué ve la red al entrenar (los cubos de 48³ dentro del cerebro de un
T1 sintético, la franja que nunca entra), un ejemplo de denoising y el lazo del ADMM.

    ../../../.venv/bin/python herramientas_abstract/fig_metodo.py <denoising.npz> <qsm.npz:clave> <salida.png>

`denoising.npz` trae `limpia`, `ruidosa` y `salida` (un corte 2D cada una, de `ejemplo_denoising.py`);
el χ es un volumen de COSMOS en un npz. La cobertura sale de `28-10-cobertura-sub-004207.npz`
(`cobertura_cubos.py`).
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

RAMA = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(RAMA))
from ddr import qsm3d  # noqa: E402

COBERTURA = RAMA / "runs/abstract-ismrm/28-10-cobertura-sub-004207.npz"
AZUL, ROJO, VERDE = "#2f6fd0", "#e0402e", "#2aa876"
GIRO = -90                   # el axial de LDM_100k (eje 1) con lo anterior arriba


def panel_cubos(ax) -> None:
    z = np.load(COBERTURA)
    t1, m, cub = z["t1"].astype(np.float32), z["mascara"], z["cubierto"]
    k = int(np.argmax(m.sum(axis=(0, 2))))              # el axial con más cerebro (eje 1)
    corte = qsm3d._gira(t1[:, k, :], GIRO)
    mc, cc = qsm3d._gira(m[:, k, :], GIRO), qsm3d._gira(cub[:, k, :], GIRO)
    ax.imshow(corte, cmap="gray", vmin=0, vmax=0.75, interpolation="nearest")
    capa = np.zeros((*corte.shape, 4))
    capa[mc & ~cc] = [0.88, 0.25, 0.18, 0.72]          # cerebro que ningún cubo cubre
    ax.imshow(capa, interpolation="nearest")
    ce = qsm3d._gira(z["centros"][:, k, :], GIRO)
    filas, cols = np.nonzero(ce)
    elegidos = [int(np.argmin(filas)), int(np.argmax(cols)), int(np.argmin(cols))]
    for i in elegidos:                      # tres cubos válidos centrados en este corte
        ax.add_patch(Rectangle((cols[i] - 24, filas[i] - 24), 48, 48, fill=False,
                               edgecolor="#29c5f6", linewidth=2.4))
    ax.set_axis_off()


def flecha(fig, a, b, **kw) -> None:
    fig.add_artist(FancyArrowPatch(a, b, transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=22, linewidth=2.2, color="#333333", **kw))


def caja_texto(fig, x, y, w, h, texto, color) -> None:
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                  transform=fig.transFigure, facecolor=color, edgecolor="none",
                                  alpha=0.95))
    fig.text(x + w / 2, y + h / 2, texto, ha="center", va="center", color="white",
             fontsize=14, fontweight="bold", linespacing=1.3)


def main() -> None:
    den = np.load(sys.argv[1])
    ruta, clave = sys.argv[2].split(":")
    chi = np.load(RAMA / ruta)[clave].astype(np.float32)
    salida = Path(sys.argv[3])
    fig = plt.figure(figsize=(18, 6.2), facecolor="white")
    # a: los cubos de entrenamiento
    ax = fig.add_axes([0.005, 0.06, 0.235, 0.80])
    panel_cubos(ax)
    fig.text(0.01, 0.93, "a  Training data", fontsize=16,
             fontweight="bold")
    fig.text(0.01, 0.015, "48³ cubes fully inside the brain (cyan) · red: never inside a cube\n"
             "1000 synthetic T1 brains · cubes reach 65% of the brain, 3% of its outermost layer",
             fontsize=12.5, color="#333333", linespacing=1.4)
    # b: el denoiser
    fig.text(0.255, 0.93, "b  3D TDV denoiser", fontsize=16, fontweight="bold")
    for j, (k, t) in enumerate((("ruidosa", "noisy (peak SNR 25)"), ("salida", "3D TDV"),
                                ("limpia", "clean"))):
        a2 = fig.add_axes([0.252 + j * 0.092, 0.27, 0.09, 0.58])
        a2.imshow(qsm3d._gira(den[f"{k}_1"], GIRO), cmap="gray", vmin=0, vmax=0.75,
                  interpolation="nearest")
        a2.set_axis_off()
        a2.set_title(t, fontsize=13)
    fig.text(0.39, 0.17, "noise level unknown to the network (blind)\n3 blocks × 3 scales · "
             "1.16 M parameters · 10 unrolled steps\ntrained once on T1 only: no QSM data",
             ha="center", va="center", fontsize=12.5, color="#333333", linespacing=1.4)
    # c: el ADMM
    fig.text(0.535, 0.93, "c  Plug-and-play QSM (ADMM)", fontsize=16,
             fontweight="bold")
    caja_texto(fig, 0.545, 0.60, 0.17, 0.20, "Data step\nχ from field φ\n(closed form, k-space)",
               AZUL)
    caja_texto(fig, 0.545, 0.18, 0.17, 0.20, "Prior step: 3D TDV\nz = M·TDV(γ·M·(χ + s)) / γ", VERDE)
    flecha(fig, (0.60, 0.595), (0.60, 0.385))
    flecha(fig, (0.66, 0.385), (0.66, 0.595))
    fig.text(0.555, 0.49, "100\niterations", fontsize=13, va="center", ha="center",
             color="#333333", linespacing=1.3)
    fig.text(0.63, 0.08, "M: brain mask · γ scales the network input · μ couples prior and data",
             fontsize=12.5, ha="center", va="center", color="#333333")
    a3 = fig.add_axes([0.80, 0.10, 0.19, 0.76])
    d = chi.shape[2] // 2
    a3.imshow(qsm3d._gira(chi[:, :, d], 90), cmap="gray", vmin=-0.1, vmax=0.1,
              interpolation="nearest")
    a3.set_axis_off()
    a3.set_title("QSM", fontsize=14, fontweight="bold")
    flecha(fig, (0.725, 0.30), (0.80, 0.42))
    fig.savefig(salida, dpi=250, facecolor="white")
    print(salida)


if __name__ == "__main__":
    main()
