"""La figura de preview del abstract (28-10), versión de concepto: a la izquierda, un T1 sintético con
tres cubos de entrenamiento (dentro del cerebro; en rojo, lo que ningún cubo cubre); a la derecha,
el χ que reconstruye el ADMM con la red 3D. Pocos elementos grandes y nada de texto chico.

    ../../../.venv/bin/python herramientas_abstract/fig_preview2.py <qsm.npz:clave> <salida.png>
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAMA = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(RAMA), str(RAMA / "herramientas_abstract")]
from ddr import qsm3d  # noqa: E402
from fig_metodo import flecha, panel_cubos  # noqa: E402


def main() -> None:
    ruta, clave = sys.argv[1].split(":")
    chi = np.load(RAMA / ruta)[clave].astype(np.float32)
    salida = Path(sys.argv[2])
    fig = plt.figure(figsize=(12, 5.6), facecolor="white")
    ax = fig.add_axes([0.01, 0.02, 0.40, 0.80])
    panel_cubos(ax)
    fig.text(0.21, 0.9, "Trained on brain-interior\nT1 cubes only", ha="center", va="center",
             fontsize=24, fontweight="bold", linespacing=1.15)
    ax = fig.add_axes([0.585, 0.02, 0.40, 0.80])
    d = chi.shape[2] // 2
    ax.imshow(qsm3d._gira(chi[:, :, d], 90), cmap="gray", vmin=-0.1, vmax=0.1,
              interpolation="nearest")
    ax.set_axis_off()
    fig.text(0.785, 0.9, "3D TDV prior\nfor QSM", ha="center", va="center", fontsize=24,
             fontweight="bold", linespacing=1.15)
    flecha(fig, (0.43, 0.42), (0.565, 0.42))
    fig.savefig(salida, dpi=200, facecolor="white")
    print(salida)


if __name__ == "__main__":
    main()
