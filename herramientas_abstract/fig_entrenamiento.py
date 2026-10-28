"""La Fig. 1 del abstract (28-10): el entrenamiento de la red 3D. a) qué ve la red: tres cubos de 48³
dentro del cerebro de un T1 sintético y, en rojo, el cerebro que ningún cubo cubre; b) un ejemplo
de denoising ciego (ruidosa, salida, limpia) a un SNR por el pico.

    ../../../.venv/bin/python herramientas_abstract/fig_entrenamiento.py <denoising.npz> <snr> <salida.png>

`denoising.npz` sale de `ejemplo_denoising.py` (cortes `limpia_k`, `ruidosa_k`, `salida_k`); la
cobertura, de `28-10-cobertura-sub-004207.npz` (`cobertura_cubos.py`), vía `fig_metodo.panel_cubos`.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RAMA = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(RAMA), str(RAMA / "herramientas_abstract")]
from ddr import qsm3d  # noqa: E402
from fig_comparacion import FONDO  # noqa: E402
from fig_metodo import GIRO, panel_cubos  # noqa: E402


def main() -> None:
    den = np.load(sys.argv[1])
    snr = sys.argv[2]
    salida = Path(sys.argv[3])
    fig = plt.figure(figsize=(15, 5.6), facecolor=FONDO)
    ax = fig.add_axes([0.005, 0.03, 0.30, 0.82])
    panel_cubos(ax)
    ax.legend(handles=[Patch(facecolor="none", edgecolor="#29c5f6", linewidth=2.4,
                             label="48³ training cubes"),
                       Patch(facecolor=(0.88, 0.25, 0.18, 0.72), label="never inside a cube")],
              loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=2, fontsize=12,
              frameon=True, facecolor="black", edgecolor="none", labelcolor="white",
              handlelength=1.4, columnspacing=1.2)
    fig.text(0.01, 0.93, "a   Training data: brain-interior cubes", color="white",
             fontsize=16, fontweight="bold")
    fig.text(0.335, 0.93, f"b   Blind 3D TDV denoising (peak SNR {snr})", color="white",
             fontsize=16, fontweight="bold")
    for j, (k, t) in enumerate((("ruidosa", "Noisy"), ("salida", "3D TDV"), ("limpia", "Clean"))):
        a2 = fig.add_axes([0.335 + j * 0.222, 0.03, 0.215, 0.80])
        a2.imshow(qsm3d._gira(den[f"{k}_1"], GIRO), cmap="gray", vmin=0, vmax=0.75,
                  interpolation="nearest")
        a2.set_axis_off()
        a2.set_title(t, color="white", fontsize=15, fontweight="bold", pad=6)
    fig.savefig(salida, dpi=200, facecolor=FONDO)
    print(salida)


if __name__ == "__main__":
    main()
