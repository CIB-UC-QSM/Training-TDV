"""Figura de mapas de error del abstract (28-10): χ − χ_GT de cada método en los tres planos, COSMOS
a la izquierda y Sim2 a la derecha, todos con la misma ventana y a sus mejores parámetros.

    ../../../.venv/bin/python herramientas_abstract/fig_errores.py <espec.json> <salida.png>

`espec.json`: {"metodos": {"COSMOS": [{"nombre", "npz", "clave"}], "Sim2": [...]} (mismos nombres
y orden), "rango"?: ppm, "dpi"?}. Los cortes, como en `fig_comparacion.py`.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

RAMA = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(RAMA), str(RAMA / "herramientas_abstract")]
from ddr import qsm3d  # noqa: E402
from fig_comparacion import FONDO, caja, carga, planos  # noqa: E402

SUJETOS = ("COSMOS", "Sim2")


def main() -> None:
    espec = json.loads(Path(sys.argv[1]).read_text())
    salida = Path(sys.argv[2])
    r = espec.get("rango", 0.05)
    nombres = [m["nombre"] for m in espec["metodos"]["COSMOS"]]
    errores, anchos = {}, []
    for s in SUJETOS:
        datos = carga(s)
        gt, me, M = datos["chi"], datos["mascara_eval"], datos["mascara"]
        recorte = caja(me)
        centro = tuple((c.start + c.stop) // 2 for c in recorte)
        errores[s] = []
        for m in espec["metodos"][s]:
            chi = np.load(RAMA / m["npz"])[m["clave"]].astype(np.float32) * M
            errores[s].append((planos(chi - gt, recorte, centro),
                               f"{qsm3d.nrmse_pct(chi, gt, me):.1f}%"))
        anchos += [p.shape[1] / p.shape[0] for p in errores[s][0][0]]
    anchos = [0.95] + anchos[:3] + [0.08] + anchos[3:]
    alto = 2.2
    n = len(nombres)
    fig = plt.figure(figsize=(alto * sum(anchos), alto * (n + 0.62)), facecolor=FONDO)
    gs = fig.add_gridspec(n + 2, len(anchos), width_ratios=anchos,
                          height_ratios=[0.3] + [1] * n + [0.32], wspace=0.02, hspace=0.03,
                          left=0.004, right=0.996, top=0.995, bottom=0.01)
    for k, s in enumerate(SUJETOS):
        j0 = 1 + 4 * k
        ax = fig.add_subplot(gs[0, j0:j0 + 3])
        ax.set_axis_off()
        ax.text(0.5, 0.25, f"{s}: error χ − χ_ref", color="white", fontsize=17,
                fontweight="bold", ha="center", va="bottom", transform=ax.transAxes)
        for i in range(n):
            pl, nrmse = errores[s][i]
            for p in range(3):
                ax = fig.add_subplot(gs[i + 1, j0 + p])
                ax.imshow(pl[p], cmap="gray", vmin=-r, vmax=r, interpolation="nearest")
                ax.set_axis_off()
                if p == 0:
                    ax.text(0.02, 0.01, f"NRMSE {nrmse}", color="white", fontsize=12.5,
                            ha="left", va="bottom", transform=ax.transAxes)
    for i, nombre in enumerate(nombres):
        ax = fig.add_subplot(gs[i + 1, 0])
        ax.set_axis_off()
        ax.text(0.95, 0.5, nombre, color="white", fontsize=16, fontweight="bold", ha="right",
                va="center", transform=ax.transAxes)
    cax = fig.add_subplot(gs[n + 1, 3:6])
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0 + 0.45 * pos.height, pos.width, 0.3 * pos.height])
    cb = fig.colorbar(ScalarMappable(Normalize(-r, r), "gray"), cax=cax, orientation="horizontal",
                      ticks=[-r, 0, r])
    cb.ax.tick_params(labelsize=12, colors="white")
    cb.ax.set_xticklabels([f"−{r:g} ppm", "0", f"+{r:g} ppm"])
    cb.outline.set_edgecolor("white")
    fig.savefig(salida, dpi=espec.get("dpi", 200), facecolor=FONDO)
    print(salida)


if __name__ == "__main__":
    main()
