"""Figura de estructura fina del abstract (28-10): una fila por región (COSMOS y Sim2), con el corte
entero y la región marcada a la izquierda y, a la derecha, la región muy ampliada en la referencia
y en cada método, todos a sus mejores parámetros.

    ../../../.venv/bin/python herramientas_abstract/fig_zoom.py <espec.json> <salida.png>

`espec.json`: {"metodos": {"COSMOS": [{"nombre", "npz", "clave"}], "Sim2": [...]} (npz =
"referencia" para χ_GT; mismos nombres y orden en los dos), "filas": [{"sujeto", "plano", "fila",
"col", "lado", "rotulo", "flechas"?}], "dpi"?}. Plano, fila y columna como en `fig_comparacion.py`
(el corte central recortado a la caja de la máscara y girado); las flechas, en coordenadas del
inserto: [fila, col, dfila, dcol] (la punta y hacia dónde va la cola).
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

RAMA = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(RAMA), str(RAMA / "herramientas_abstract")]
from fig_comparacion import (COLOR_ROI, FONDO, RANGO, caja, carga, flechas,  # noqa: E402
                             planos, recorta)


def main() -> None:
    espec = json.loads(Path(sys.argv[1]).read_text())
    salida = Path(sys.argv[2])
    filas = espec["filas"]
    nombres = [m["nombre"] for m in espec["metodos"]["COSMOS"]]
    cortes = {}
    for s in {f["sujeto"] for f in filas}:
        datos = carga(s)
        gt, me, M = datos["chi"], datos["mascara_eval"], datos["mascara"]
        recorte = caja(me)
        centro = tuple((r.start + r.stop) // 2 for r in recorte)
        cortes[s] = [planos(gt if m["npz"] == "referencia" else
                            np.load(RAMA / m["npz"])[m["clave"]].astype(np.float32) * M,
                            recorte, centro) for m in espec["metodos"][s]]
    n = len(nombres)
    anchos = [1.25] + [1.0] * n
    lado = 2.25
    fig = plt.figure(figsize=(lado * sum(anchos), lado * (len(filas) + 0.22)), facecolor=FONDO)
    gs = fig.add_gridspec(len(filas) + 1, n + 1, width_ratios=anchos,
                          height_ratios=[0.22] + [1] * len(filas), wspace=0.03, hspace=0.04,
                          left=0.004, right=0.996, top=0.995, bottom=0.004)
    for j, c in enumerate(["", *nombres]):
        ax = fig.add_subplot(gs[0, j])
        ax.set_axis_off()
        ax.text(0.5, 0.15, c, color="white", fontsize=17, fontweight="bold", ha="center",
                va="bottom", transform=ax.transAxes)
    for i, f in enumerate(filas):
        ref = cortes[f["sujeto"]][0][f["plano"]]
        ax = fig.add_subplot(gs[i + 1, 0])
        ax.imshow(ref, cmap="gray", vmin=RANGO[0], vmax=RANGO[1], interpolation="nearest")
        h = f["lado"] / 2
        ax.add_patch(Rectangle((f["col"] - h, f["fila"] - h), f["lado"], f["lado"], fill=False,
                               edgecolor=COLOR_ROI, linewidth=2.2))
        ax.set_axis_off()
        ax.text(0.02, 0.01, f["rotulo"], color="white", fontsize=13.5, fontweight="bold",
                ha="left", va="bottom", transform=ax.transAxes)
        for j in range(n):
            ax = fig.add_subplot(gs[i + 1, j + 1])
            ax.imshow(recorta(cortes[f["sujeto"]][j][f["plano"]], f), cmap="gray",
                      vmin=RANGO[0], vmax=RANGO[1], interpolation="nearest")
            flechas(ax, f)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(COLOR_ROI)
                sp.set_linewidth(1.6)
    fig.savefig(salida, dpi=espec.get("dpi", 200), facecolor=FONDO)
    print(salida)


if __name__ == "__main__":
    main()
