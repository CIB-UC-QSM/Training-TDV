"""Las Figs. 2 y 3 del abstract (28-10): una fila por método (la referencia arriba) con χ en los tres
planos, insertos muy ampliados de estructura fina y el error en un plano, todos a sus mejores
parámetros.

    ../../../.venv/bin/python herramientas_abstract/fig_comparacion.py <espec.json> <salida.png>

`espec.json`: {"sujeto": "COSMOS" | "Sim2", "titulo": str, "metodos": [{"nombre", "npz", "clave",
"params"?}] (npz = "referencia" para χ_GT), "rois": [{"letra", "plano" (0 sagital, 1 coronal,
2 axial), "fila", "col", "lado", "flechas"?: [[fila, col, dfila, dcol]]}], "centro"?: [d, h, w],
"error_plano"?: 0|1|2}. Los cortes salen de la caja de la máscara de evaluación, con la orientación
de `qsm3d.cortes_centrales`; fila y columna de un ROI son las del corte ya recortado y girado, y
las de una flecha, las del inserto (la punta; la cola va a dfila, dcol de ella). NRMSE y XSIM
(máscara de evaluación entera) se calculan del mapa mismo, con `qsm3d`; el mapa de cada método es
M·χ, como en el ADMM.
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
from matplotlib.patches import Rectangle  # noqa: E402

RAMA = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(RAMA))
from ddr import qsm3d  # noqa: E402
sys.path.insert(0, str(RAMA / "herramientas_abstract"))
import sujetos  # noqa: E402

FONDO = "#808080"
RANGO = (-0.1, 0.1)
RANGO_ERROR = (-0.05, 0.05)
COLOR_ROI = "#ffd21f"
COLOR_FLECHA = "#00e5ff"
PLANOS = ("Sagittal", "Coronal", "Axial")


def carga(sujeto: str) -> dict:
    return sujetos.carga_cosmos() if sujeto == "COSMOS" else sujetos.carga_sim2()


def caja(m: np.ndarray, margen: int = 2) -> tuple:
    idx = np.argwhere(m)
    lo = np.maximum(idx.min(0) - margen, 0)
    hi = np.minimum(idx.max(0) + margen + 1, m.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def planos(v: np.ndarray, recorte: tuple, centro: tuple) -> list[np.ndarray]:
    vr = v[recorte]
    d, h, w = (c - s.start for c, s in zip(centro, recorte))
    return [qsm3d._gira(p, a) for p, a in zip((vr[d], vr[:, h], vr[:, :, w]),
                                              qsm3d.ORIENTACION_COSMOS)]


def recorta(im: np.ndarray, r: dict) -> np.ndarray:
    h = r["lado"] // 2
    f0, c0 = max(r["fila"] - h, 0), max(r["col"] - h, 0)
    return im[f0:f0 + r["lado"], c0:c0 + r["lado"]]


def flechas(ax, r: dict) -> None:
    for f, c, df, dc in r.get("flechas", []):
        ax.annotate("", xy=(c, f), xytext=(c + dc, f + df),
                    arrowprops=dict(arrowstyle="-|>", color=COLOR_FLECHA, lw=2.2,
                                    mutation_scale=16))


def barra(fig, caja_ax, rango, etiqueta: str, alto: float) -> None:
    """Una barra de color gris horizontal dentro de la celda `caja_ax`, a la altura `alto` (0–1)."""
    pos = caja_ax.get_position()
    cax = fig.add_axes([pos.x0 + 0.12 * pos.width, pos.y0 + alto * pos.height,
                        0.76 * pos.width, 0.09 * pos.height])
    cb = fig.colorbar(ScalarMappable(Normalize(*rango), "gray"), cax=cax, orientation="horizontal",
                      ticks=[rango[0], 0, rango[1]])
    cb.ax.tick_params(labelsize=11.5, colors="white", length=3)
    cb.ax.set_xticklabels([f"{rango[0]:+.2g}", "0", f"{rango[1]:+.2g}"])
    cb.outline.set_edgecolor("white")
    cax.set_title(etiqueta, color="white", fontsize=12.5, pad=4)


def main() -> None:
    espec = json.loads(Path(sys.argv[1]).read_text())
    salida = Path(sys.argv[2])
    datos = carga(espec["sujeto"])
    gt, me, M = datos["chi"], datos["mascara_eval"], datos["mascara"]
    recorte = caja(me)
    centro = tuple(espec.get("centro", [(s.start + s.stop) // 2 for s in recorte]))
    filas = []
    for m in espec["metodos"]:
        ref = m["npz"] == "referencia"
        chi = gt if ref else np.load(RAMA / m["npz"])[m["clave"]].astype(np.float32) * M
        if ref:
            rotulo = (m["nombre"], "", "")
        else:
            rotulo = (m["nombre"], f"NRMSE {qsm3d.nrmse_pct(chi, gt, me):.1f}%  "
                      f"XSIM {qsm3d.xsim(chi, gt, me):.3f}", m.get("params", ""))
            print(m["nombre"], rotulo[1], flush=True)
        filas.append((rotulo, planos(chi, recorte, centro),
                      None if ref else planos(chi - gt, recorte, centro)))
    rois = espec.get("rois", [])
    ep = espec.get("error_plano", 0)
    p0 = filas[0][1]
    anchos = ([1.5] + [p.shape[1] / p.shape[0] for p in p0] + [1.0] * len(rois)
              + [p0[ep].shape[1] / p0[ep].shape[0]])
    alto_fila = 2.3
    fig = plt.figure(figsize=(alto_fila * sum(anchos), alto_fila * (len(filas) + 0.3)),
                     facecolor=FONDO)
    gs = fig.add_gridspec(len(filas) + 1, len(anchos), width_ratios=anchos,
                          height_ratios=[0.3] + [1] * len(filas), wspace=0.025, hspace=0.03,
                          left=0.004, right=0.996, top=0.995, bottom=0.004)
    cabeceras = ([espec.get("titulo", ""), *PLANOS]
                 + [f"{r['letra']}  ({PLANOS[r['plano']].lower()[:3]}.)" for r in rois]
                 + [f"Error ({PLANOS[ep].lower()})"])
    for j, c in enumerate(cabeceras):
        ax = fig.add_subplot(gs[0, j])
        ax.set_axis_off()
        es_roi = 4 <= j < 4 + len(rois)
        ax.text(0.5 if j else 0.04, 0.2, c, color=COLOR_ROI if es_roi else "white",
                fontsize=16 if j else 17, fontweight="bold", ha="center" if j else "left",
                va="bottom", transform=ax.transAxes)
    for i, (rotulo, pl, err) in enumerate(filas):
        ax = fig.add_subplot(gs[i + 1, 0])
        ax.set_axis_off()
        nombre, metricas, params = rotulo
        ax.text(0.96, 0.62 if metricas else 0.5, nombre, color="white", fontsize=17,
                fontweight="bold", ha="right", va="center", transform=ax.transAxes)
        if metricas:
            ax.text(0.96, 0.42, metricas, color="white", fontsize=13.5, ha="right", va="center",
                    transform=ax.transAxes)
            ax.text(0.96, 0.25, params, color="#e8e8e8", fontsize=12, ha="right", va="center",
                    transform=ax.transAxes)
        for p in range(3):
            ax = fig.add_subplot(gs[i + 1, 1 + p])
            ax.imshow(pl[p], cmap="gray", vmin=RANGO[0], vmax=RANGO[1], interpolation="nearest")
            ax.set_axis_off()
            for r in rois:
                if r["plano"] == p and i == 0:
                    h = r["lado"] / 2
                    ax.add_patch(Rectangle((r["col"] - h, r["fila"] - h), r["lado"], r["lado"],
                                           fill=False, edgecolor=COLOR_ROI, linewidth=2))
                    ax.text(r["col"] + h + 1.5, r["fila"] - h, r["letra"], color=COLOR_ROI,
                            fontsize=15, fontweight="bold", va="top")
        for k, r in enumerate(rois):
            ax = fig.add_subplot(gs[i + 1, 4 + k])
            ax.imshow(recorta(pl[r["plano"]], r), cmap="gray", vmin=RANGO[0], vmax=RANGO[1],
                      interpolation="nearest")
            flechas(ax, r)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(COLOR_ROI)
                s.set_linewidth(1.6)
        ax = fig.add_subplot(gs[i + 1, len(anchos) - 1])
        ax.set_axis_off()
        if err is not None:
            ax.imshow(err[ep], cmap="gray", vmin=RANGO_ERROR[0], vmax=RANGO_ERROR[1],
                      interpolation="nearest")
        else:
            fig.canvas.draw()
            barra(fig, ax, RANGO, "χ (ppm)", 0.62)
            barra(fig, ax, RANGO_ERROR, "error (ppm)", 0.2)
    fig.savefig(salida, dpi=espec.get("dpi", 220), facecolor=FONDO)
    print(salida)


if __name__ == "__main__":
    main()
