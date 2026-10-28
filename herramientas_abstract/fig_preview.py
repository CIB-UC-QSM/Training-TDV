"""La figura de preview del abstract (28-10): pocos paneles grandes de un mismo recorte, con un rótulo
grande por panel y nada de texto chico (pide ISMRM: legible en un teléfono, sin caption).

    ../../../.venv/bin/python herramientas_abstract/fig_preview.py <espec.json> <salida.png>

`espec.json`: {"sujeto", "metodos": [{"nombre", "npz", "clave"}], "roi": {"plano", "fila", "col",
"lado"}, "titulo"?}; plano y coordenadas como en `fig_comparacion.py` (corte central recortado y
girado).
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAMA = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(RAMA), str(RAMA / "herramientas_abstract")]
from fig_comparacion import caja, carga, planos, recorta  # noqa: E402

FONDO = "#808080"


def main() -> None:
    espec = json.loads(Path(sys.argv[1]).read_text())
    salida = Path(sys.argv[2])
    datos = carga(espec["sujeto"])
    gt, me, M = datos["chi"], datos["mascara_eval"], datos["mascara"]
    recorte = caja(me)
    centro = tuple((s.start + s.stop) // 2 for s in recorte)
    r = espec["roi"]
    n = len(espec["metodos"])
    fig, ejes = plt.subplots(1, n, figsize=(5.2 * n, 6.0), facecolor=FONDO)
    for ax, m in zip(ejes, espec["metodos"]):
        chi = gt if m["npz"] == "referencia" else (
            np.load(RAMA / m["npz"])[m["clave"]].astype(np.float32) * M)
        ax.imshow(recorta(planos(chi, recorte, centro)[r["plano"]], r), cmap="gray",
                  vmin=-0.1, vmax=0.1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("white")
            s.set_linewidth(3)
        ax.set_title(m["nombre"], color="white", fontsize=34, fontweight="bold", pad=14)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.03, wspace=0.06)
    fig.savefig(salida, dpi=espec.get("dpi", 200), facecolor=FONDO)
    print(salida)


if __name__ == "__main__":
    main()
