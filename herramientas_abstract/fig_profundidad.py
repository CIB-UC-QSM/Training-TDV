"""La Fig. 5 del abstract (28-10): el cambio del error por capa de profundidad (la erosión e menos la
e + 1 de la máscara de evaluación, 6-conexa; 0 es el borde, «≥12» el interior) de la red 3D frente a
cada línea base, y arriba la parte de cada capa que entra en los cubos de entrenamiento (48³ con
margen 1, medida en las máscaras de los 12 T1 de validación: `28-10-cobertura-por-capa.json`, una
aproximación, porque las capas de abajo salen de las máscaras de QSM).

    ../../../.venv/bin/python herramientas_abstract/fig_profundidad.py <espec.json> <salida.png>

`espec.json`: {"red": {"COSMOS": fuente, "Sim2": fuente}, "bases": [{"nombre", "color", "COSMOS":
fuente, "Sim2": fuente}], "extra"?: {"titulo", "color", "etiqueta", "nota", "num": fuente, "den":
fuente}, "ylim"?: [lo, hi]}, fuente = {"json": ruta, "clave": [claves hasta `rmse_capa`]} con los 13
RMSE (ppm) en t = 100. Se grafica 100·(RMSE_3D / RMSE_base − 1).
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAMA = Path(__file__).absolute().parents[1]
COBERTURA = RAMA / "runs/abstract-ismrm/28-10-cobertura-por-capa.json"
TITULOS = {"COSMOS": "COSMOS-based simulation", "Sim2": "QSM Challenge 2.0 Sim2"}
ETIQUETAS = [str(i) for i in range(12)] + ["≥12"]


def lee(f: dict) -> np.ndarray:
    v = json.loads((RAMA / f["json"]).read_text())
    for c in f["clave"]:
        v = v[c]
    return np.asarray(v, dtype=float)


def main() -> None:
    espec = json.loads(Path(sys.argv[1]).read_text())
    salida = Path(sys.argv[2])
    cob = 100 * np.asarray(json.loads(COBERTURA.read_text())["capas_0_11_e_interior"])
    x = np.arange(13)
    extra = espec.get("extra")
    n = 2 + (extra is not None)
    fig = plt.figure(figsize=(6.6 * n, 6.4), facecolor="white")
    gs = fig.add_gridspec(2, n, height_ratios=[0.28, 1.0], hspace=0.08, wspace=0.22,
                          left=0.055, right=0.99, top=0.9, bottom=0.11)
    ylim = espec.get("ylim", [-15, 6])

    def arriba(j: int) -> None:
        ax = fig.add_subplot(gs[0, j])
        ax.bar(x, cob, color="#9ecae1", width=0.8)
        ax.set_xticks(x, [])
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
        ax.tick_params(labelsize=10.5)
        ax.set_ylabel("in training\ncubes (%)", fontsize=11, color="#2b6ca3")
        ax.spines[["top", "right"]].set_visible(False)

    def abajo(j: int, titulo: str, ylabel: str):
        ax = fig.add_subplot(gs[1, j])
        ax.axhline(0, color="#444444", lw=1)
        ax.set_xticks(x, ETIQUETAS)
        ax.set_xlabel("depth layer (voxels from the brain-mask boundary)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        fig.text((j + 0.5) / n, 0.955, titulo, ha="center", fontsize=14, fontweight="bold")
        return ax

    for j, s in enumerate(("COSMOS", "Sim2")):
        arriba(j)
        ax = abajo(j, TITULOS[s], "RMSE change of 3D TDV vs baseline (%)")
        red = lee(espec["red"][s])
        for b in espec["bases"]:
            if s in b:
                ax.plot(x, 100 * (red / lee(b[s]) - 1), "-o", color=b["color"], lw=2.4, ms=5,
                        label=f"vs {b['nombre']}")
        ax.set_ylim(ylim)
        ax.legend(fontsize=11.5, frameon=False, loc="lower left")
    if extra is not None:
        arriba(2)
        ax = abajo(2, extra["titulo"], extra.get("ylabel", "RMSE change (%)"))
        ax.plot(x, 100 * (lee(extra["num"]) / lee(extra["den"]) - 1), "-o", color=extra["color"],
                lw=2.4, ms=5, label=extra["etiqueta"])
        ax.legend(fontsize=11.5, frameon=False, loc="upper right")
        if extra.get("nota"):
            ax.text(0.98, 0.62, extra["nota"], transform=ax.transAxes, ha="right", fontsize=11.5,
                    color="#444444", linespacing=1.4)
    fig.savefig(salida, dpi=espec.get("dpi", 250), facecolor="white")
    print(salida)
    for s in ("COSMOS", "Sim2"):
        red = lee(espec["red"][s])
        for b in espec["bases"]:
            if s in b:
                r = 100 * (red / lee(b[s]) - 1)
                print(s, b["nombre"], "por capa (%):", [round(v, 1) for v in r])
    if extra is not None:
        r = 100 * (lee(extra["num"]) / lee(extra["den"]) - 1)
        print("extra por capa (%):", [round(v, 1) for v in r])


if __name__ == "__main__":
    main()
