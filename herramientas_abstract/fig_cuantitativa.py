"""La Fig. 4 del abstract (28-10): el NRMSE a lo largo de las iteraciones de cada método en COSMOS y
Sim2, y la tabla de NRMSE y XSIM en t = 100 con la máscara entera y sin las 4 capas del borde.

    ../../../.venv/bin/python herramientas_abstract/fig_cuantitativa.py <espec.json> <salida.png>

`espec.json`: {"metodos": [{"nombre", "color", "ancho"?, "estilo"?, "COSMOS": {...}, "Sim2": {...}}]},
cada sujeto con "tray": {"json": ruta, "clave": [claves hasta la lista de NRMSE]} y "chi": {"npz":
ruta, "clave": clave}. La tabla se calcula del χ (M·χ en t = 100) con `qsm3d`: NRMSE y XSIM en la
máscara de evaluación entera y en la erosionada 4 vóxeles (6-conexa; el XSIM, el mapa de la
máscara entera promediado en la erosionada, como `grid_msk_27_10.py`).
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
from ddr import qsm3d  # noqa: E402
import sujetos  # noqa: E402
from grid_tdv2d import erosiones, xsim_mapa  # noqa: E402

SUJETOS = ("COSMOS", "Sim2")
TITULOS = {"COSMOS": "COSMOS-based simulation", "Sim2": "QSM Challenge 2.0 Sim2"}
E = 4


def trayectoria(t: dict) -> list:
    v = json.loads((RAMA / t["json"]).read_text())
    for c in t["clave"]:
        v = v[c]
    return v


def metricas(chi: np.ndarray, datos: dict, eros: list) -> tuple:
    gt, me = datos["chi"], datos["mascara_eval"]
    mapa = xsim_mapa(chi, gt, me)
    return (qsm3d.nrmse_pct(chi, gt, me), float(mapa[me].mean()),
            qsm3d.nrmse_pct(chi, gt, eros[E]), float(mapa[eros[E]].mean()))


def main() -> None:
    espec = json.loads(Path(sys.argv[1]).read_text())
    salida = Path(sys.argv[2])
    datos = sujetos.carga_sujetos()
    eros = {s: erosiones(datos[s]["mascara_eval"])[:E + 1] for s in SUJETOS}
    fig = plt.figure(figsize=(15, 8.2), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.62], hspace=0.32, wspace=0.16,
                          left=0.06, right=0.985, top=0.93, bottom=0.075)
    filas = []
    for j, s in enumerate(SUJETOS):
        ax = fig.add_subplot(gs[0, j])
        for m in espec["metodos"]:
            if s not in m:
                continue
            n = trayectoria(m[s]["tray"])
            ax.plot(np.arange(1, len(n) + 1), n, color=m["color"], lw=m.get("ancho", 2.2),
                    ls=m.get("estilo", "-"), label=m["nombre"])
        ax.set_xscale("log")
        ax.set_xlim(1, 100)
        ax.set_ylim(espec.get("ylim", {}).get(s, [20, 80]))
        ax.set_xlabel("ADMM iteration", fontsize=13)
        ax.set_ylabel("NRMSE (%)", fontsize=13)
        ax.set_title(TITULOS[s], fontsize=14, fontweight="bold")
        ax.tick_params(labelsize=11.5)
        ax.grid(alpha=0.25)
        if j == 0:
            ax.legend(fontsize=11.5, frameon=False)
    for m in espec["metodos"]:
        fila = [m["nombre"]]
        for s in SUJETOS:
            if s not in m:
                fila += ["—"] * 4
                continue
            chi = np.load(RAMA / m[s]["chi"]["npz"])[m[s]["chi"]["clave"]].astype(np.float32)
            chi = chi * datos[s]["mascara"]
            n0, x0, n4, x4 = metricas(chi, datos[s], eros[s])
            fila += [f"{n0:.1f}", f"{x0:.3f}", f"{n4:.1f}", f"{x4:.3f}"]
        filas.append(fila)
    ax = fig.add_subplot(gs[1, :])
    ax.set_axis_off()
    cab = ["Method"] + [f"{q}{r}" for _ in SUJETOS for r in ("", "\nrim excl.")
                        for q in ("NRMSE (%)", "XSIM")]
    t = ax.table(cellText=filas, colLabels=cab, cellLoc="center", colWidths=[0.2] + [0.1] * 8,
                 bbox=[0, 0, 1, 0.82])
    for j, s in enumerate(SUJETOS):
        ax.text(0.4 + 0.4 * j, 0.88, TITULOS[s], ha="center", va="bottom", fontsize=13,
                fontweight="bold", transform=ax.transAxes)
    t.auto_set_font_size(False)
    t.set_fontsize(12.5)

    for (r, c), celda in t.get_celld().items():
        celda.set_edgecolor("#bbbbbb")
        if r == 0:
            celda.set_text_props(fontweight="bold")
            celda.set_facecolor("#eeeeee")
    # el mejor de cada columna (NRMSE menor, XSIM mayor), en negrita
    for c in range(1, 9):
        signo = 1 if c % 2 else -1
        vals = [signo * float(f[c]) if f[c] != "—" else np.inf for f in filas]
        t[(int(np.argmin(vals)) + 1, c)].set_text_props(fontweight="bold")
    ax.text(0.5, -0.06, "After 100 iterations · rim excl. = brain mask eroded by "
            f"{E} voxels · best per column in bold", ha="center", va="top", fontsize=12,
            transform=ax.transAxes)
    fig.savefig(salida, dpi=espec.get("dpi", 250), facecolor="white")
    print(salida)
    print(json.dumps(filas, ensure_ascii=False))


if __name__ == "__main__":
    main()
