"""Fig. 4 del abstract (28-10): un solo χ con el borde sacado, la 3D no lineal en Sim2.

    ../../../.venv/bin/python herramientas_abstract/fig_erosion.py <npz:clave> <salida.png> [e]

Muestra χ en los tres planos dentro de la máscara de evaluación erosionada `e` vóxeles (4 por
defecto, 6-conexa, la de `grid_tdv2d.erosiones`), con el borde de la máscara entera en línea
punteada, y rotula NRMSE y XSIM con la máscara erosionada y con la entera. El XSIM sin borde es
el mapa de XSIM de la máscara entera promediado en la erosionada, como `por_profundidad`.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAMA = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(RAMA), str(RAMA / "herramientas_abstract")]
import sujetos  # noqa: E402
from ddr import qsm3d  # noqa: E402
from fig_comparacion import FONDO, PLANOS, RANGO, barra, caja, planos  # noqa: E402
from grid_tdv2d import erosiones, xsim_mapa  # noqa: E402


def main() -> None:
    npz, clave = sys.argv[1].split(":", 1)
    salida = Path(sys.argv[2])
    e = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    d = sujetos.carga_sim2()
    gt, me, M = d["chi"], d["mascara_eval"], d["mascara"]
    chi = np.load(npz)[clave].astype(np.float32) * M
    me_e = erosiones(me)[e]
    entera = (qsm3d.nrmse_pct(chi, gt, me), qsm3d.xsim(chi, gt, me))
    sin_borde = (qsm3d.nrmse_pct(chi, gt, me_e), float(xsim_mapa(chi, gt, me)[me_e].mean()))
    print(f"máscara entera {entera[0]:.2f} / {entera[1]:.4f}; sin {e} vóxeles "
          f"{sin_borde[0]:.2f} / {sin_borde[1]:.4f}", flush=True)
    recorte = caja(me)
    centro = tuple((s.start + s.stop) // 2 for s in recorte)
    pl = planos(np.where(me_e, chi, np.nan), recorte, centro)
    pl_m = planos(me.astype(float), recorte, centro)

    anchos = [p.shape[1] / p.shape[0] for p in pl]
    fig = plt.figure(figsize=(12, 5.2), facecolor=FONDO)
    gs = fig.add_gridspec(3, 3, width_ratios=anchos, height_ratios=[0.2, 1, 0.2], wspace=0.03,
                          hspace=0.03, left=0.01, right=0.99, top=0.98, bottom=0.02)
    t = fig.add_subplot(gs[0, :])
    t.set_axis_off()
    t.text(0.5, 0.62, f"Sim2 · 3D TDV (ours) · outer {e} voxels excluded", color="white",
           fontsize=16, fontweight="bold", ha="center", va="center", transform=t.transAxes)
    t.text(0.5, 0.08, f"NRMSE {sin_borde[0]:.1f}%   XSIM {sin_borde[1]:.3f}      "
           f"(whole brain: {entera[0]:.1f}%, {entera[1]:.3f})", color="white", fontsize=14,
           ha="center", va="center", transform=t.transAxes)
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad(FONDO)
    for p in range(3):
        ax = fig.add_subplot(gs[1, p])
        ax.imshow(pl[p], cmap=cmap, vmin=RANGO[0], vmax=RANGO[1], interpolation="nearest")
        ax.contour(pl_m[p], levels=[0.5], colors="white", linewidths=1.0, linestyles="--")
        ax.set_axis_off()
        ax.text(0.03, 0.03, PLANOS[p], color="white", fontsize=12, transform=ax.transAxes)
    pie = fig.add_subplot(gs[2, 1])
    pie.set_axis_off()
    barra(fig, pie, RANGO, "χ (ppm)", 0.3)
    fig.text(0.985, 0.03, "dashed: whole-brain mask boundary", color="white", fontsize=12,
             ha="right", va="bottom")
    fig.savefig(salida, dpi=200, facecolor=FONDO)
    print(salida)


if __name__ == "__main__":
    main()
