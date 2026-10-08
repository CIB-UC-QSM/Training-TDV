"""Qué aprendió la red, en pesos: filtros de K1 y cuánto se alejó cada capa de su init.

    uv run python docs/presentacion/aprendido.py      # CPU, segundos

Mismo cálculo que `docs/avance/figuras.py` (f6) y `docs/avance/movimiento.py` (f7), con la final
agregada. El init es el de la semilla 0 (`torch.manual_seed(0)` antes de construir `VNet2D`), que
es el de `t8delta-100k-s0`; el oficial viene de otro init, así que su distancia sólo dice «está
tan lejos como dos inits independientes».
"""
import sys

import numpy as np

from estilo import RAIZ, RUNS, OFICIAL, C, plt, guardar, env_cpu, eje_capas

torch = env_cpu()
sys.path.insert(0, str(RAIZ))
from evaluate import load_vnet  # noqa: E402
from ddr.model import VNet2D  # noqa: E402

ck = torch.load(OFICIAL, map_location="cpu", weights_only=False)
torch.manual_seed(0)
w0 = VNet2D(ck["config"]).state_dict()
MODELOS = [("init aleatorio (semilla 0)", None, C["texto"]),
           ("final s0", RUNS / "t8delta-100k-s0/best.pth", C["final"]),
           ("v4 · it 68k", RUNS / "replica-v4/checkpoint.pth", C["v4"]),
           ("checkpoint oficial", OFICIAL, C["oficial"])]
sd = {n: (w0 if p is None else load_vnet(str(p))[0].state_dict()) for n, p, _ in MODELOS}

# --- FILTROS DE K1 ------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(15, 3.9))
for ax, (n, _, col) in zip(axs, MODELOS):
    w = sd[n]["R.K1.weight"].numpy()
    w = w - w.mean(axis=(1, 2, 3), keepdims=True)
    w = w / (np.abs(w).max(axis=(1, 2, 3), keepdims=True) + 1e-12) * 0.5 + 0.5
    grilla = np.ones((4 * 4 + 1, 8 * 4 + 1, 3))
    for k in range(32):
        f, c = divmod(k, 8)
        grilla[1 + f * 4:4 + f * 4, 1 + c * 4:4 + c * 4] = w[k].transpose(1, 2, 0)
    ax.imshow(grilla, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(col); s.set_linewidth(2.5)
    ax.set_title(n, fontsize=13, loc="center")
fig.suptitle("Primera capa K1: 32 filtros 3×3 RGB, cada uno normalizado", x=0.01, ha="left",
             fontsize=16, fontweight="bold")
guardar(fig, "d1_filtros_K1")

# --- MOVIMIENTO POR CAPA ------------------------------------------------------------------------
claves = [k for k in w0 if k.endswith("weight")]
fig, ax = plt.subplots(figsize=(12, 4.6))
for n, _, col in MODELOS[1:]:
    mv = [((sd[n][k] - w0[k]).norm() / w0[k].norm()).item() for k in claves]
    # El oficial salió de OTRO init: su distancia al init de la semilla 0 no es movimiento, es la
    # distancia entre dos inits (mediana 1.59 = √(1 + 1.24²)). Va gris y rotulado así.
    of = "oficial" in n
    ax.plot(range(len(claves)), mv, marker="o", ms=4, color=C["tenue"] if of else col, ls="--" if of else "-",
            label="oficial (otro init: no es movimiento)" if of else n)
    print(f"  {n:22s} K1 {mv[0]:.3f}  mediana {np.median(mv):.3f}")
ax.axhline(np.sqrt(2), color=C["oficial"], ls=":", lw=1.2)
ax.text(len(claves) * 0.55, np.sqrt(2) - 0.05, "√2 ≈ distancia entre dos inits independientes",
        fontsize=10.5, color=C["oficial"], va="top",
        bbox=dict(fc=C["fondo"], ec="none", pad=1.5))
eje_capas(ax, claves)
ax.set_ylabel("‖W − W₀‖ / ‖W₀‖"); ax.set_xlabel("capa, de la entrada a la salida", labelpad=22)
ax.set_title("Cuánto se alejó cada capa de su inicialización (semilla 0)")
ax.legend(frameon=True, facecolor=C["fondo"], edgecolor="none", loc="upper right", fontsize=10.5)
guardar(fig, "d2_movimiento_capas")
