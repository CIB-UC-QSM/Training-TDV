"""Los cuatro checkpoints dentro del ADMM de QSM: el síntoma, lo que NO lo explica, y lo que sí.

    uv run python docs/presentacion/qsm_admm.py     # img_qsm_admm_4modelos

Los `.npy` los deja `scripts/corre_qsm_admm.py`, copia fiel de `../tdv/wTDV_QSM_torch.py` —mismos
μ = 0.008, μ₂ = 1, tolUpdate = 0.39, máscara, autocast bf16 y los tres cortes adyacentes como RGB—
con tres cambios iguales para todos: sin `torch.compile`, sin escribir en `results/`, y
`cudnn.benchmark = False`.

Los números de estabilidad salen de `scripts/diag_qsm_estabilidad.py`: iteración de potencia sobre
el Hessiano de R por diferencias centradas, en float64.
"""
import re
from pathlib import Path

import numpy as np

from estilo import RAIZ, C, plt, guardar

SC = RAIZ / "scripts"
CORRIDAS = [("oficial (Kobler)", "tdv3-3-25-f32-color", C["oficial"]),
            ("T₀ = 1.1δ · rápido", "t1p1delta-rapido-100k", C["v4"]),
            ("T₀ = 2δ · referencia", "t2delta-100k-s1", C["fallida"]),
            ("T₀ = 8δ · final s1", "t8delta-100k-s1", C["final"])]
# (T/S)·λ_max(∇²R), de scripts/diag_qsm_estabilidad.py. Estable mientras < 2.
ESTAB = {"oficial (Kobler)": (0.9942, 1.3778), "T₀ = 1.1δ · rápido": (5.7623, 1.6063),
         "T₀ = 2δ · referencia": (2.7906, 1.8941), "T₀ = 8δ · final s1": (23.2169, 2.0134)}


def curva(tag, std=None):
    p = SC / (f"admm_{tag}.npy" if std is None else f"admm_{tag}_std{std}.npy")
    return np.load(p) if p.exists() else None


def barrido(tag):
    """(target_std, RMSE final, convergió) de todos los .npy que haya, el 0.06 incluido.

    «Convergió» = paró antes de las 100 por `tolUpdate`, que es lo que gobierna la estabilidad.
    """
    pts = []
    for p in sorted(SC.glob(f"admm_{tag}*.npy")):
        m = re.search(r"_std([\d.]+)\.npy$", p.name)
        d = np.load(p)
        pts.append((float(m[1]) if m else 0.06, float(d[-1, 2]), float(len(d) < 100)))
    return np.array(sorted(pts)) if pts else None


fig = plt.figure(figsize=(17, 7.2))
g = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.0], wspace=0.24,
                     left=0.05, right=0.99, top=0.74, bottom=0.13)

# --- 1. EL SÍNTOMA --------------------------------------------------------------------------
a = fig.add_subplot(g[0, 0])
for etq, tag, col in CORRIDAS:
    d = curva(tag)
    if d is None:
        continue
    a.plot(d[:, 0], d[:, 2], color=col, lw=2.2, label=f"{etq}   {d[-1, 2]:.2f}%")
    a.plot(d[-1, 0], d[-1, 2], "o", color=col, ms=7, mec=C["fondo"], mew=1.2)
a.set_xlabel("iteración del ADMM"); a.set_ylabel("RMSE contra COSMOS (%)")
a.set_ylim(26.9, 33.5)          # sin esto el transitorio de las primeras 5 it aplasta el resto
a.set_title("1 · El síntoma\nel oficial converge en 33 it; los nuestros no en 100", fontsize=11.5)
a.legend(frameon=False, loc="upper right"); a.set_xlim(left=0)

# --- 2. LO QUE NO LO EXPLICA ----------------------------------------------------------------
b = fig.add_subplot(g[0, 1])
for etq, tag, col in CORRIDAS:
    p = barrido(tag)
    if p is None or len(p) < 2:
        continue
    b.plot(p[:, 0], p[:, 1], "-", color=col, lw=2.0, label=etq)
    cv = p[:, 2] > 0
    b.plot(p[cv, 0], p[cv, 1], "o", color=col, ms=8, mec=C["fondo"], mew=1.2)      # convergió
    b.plot(p[~cv, 0], p[~cv, 1], "o", mfc=C["fondo"], mec=col, mew=2.0, ms=8)      # 100 it sin converger
b.axvline(0.06, color=C["evento"], ls="--", lw=1.3)
b.text(0.064, b.get_ylim()[1], "el 0.06\ndel solver", color=C["evento"], fontsize=10, va="top")
b.set_xscale("log"); b.set_xlabel("target_std (fija el α de entrada)")
b.set_ylabel("RMSE final (%)")
b.set_xticks([0.03, 0.06, 0.12, 0.25], ["0.03", "0.06", "0.12", "0.25"])
b.minorticks_off()
b.set_title("2 · El óptimo de RMSE es 0.06 para todos\n● convergió   ○ 100 it sin converger", fontsize=11.5)

# --- 3. LO QUE SÍ ---------------------------------------------------------------------------
c = fig.add_subplot(g[0, 2])
etqs = [e for e, *_ in CORRIDAS]
x = np.arange(len(etqs))
qsm = [ESTAB[e][0] for e in etqs]
bsds = [ESTAB[e][1] for e in etqs]
c.bar(x - 0.2, bsds, 0.38, color=C["tenue"], label="a σ = 25 (donde entrena)")
c.bar(x + 0.2, qsm, 0.38, color=[col for *_, col in CORRIDAS], label="a std = 0.06 (donde lo usa QSM)")
for i, v in enumerate(bsds):
    c.text(i - 0.2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
for i, v in enumerate(qsm):
    c.text(i + 0.2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="semibold")
c.axhline(2.0, color=C["evento"], ls="--", lw=1.6)
c.text(-0.45, 2.25, "umbral de estabilidad = 2", color=C["evento"], fontsize=10.5)
c.set_yscale("log"); c.set_ylim(0.6, 60)
c.set_xticks(x, [e.replace(" · ", "\n").replace(" (Kobler)", "\n(Kobler)") for e in etqs], fontsize=9.5)
c.set_ylabel("(T/S)·λ_max(∇²R)")
c.grid(axis="x", visible=False)
c.legend(frameon=False, fontsize=10, loc="upper left")
c.set_title("3 · La causa\nel paso explícito sólo es estable bajo 2", fontsize=11.5)

fig.suptitle("Los cuatro denoisers coinciden en 0.027 dB a σ = 25; como prior PnP de QSM no se "
             "parecen en nada", x=0.012, ha="left", fontsize=16, fontweight="bold")
fig.text(0.012, 0.915,
         "El esquema desenrollado son S = 10 pasos EXPLÍCITOS de descenso sobre T·R, estables sólo "
         "mientras (T/S)·λ_max(∇²R) < 2.\n"
         "El entrenamiento fija ese número justo bajo 2 a la amplitud a la que entrena: los cuatro "
         "caen en 1.38–2.01. Qué pasa 4× más abajo, que es donde el solver corre la red, no lo "
         "controla nada:\nel oficial baja a 0.99 y los tres nuestros se van arriba de 2.",
         fontsize=11, color=C["texto"], ha="left", va="top", linespacing=1.45)
guardar(fig, "img_qsm_admm_4modelos")

print(f"{'corrida':22s} {'its':>5s} {'RMSE final':>12s} {'mejor':>10s} {'estab QSM':>11s} {'estab BSDS':>11s}")
for etq, tag, col in CORRIDAS:
    d = curva(tag)
    if d is None:
        print(f"{etq:22s}  (falta)"); continue
    print(f"{etq:22s} {int(d[-1,0])+1:5d} {d[-1,2]:11.4f}% {d[:,2].min():9.4f}% "
          f"{ESTAB[etq][0]:11.4f} {ESTAB[etq][1]:11.4f}")
print()
for etq, tag, col in CORRIDAS:
    p = barrido(tag)
    if p is not None and len(p) > 1:
        print(f"{etq:22s} barrido target_std: " +
              "  ".join(f"{s:g}->{r:.2f}%{'*' if c else ''}" for s, r, c in p))
print("\n(* convergió antes de las 100 iteraciones)")
