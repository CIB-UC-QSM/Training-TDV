"""Grilla de eigenfunciones no lineales: filas = modelos, columnas = iteraciones del flujo."""
import numpy as np, matplotlib, os
matplotlib.use("Agg"); import matplotlib.pyplot as plt
O = "docs/avance/salida"
filas = [("oraculo", "checkpoint oficial (31.53 dB)"), ("v4", "v4, it 68 000 (31.48 dB)"),
         ("actual_viva", "actual viva, it 4 000 (31.19 dB)"), ("ref100k", "referencia-100k mejor (21.7 dB)"),
         ("actual_explotada", "actual tras estallar (−2.6 dB)"), ("oraculo_img2", "oficial, otra imagen"),
         ("v4_img2", "v4, otra imagen")]
filas = [f for f in filas if os.path.exists(f"{O}/eig_{f[0]}.npz")]
fig, axs = plt.subplots(len(filas), 4, figsize=(8.6, 2.25 * len(filas)))
for r, (tag, name) in enumerate(filas):
    d = np.load(f"{O}/eig_{tag}.npz"); its = sorted(int(k[2:]) for k in d.files if k.startswith("it"))
    lam = d["log"][-1, 2]
    for c, k in enumerate(its[:4]):
        ax = axs[r, c]; ax.imshow(np.clip(d[f"it{k}"], 0, 1)); ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title("tras 1 paso" if k == 0 else f"iteración {k}", fontsize=10)
    axs[r, 0].set_ylabel(name, fontsize=8.5, rotation=0, ha="right", va="center")
    axs[r, 3].text(1.04, 0.5, f"Λ = {lam:.3g}", transform=axs[r, 3].transAxes, fontsize=9, va="center")
fig.suptitle("Eigenfunciones no lineales de R:  ∇R(x) = Λ x", x=0.02, ha="left")
fig.tight_layout(); fig.savefig(f"{O}/f9_eigen.png", dpi=130, bbox_inches="tight"); print(len(filas))
