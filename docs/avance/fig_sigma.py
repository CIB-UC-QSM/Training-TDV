"""σ_max del operador adjunto por checkpoint, desde docs/mediciones/adjunta-sigma-max.txt (7.1)."""
import re, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
rows = []
for l in open("docs/mediciones/adjunta-sigma-max.txt"):
    m = re.match(r"(.{24})\s*T=\S+\s+sigma_max\[min/med/max\] =\s+(\S+)\s+(\S+)\s+(\S+)", l)
    if m: rows.append((m[1].strip(), *map(float, m.groups()[1:])))
col = lambda n: "#52514e" if "oraculo" in n else "#1baf7a" if "VIVA" in n else "#eda100" if "pre" in n else "#e34948"
fig, ax = plt.subplots(figsize=(9, 4.8))
for i, (n, lo, med, hi) in enumerate(rows):
    ax.plot([lo, hi], [i, i], color=col(n), lw=2); ax.plot(med, i, "o", ms=8, color=col(n))
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9); ax.invert_yaxis()
ax.set_xscale("log"); ax.grid(axis="x", color="#e6e5e0")
ax.axvline(16.5, color="#52514e", ls=":", lw=1)
ax.text(18, 0.3, "ninguna de 38 corridas vivas pasa de 16.5", fontsize=8.5, color="#52514e")
ax.set_xlabel("σ_max del operador adjunto (16 parches: ● mediana, línea min–max)")
ax.set_title("El amplificador no está en la arquitectura: lo construye el entrenamiento", loc="left")
for c, t in (("#52514e", "oficial"), ("#1baf7a", "viva"), ("#eda100", "antes de su evento"), ("#e34948", "referencia-100k")):
    ax.plot([], [], "o", color=c, label=t)
ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1, 0.92))
fig.savefig("docs/avance/salida/f8_sigma_max.png")
print(len(rows))
