"""σ_max del operador adjunto por checkpoint, desde docs/mediciones/adjunta-sigma-max.txt (7.1).

    uv run python docs/presentacion/fig_sigma.py      # d3_sigma_max

Misma figura que docs/avance/fig_sigma.py, con el estilo de la presentación. Ojo con el rango que
se cita: las MEDIANAS de los nueve estados sanos van de 1.78 a 2.09; el máximo por parche llega a
61.1 en `tinit01-s3` antes de su evento. «1.5–5.1» sólo vale excluyendo ese parche.
"""
import re

from estilo import RAIZ, C, plt, guardar

filas = []
for l in open(RAIZ / "docs/mediciones/adjunta-sigma-max.txt"):
    m = re.match(r"(.{24})\s*T=\S+\s+sigma_max\[min/med/max\] =\s+(\S+)\s+(\S+)\s+(\S+)", l)
    if m:
        filas.append((m[1].strip().replace("oraculo", "oficial"), *map(float, m.groups()[1:])))


def color(n):
    return (C["oficial"] if "oficial" in n else C["final"] if "VIVA" in n
            else C["cercana"] if "pre" in n else C["fallida"])


fig, ax = plt.subplots(figsize=(12, 6))
for i, (n, lo, med, hi) in enumerate(filas):
    ax.plot([lo, hi], [i, i], color=color(n), lw=2.5, solid_capstyle="round")
    ax.plot(med, i, "o", ms=10, color=color(n), mec="white", mew=1)
ax.set_yticks(range(len(filas)), [f[0] for f in filas], fontsize=11.5)
ax.invert_yaxis(); ax.set_xscale("log"); ax.grid(axis="y", visible=False)
ax.axvline(16.5, color=C["oficial"], ls=":", lw=1.2)
ax.text(19, len(filas) - 2.6, "ninguna de 38 corridas\nvivas pasa de 16.5", fontsize=11,
        color=C["oficial"])
ax.set_xlabel("σ_max del operador adjunto (16 parches: ● mediana, línea mín–máx)")
ax.set_title("El amplificador no está en la arquitectura: lo construye el entrenamiento")
for c, t in ((C["oficial"], "checkpoint oficial"), (C["final"], "viva"),
             (C["cercana"], "antes de su evento"), (C["fallida"], "referencia-100k")):
    ax.plot([], [], "o", ms=9, color=c, label=t)
ax.legend(frameon=False, loc="lower right")
guardar(fig, "d3_sigma_max")
