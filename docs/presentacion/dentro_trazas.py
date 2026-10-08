"""Lo que pasa adentro del entrenamiento, iteración por iteración, leído de `traza.npy`.

    uv run python docs/presentacion/dentro_trazas.py      # g1, g2, g3; segundos, CPU

g1: una sana (las finales) contra la estallada (`referencia-100k`) en los mismos ejes: pérdida,
    ‖g‖, signo de ∂L/∂T y el peso del dato w = λ/T.
g2: los dos estallidos de la lámina de anatomía, ampliados a ±150 iteraciones.
g3: cuán grande es un estallido: la escala de ‖g‖ y de la pérdida, de lo sano a lo peor medido.

El panel del signo es la lectura nueva: en una corrida sana ∂L/∂T es positivo en el 50 % de las
iteraciones —ruido puro, por eso T no se mueve—. En `referencia-100k` sube a ~1 (mediana 0.987
después del evento) en la mayor parte del estallido: un empuje sistemático, el «régimen
sostenido» de la bitácora (por qué el clip no rescata). Pero NO en todo: entre la it 17 000 y la 38 500
vuelve a 0.51–0.75 con ‖g‖ ~4·10⁵, así que el signo trabado no acompaña todo el estallido.
"""
import numpy as np

from estilo import C, UMBRAL_G, RUNS, plt, guardar, lee_corrida, miles, primer_evento

VENTANA = 500
FIN = {n: lee_corrida(n) for n in ("t8delta-100k-s0", "t8delta-100k-s1")}
REF = lee_corrida("referencia-100k")


def bloques(it, y, f):
    n = len(y) // VENTANA
    yy = np.asarray(y)[: n * VENTANA].reshape(n, VENTANA)
    return np.asarray(it)[: n * VENTANA].reshape(n, VENTANA)[:, -1], f(yy)


def mediana(yy):
    return np.nanmedian(np.where(np.isfinite(yy), yy, np.nan), axis=1)


# --- G1: ADENTRO, SANA CONTRA ESTALLADA --------------------------------------------------------
def interior():
    fig, axs = plt.subplots(4, 1, figsize=(13, 12.5), sharex=True)
    series = [(FIN["t8delta-100k-s0"], "final s0", C["final"], 1.0),
              (FIN["t8delta-100k-s1"], "final s1", C["final"], 0.55),
              (REF, "referencia-100k", C["fallida"], 1.0)]
    for r, etq, col, a in series:
        it = r["it"]
        x, y = bloques(it, r["loss"], mediana)
        axs[0].semilogy(x, y, color=col, alpha=a, label=etq)
        x, y = bloques(it, r["g"], mediana)
        axs[1].semilogy(x, y, color=col, alpha=a)
        x, y = bloques(it, r["dT"], lambda yy: np.mean(yy > 0, axis=1))
        axs[2].plot(x, y, color=col, alpha=a)
        x, y = bloques(it, r["lam"] / r["T"], mediana)
        axs[3].plot(x, y, color=col, alpha=a)
    ev = primer_evento(REF)
    for ax in axs:
        ax.axvline(ev, color=C["evento"], lw=1, ls=":")
    axs[0].text(ev + 800, 3e6, f"estalla en la it {ev:,}".replace(",", " "), color=C["evento"],
                fontsize=11)
    axs[0].set_title("Pérdida de entrenamiento (mediana cada 500 it)")
    axs[0].set_ylabel("‖x_S − y‖²")
    axs[0].legend(frameon=False, loc="center right")
    axs[1].set_title("Norma del gradiente ‖g‖ (mediana cada 500 it)")
    axs[1].axhline(UMBRAL_G, color=C["evento"], ls="--", lw=1)
    axs[1].text(60_000, UMBRAL_G * 1.6, "umbral de evento 10⁴", color=C["evento"], fontsize=10.5)
    axs[2].set_title("Fracción de iteraciones con ∂L/∂T > 0 (cada 500 it)")
    axs[2].axhline(0.5, color=C["oficial"], ls=(0, (2, 2)), lw=1.2)
    axs[2].text(60_000, 0.52, "0.5 = ruido puro: T no recibe empuje neto", fontsize=10.5,
                color=C["oficial"])
    axs[2].text(60_000, 0.93, "≈1 = empuje sistemático: T baja sin parar", fontsize=10.5,
                color=C["evento"])
    axs[2].set_ylim(0.3, 1.05)
    axs[3].set_title("Peso del dato w = λ / T (mediana cada 500 it)")
    axs[3].set_xlabel("iteración"); miles(axs[3]); axs[3].set_xlim(0, 100_000)
    fig.suptitle("Adentro de una corrida: sana (azul) contra estallada (naranja)", x=0.01,
                 ha="left", fontsize=17, fontweight="bold")
    fig.tight_layout()
    guardar(fig, "g1_interior_sana_vs_estallada")


# --- G2: EL ESTALLIDO AMPLIADO ------------------------------------------------------------------
def zoom_evento():
    casos = [(REF, "referencia-100k"), (lee_corrida("precision/rapido-seed10"), "rapido-seed10")]
    fig, axs = plt.subplots(4, 2, figsize=(14, 10), sharex="col")
    for c, (r, etq) in enumerate(casos):
        ev = primer_evento(r)
        k = slice(max(0, ev - 151), ev + 150)
        it = r["it"][k]
        for f, (y, t, log) in enumerate(((r["loss"], "pérdida", True), (r["g"], "‖g‖", True),
                                         (r["T"], "T", False), (r["lam"], "λ", False))):
            ax = axs[f, c]
            ax.plot(it, np.asarray(y)[k], ".-", ms=2.5, lw=0.8, color=C["fallida"])
            ax.axvline(ev, color=C["evento"], lw=1.2)
            if log:
                ax.set_yscale("log")
            ax.set_ylabel(t)
        axs[0, c].set_title(f"{etq}: evento en la it {ev:,}".replace(",", " "))
        axs[1, c].axhline(UMBRAL_G, color=C["evento"], ls="--", lw=1)
        axs[3, c].set_xlabel("iteración")
        L0 = np.nanmedian(r["loss"][ev - 150:ev - 1])
        G0 = np.nanmedian(r["g"][ev - 150:ev - 1])
        axs[0, c].text(0.02, 0.9, f"en la it del evento: pérdida ×{r['loss'][ev - 1] / L0:.1f}, "
                                  f"‖g‖ ×{r['g'][ev - 1] / G0:.0f}\n(contra la mediana de las 150 previas)",
                       transform=axs[0, c].transAxes, fontsize=10.5, va="top", color=C["texto"])
    fig.suptitle("El estallido de cerca: ‖g‖ salta ×300–500 mientras la pérdida apenas se duplica o quintuplica",
                 x=0.01, ha="left", fontsize=16, fontweight="bold")
    fig.tight_layout()
    guardar(fig, "g2_estallido_ampliado")


# --- G3: CUÁN GRANDE ES UN ESTALLIDO ------------------------------------------------------------
def magnitud():
    """Todas las trayectorias distintas con traza: máx. ‖g‖ por corrida.

    Sin duplicados: repeat5/run1..4 son copias bit a bit de run0, y lambdainit/B-seed0 es el
    prefijo bit a bit de referencia-100k. Con disyuntor = traza de 9 o 10 columnas
    (scripts/resumen.py:111); se separa la que siguió sana de la que igual se perdió.
    """
    from matplotlib.ticker import LogLocator, NullFormatter
    copias = {f"repeat5/run{i}" for i in range(1, 5)} | {"lambdainit/B-seed0"}
    filas = []
    for d in sorted(RUNS.rglob("traza.npy")):
        nombre = d.parent.relative_to(RUNS).as_posix()
        tr = np.load(d)
        if nombre in copias or tr.ndim != 2 or tr.shape[1] not in (6, 9, 10) or len(tr) < 1000:
            continue
        g = tr[:, 5][np.isfinite(tr[:, 5])]
        disy = tr.shape[1] >= 9
        acep = g if not disy else tr[:, 5][(tr[:, 8] == 0) & np.isfinite(tr[:, 5])]
        # «siguió sana»: las últimas 1 000 it aceptadas tienen mediana por debajo de 10³
        ult = tr[-1000:, 5][(tr[-1000:, 8] == 0)] if disy else g[-1000:]
        sigue = len(ult) > 0 and np.median(ult) < UMBRAL_G / 10
        filas.append(dict(n=nombre, med=np.median(acep), max=g.max(), disy=disy, sigue=sigue))
    grupos = [(3, [f for f in filas if f["max"] <= UMBRAL_G], C["final"], "sin evento"),
              (2, [f for f in filas if f["max"] > UMBRAL_G and f["disy"] and f["sigue"]], C["cercana"],
               "con disyuntor: pico rechazado, siguió sana"),
              (1, [f for f in filas if f["max"] > UMBRAL_G and f["disy"] and not f["sigue"]],
               "#a33b1c", "con disyuntor: igual se perdió"),
              (0, [f for f in filas if f["max"] > UMBRAL_G and not f["disy"]], C["fallida"],
               "estalladas (sin disyuntor)")]
    fig, ax = plt.subplots(figsize=(13, 6))
    rng = np.random.default_rng(0)
    for y, grupo, col, t in grupos:
        v = np.array([f["max"] for f in grupo])
        ax.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=50, color=col, alpha=0.9,
                   edgecolor="white", lw=0.6, label=f"{t} ({len(v)})", zorder=3)
    med = np.median([f["med"] for f in grupos[0][1]])
    ax.axvline(med, color=C["final"], lw=1.2, ls=":")
    ax.text(med * 0.8, 3.55, f"‖g‖ típico de una sana ≈ {med:.0f}", color=C["final"], fontsize=11,
            ha="right")
    ax.axvline(UMBRAL_G, color=C["evento"], ls="--", lw=1)
    ax.text(UMBRAL_G * 1.3, 3.55, "umbral 10⁴", color=C["evento"], fontsize=11)
    peor = max(grupos[3][1], key=lambda f: f["max"])
    ax.annotate(f"{peor['n']}: {peor['max']:.1e}\n≈ 10^{np.log10(peor['max'] / med):.0f} × lo típico",
                xy=(peor["max"], 0), xytext=(peor["max"] / 3e5, -0.72), fontsize=11, color=C["texto"],
                arrowprops=dict(arrowstyle="-", color=C["texto"], lw=0.8))
    ax.set_xscale("log"); ax.set_xlim(10, 1e20)
    ax.xaxis.set_major_locator(LogLocator(base=1000))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(1.0,), numticks=30))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0, 1, 2, 3], ["estalladas", "disyuntor,\nigual perdida", "disyuntor,\nsiguió", "sanas"])
    ax.set_ylim(-1.0, 3.8)
    ax.set_xlabel("máximo de ‖g‖ en la corrida (cada número es ×1000; cada rayita, ×10)")
    ax.set_title(f"Cuán grande es un estallido: máximo de ‖g‖ en {len(filas)} trayectorias distintas")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.88), fontsize=10.5)
    guardar(fig, "g3_magnitud_estallido")
    print(f"  {len(filas)} trayectorias; " + ", ".join(f"{t}: {len(g)}" for _, g, _, t in grupos)
          + f"; típico {med:.1f}; peor {peor['n']} {peor['max']:.3g}")


if __name__ == "__main__":
    interior()
    zoom_evento()
    magnitud()
