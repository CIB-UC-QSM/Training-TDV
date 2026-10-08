"""Figuras de trayectoria de la presentación: PSNR, T, λ, δ_Adam, ‖g‖ y el disyuntor.

    uv run python docs/presentacion/figuras.py      # ~1 min, CPU, sólo lee runs/

Lee las dos corridas finales (`runs/t8delta-100k-s0`, `-s1`) en el estado en que estén: hoy en
vuelo, mañana terminadas. Las etiquetas dicen «en curso, it N» o «terminada» solas, así que para
meter el resultado final alcanza con volver a correr este script.

Criterio de lectura que vale para todas: lo que estalla se CORTA en su primer evento (‖g‖ > 1e4,
el umbral de `scripts/resumen.py:27`) y se marca con ✕; no se dibuja lo que viene después, que es
lo que hacía ilegibles las figuras de `docs/avance/`. El oficial va siempre gris punteado.
"""
import numpy as np

from estilo import (C, ORAC_PSNR, T_STAR, L_STAR, DELTA_ADAM, UMBRAL_G, RUNS, plt, guardar,
                    lee_corrida, primer_evento, terminada, cortada, estado_txt, delta_adam,
                    envolvente, miles)

# --- CATÁLOGO ---------------------------------------------------------------------------------
# Las corridas de la receta vieja (T₀ = 0.001) murieron por T → 0 y esa fecha sale de la
# bitácora, no del umbral de ‖g‖: el log es cada 250 it, v1 ni guarda ‖g‖ y v6 ya pasa 10⁴ en la
# ventana [1, 250] (max|g| = 14 502) sin haber muerto. Por eso en las figuras van con otro marcador.
VIEJAS = {  # nombre: (etiqueta, muerte por T → 0, baches de los que se recuperó)
    "replica-bf16": ("v1 · 31 ago", 19_750, [5_000]),          # docs/bitacora.md:137, 184-189
    "replica-v5": ("v5 · 1–2 sep", 2_500, []),                # docs/bitacora.md:307-318
    "v6-fp32-ieee": ("v6 · 2 sep (control de 5k)", 1_750, []),  # docs/bitacora.md:142, 355-359
}
V4_EVENTO, V4_BACHES = 69_250, [8_000]                         # docs/bitacora.md:285-293
FINALES = ["t8delta-100k-s0", "t8delta-100k-s1"]
CERCANAS = ["tinit-10delta-10k", "tinit-7p9delta-10k", "t8delta-100k-corte-s0",
            "t8delta-100k-corte-s1"]
MARCA = dict(estalla=dict(marker="X", color=C["evento"], ms=13, mec="white", mew=1.2),
             t0=dict(marker="D", color="#7a1f16", ms=10, mec="white", mew=1.2),
             bache=dict(marker="o", color="none", ms=11, mec=C["evento"], mew=1.8),
             rebobina=dict(marker="v", color="#f4d03f", ms=11, mec="#2b2a27", mew=0.8))

R = {n: lee_corrida(n) for n in
     [*VIEJAS, "replica-v4", "referencia-100k", "disyuntor-100k", *CERCANAS, *FINALES]}
CORTAS = {}
for tanda in ("precision", "tinit01", "lambdainit", "clip", "nodDP"):
    for d in sorted((RUNS / tanda).glob("*")):
        if (d / "traza.npy").exists():
            r = lee_corrida(f"{tanda}/{d.name}")
            # Fuera las de diagnóstico de 300 it (nodDP/solo-*, det-*) y la truncada clip-seed1:
            # no son corridas de 4 000 y contarlas como «sobrevivió» infla la supervivencia.
            if len(r["it"]) >= 4000 or primer_evento(r) is not None:
                CORTAS[f"{tanda}/{d.name}"] = r


def iteracion_actual(r):
    return int(r["it"][-1])


def etiqueta_final(n):
    return f"final {n[-2:]} · {estado_txt(R[n])}"


def muerte(n):
    """Iteración en que la corrida queda perdida, o None si sigue sana.

    Sin disyuntor, el primer evento (‖g‖ > 1e4). Con disyuntor, evento por evento: cuenta como
    muerte el primero después del cual la corrida NO vuelve a lo sano — mediana de ‖g‖ aceptado
    entre 300 y 800 it después por encima de 1e3, o ningún lote aceptado (trabada, como
    `disyuntor-100k`). Un evento rebobinado o contenido con rechazos no es una muerte.
    """
    if n in VIEJAS:
        return VIEJAS[n][1]
    if n == "replica-v4":
        return V4_EVENTO
    r = R.get(n) or CORTAS[n]
    ev = primer_evento(r)
    if ev is None or "rech" not in r:
        return ev
    g, rech = np.asarray(r["g"]), np.asarray(r["rech"])
    for i in np.flatnonzero(np.nan_to_num(g, nan=np.inf) > UMBRAL_G):
        if i + 800 > len(g):
            return None if i < len(g) - 1 else int(i + 1)   # sin ventana todavía: no se decide
        acep = g[i + 300:i + 800][~rech[i + 300:i + 800]]
        if len(acep) == 0 or np.median(acep) > UMBRAL_G / 10:
            return int(i + 1)
    return None


def psnr_hasta(r, tope):
    p = r["psnr"]
    return p if tope is None else p[p[:, 0] <= tope]


def linea_oficial(ax, y, texto, x=0.99, va="bottom"):
    ax.axhline(y, color=C["oficial"], ls=(0, (2, 2)), lw=1.4, zorder=1)
    ax.text(x, y, texto, transform=ax.get_yaxis_transform(), ha="right", va=va,
            color=C["oficial"], fontsize=11)


def leyenda_marcas(fig, claves, y=-0.02, ncol=3):
    from matplotlib.lines import Line2D
    t = dict(estalla="estalla: ‖g‖ > 10⁴ y no vuelve a lo sano",
             t0="muere por T → 0 (receta vieja; fecha de la bitácora)",
             bache="bache del que se recuperó", rebobina="el disyuntor rebobina y sigue sana",
             falta="lo que falta correr", sana="sana (el color es el grupo)")
    h = []
    for k in claves:
        if k in MARCA:
            h.append(Line2D([], [], ls="none", **MARCA[k], label=t[k]))
        elif k == "falta":
            h.append(plt.Rectangle((0, 0), 1, 1, fc="white", ec=C["final"], ls=(0, (2, 2)), label=t[k]))
        else:
            h.append(plt.Rectangle((0, 0), 1, 1, fc=C["texto"], label=t[k]))
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
               fontsize=11)


# --- 1. LÍNEAS DE VIDA: TODA LA HISTORIA EN UNA FIGURA ---------------------------------------
def lineas_de_vida():
    grupos = [
        [("replica-bf16", C["fallida"]), ("replica-v4", C["v4"]), ("replica-v5", C["fallida"]),
         ("v6-fp32-ieee", C["fallida"])],
        [("referencia-100k", C["fallida"]), ("disyuntor-100k", C["fallida"])],
        [(n, C["cercana"]) for n in CERCANAS],
        [(n, C["final"]) for n in FINALES],
    ]
    nombres_grupo = ["receta vieja: T₀ = 0.001", "T₀ = 0.1, sin rebobinado",
                     "T₀ = 7.9–10δ, disyuntor v1", "las finales: T₀ = 8δ, disyuntor v2"]
    etq = {**{n: v[0] for n, v in VIEJAS.items()}, "replica-v4": "v4 · 1 sep",
           "referencia-100k": "referencia-100k · 14 sep",
           "disyuntor-100k": "1er disyuntor (sólo rechazo) · 3 oct",
           "tinit-7p9delta-10k": "T₀ = 7.9δ, 10k · 4 oct", "tinit-10delta-10k": "T₀ = 10δ, 10k · 4 oct",
           "t8delta-100k-corte-s0": "8δ s0 · 5 oct (corte de luz)",
           "t8delta-100k-corte-s1": "8δ s1 · 5 oct (corte de luz)",
           "t8delta-100k-s0": "final s0 · 5–6 oct", "t8delta-100k-s1": "final s1 · 5–7 oct"}
    filas = [f for g in grupos for f in g]
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    y = 0
    ticks, cortes = [], []
    for gi, g in enumerate(grupos):
        ax.text(-1_000, y - 0.62, nombres_grupo[gi], ha="right", va="center", fontsize=10.5,
                color=C["texto"], style="italic")
        for n, col in g:
            r = R[n]
            fin = iteracion_actual(r)   # v6 corrió 2 000 de 5 000; el resto de las viejas, 100 000
            m = muerte(n)
            ax.barh(y, fin if m is None else m, color=col, height=0.56, alpha=0.95)
            if m is not None and fin > m:
                ax.barh(y, fin - m, left=m, color="none", height=0.56, hatch="////",
                        edgecolor=C["evento"], lw=0)
            if m is not None:
                ax.plot(m, y, **MARCA["t0" if n in VIEJAS else "estalla"], zorder=4)
            for b in (VIEJAS[n][2] if n in VIEJAS else V4_BACHES if n == "replica-v4" else []):
                ax.plot(b, y, **MARCA["bache"], zorder=4)
            if n in FINALES and not terminada(r) and not cortada(r):
                ax.barh(y, 100_000 - fin, left=fin, color="none", height=0.56,
                        edgecolor=C["final"], ls=(0, (2, 2)), lw=1.2)
            if cortada(r):
                motivo = r["estado"]["cortada"].split(",")[0].split(":")[0]
                ax.text(fin + 800, y, f"cortada: {motivo}", va="center", fontsize=10,
                        color=C["texto"], style="italic")
            if "reb" in r:
                for it in np.flatnonzero(r["reb"]) + 1:
                    ax.plot(it, y, **MARCA["rebobina"], zorder=5)
            # Mejor PSNR SANO: sólo evaluaciones antes de la muerte. Después del evento el número
            # no es «lo que rindió» (v6 da 20.173, que es la imagen ruidosa).
            p = psnr_hasta(r, m)
            if len(p):
                b = p[np.nanargmax(p[:, 1])][1]
                ax.text(101_500, y, f"{b:.3f} dB", va="center", fontsize=12, color="#2b2a27",
                        fontweight="bold" if b > 31.4 else "normal")
            else:
                ax.text(101_500, y, "—  (sin evaluación sana)", va="center", fontsize=10.5,
                        color=C["texto"])
            ticks.append((y, etq[n]))
            y += 1
        cortes.append(y - 0.5)
        y += 0.7
    for c in cortes[:-1]:
        ax.axhline(c + 0.35, color="#d6d4cc", lw=1)
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks])
    ax.set_ylim(y - 0.6, -1.3)
    ax.set_xlim(0, 100_000); miles(ax); ax.grid(axis="y", visible=False)
    ax.set_xlabel("iteración de entrenamiento (la receta del paper son 100k)")
    ax.text(101_500, -1.0, "mejor PSNR sano", fontsize=11, color=C["texto"])
    ax.set_title("Cada corrida: cuánto vivió sana y lo mejor que rindió antes de perderse")
    en_vuelo = any(not terminada(R[n]) and not cortada(R[n]) for n in FINALES)
    leyenda_marcas(fig, ["sana", "estalla", "t0", "bache", "rebobina"] + (["falta"] if en_vuelo else []),
                   y=0.03)
    guardar(fig, "a1_lineas_de_vida")


# --- 2. PSNR: FALLIDAS CONTRA LAS QUE LLEGARON ------------------------------------------------
def psnr_dos_paneles():
    fig, axs = plt.subplots(1, 2, figsize=(14.5, 5.6), sharey=True,
                            gridspec_kw=dict(width_ratios=[1, 1.3]))
    lo, hi = 29.6, 31.75
    # Izquierda: las que se perdieron, hasta su muerte. Lo que cae debajo del eje va con flecha y
    # valor; una corrida sin evaluación antes de su muerte no se dibuja (no hay PSNR que mostrar).
    ax = axs[0]
    for n, r, col, lw, al in ([(n, R[n], C["fallida"], 1.6, 0.9) for n in
                                (*VIEJAS, "referencia-100k", "disyuntor-100k")]
                              + [(n, r, C["tenue"], 0.9, 0.8) for n, r in CORTAS.items()]):
        m = muerte(n)
        p = psnr_hasta(r, m)
        if not len(p):
            continue
        ax.plot(p[:, 0], np.maximum(p[:, 1], lo), "-o" if lw > 1 else "-", ms=3.5, lw=lw,
                color=col, alpha=al, zorder=2 if lw > 1 else 0)
        bajo = p[p[:, 1] < lo]
        if lw > 1 and len(bajo):
            i = np.argmin(bajo[:, 1])
            ax.annotate(f"{bajo[i, 1]:.1f}", xy=(bajo[i, 0], lo), xytext=(bajo[i, 0], lo + 0.28),
                        ha="center", fontsize=10.5, color=C["evento"],
                        arrowprops=dict(arrowstyle="-|>", color=C["evento"], lw=1.3))
        if m is not None:
            mk = MARCA["t0" if n in VIEJAS else "estalla"]
            ax.plot(m, max(lo, p[-1, 1]), **({**mk, "ms": mk["ms"] * (1 if lw > 1 else 0.45)}))
    ax.annotate("v1: 31.155 en la it 4 000, bache en la ~5 000,\nvolvió a 30.7 y murió en la 19 750",
                xy=(19_750, 30.70), xytext=(6_500, 31.36), fontsize=10.5, color=C["texto"],
                arrowprops=dict(arrowstyle="-", color=C["texto"], lw=0.8))
    nm = sum(muerte(n) is not None for n in CORTAS)
    ax.text(0.98, 0.56, f"gris: {len(CORTAS)} corridas cortas de 4 000 it ({nm} estallan)\n"
                        "ref-100k y v6 se pierden antes\nde su primera evaluación",
            transform=ax.transAxes, ha="right", fontsize=10, color=C["texto"], zorder=6,
            bbox=dict(fc=C["fondo"], ec="none", pad=2))
    ax.set_title("Las que se perdieron (se cortan en ✕ / ◆)")
    # Derecha: las que llegaron. Las violetas van ARRIBA: las de 18k comparten semilla y T₀ con las
    # finales y en `referencia` son la misma trayectoria, así que debajo no se verían.
    ax = axs[1]
    p = R["replica-v4"]["psnr"]; p = p[p[:, 0] <= 68_000]
    ax.plot(p[:, 0], np.maximum(p[:, 1], lo), "-", color=C["v4"], lw=2.2,
            label="v4 · 1 sep (estalla en la 69 250)")
    ax.plot(V4_EVENTO, p[-1, 1], **MARCA["estalla"])
    for n in FINALES:
        p = R[n]["psnr"]
        ax.plot(p[:, 0], np.maximum(p[:, 1], lo), "-", color=C["final"], lw=2.6, marker="o",
                ms=3.5, label=etiqueta_final(n), alpha=1 if n.endswith("s0") else 0.65)
    for n in CERCANAS:
        p = R[n]["psnr"]
        ax.plot(p[:, 0], p[:, 1], "--", color=C["cercana"], lw=1.7, zorder=5)
    ax.plot([], [], "--", color=C["cercana"],
            label="T₀ = 7.9–10δ, 10k y 18k it (las de 18k = las finales)")
    ax.set_title("Las que llegaron")
    ax.legend(loc="lower right", frameon=False, fontsize=10.5)
    for ax in axs:
        linea_oficial(ax, ORAC_PSNR, "checkpoint oficial 31.53 dB")
        ax.set_ylim(lo, hi); ax.set_xlim(0, 100_000); miles(ax)
        ax.set_xlabel("iteración")
    axs[0].set_xlim(0, 22_000)   # mueren todas antes de la 20k: el resto del eje sería vacío
    axs[0].set_ylabel("PSNR en CBSD68, σ = 25 (dB)")
    leyenda_marcas(fig, ["estalla", "t0"], y=0.0, ncol=2)
    guardar(fig, "a2_psnr_fallidas_vs_buenas")


# --- 3. PSNR FINAL, AMPLIADO -------------------------------------------------------------------
def psnr_final_zoom():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    p = R["replica-v4"]["psnr"]; p = p[p[:, 0] <= 68_000]
    ax.plot(p[:, 0], p[:, 1], "-", color=C["v4"], lw=2, label="v4 (sep), mejor 31.478")
    for n in FINALES:
        p = R[n]["psnr"]
        b = p[np.nanargmax(p[:, 1])]
        ax.plot(p[:, 0], p[:, 1], "-o", ms=4, color=C["final"], lw=2.4,
                alpha=1 if n.endswith("s0") else 0.6, label=f"{etiqueta_final(n)}, mejor {b[1]:.3f}")
    linea_oficial(ax, ORAC_PSNR, "checkpoint oficial 31.534 dB", x=0.30)
    for x in (25_000, 50_000, 75_000):
        ax.axvline(x, color="#d6d4cc", lw=1, zorder=0)
        ax.text(x + 400, 31.575, "lr ÷ 2", fontsize=10, color=C["texto"])
    ax.set_ylim(31.0, 31.6); ax.set_xlim(0, 100_000); miles(ax)
    ax.set_xlabel("iteración"); ax.set_ylabel("PSNR en CBSD68 (dB)")
    ax.set_title("Las finales contra v4 y el oficial (ampliado a 31.0–31.6 dB)")
    ax.legend(loc="lower right", frameon=False)
    guardar(fig, "a3_psnr_final_zoom")


# --- 4. ANATOMÍA: SANA CONTRA ESTALLADA --------------------------------------------------------
def anatomia(sana="precision/rapido-seed0", rota="precision/rapido-seed10"):
    rs, rr = CORTAS[sana], CORTAS[rota]
    ev = primer_evento(rr)
    fig, axs = plt.subplots(3, 2, figsize=(13.5, 8.6), sharex=True, sharey="row")
    for c, (r, titulo, col) in enumerate(((rs, "Deseado: una corrida sana", C["final"]),
                                          (rr, f"Indeseado: estalla en la it {ev:,}".replace(",", " "),
                                           C["fallida"]))):
        p = r["psnr"]
        lo = 30.4
        dentro = p[:, 1] >= lo
        axs[0, c].plot(p[dentro, 0], p[dentro, 1], "-o", color=col, ms=6)
        for it_, v in p[~dentro]:
            # Fuera de escala: flecha hacia abajo con el valor, en vez de una línea que cae 30 dB.
            axs[0, c].annotate(f"{v:.1f} dB", xy=(it_, lo), xytext=(it_, lo + 0.55), ha="center",
                               fontsize=12, color=C["evento"], fontweight="bold",
                               arrowprops=dict(arrowstyle="-|>", color=C["evento"], lw=1.6))
        x, med, mx = envolvente(r["it"], r["g"], 25)
        axs[1, c].fill_between(x, med, mx, color=col, alpha=0.25, lw=0)
        axs[1, c].plot(x, med, color=col, lw=1.6)
        axs[2, c].plot(r["it"], r["T"], color=col)
        axs[0, c].set_title(titulo)
        if c == 1:
            for ax in axs[:, 1]:
                ax.axvspan(ev, r["it"][-1], color=C["evento"], alpha=0.07, lw=0)
                ax.axvline(ev, color=C["evento"], lw=1.2)
    axs[0, 0].set_ylabel("PSNR (dB)"); axs[0, 0].set_ylim(30.4, 31.65)
    linea_oficial(axs[0, 0], ORAC_PSNR, "oficial"); linea_oficial(axs[0, 1], ORAC_PSNR, "oficial")
    for ax in axs[1]:
        ax.set_yscale("log"); ax.axhline(UMBRAL_G, color=C["evento"], ls="--", lw=1)
    axs[1, 1].text(80, UMBRAL_G * 1.6, "umbral de evento 10⁴", color=C["evento"], fontsize=10.5)
    axs[1, 0].set_ylabel("‖g‖ (mediana y máx.\ncada 25 it)")
    for ax in axs[2]:
        ax.axhline(DELTA_ADAM, color=C["oficial"], ls=":", lw=1.2)
        ax.set_xlabel("iteración")
    axs[2, 0].text(80, DELTA_ADAM * 1.15, "δ_Adam", fontsize=10.5, color=C["oficial"])
    axs[2, 0].set_ylabel("T"); axs[2, 0].set_ylim(0, 0.13)
    guardar(fig, "b1_anatomia_sana_vs_estallada")


# --- 5. PARÁMETROS: FALLIDAS Y FINALES CON LOS MISMOS EJES -------------------------------------
PISO_T, PISO_L = 2e-4, 1.2e-3   # debajo de esto se dibuja en el piso: es el clamp, T o λ ≈ 0
PISO_D = 0.012                  # T/δ: con T en el piso de 2e-4, T/δ = 0.016


def parametros(grupo):
    """T, λ, T/δ y ‖g‖ en cuatro paneles con los mismos ejes Y en los dos grupos.

    `grupo` = 'fallidas' (eje X hasta 22k: mueren todas antes) o 'finales' (hasta 100k).
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 8.4), sharex=True)
    (aT, aL), (aD, aG) = axs
    vieja = dict(color="#f2a27e", ls=(0, (4, 1.6)))
    if grupo == "fallidas":
        series = [(n, VIEJAS[n][0], vieja, VIEJAS[n][1]) for n in VIEJAS]
        series += [(n, etq, dict(color=C["fallida"], ls="-"), muerte(n))
                   for n, etq in (("referencia-100k", "referencia-100k"),
                                  ("disyuntor-100k", "1er disyuntor"))]
        titulo = "Corridas fallidas: los escalares hasta su muerte (✕ estalla, ◆ T → 0)"
        xmax = 22_000
    else:
        series = [("replica-v4", "v4 · sep", dict(color=C["v4"], ls="-"), 69_250)]
        series += [(n, etiqueta_final(n), dict(color=C["final"], ls="-"), None) for n in FINALES]
        titulo = "v4 y las finales: los mismos cuatro paneles"
        xmax = 100_000
    for n, etq, est, m in series:
        r = R[n]
        it = np.asarray(r["it"])
        k = slice(None) if m is None else (it <= m)
        cada = max(1, len(it[k]) // 4000)
        alpha = 0.6 if n.endswith("s1") else 1
        T = np.maximum(r["T"], PISO_T); L = np.maximum(r["lam"], PISO_L)
        for ax, y in ((aT, T), (aL, L), (aD, np.maximum(r["T"] / delta_adam(it), PISO_D))):
            ax.plot(it[k][::cada], np.asarray(y)[k][::cada], lw=1.9, alpha=alpha, **est)
            if m is not None and (it <= m).any():
                j = np.flatnonzero(it <= m)[-1]
                mk = MARCA["t0" if n in VIEJAS else "estalla"]
                ax.plot(it[j], np.asarray(y)[j], **{**mk, "ms": mk["ms"] * 0.8}, zorder=5)
        x, med, mx = envolvente(it[k], np.asarray(r["g"])[k], 250 if "traza" in r else 1)
        aG.plot(x, med, lw=1.7 if "traza" in r else 1.1, alpha=alpha, **est)
        if "traza" in r:
            aG.fill_between(x, med, mx, color=est["color"], alpha=0.15 * alpha, lw=0)
        if "reb" in r:
            for i in np.flatnonzero(r["reb"]) + 1:
                aG.plot(i, 3e5, "v", color="#f4d03f", ms=10, mec="#2b2a27", mew=0.8)
    if grupo == "fallidas":
        aT.plot([], [], **vieja, label="receta vieja, T₀ = 0.001 (v1, v5, v6)")
        aT.plot([], [], color=C["fallida"], label="receta actual, T₀ = 0.1 (ref-100k, 1er disyuntor)")
        aT.annotate("v1: T llega a 0\nen la it 19 750", xy=(19_750, PISO_T), xytext=(7_500, 4.5e-4),
                    fontsize=11, color=C["texto"], arrowprops=dict(arrowstyle="-", color=C["texto"], lw=0.8))
        aG.text(0.98, 0.04, "v5, v6: ‖g‖ del log, una muestra cada 250 it (v1 no lo registraba)",
                transform=aG.transAxes, ha="right", fontsize=10, color=C["texto"])
    else:
        for n, etq, est, m in series:
            aT.plot([], [], lw=2, alpha=0.6 if n.endswith("s1") else 1, **est, label=etq)
        aG.text(0.98, 0.04, "v4: ‖g‖ del log, una muestra cada 250 it (por eso es más ruidosa)",
                transform=aG.transAxes, ha="right", fontsize=10, color=C["texto"])
    for ax, piso, t in ((aT, PISO_T, "T ≈ 0"), (aL, PISO_L, "λ ≈ 0"), (aD, PISO_D, "T ≈ 0")):
        ax.axhspan(piso * 0.5, piso * 1.15, color=C["evento"], alpha=0.08, lw=0)
        ax.text(0.01, piso * 1.25, t + " (clamp)", transform=ax.get_yaxis_transform(),
                fontsize=10, color=C["evento"])
    linea_oficial(aT, T_STAR, "T del oficial", x=0.99, va="top")
    linea_oficial(aL, L_STAR, "λ del oficial", x=0.99)
    linea_oficial(aD, T_STAR / DELTA_ADAM, "oficial (δ inicial)", x=0.99)
    aD.axhline(1, color=C["evento"], lw=1.2, ls="--")
    aD.text(0.01, 1.2, "T = δ: un solo pico puede llevar T a 0", color=C["evento"],
            fontsize=10.5, transform=aD.get_yaxis_transform())
    aT.set_yscale("log"); aT.set_ylim(PISO_T * 0.5, 0.3); aT.set_title("T: tiempo de parada")
    aL.set_yscale("log"); aL.set_ylim(PISO_L * 0.5, 0.6); aL.set_title("λ: peso del dato")
    aD.set_yscale("log"); aD.set_ylim(PISO_D * 0.5, 60); aD.set_title("T medido en pasos de Adam: T / δ(t)")
    aG.set_yscale("log"); aG.set_ylim(3, 1e7)
    aG.set_title("‖g‖: norma del gradiente (mediana por 250 it)")
    aG.axhline(UMBRAL_G, color=C["evento"], ls="--", lw=1)
    aG.text(0.01, UMBRAL_G * 1.5, "umbral de evento", color=C["evento"], fontsize=10.5,
            transform=aG.get_yaxis_transform())
    for ax in (aD, aG):
        ax.set_xlabel("iteración"); miles(ax)
    aG.set_xlim(0, xmax)
    aT.legend(loc="center right" if grupo == "fallidas" else "lower right", frameon=False,
              fontsize=10.5)
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=17, fontweight="bold")
    guardar(fig, f"b2_parametros_{grupo}")


# --- 6. EL DISYUNTOR EN ACCIÓN -----------------------------------------------------------------
def disyuntor(n="t8delta-100k-s1", ventana=900):
    r = R[n]
    reb = np.flatnonzero(r["reb"]) + 1
    if not len(reb):
        return
    fig, axs = plt.subplots(1, len(reb), figsize=(5.2 * len(reb), 4.8), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, b in zip(axs, reb):
        # El evento de cada rebobinado es el primer ‖g‖ > 1e4 en las 300 it previas: un pico
        # suelto contenido sólo con rechazos (la it 51 575 de la s1) no es el de ningún rebobinado.
        cerca = (r["g"] > UMBRAL_G) & (r["it"] <= b) & (r["it"] > b - 300)
        ev = int(np.flatnonzero(cerca)[0] + 1)
        a, z = max(1, ev - ventana // 3), min(len(r["it"]), ev + 2 * ventana // 3)
        k = slice(a - 1, z)
        it, g, rech = r["it"][k], r["g"][k], r["rech"][k]
        cen = r["traza"][k, 6]
        ax.semilogy(it[~rech], g[~rech], ".", ms=3, color=C["final"], label="lote aceptado")
        ax.semilogy(it[rech], g[rech], "x", ms=6, color=C["evento"], label="lote rechazado")
        ax.semilogy(it, 5 * cen, color=C["texto"], lw=1, label="5 × mediana móvil")
        ax.axhline(UMBRAL_G, color=C["evento"], ls="--", lw=1, label="cota absoluta 10⁴")
        for i in np.flatnonzero(r["reb"][k]):
            ax.axvline(it[i], color="#d4a106", lw=2.2, label="rebobinado")
        ax.set_title(f"evento en la it {ev:,}".replace(",", " "))
        ax.set_xlabel("iteración")
    axs[0].set_ylabel("‖g‖")
    h, l = axs[0].get_legend_handles_labels()
    u = dict(zip(l, h))
    fig.legend(u.values(), u.keys(), loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"El disyuntor en la final semilla 1: {len(reb)} estallidos, {len(reb)} rebobinados, "
                 f"y terminó sana", x=0.01, ha="left", fontsize=16, fontweight="bold")
    guardar(fig, "c1_disyuntor_en_accion")


if __name__ == "__main__":
    lineas_de_vida()
    psnr_dos_paneles()
    psnr_final_zoom()
    anatomia()
    parametros("fallidas")
    parametros("finales")
    disyuntor()
