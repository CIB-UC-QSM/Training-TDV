"""El salto de PSNR en cada halving del lr (25k, 50k, 75k): qué cambia en el modelo y en la imagen.

    uv run python docs/presentacion/salto_lr.py             # s1 (traza) + s2 si hay instantáneas
    uv run python docs/presentacion/salto_lr.py 64000       # s2 sobre otro centro (prueba)

Pregunta: la PSNR de las finales salta en la it 52 000 (+0.010 / +0.008 dB) después de 8 000 it
casi planas. ¿Qué le pasa al modelo en ese salto y cómo se ven las imágenes?

Lo que se puede medir y con qué:
- 25k y 50k, SÓLO con la traza (los pesos de esas iteraciones no se guardaron; por_implementar.md):
  pérdida, ‖g‖, T y λ por iteración, y la PSNR cada 2 000.
- 75k, CON PESOS: `vigia_instantaneas.py` copia los estados 72k, 74k, 76k y 78k. El halving cae
  entre la 74k y la 76k, así que el par (74k → 76k) contiene el salto y (72k → 74k) / (76k → 78k)
  son los controles de antes y después: cambio de pesos por capa, de la salida, y PSNR por imagen.

Lo que NO sirve como prueba: «en el par que contiene el halving los pesos se mueven menos». Con
el lr a la mitad, Adam desplaza la mitad por paso de forma mecánica, así que eso pasa con o sin
«asentamiento» (la verificación del 10-06 lo marcó: es una prueba que no puede fallar). Lo que sí
distingue: la mejora de la salida POR UNIDAD de desplazamiento de los pesos (ΔPSNR / ‖ΔW‖). Si el
halving sólo acorta los pasos, esa razón es la de los pares vecinos; si la red «se asienta» en un
lugar mejor, es mayor en el par que contiene el halving.

Estado al 2026-10-07: los estados de la 72k–78k NO existen. El vigía se cortó en la 68k junto con la
sesión, y la s0 se cortó a mano en la 66k. La 75k queda medida sólo en la traza de la s1 (s1).
"""
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from estilo import RAIZ, SALIDA, C, plt, guardar, lee_corrida, env_cpu, eje_capas

HALVINGS = [25_000, 50_000, 75_000]
VENTANA = 5_000
INST = Path(os.environ.get("INST", SALIDA / "ckpt" / "inst"))   # INST=... sólo para probar
FIN = {s: lee_corrida(f"t8delta-100k-s{s}") for s in (0, 1)}


def movil(y, k):
    """Mediana móvil centrada de k iteraciones (k impar), con NaN en los bordes."""
    y = np.asarray(y, float)
    out = np.full(len(y), np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(y, k)
    out[k // 2:k // 2 + len(w)] = np.nanmedian(w, axis=1)
    return out


# --- S1: CADA HALVING EN LA TRAZA ----------------------------------------------------------------
def traza():
    # Por semilla: la s0 se cortó a mano en la 66k, así que la 75k sólo existe en la s1.
    hs = [h for h in HALVINGS if any(len(FIN[s]["it"]) > h + 2_000 for s in FIN)]
    fig, axs = plt.subplots(4, len(hs), figsize=(5.4 * len(hs), 11), sharex="col", squeeze=False)
    filas = []
    for c, H in enumerate(hs):
        for s, a in ((0, 1.0), (1, 0.55)):
            r = FIN[s]
            if len(r["it"]) <= H + 2_000:
                continue
            k = slice(H - VENTANA, min(H + VENTANA, len(r["it"])))
            it = r["it"][k]
            g = np.where(r["rech"][k], np.nan, r["g"][k])
            axs[1, c].plot(it, movil(g, 201), color=C["final"], alpha=a, lw=1.6)
            axs[2, c].plot(it, movil(r["loss"][k], 501), color=C["final"], alpha=a, lw=1.6)
            axs[3, c].plot(it, r["lam"][k] / r["T"][k], color=C["final"], alpha=a, lw=1.6,
                           label=f"s{s}")
            p = r["psnr"]; p = p[(p[:, 0] >= H - VENTANA) & (p[:, 0] <= H + VENTANA)]
            axs[0, c].plot(p[:, 0], p[:, 1], "o-", color=C["final"], alpha=a, ms=6)
            antes = lambda y: np.nanmedian(y[H - 1000:H])  # noqa: E731
            despues = lambda y: np.nanmedian(y[H:H + 300])  # noqa: E731
            gg = np.where(r["rech"], np.nan, r["g"])
            hp = dict((int(i), v) for i, v in r["psnr"])
            # La evaluación es cada 2 000: en la 25k y la 75k no hay una justo en el halving.
            pa = hp.get(H, hp.get(H - 1000)); pd = hp.get(H + 2000 if H in hp else H + 1000)
            filas.append(dict(H=H, s=s, psnr_antes=pa, psnr_despues=pd,
                              g_antes=antes(gg), g_despues=despues(gg),
                              loss_antes=antes(r["loss"]), loss_despues=despues(r["loss"]),
                              T_antes=float(r["T"][H - 1]), T_despues=float(r["T"][H + 299])))
        for ax in axs[:, c]:
            ax.axvline(H, color=C["evento"], lw=1.2, ls="--")
        axs[0, c].set_title(f"lr ÷ 2 en la it {H:,}".replace(",", " "))
        axs[3, c].set_xlabel("iteración")
        axs[3, c].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    axs[0, 0].set_ylabel("PSNR (dB)")
    axs[1, 0].set_ylabel("‖g‖ aceptado\n(mediana móvil 201)")
    axs[2, 0].set_ylabel("pérdida\n(mediana móvil 501)")
    axs[3, 0].set_ylabel("w = λ / T")
    axs[3, 0].legend(frameon=False)
    fig.suptitle("Cada halving del lr: ‖g‖ cae de golpe y la PSNR sube un escalón; T no se mueve",
                 x=0.01, ha="left", fontsize=16, fontweight="bold")
    fig.tight_layout()
    guardar(fig, "s1_salto_lr_traza")
    return filas


# --- S2: EL HALVING CON PESOS E IMÁGENES ---------------------------------------------------------
def estados(H):
    """Pares (H−3k→H−1k), (H−1k→H+1k), (H+1k→H+3k) con instantáneas cada 2 000 it."""
    torch = env_cpu()
    sys.path.insert(0, str(RAIZ))
    from evaluate import load_vnet
    from ddr.data import add_noise
    # El lr cambia desde la iteración H+1 (StepLR después de cada paso). Con H impar en miles
    # (25k, 75k) y estados cada 2 000, el par del medio (H−1k → H+1k) lo contiene; con H par el que
    # lo contiene sería el tercero, así que se usan pares desplazados.
    its = [H - 3000, H - 1000, H + 1000, H + 3000] if H % 2000 else [H - 2000, H, H + 2000, H + 4000]
    rutas = {s: {i: INST / f"t8delta-100k-s{s}" / f"it{i:06d}.pth" for i in its} for s in (0, 1)}
    # Se usan las semillas que tengan los cuatro estados (la s0, cortada en la 66k, nunca los tiene).
    rutas = {s: r for s, r in rutas.items() if all(p.exists() for p in r.values())}
    if not rutas:
        print(f"  s2: ninguna semilla tiene las instantáneas {', '.join(str(i) for i in its)}")
        return None
    nombres = [l.strip() for l in open(RAIZ / "datasets/bsds500/cbsd68.txt") if l.strip()]

    def img(n):
        p = next((RAIZ / "datasets/bsds500/images").glob(f"*/{n}.jpg"))
        x = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255
        return torch.from_numpy(x.transpose(2, 0, 1)[None].copy())

    def rec(t, l=96):
        h, w = t.shape[-2:]
        return t[..., h // 2 - l // 2:h // 2 + l // 2, w // 2 - l // 2:w // 2 + l // 2]

    Y = torch.cat([rec(img(n)) for n in nombres[:16]])
    Z = add_noise(Y, 25.0, torch.Generator().manual_seed(3))
    CE = img("253027"); ZC = add_noise(CE, 25.0, torch.Generator().manual_seed(1235))
    zoom = (slice(60, 170), slice(120, 230))
    M = {}
    with torch.no_grad():
        for s in rutas:
            for i, p in rutas[s].items():
                m, _ = load_vnet(str(p))
                x = m(Z, Z)[-1]
                M[(s, i)] = dict(sd={k: v for k, v in m.state_dict().items() if k.endswith("weight")},
                                 x=x, xc=m(ZC, ZC)[-1],
                                 psnr=[(-10 * torch.log10(((x[j] - Y[j]) ** 2).mean())).item()
                                       for j in range(len(Y))])
    sem = sorted(rutas)
    S0 = sem[0]
    claves = list(M[(S0, its[0])]["sd"])
    pares = list(zip(its, its[1:]))
    nombre_par = ["antes", "CONTIENE EL HALVING", "después"]
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.15, 1.15], hspace=0.42, wspace=0.08)
    ax = fig.add_subplot(gs[0, :3])
    for (a, b), t, col in zip(pares, nombre_par, (C["tenue"], C["evento"], C["final"])):
        mv = [((M[(S0, b)]["sd"][k] - M[(S0, a)]["sd"][k]).norm() / M[(S0, a)]["sd"][k].norm()).item()
              for k in claves]
        ax.plot(mv, marker="o", ms=3, color=col, label=f"{a // 1000}k → {b // 1000}k ({t})")
    eje_capas(ax, claves)
    ax.set_ylabel("‖ΔW‖/‖W‖ en 2 000 it"); ax.set_yscale("log")
    ax.set_title(f"Pesos (s{S0}): cuánto se mueve cada capa en cada par"); ax.legend(frameon=False, fontsize=10)
    ax = fig.add_subplot(gs[0, 3])
    for s, a in ((s, 1.0 if s == S0 else 0.55) for s in sem):
        d = np.array(M[(s, pares[1][1])]["psnr"]) - np.array(M[(s, pares[1][0])]["psnr"])
        ax.scatter(M[(s, pares[1][0])]["psnr"], d, color=C["final"], alpha=a, s=30, label=f"s{s}")
    ax.axhline(0, color="#8a8984", lw=0.8)
    ax.set_xlabel("PSNR del recorte antes (dB)"); ax.set_ylabel("Δ PSNR en el salto (dB)")
    ax.set_title("16 recortes: ¿mejoran todos?", fontsize=12); ax.legend(frameon=False, fontsize=9)
    for r, (a, b) in enumerate(pares[:2] if len(pares) > 1 else pares):
        pass
    for f, s in zip((1, 2), sem):
        for c, (a, b) in enumerate(pares):
            ax = fig.add_subplot(gs[f, c])
            d = (M[(s, b)]["xc"] - M[(s, a)]["xc"])[0].numpy().transpose(1, 2, 0)[zoom]
            ax.imshow(np.clip(0.5 + 20 * d, 0, 1)); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"s{s}: {a // 1000}k → {b // 1000}k, ×20", fontsize=11.5, loc="center")
            ax.set_xlabel(f"RMS {255 * np.sqrt((d ** 2).mean()):.2f} niveles de gris", fontsize=10.5)
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_linewidth(3)
                sp.set_color(C["evento"] if c == 1 else C["tenue"])
        ax = fig.add_subplot(gs[f, 3])
        ax.imshow(M[(s, pares[1][1])]["xc"][0].clamp(0, 1).numpy().transpose(1, 2, 0)[zoom])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"s{s}: salida en la {pares[1][1] // 1000}k", fontsize=11.5, loc="center")
    fig.suptitle(f"El halving de la it {H:,} con pesos e imágenes: el par del medio lo contiene"
                 .replace(",", " "), x=0.01, ha="left", fontsize=16, fontweight="bold")
    guardar(fig, f"s2_salto_lr_estados_{H // 1000}k" + ("_prueba" if PRUEBA else ""))
    return [dict(par=f"{a}→{b}", s=s,
                 dW=float(np.median([((M[(s, b)]["sd"][k] - M[(s, a)]["sd"][k]).norm()
                                      / M[(s, a)]["sd"][k].norm()).item() for k in claves])),
                 dx=255 * float(((M[(s, b)]["x"] - M[(s, a)]["x"]) ** 2).mean().sqrt()),
                 dpsnr=float(np.mean(np.array(M[(s, b)]["psnr"]) - np.array(M[(s, a)]["psnr"]))))
            for s in sem for a, b in pares]


# Con un centro distinto o un INST de prueba, la salida va a archivos *_prueba: no pisa la real.
PRUEBA = len(sys.argv) > 1 or "INST" in os.environ
filas = traza()
centro = int(sys.argv[1]) if len(sys.argv) > 1 else 75_000
est = estados(centro)
L = ["# El salto en cada halving del lr", "",
     "Ventanas: «antes» = mediana de las 1 000 it previas; «después» = de las 300 siguientes.", "",
     "| halving | semilla | PSNR antes (it H o H−1k) | PSNR después (H+2k o H+1k) | ‖g‖ antes → después | pérdida antes → después | T antes → después |",
     "|---|---|---|---|---|---|---|"]
for f in filas:
    pa = "—" if f["psnr_antes"] is None else f"{f['psnr_antes']:.4f}"
    pd = "—" if f["psnr_despues"] is None else f"{f['psnr_despues']:.4f}"
    L.append(f"| {f['H']:,} | s{f['s']} | {pa} | {pd} | {f['g_antes']:.2f} → {f['g_despues']:.2f} "
             f"({100 * (f['g_despues'] / f['g_antes'] - 1):+.0f} %) | {f['loss_antes']:.3f} → "
             f"{f['loss_despues']:.3f} | {f['T_antes']:.5f} → {f['T_despues']:.5f} |")
if est:
    L += ["", f"## Con pesos, alrededor de la it {centro:,}", "",
          "| par | semilla | mediana ‖ΔW‖/‖W‖ | RMS Δsalida (niveles de gris) | Δ PSNR medio (16 recortes) | ΔPSNR / ‖ΔW‖ |",
          "|---|---|---|---|---|---|"]
    L += [f"| {e['par']} | s{e['s']} | {e['dW']:.2e} | {e['dx']:.3f} | {e['dpsnr']:+.4f} | "
          f"{e['dpsnr'] / e['dW']:+.2f} |" for e in est]
else:
    L += ["", f"Con pesos (it {centro:,}): esos estados no existen (el vigía se cortó en la 68k y la s0 en "
          "la 66k). Para tenerlos hay que re-correr con estados guardados: por_implementar.md."]
(SALIDA / ("salto_lr_prueba.md" if PRUEBA else "salto_lr.md")).write_text("\n".join(L) + "\n")
print("\n".join(L))
