"""Por qué T₀ = 8δ y no 2δ: la corrida `tinit-2delta-50k` contra las finales, a iteración pareada.

    uv run python docs/presentacion/tinit_2delta.py     # t1, t2 y salida/tinit_2delta.md; CPU, ~2 min

`tinit-2delta-50k` (pre-registro: docs/bitacora.md:3733) es la final s0 con UNA variable cambiada:
T_init = 2δ = 0.0253 en vez de 8δ = 0.1012 (misma semilla 0, mismo código, mismo disyuntor v2). Se
cortó a mano en la it 4 000 (estado.json) para usar la GPU en otra tanda, así que el contraste es
de 0 a 4 000 iteraciones. Del estado de la 2 000 sólo hay traza y history: los pesos se pisaron.

Para el estado @4k, el par con la misma semilla y el mismo init es `disyuntor-100k` @4k (T₀ = 0.1 =
7.9δ, el primer disyuntor, que todavía no había estallado: su evento es en la 4 075). No es la final
s0 @4k —esos pesos no se guardaron— pero comparte init y T₀ ≈ 8δ.

n = 1 por brazo: esto es descriptivo (el propio pre-registro lo dice, sección 6).
"""
import json
import sys

import numpy as np
from PIL import Image

from estilo import RAIZ, RUNS, OFICIAL, SALIDA, C, DELTA_ADAM, UMBRAL_G, plt, guardar, env_cpu

N = 4000
CORRIDAS = [("tinit-2delta-50k", "T₀ = 2δ (0.0253)", C["fallida"], "-"),
            ("t8delta-100k-s0", "T₀ = 8δ, final s0 (misma semilla)", C["final"], "-"),
            ("t8delta-100k-s1", "T₀ = 8δ, final s1", C["final"], ":")]


def carga(n):
    tr = np.load(RUNS / n / "traza.npy")[:N]
    h = {x["iteration"]: x["psnr"] for x in json.load(open(RUNS / n / "history.json"))}
    return dict(tr=tr, it=np.arange(1, len(tr) + 1), g=tr[:, 5], rech=tr[:, 8] > 0,
                reb=np.flatnonzero(tr[:, 9] > 0) + 1, T=tr[:, 1], lam=tr[:, 3], h=h)


R = {n: carga(n) for n, *_ in CORRIDAS}


# --- T1: LA TRAZA PAREADA, 0–4 000 --------------------------------------------------------------
def traza():
    fig, axs = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    (aT, aG), (aL, aR) = axs
    for n, etq, col, ls in CORRIDAS:
        r = R[n]
        aT.plot(r["it"], r["T"] / DELTA_ADAM, color=col, ls=ls, lw=2, label=etq)
        acep = np.where(r["rech"], np.nan, r["g"])
        k = 50
        m = len(acep) // k
        aG.plot(r["it"][: m * k].reshape(m, k)[:, -1],
                np.nanmedian(acep[: m * k].reshape(m, k), axis=1), color=col, ls=ls, lw=1.8)
        aG.plot(r["it"][r["rech"]], r["g"][r["rech"]], "x", color=col, ms=5, alpha=0.8)
        aL.plot(r["it"], r["lam"], color=col, ls=ls, lw=2)
        aR.plot(r["it"], np.cumsum(r["rech"]), color=col, ls=ls, lw=2.2)
        for b in r["reb"]:
            aR.plot(b, np.cumsum(r["rech"])[b - 1], "v", color="#f4d03f", ms=12, mec="#2b2a27")
            aG.axvline(b, color="#d4a106", lw=1.4, alpha=0.8)
    aT.axhline(1, color=C["evento"], ls="--", lw=1.2)
    aT.text(60, 1.1, "pared: T = δ", color=C["evento"], fontsize=10.5)
    aT.set_yscale("log"); aT.set_ylim(0.7, 12); aT.set_title("T medido en pasos de Adam (T/δ)")
    aT.legend(frameon=False, loc="center right")
    aG.set_yscale("log"); aG.axhline(UMBRAL_G, color=C["evento"], ls="--", lw=1)
    aG.set_title("‖g‖: mediana aceptada cada 50 it; ✕ lotes rechazados; | rebobinado")
    aL.set_title("λ (peso del dato)")
    aR.set_title("Lotes rechazados acumulados (▼ = rebobinado)")
    for ax in (aL, aR):
        ax.set_xlabel("iteración")
    fig.suptitle("T₀ = 2δ contra 8δ en las mismas 4 000 it: misma PSNR (31.155 contra 31.191 a la "
                 "4k), mucho menos estable", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    guardar(fig, "t1_2delta_vs_8delta")


# --- T2: EL ESTADO A LA IT 4 000 ----------------------------------------------------------------
def estado():
    torch = env_cpu()
    sys.path.insert(0, str(RAIZ))
    from evaluate import load_vnet
    from ddr.data import add_noise, psnr

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
    nn = (Z - Y).norm().item()
    CE = img("253027"); ZC = add_noise(CE, 25.0, torch.Generator().manual_seed(1235))
    zoom = (slice(60, 170), slice(120, 230))
    modelos = [("2δ @4k", RUNS / "tinit-2delta-50k/best.pth", C["fallida"], "t2d_4k"),
               ("≈8δ @4k\n(misma semilla)", RUNS / "disyuntor-100k/best.pth", C["final"], "dis_4k"),
               ("final s1\n@100k", RUNS / "t8delta-100k-s1/best.pth", C["final"], "final_s1"),
               ("oficial", OFICIAL, C["oficial"], "oficial")]
    M = []
    with torch.no_grad():
        for etq, p, col, eig in modelos:
            m, _ = load_vnet(str(p))
            est = m(Z, Z)
            # Paso efectivo con la definición de scripts/measure_adjoint_chain.py (y dentro.py).
            gR = float(np.mean([m.R.grad(est[s]).norm().item() for s in range(est.shape[0] - 1)]))
            M.append(dict(etq=etq, col=col, T=float(m.T), lam=float(m.lmbda), gR=gR,
                          paso=float(m.T) * gR / nn, psnr=psnr(est[-1], Y),
                          xc=m(ZC, ZC)[-1][0].clamp(0, 1).numpy().transpose(1, 2, 0)[zoom],
                          psnr_c=psnr(m(ZC, ZC)[-1], CE), eig=eig))
            print(f"  {etq:26s} T {M[-1]['T']:.4f}  ‖∇R‖ {gR:8.1f}  paso {M[-1]['paso']:.3f}  "
                  f"PSNR(16 recortes) {M[-1]['psnr']:.2f}", flush=True)
    fig = plt.figure(figsize=(16, 9.5))
    g = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.35, wspace=0.12)
    for c, (k, t, fmt) in enumerate((("T", "T", "{:.4f}"), ("gR", "‖∇R‖ (cuánto empuja R)", "{:.0f}"),
                                     ("paso", "paso total T·‖∇R‖/‖ruido‖", "{:.2f}"))):
        ax = fig.add_subplot(g[0, c])
        v = [d[k] for d in M]
        ax.bar(range(len(M)), v, color=[d["col"] for d in M], width=0.6)
        for i, x in enumerate(v):
            ax.text(i, x, fmt.format(x), ha="center", va="bottom", fontsize=11)
        ax.set_xticks(range(len(M)), [d["etq"] for d in M], fontsize=10)
        ax.set_title(t, fontsize=13); ax.grid(axis="x", visible=False)
    ax = fig.add_subplot(g[0, 3]); ax.axis("off")
    ax.text(0, 0.95, "Misma PSNR, otro camino:", fontsize=12.5, fontweight="bold", va="top")
    ax.text(0, 0.78, "con T 3.6× más chica, R tiene que empujar\n~3.6× más para dar el mismo paso\n"
                     "total. Más ‖∇R‖ = gradientes más grandes\n= más rechazos y rebobinados.",
            fontsize=11.5, va="top", color=C["texto"])
    for c, d in enumerate(M):
        ax = fig.add_subplot(g[1, c])
        p = SALIDA / "eig" / f"eig_{d['eig']}.npz"
        if p.exists():
            e = np.load(p)
            ax.imshow(np.clip(e["it1000"], 0, 1))
            conv = "" if e["log"][-1, 1] < 0.1 else " · no converge"
            ax.set_xlabel(f"eigenfunción · Λ = {e['log'][-1, 2]:.3g}{conv}", fontsize=10.5)
        else:
            ax.imshow(d["xc"]); ax.set_xlabel(f"salida · {d['psnr_c']:.2f} dB", fontsize=10.5)
        ax.set_title(d["etq"].replace("\n", " "), fontsize=12, loc="center")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(d["col"]); s.set_linewidth(3)
    fig.suptitle("El estado a la it 4 000: T₀ = 2δ llega a la misma PSNR con ‖∇R‖ mucho mayor",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    guardar(fig, "t2_2delta_estado_4k")
    return M


traza()
M = estado()
d2, s0 = R["tinit-2delta-50k"], R["t8delta-100k-s0"]
e = json.load(open(RUNS / "tinit-2delta-50k/estado.json"))
L = ["# T₀ = 2δ contra 8δ (`tinit-2delta-50k`, cortada a mano en la it 4 000)", "",
     "| 0–4 000 it | 2δ (s0) | 8δ final s0 | 8δ final s1 |", "|---|---|---|---|"]
for t, f in (("PSNR @2 000", lambda r: f"{r['h'][2000]:.3f}"), ("PSNR @4 000", lambda r: f"{r['h'][4000]:.3f}"),
             ("lotes rechazados", lambda r: str(int(r["rech"].sum()))),
             ("rebobinados", lambda r: str(len(r["reb"])) + (f" ({', '.join(str(b) for b in r['reb'])})" if len(r["reb"]) else "")),
             ("eventos ‖g‖ > 10⁴", lambda r: str(int((r["g"] > UMBRAL_G).sum()))),
             ("máx ‖g‖", lambda r: f"{r['g'].max():.2g}"),
             ("mediana ‖g‖ aceptado", lambda r: f"{np.median(r['g'][~r['rech']]):.0f}"),
             ("T/δ mínima", lambda r: f"{r['T'].min() / DELTA_ADAM:.2f}"),
             ("T @4 000 (en δ)", lambda r: f"{r['T'][-1]:.4f} ({r['T'][-1] / DELTA_ADAM:.2f} δ)")):
    L.append(f"| {t} | " + " | ".join(f(R[n]) for n, *_ in CORRIDAS) + " |")
L += ["", "Estado @4 000 (16 recortes de CBSD68; paso efectivo como en dentro.py):", "",
      "| modelo | T | ‖∇R‖ | paso T·‖∇R‖/‖ruido‖ |", "|---|---|---|---|"]
L += [f"| {d['etq']} | {d['T']:.4f} | {d['gR']:.0f} | {d['paso']:.3f} |" for d in M]
L += ["", "## Predicciones registradas (docs/bitacora.md:3733) contra lo medido", "",
      "| # | predicción | medido | veredicto |", "|---|---|---|---|",
      f"| 1 | T no toca cero en las 50 000 | mín {d2['T'].min():.4f} en 4 000 it | sin resultado (cortada en la 4 000; no tocó en ese tramo) |",
      f"| 2 | PSNR @4 000 en [31.05, 31.25] | {d2['h'][4000]:.3f} | ✔ |",
      f"| 3 | T @4 000 en [0.023, 0.028] | {d2['T'][-1]:.4f} | ✔ |",
      "| 4 | PSNR @16 000 ≥ 31.09 | — | sin resultado |",
      "| 5 | PSNR @50 000 en [31.48, 31.58] | — | sin resultado |",
      "| 6 | T @50 000 > 0.0300 | — | sin resultado |",
      f"| 7 | rechazos > 18 en las 50 000 | {int(d2['rech'].sum())} en 4 000 | ✔ (ya en la 4 000) |"]
(SALIDA / "tinit_2delta.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
