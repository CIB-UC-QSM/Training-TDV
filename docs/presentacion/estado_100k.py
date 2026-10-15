"""Qué aprendieron las cuatro redes al término de sus 100 000 iteraciones, y con qué fuerza.

    uv run python docs/presentacion/estado_100k.py     # img_estado_100k_4modelos; CPU, ~4 min

Misma lámina que `tinit_2delta.py` (t2_2delta_estado_4k) pero al final del entrenamiento y con los
cuatro modelos que importan: T₀ = 1.1δ, 2δ, 8δ (la final s1) y el checkpoint oficial.

Los checkpoints son los `last.pth`, que es lo que está copiado en `../tdv/checkpoints/`: para 2δ eso
NO es el `best.pth` (it 94 000, 31.5360) sino el de la it 100 000, 31.5357 — 0.0004 dB, pero que la
lámina describa el archivo que se probó en QSM y no otro.

Ojo con lo que se puede leer de acá: `T` y `Λ` son **coordenadas de gauge**, no observables. El
modelo es idéntico bajo (T, K_N) → (cT, K_N/c) —verificado a 1e-15 en float64,
docs/mediciones/degeneracion-T-KN.txt— así que sólo los productos `T·‖∇R‖` y `T·Λ` y el escalar `λ`
dicen algo. Las eigenfunciones van como CUALITATIVAS: la correlación directa entre ellas se
descartó el 2026-10-10.
"""
import sys

import numpy as np
from PIL import Image

from estilo import RAIZ, RUNS, OFICIAL, SALIDA, C, DELTA_ADAM, plt, guardar, env_cpu

EIG = SALIDA / "eig"
# (etiqueta, checkpoint, color, tag de la eigenfunción, T_init en δ)
MODELOS = [
    ("oficial\n(Kobler)", OFICIAL, C["oficial"], "oficial", None),
    ("T₀ = 1.1δ\nrápido", RUNS / "t1p1delta-rapido-100k/last.pth", C["v4"], "t1p1delta", 1.1),
    ("T₀ = 2δ\nreferencia", RUNS / "t2delta-100k-s1/last.pth", C["fallida"], "t2delta100k", 2.0),
    ("T₀ = 8δ · final s1\nreferencia", RUNS / "t8delta-100k-s1/last.pth", C["final"], "final_s1", 8.0),
]

torch = env_cpu()
sys.path.insert(0, str(RAIZ))
from evaluate import load_vnet                      # noqa: E402
from ddr.data import add_noise, psnr                # noqa: E402

nombres = [l.strip() for l in open(RAIZ / "datasets/bsds500/cbsd68.txt") if l.strip()]


def img(n):
    p = next((RAIZ / "datasets/bsds500/images").glob(f"*/{n}.jpg"))
    x = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(x.transpose(2, 0, 1)[None].copy())


def rec(t, l=96):
    h, w = t.shape[-2:]
    return t[..., h // 2 - l // 2:h // 2 + l // 2, w // 2 - l // 2:w // 2 + l // 2]


# El mismo punto para los cuatro: 16 recortes de CBSD68 con una realización de ruido fija.
Y = torch.cat([rec(img(n)) for n in nombres[:16]])
Z = add_noise(Y, 25.0, torch.Generator().manual_seed(3))
CE = img("253027")
ZC = add_noise(CE, 25.0, torch.Generator().manual_seed(1235))
zoom = (slice(60, 170), slice(120, 230))

M = []
with torch.no_grad():
    for etq, p, col, tag, t0 in MODELOS:
        m, _ = load_vnet(str(p))
        est = m(Z, Z)
        T, lam = float(m.T), float(m.lmbda)
        gR = float(np.mean([m.R.grad(est[s]).norm().item() for s in range(est.shape[0] - 1)]))
        fd = float(np.mean([lam * (est[s] - Z).norm().item() for s in range(est.shape[0] - 1)]))
        xc = m(ZC, ZC)[-1]
        e = np.load(EIG / f"eig_{tag}.npz") if (EIG / f"eig_{tag}.npz").exists() else None
        e2 = np.load(EIG / f"eig_{tag}_img2.npz") if (EIG / f"eig_{tag}_img2.npz").exists() else None
        M.append(dict(etq=etq, col=col, T=T, lam=lam, gR=gR, fuerza=T * gR, fuerza_dato=fd,
                      rho=fd / (T * gR), psnr=psnr(est[-1], Y), t0=t0,
                      xc=xc[0].clamp(0, 1).numpy().transpose(1, 2, 0)[zoom], psnr_c=psnr(xc, CE),
                      eig=e, eig2=e2))
        d = M[-1]
        print(f"  {etq.replace(chr(10),' '):30s} T {T:.5f}  lam {lam:.5f}  T|gR| {T*gR:8.3f}  "
              f"rho {d['rho']:.4f}  PSNR16 {d['psnr']:.3f}", flush=True)

# --- LA LÁMINA ----------------------------------------------------------------------------------
fig = plt.figure(figsize=(17, 17.2))
g = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.12, 1.12, 1.12],
                     hspace=0.30, wspace=0.10, left=0.055, right=0.985, top=0.885, bottom=0.035)
etqs = [d["etq"] for d in M]
cols = [d["col"] for d in M]

# Los títulos van en dos líneas porque a una sola se pisan entre paneles contiguos.
PANELES = [
    ("T", "1 · T\ncoordenada de gauge · NO observable", "{:.4f}", False),
    ("lam", "2 · λ\npeso del dato · sí invariante", "{:.4f}", False),
    ("fuerza", "3 · T·‖∇R‖ a σ = 25\nla fuerza real · invariante", "{:.1f}", False),
]
for c, (k, t, fmt, logy) in enumerate(PANELES):
    ax = fig.add_subplot(g[0, c])
    v = [d[k] for d in M]
    ax.bar(range(4), v, color=cols, width=0.62)
    for i, x in enumerate(v):
        ax.text(i, x, fmt.format(x), ha="center", va="bottom", fontsize=11.5, fontweight="semibold")
    ax.set_xticks(range(4), etqs, fontsize=9.5)
    ax.set_title(t, fontsize=11.5)
    ax.set_ylim(0, max(v) * 1.22)
    ax.grid(axis="x", visible=False)

ax = fig.add_subplot(g[0, 3]); ax.axis("off"); ax.grid(False)
ax.text(0, 1.04, "4 · Lo que dice la lámina", fontsize=11.5, fontweight="semibold", va="top")
sp = max(d["fuerza"] for d in M) / min(d["fuerza"] for d in M)
spT = max(d["T"] for d in M) / min(d["T"] for d in M)
spP = max(d["psnr"] for d in M) - min(d["psnr"] for d in M)
tl2 = [d["T"] * d["eig2"]["log"][-1, 2] for d in M if d["eig2"] is not None]
tl1 = [d["T"] * d["eig"]["log"][-1, 2] for d in M if d["eig"] is not None]
ax.text(0, 0.90,
        f"`T` se abre {spT:.1f}× entre los cuatro y `T·‖∇R‖`\n"
        f"sólo {sp:.2f}×: el primero es una coordenada\n"
        f"libre, el segundo es la fuerza.\n\n"
        f"Las PSNR caben en {spP:.3f} dB, bajo el piso\n"
        f"de detección de 0.33 dB, y los T·Λ de la\n"
        f"imagen 2 dentro de {max(tl2)/min(tl2):.2f}×.\n\n"
        f"O sea: los cuatro aprendieron el mismo\n"
        f"regularizador y lo escribieron en gauges\n"
        f"distintos. Lo único que de verdad los\n"
        f"separa es λ — el `rápido` ancla al dato\n"
        f"{M[3]['lam']/M[1]['lam']:.1f}× menos que la referencia.",
        fontsize=10.5, va="top", color=C["texto"], linespacing=1.5)

for fila, (clave, etq_fila) in enumerate((("eig", "imagen 1"), ("eig2", "imagen 2")), start=1):
    for c, d in enumerate(M):
        ax = fig.add_subplot(g[fila, c])
        e = d[clave]
        if e is None:
            ax.text(0.5, 0.5, "falta", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.imshow(np.clip(e["it1000"], 0, 1))
            conv = "" if e["log"][-1, 1] < 0.1 else "  · no converge"
            L = e["log"][-1, 2]
            ax.set_xlabel(f"Λ = {L:.3g} (gauge){conv}\nT·Λ = {d['T']*L:.2e}  ← invariante",
                          fontsize=10, color="#2b2a27")
        if fila == 1:
            ax.set_title(d["etq"].replace("\n", " · "), fontsize=11.5, loc="center", color="#2b2a27")
        if c == 0:
            ax.set_ylabel(f"eigenfunción\n{etq_fila}", fontsize=11.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(d["col"]); s.set_linewidth(3)

for c, d in enumerate(M):
    ax = fig.add_subplot(g[3, c])
    ax.imshow(d["xc"], interpolation="nearest")
    ax.set_xlabel(f"{d['psnr_c']:.2f} dB", fontsize=13, fontweight="bold", color="#2b2a27")
    if c == 0:
        ax.set_ylabel("denoise σ = 25\n(recorte)", fontsize=11.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(d["col"]); s.set_linewidth(3)

fig.suptitle("Al término de las 100 000 iteraciones: los cuatro aprendieron lo mismo, escrito en "
             "gauges distintos", x=0.012, ha="left", fontsize=17, fontweight="bold")
fig.text(0.012, 0.962,
         "Filas 2–3 CUALITATIVAS: la eigenfunción se dibuja con la escala de cada modelo y Λ es "
         "gauge-dependiente — la comparación válida es T·Λ, abajo de cada una.\n"
         f"En la imagen 2 los cuatro T·Λ caben en {max(tl2)/min(tl2):.2f}×; en la imagen 1 se abren "
         f"{max(tl1)/min(tl1):.1f}× porque el flujo cayó en modos distintos, no porque los modelos difieran.\n"
         "Fila 4: misma imagen, mismo ruido, mismo σ para los cuatro.",
         fontsize=10.5, color=C["texto"], ha="left", va="top", linespacing=1.45)
guardar(fig, "img_estado_100k_4modelos")

L = ["# Estado al término de las 100 000 iteraciones", "",
     "| | " + " | ".join(d["etq"].replace("\n", " ") for d in M) + " |",
     "|---|" + "---|" * 4]
for t, f in (("T₀ (en δ)", lambda d: "—" if d["t0"] is None else f"{d['t0']:.1f}δ"),
             ("T", lambda d: f"{d['T']:.5f}"), ("λ", lambda d: f"{d['lam']:.5f}"),
             ("T/δ final", lambda d: f"{d['T']/DELTA_ADAM:.2f}"),
             ("‖∇R‖ (σ=25)", lambda d: f"{d['gR']:.1f}"),
             ("T·‖∇R‖ (invariante)", lambda d: f"{d['fuerza']:.2f}"),
             ("λ·‖x−z‖ (invariante)", lambda d: f"{d['fuerza_dato']:.3f}"),
             ("ρ = dato/reg", lambda d: f"{d['rho']:.4f}"),
             ("Λ (gauge)", lambda d: "—" if d["eig"] is None else f"{d['eig']['log'][-1,2]:.3g}"),
             ("T·Λ (invariante)", lambda d: "—" if d["eig"] is None else f"{d['T']*d['eig']['log'][-1,2]:.3e}"),
             ("PSNR 16 recortes", lambda d: f"{d['psnr']:.3f}"),
             ("PSNR cebras", lambda d: f"{d['psnr_c']:.3f}")):
    L.append(f"| {t} | " + " | ".join(f(d) for d in M) + " |")
(SALIDA / "estado_100k.md").write_text("\n".join(L) + "\n")
print("\n" + "\n".join(L))
