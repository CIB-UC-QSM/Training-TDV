"""La meseta: ¿sigue aprendiendo la red cuando la PSNR ya casi no sube?

    uv run python docs/presentacion/meseta.py        # m1, m2, m3 + salida/meseta.md; CPU, ~5 min

Pregunta: entre la it 52 000 y la 60 000 las finales suben < 0.01 dB (+0.002 s0, +0.007 s1; de la
50k a la 60k, +0.012 y +0.015, casi todo el escalón del lr ÷ 2). ¿Se ve una evolución en lo que
aprenden (pesos, salida, imagen) o la red está quieta?

Estados disponibles de UNA misma trayectoria por semilla (bit-reproducible, ver trayectorias.py):
- 16k y 18k: `t8delta-100k-corte-s{0,1}` (la 16k sólo en s0, que tiene best ≠ last);
- 62k/64k/66k/68k: instantáneas que copió `vigia_instantaneas.py` en salida/ckpt/inst/ (el vigía
  se cortó en la 68k con la sesión, y la s0 se cortó a mano en la 66k);
- la it 100 000 de la s1: su `last.pth`, porque terminó.
Los de la 50k–60k NO existen (train.py pisa last.pth); ver por_implementar.md.

Lecturas, para cada par de estados consecutivos (a, b):
- cambio de pesos por tensor: ‖W_b − W_a‖/‖W_a‖, normalizado por 1 000 it;
- cambio de la salida: RMS(x̂_b − x̂_a) en niveles de gris de 8 bits, sobre 16 recortes de CBSD68;
  como escala, la misma cifra entre las dos semillas (s0 contra s1) y entre la final y el oficial.
Un cambio de salida de < 1 nivel de gris es invisible a ojo; la PSNR no lo detecta si no mejora
el error, pero la red sí se está moviendo.
"""
import sys

import numpy as np
from PIL import Image

from estilo import RAIZ, RUNS, OFICIAL, SALIDA, C, plt, guardar, env_cpu, miles, lee_corrida

torch = env_cpu()
sys.path.insert(0, str(RAIZ))
from evaluate import load_vnet  # noqa: E402
from ddr.data import add_noise  # noqa: E402

INST = SALIDA / "ckpt" / "inst"
NOMBRES = [l.strip() for l in open(RAIZ / "datasets/bsds500/cbsd68.txt") if l.strip()]


def carga(nombre):
    p = next((RAIZ / "datasets/bsds500/images").glob(f"*/{nombre}.jpg"))
    x = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(x.transpose(2, 0, 1)[None].copy())


def recorte(t, lado=96):
    h, w = t.shape[-2:]
    return t[..., h // 2 - lado // 2:h // 2 + lado // 2, w // 2 - lado // 2:w // 2 + lado // 2]


Y = torch.cat([recorte(carga(n)) for n in NOMBRES[:16]])
Z = add_noise(Y, 25.0, torch.Generator().manual_seed(3))
CEBRA = carga("253027")
Z_CEBRA = add_noise(CEBRA, 25.0, torch.Generator().manual_seed(1235))
ZOOM = (slice(60, 170), slice(120, 230))


def estados(semilla):
    """[(iteración, ruta)] de la trayectoria de una semilla, ordenados."""
    corte = RUNS / f"t8delta-100k-corte-s{semilla}"
    e = {}
    for f in ("best.pth", "last.pth"):
        it = torch.load(corte / f, map_location="cpu", weights_only=False)["iteration"]
        e[it] = corte / f
    for p in sorted((INST / f"t8delta-100k-s{semilla}").glob("it*.pth")):
        e[int(p.stem[2:])] = p
    fin = RUNS / f"t8delta-100k-s{semilla}" / "last.pth"
    e.setdefault(torch.load(fin, map_location="cpu", weights_only=False)["iteration"], fin)
    return sorted(e.items())


@torch.no_grad()
def corre(ruta):
    m, _ = load_vnet(str(ruta))
    return dict(sd={k: v for k, v in m.state_dict().items() if k.endswith("weight")},
                x=m(Z, Z)[-1], xc=m(Z_CEBRA, Z_CEBRA)[-1])


def rms_gris(a, b):
    return 255 * ((a - b) ** 2).mean().sqrt().item()


E = {s: estados(s) for s in (0, 1)}
R = {(s, it): corre(p) for s in E for it, p in E[s]}
OF = corre(OFICIAL)
for s in E:
    print(f"  semilla {s}: estados en " + ", ".join(f"{it // 1000}k" for it, _ in E[s]))

# --- PARES CONSECUTIVOS ---------------------------------------------------------------------------
pares = []
for s in E:
    its = [it for it, _ in E[s]]
    for a, b in zip(its, its[1:]):
        A, B = R[(s, a)], R[(s, b)]
        dw = np.median([((B["sd"][k] - A["sd"][k]).norm() / A["sd"][k].norm()).item() for k in A["sd"]])
        pares.append(dict(s=s, a=a, b=b, dw_por_1k=dw / (b - a) * 1000,
                          dx=rms_gris(A["x"], B["x"]), dx_por_1k=rms_gris(A["x"], B["x"]) / (b - a) * 1000))
comunes = sorted(set(it for it, _ in E[0]) & set(it for it, _ in E[1]))
semillas = {it: rms_gris(R[(0, it)]["x"], R[(1, it)]["x"]) for it in comunes}
ult = {s: E[s][-1][0] for s in E}
vs_oficial = {s: rms_gris(R[(s, ult[s])]["x"], OF["x"]) for s in E}

# --- M1: GANANCIA POR TRAMO DE 10 000 IT (de history.json y traza) ------------------------------
def ganancia():
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for s, a in ((0, 1.0), (1, 0.55)):
        r = lee_corrida(f"t8delta-100k-s{s}")
        p = dict((int(i), v) for i, v in r["psnr"])
        tramos = [(i, i + 10_000) for i in range(0, 100_000, 10_000)]
        x, dp, dl, dw = [], [], [], []
        for i, j in tramos:
            ini, fin = p.get(i) if i else None, p.get(j)
            if fin is None:
                continue
            x.append(j / 1000 - 5)
            dp.append(fin - (ini if ini is not None else p[2_000]))
            L, w = r["loss"], r["lam"] / r["T"]
            k = slice(i, j)
            dl.append(np.nanmedian(L[j - 1000:j]) - np.nanmedian(L[max(i, 0):i + 1000]))
            dw.append(np.median(w[j - 500:j]) - np.median(w[i:i + 500]))
        off = -1.6 if s == 0 else 1.6
        for ax, v in zip(axs, (dp, dl, dw)):
            ax.bar(np.array(x) + off, v, width=3.2, color=C["final"], alpha=a,
                   label=f"final s{s}")
    axs[0].set_title("ΔPSNR por tramo de 10k it (dB)"); axs[0].set_yscale("log")
    for cont in axs[0].containers:
        axs[0].bar_label(cont, fmt="%.3f", fontsize=8, rotation=90, padding=2)
    axs[1].set_title("Δ pérdida de entrenamiento (mediana)")
    axs[2].set_title("Δ peso del dato w = λ/T")
    for ax in axs:
        ax.axhline(0, color="#8a8984", lw=0.8)
        ax.set_xticks(range(5, 100, 10), [f"{i}–{i + 10}" for i in range(0, 100, 10)], fontsize=9.5,
                      rotation=45)
        ax.set_xlabel("tramo (miles de it; el 1º, desde la 2k)" if ax is axs[0] else "tramo (miles de it)")
        ax.axvspan(50, 60, color="#f4d03f", alpha=0.18, lw=0)
    axs[0].legend(frameon=False, loc="upper right")
    fig.suptitle("La ganancia por tramo cae: ~0.01 dB por 10k it entre la 30k y la 60k, ~0.002 después "
                 "(salvo el halving de la 75k)", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    guardar(fig, "m1_ganancia_por_tramo")


# --- M2: CUÁNTO CAMBIA LA RED ENTRE ESTADOS CONSECUTIVOS ----------------------------------------
def cambio():
    """Sólo marcadores: entre la 18k y la 62k no hay estados, y una línea sugeriría una caída suave."""
    fig, axs = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
    for s, mk in ((0, "o"), (1, "s")):
        # Sólo pares de ≤ 4 000 it: con saltos largos el cambio no se acumula en línea recta.
        ps = [q for q in pares if q["s"] == s and q["b"] - q["a"] <= 4000]
        x = [(q["a"] + q["b"]) / 2 for q in ps]
        axs[0].plot(x, [q["dw_por_1k"] for q in ps], mk, ms=9, color=C["final"],
                    alpha=1 if s == 0 else 0.6, label=f"final s{s}", ls="none")
        axs[1].plot(x, [q["dx_por_1k"] for q in ps], mk, ms=9, color=C["final"],
                    alpha=1 if s == 0 else 0.6, ls="none")
    for x, lr in ((12_500, "lr 4·10⁻⁴"), (37_500, "2·10⁻⁴"), (62_500, "1·10⁻⁴"), (87_500, "5·10⁻⁵")):
        for ax in axs[:2]:
            ax.text(x, 0.03, lr, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=10, color=C["texto"])
    for ax in axs[:2]:
        for h in (25_000, 50_000, 75_000):
            ax.axvline(h, color="#d6d4cc", lw=1)
        ax.set_yscale("log"); ax.set_xlabel("iteración"); miles(ax); ax.set_xlim(0, 100_000)
        ax.axvspan(50_000, 60_000, color="#f4d03f", alpha=0.18, lw=0)
    axs[1].axhline(1, color=C["evento"], lw=1, ls="--")
    axs[1].text(0.99, 0.93, "1 nivel de gris", fontsize=10.5, color=C["evento"], ha="right",
                va="top", transform=axs[1].get_yaxis_transform())
    axs[0].set_title("Pesos: mediana de ‖ΔW‖/‖W‖ cada 1 000 it")
    axs[1].set_title("Salida: RMS(Δx̂) cada 1 000 it (niveles de gris)")
    axs[0].legend(frameon=False, loc="lower left")
    # Totales, aparte: no son tasas.
    tot = [(f"s0 contra s1\nen la {it // 1000}k", v, C["cercana"]) for it, v in semillas.items()][-1:]
    tot += [(f"final s{s} ({ult[s] // 1000}k)\ncontra oficial", v, C["oficial"]) for s, v in vs_oficial.items()]
    largos = [q for q in pares if q["b"] - q["a"] > 4000 and q["a"] >= 60_000]
    tot += [(f"s{q['s']}: {q['a'] // 1000}k → {q['b'] // 1000}k", q["dx"], C["final"]) for q in largos]
    axs[2].barh(range(len(tot)), [t[1] for t in tot], color=[t[2] for t in tot], height=0.6)
    for i, t in enumerate(tot):
        axs[2].text(t[1] + 0.03, i, f"{t[1]:.2f}", va="center", fontsize=11)
    axs[2].set_yticks(range(len(tot)), [t[0] for t in tot], fontsize=10.5)
    axs[2].invert_yaxis(); axs[2].axvline(1, color=C["evento"], lw=1, ls="--")
    axs[2].set_title("Totales (niveles de gris, RMS)"); axs[2].grid(axis="y", visible=False)
    fig.suptitle("¿Se mueve la red en la meseta? Cambio entre estados guardados de la misma trayectoria",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    guardar(fig, "m2_cambio_entre_estados")


# --- M3: LA DIFERENCIA, A LA VISTA ---------------------------------------------------------------
def visual(s=0, ganancia_vis=20):
    its = [it for it, _ in E[s]]
    sel = [its[0], its[1], its[-2], its[-1]] if len(its) >= 4 else its
    pares_v = [(sel[0], sel[1]), (sel[-2], sel[-1])]
    fig, axs = plt.subplots(2, 4, figsize=(15, 8.2), gridspec_kw=dict(wspace=0.05, hspace=0.25))
    for r, (a, b) in enumerate(pares_v):
        xa, xb = R[(s, a)]["xc"], R[(s, b)]["xc"]
        d = (xb - xa)[0].numpy().transpose(1, 2, 0)[ZOOM]
        imgs = [(xa[0].clamp(0, 1).numpy().transpose(1, 2, 0)[ZOOM], f"it {a // 1000}k"),
                (xb[0].clamp(0, 1).numpy().transpose(1, 2, 0)[ZOOM], f"it {b // 1000}k"),
                (np.clip(0.5 + ganancia_vis * d, 0, 1), f"diferencia ×{ganancia_vis}"),
                (np.clip(0.5 + ganancia_vis * (OF["xc"] - xb)[0].numpy().transpose(1, 2, 0)[ZOOM], 0, 1),
                 f"oficial − it {b // 1000}k, ×{ganancia_vis}")]
        for c, (im, t) in enumerate(imgs):
            ax = axs[r, c]
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.set_title(t, fontsize=12, loc="center")
        rms = 255 * np.sqrt((d ** 2).mean())
        axs[r, 2].set_xlabel(f"RMS = {rms:.2f} niveles de gris en {b - a:,} it (este recorte)".replace(",", " "),
                             fontsize=11)
    fig.suptitle(f"final s{s}: lo que cambia la salida en 2 000 it al principio y en la meseta "
                 f"(gris = sin cambio)", x=0.01, ha="left", fontsize=15, fontweight="bold")
    guardar(fig, "m3_diferencia_visual")


ganancia()
cambio()
visual()
L = ["# La meseta: cambio entre estados guardados", "",
     "| semilla | de | a | mediana ‖ΔW‖/‖W‖ cada 1k it | RMS(Δx̂) total (niveles de gris) | RMS cada 1k it |",
     "|---|---|---|---|---|---|"]
L += [f"| s{q['s']} | {q['a']:,} | {q['b']:,} | {q['dw_por_1k']:.2e} | {q['dx']:.3f} | {q['dx_por_1k']:.3f} |"
      for q in pares]
L += ["", "Escalas de referencia (RMS de la salida, niveles de gris):"]
L += [f"- s0 contra s1 en la it {it:,}: {v:.3f}" for it, v in semillas.items()]
L += [f"- final s{s} (it {ult[s]:,}) contra el oficial: {v:.3f}" for s, v in vs_oficial.items()]
(SALIDA / "meseta.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
