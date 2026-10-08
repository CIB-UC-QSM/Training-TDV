"""Qué aprende la red y cómo, medido en los estados guardados de cada trayectoria (trayectorias.py).

    uv run python docs/presentacion/dentro.py        # g4, g5, g7, g8 + salida/dentro.json; CPU, ~3 min

Por estado: T, λ, ‖∇R‖ (de lote, promedio sobre los 10 x_s), el paso efectivo T·‖∇R‖/‖n‖ (el invariante de `docs/metricas.md`:
1.1–1.3 en el oráculo y en las vivas), `corr` = ‖x_S − z‖/‖n‖, la PSNR de cada uno de los 11
estados del esquema, la distancia de cada capa a su init y la imagen denoisada.

El lote de medición es fijo: 16 recortes de 96×96 del centro de las primeras 16 imágenes de
CBSD68, ruido σ = 25 con semilla 3. No es el lote de `scripts/measure_adjoint_chain.py` (aquel
dio ‖n‖ = 63.23 sobre otros parches), así que ‖∇R‖ no se compara número a número con la bitácora;
el cociente T·‖∇R‖/‖n‖ sí, porque está en unidades del ruido (verificado: oficial 1.1).
"""
import json
import sys

import numpy as np
from PIL import Image

from estilo import RAIZ, C, SALIDA, plt, guardar, env_cpu, miles, eje_capas
from trayectorias import ESTADOS, TRAYECTORIAS, ruta

torch = env_cpu()
sys.path.insert(0, str(RAIZ))
from evaluate import load_vnet  # noqa: E402
from ddr.data import add_noise, psnr  # noqa: E402

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
N_NORM = (Z - Y).norm().item()                 # de lote, como el script de la bitácora
CEBRA = carga("253027")
Z_CEBRA = add_noise(CEBRA, 25.0, torch.Generator().manual_seed(1235))   # = denoise.py, k = 1
ZOOM = (slice(60, 170), slice(120, 230))
init = {}


@torch.no_grad()
def mide(tag):
    ck = torch.load(ruta(tag), map_location="cpu", weights_only=False)
    m, _ = load_vnet(str(ruta(tag)))
    estados = m(Z, Z)
    # Misma definición que scripts/measure_adjoint_chain.py:38-42,80: ‖∇R‖ de lote en cada x_s,
    # promediada sobre los S pasos, contra ‖n‖ de lote. Medida sólo en z da ~3× más (3.12 en el
    # oficial contra el 1.10 de la bitácora): ∇R es máximo sobre la imagen ruidosa.
    gR = float(np.mean([m.R.grad(estados[s]).norm().item() for s in range(estados.shape[0] - 1)]))
    p_paso = [psnr(estados[s], Y) for s in range(estados.shape[0])]
    corr = (estados[-1] - Z).norm().item() / N_NORM
    xc = m(Z_CEBRA, Z_CEBRA)[-1]
    sd = m.state_dict()
    # Movimiento sólo donde el init es reconstruible: s1, s0 y la semilla 0 de referencia-100k y
    # del 1er disyuntor. rapido-seed10, v4 y el oficial salieron de otros inits: contra el de la
    # semilla 0 darían ~√2, que es distancia entre inits y no movimiento.
    semilla = ("init_s1" if tag.startswith(("init_s1", "s1_")) else
               "init_s0" if tag.startswith(("init_s0", "s0_")) else
               "init_ref" if tag.startswith(("init_ref", "ref_", "dis_")) else None)
    mov = None
    if semilla:
        w0 = init.setdefault(semilla, load_vnet(str(ruta(semilla)))[0].state_dict())
        mov = [((sd[k] - w0[k]).norm() / w0[k].norm()).item() for k in w0 if k.endswith("weight")]
    T, lam = float(m.T), float(m.lmbda)
    return dict(tag=tag, it=ck.get("iteration") or ESTADOS[tag][1], T=T, lam=lam, gradR=gR,
                paso=T * gR / N_NORM, corr=corr, psnr_pasos=p_paso, psnr_cebra=psnr(xc, CEBRA),
                img=xc[0].clamp(0, 1).numpy().transpose(1, 2, 0)[ZOOM], mov=mov,
                pasos_img=[estados[s, 0].clamp(0, 1).numpy().transpose(1, 2, 0)
                           for s in range(estados.shape[0])])


TAGS = list(dict.fromkeys(t for _, _, ts in TRAYECTORIAS for t in ts)) + ["oficial"]
M = {}
for t in TAGS:
    M[t] = mide(t)
    d = M[t]
    print(f"  {t:9s} it {d['it']!s:>6}  T {d['T']:.4f}  ‖∇R‖ {d['gradR']:10.3g}  "
          f"T‖∇R‖/‖n‖ {d['paso']:8.3g}  corr {d['corr']:7.3g}  PSNR x_S {d['psnr_pasos'][-1]:7.2f}",
          flush=True)
json.dump({t: {k: v for k, v in d.items() if k not in ("img", "pasos_img")} for t, d in M.items()},
          open(SALIDA / "dentro.json", "w"), indent=1, default=float)
CLAVES = [k for k in load_vnet(str(ruta("oficial")))[0].state_dict() if k.endswith("weight")]


def etq(t):
    d = M[t]
    return ESTADOS[t][0] if d["it"] in (None, 0) or ESTADOS[t][0] != "final" else f"it {d['it'] // 1000}k"


# --- G4: QUÉ SALE DE LA RED EN CADA ESTADO -----------------------------------------------------
def salidas():
    filas = [(n, col, [t for t in ts if not t.startswith("init")]) for n, col, ts in TRAYECTORIAS]
    ncol = max(len(ts) for *_, ts in filas)
    fig, axs = plt.subplots(len(filas), ncol, figsize=(2.6 * ncol + 2.4, 2.75 * len(filas)),
                            gridspec_kw=dict(wspace=0.05, hspace=0.35))
    for r, (nombre, col, ts) in enumerate(filas):
        for c in range(ncol):
            ax = axs[r, c]
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if c >= len(ts):
                ax.set_axis_off(); continue
            d = M[ts[c]]
            ax.imshow(d["img"])
            ax.set_title(f"{etq(ts[c])} · {d['psnr_cebra']:.1f} dB", fontsize=11.5, loc="center")
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(col); s.set_linewidth(3)
        axs[r, 0].set_ylabel(nombre.replace(" · ", "\n"), fontsize=11, rotation=0, ha="right",
                             va="center", labelpad=10)
    fig.suptitle("Qué devuelve la red en cada estado guardado (σ = 25, misma imagen y mismo ruido)",
                 x=0.01, ha="left", fontsize=16, fontweight="bold")
    guardar(fig, "g4_salida_por_estado")


# --- G5: LOS DIEZ PASOS DEL ESQUEMA -------------------------------------------------------------
def pasos():
    casos = [("oficial", "checkpoint oficial", C["oficial"]), ("s1_fin", "final s1", C["final"]),
             ("ref_1k", "referencia-100k, it 1k (antes del evento)", C["cercana"]),
             ("ref_4k", "referencia-100k, it 4k (después)", C["fallida"])]
    mostrar = [0, 1, 2, 3, 5, 7, 10]
    fig = plt.figure(figsize=(16, 10.5))
    g = fig.add_gridspec(len(casos), len(mostrar) + 3, wspace=0.05, hspace=0.25)
    for r, (t, nombre, col) in enumerate(casos):
        for c, s in enumerate(mostrar):
            ax = fig.add_subplot(g[r, c])
            ax.imshow(M[t]["pasos_img"][s]); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(col); sp.set_linewidth(2)
            if r == 0:
                ax.set_title("x₀ = z (ruidosa)" if s == 0 else f"paso {s}", fontsize=11, loc="center")
            ax.set_xlabel(f"{M[t]['psnr_pasos'][s]:.1f}", fontsize=10)   # media de los 16 recortes
            if c == 0:
                ax.set_ylabel(nombre.replace(", ", "\n"), fontsize=10.5, rotation=0, ha="right",
                              va="center", labelpad=8)
    ax = fig.add_subplot(g[:, len(mostrar) + 1:])
    for t, nombre, col in casos:
        p = np.array(M[t]["psnr_pasos"])
        ax.plot(range(len(p)), np.where(np.isfinite(p), p, np.nan), "-o", ms=4, color=col,
                label=nombre)
    ax.set_ylim(10, 33); ax.set_xlabel("paso s del esquema"); ax.set_ylabel("PSNR de x_s (dB)")
    ax.set_title("PSNR paso a paso", loc="left")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    fig.suptitle("Cómo denoisa: los 10 pasos de descenso, de la imagen ruidosa a x_S "
                 "(se muestra 1 recorte; el número es la PSNR media de 16)", x=0.01, ha="left",
                 fontsize=15, fontweight="bold")
    guardar(fig, "g5_pasos_del_esquema")


# --- G7: EL PASO EFECTIVO A LO LARGO DE CADA TRAYECTORIA ----------------------------------------
def paso_efectivo():
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4))
    estilos = [dict(ls="-", marker="o"), dict(ls="-", marker="s"), dict(ls="--", marker="X"),
               dict(ls=":", marker="D")]
    for (nombre, col, ts), est in zip(TRAYECTORIAS[:4], estilos):
        x = [max(M[t]["it"] or 0, 300) for t in ts]
        for ax, k in zip(axs, ("T", "gradR", "paso")):
            ax.plot(x, [M[t][k] for t in ts], color=col, lw=1.8, ms=7, **est,
                    label=nombre if ax is axs[0] else None)
    for ax, t in zip(axs, ("T", "‖∇R‖: cuánto empuja el regularizador",
                           "T·‖∇R‖ / ‖ruido‖: el paso total, en unidades del ruido")):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(t, fontsize=13)
        ax.set_xlabel("iteración (init dibujado en 300)")
    d = M["oficial"]
    for ax, k in zip(axs, ("T", "gradR", "paso")):
        ax.axhline(d[k], color=C["oficial"], ls=(0, (2, 2)), lw=1.3,
                   label="checkpoint oficial" if ax is axs[0] else None)
    sanas = [M[t]["paso"] for t in ("s1_18k", "s1_fin", "s0_16k", "s0_fin")]
    lo, hi = min(sanas + [d["paso"]]), max(sanas + [d["paso"]])
    axs[2].axhspan(lo * 0.97, hi * 1.03, color=C["final"], alpha=0.12, lw=0)
    axs[2].text(320, hi * 1.25, f"{lo:.2f}–{hi:.2f}: oficial y estados sanos", fontsize=10.5,
                color=C["final"])
    axs[0].legend(frameon=False, fontsize=9.5, loc="lower left")
    dT = 100 * (1 - M["s1_fin"]["T"] / M["init_s1"]["T"])
    crece = M["s1_fin"]["gradR"] / M["init_s1"]["gradR"]
    fig.suptitle(f"En la sana T baja {dT:.0f} %; lo que crece ×{crece:.0f} es ‖∇R‖, hasta que el "
                 "paso total iguala al ruido", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    guardar(fig, "g7_paso_efectivo")


# --- G8: CÓMO SE MUEVEN LAS CAPAS A LO LARGO DEL ENTRENAMIENTO ----------------------------------
def movimiento():
    filas = [("final s1 (sana)", ["s1_18k", "s1_fin"], C["final"]),
             ("referencia-100k (estalla en la 1 958)", ["ref_1k", "ref_4k", "ref_38k", "ref_100k"],
              C["fallida"])]
    fig, axs = plt.subplots(2, 1, figsize=(13, 8.6), sharex=True, sharey=True,
                            gridspec_kw=dict(hspace=0.42))
    for ax, (nombre, ts, col) in zip(axs, filas):
        for i, t in enumerate(ts):
            a = 0.35 + 0.65 * i / max(1, len(ts) - 1)
            ax.plot(M[t]["mov"], marker="o", ms=3, color=col, alpha=a, label=etq(t))
        ax.axhline(np.sqrt(2), color=C["oficial"], ls=":", lw=1.2)
        ax.set_title(nombre); ax.set_ylabel("‖W − W₀‖ / ‖W₀‖")
        ax.legend(frameon=False, ncol=len(ts), loc="upper right", fontsize=10.5)
    eje_capas(axs[1], CLAVES)
    eje_capas(axs[0], CLAVES)
    axs[1].set_xlabel("capa, de la entrada a la salida  (punteado: √2, la distancia entre dos inits)", labelpad=22)
    fig.suptitle("Cuánto se alejó cada capa de su init, en cada estado guardado", x=0.01, ha="left",
                 fontsize=16, fontweight="bold")
    guardar(fig, "g8_movimiento_por_estado")


salidas()
pasos()
paso_efectivo()
movimiento()
