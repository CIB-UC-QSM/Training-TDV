"""PSNR, NRMSE y SNR sobre CBSD68 a varios niveles de ruido, sin re-entrenar.

    uv run python docs/presentacion/metricas_sigma.py              # todo lo que falte (caché)
    uv run python docs/presentacion/metricas_sigma.py final        # rehace sólo las finales
    MODELO="final s1" uv run python docs/presentacion/metricas_sigma.py   # un modelo (en paralelo)

«Distintos SNR» = distintos niveles de ruido σ: la red se entrenó SÓLO a σ = 25, y para otro σ se
usa el reescalado del paper (sec. 5.2, `evaluate.py:6-9`): se denoisa z·25/σ y se divide la salida
por lo mismo. La red NO es escala-equivariante (AGENTS.md: se desvía 15–28 %), así que esto mide
cuánto aguanta esa aproximación, no una simetría.

Métricas por imagen, promediadas sobre las 68 (ruido con semilla 1234 + i, como evaluate.py):
    PSNR  = 20·log10(1/RMSE)                           (rango de datos 1)
    NRMSE = ‖x̂ − y‖ / ‖y‖                              (adimensional; se reporta en %)
    SNR   = 10·log10(‖y‖² / ‖error‖²) = −20·log10(NRMSE) (de entrada: error = z − y)
NRMSE y SNR de salida son la misma información en otra escala; PSNR difiere de ellas porque
normaliza por el máximo (1) y no por la energía de cada imagen.

Valores publicados (tests/test_reproduction.py:23-25): 34.12 / 31.53 / 28.26 dB a σ 15/25/50.
CPU a propósito; ~8 s por imagen y modelo con 8 hilos, así que una celda (modelo, σ) son ~10 min.
"""
import json
import sys

import numpy as np

from estilo import RAIZ, RUNS, OFICIAL, SALIDA, C, plt, guardar, env_cpu

torch = env_cpu()
sys.path.insert(0, str(RAIZ))
from evaluate import load_vnet  # noqa: E402
from ddr.data import ImageDataset2D, add_noise  # noqa: E402

SIGMAS = [15, 25, 50]
PUBLICADO = {15: 34.12, 25: 31.53, 50: 28.26}
MODELOS = [("oficial", OFICIAL, C["oficial"]), ("v4", RUNS / "replica-v4/checkpoint.pth", C["v4"]),
           ("final s0", RUNS / "t8delta-100k-s0/best.pth", C["final"]),
           ("final s1", RUNS / "t8delta-100k-s1/best.pth", C["final"])]
# Una caché por modelo: así cuatro procesos (MODELO=...) pueden correr en paralelo sin pisarse.
CACHE = lambda nombre: SALIDA / f"metricas_sigma_{nombre.replace(' ', '_')}.json"  # noqa: E731

nombres = [l.strip() for l in open(RAIZ / "datasets/bsds500/cbsd68.txt") if l.strip()]
ds = ImageDataset2D.from_root(str(RAIZ / "datasets/bsds500"), "val", channels=3, names=nombres)
cache = {}
for nombre, *_ in MODELOS:
    if CACHE(nombre).exists():
        cache.update(json.load(open(CACHE(nombre))))
rehacer = sys.argv[1] if len(sys.argv) > 1 else None
import os  # noqa: E402
solo = os.environ.get("MODELO")       # MODELO="final s1" corre sólo ese modelo (para paralelizar)


@torch.no_grad()
def evalua(m, sigma):
    escala = 25.0 / sigma
    filas = []
    for i in range(len(ds)):
        y = ds[i][0].unsqueeze(0)
        z = add_noise(y, sigma, torch.Generator().manual_seed(1234 + i), 255.0)
        x = m(z * escala, z * escala)[-1] / escala
        ny = y.norm().item()
        e_in, e_out = (z - y).norm().item(), (x - y).norm().item()
        mse = ((x - y) ** 2).mean().item()
        filas.append((20 * np.log10(1 / np.sqrt(mse)), e_out / ny, 20 * np.log10(ny / e_in),
                      20 * np.log10(ny / e_out)))
    f = np.array(filas)
    return dict(psnr=f[:, 0].mean(), nrmse=f[:, 1].mean(), snr_in=f[:, 2].mean(),
                snr_out=f[:, 3].mean(), psnr_img=f[:, 0].tolist(), nrmse_img=f[:, 1].tolist())


for nombre, ck, _ in MODELOS:
    if solo and nombre != solo:
        continue
    it = torch.load(ck, map_location="cpu", weights_only=False).get("iteration")
    m = None
    for s in SIGMAS:
        clave = f"{nombre}|{s}"
        viejo = cache.get(clave)
        if viejo and viejo.get("it") == it and not (rehacer and nombre.startswith(rehacer)):
            continue
        m = m or load_vnet(str(ck))[0]
        cache[clave] = {**evalua(m, s), "it": it}
        propio = {k: v for k, v in cache.items() if k.startswith(nombre + "|")}
        json.dump(propio, open(CACHE(nombre), "w"), indent=1, default=float)
        d = cache[clave]
        print(f"  {nombre:9s} σ={s:2d}  PSNR {d['psnr']:.3f}  NRMSE {100 * d['nrmse']:.2f} %  "
              f"SNR in {d['snr_in']:.2f} → out {d['snr_out']:.2f} dB", flush=True)

if solo:
    sys.exit(0)   # la tabla y la figura las arma la corrida sin MODELO, con todas las cachés

# --- TABLA ---------------------------------------------------------------------------------------
L = ["# Métricas a distintos niveles de ruido (CBSD68, 68 imágenes, sin re-entrenar)", "",
     "La red se entrenó a σ = 25; los otros σ usan el reescalado del paper (z·25/σ).", "",
     "| modelo | it | " + " | ".join(f"σ = {s}: PSNR · NRMSE · SNR salida" for s in SIGMAS) + " |",
     "|---|---|" + "---|" * len(SIGMAS)]
for nombre, *_ in MODELOS:
    celdas = []
    for s in SIGMAS:
        d = cache.get(f"{nombre}|{s}")
        celdas.append("—" if d is None else
                      f"{d['psnr']:.2f} dB · {100 * d['nrmse']:.2f} % · {d['snr_out']:.2f} dB")
    it = next((cache[f"{nombre}|{s}"]["it"] for s in SIGMAS if f"{nombre}|{s}" in cache), None)
    L.append(f"| {nombre} | {it or '—'} | " + " | ".join(celdas) + " |")
snr_in = {s: cache[f"oficial|{s}"]["snr_in"] for s in SIGMAS if f"oficial|{s}" in cache}
L += ["| publicado (paper) | | " + " | ".join(f"{PUBLICADO[s]:.2f} dB" for s in SIGMAS) + " |", "",
      "SNR de entrada (imagen ruidosa): " + ", ".join(f"σ = {s} → {v:.2f} dB" for s, v in snr_in.items()),
      "", "NRMSE = ‖x̂ − y‖/‖y‖ por imagen, promediado. SNR de salida = −20·log10(NRMSE) por imagen."]
(SALIDA / "metricas_sigma.md").write_text("\n".join(L) + "\n")

# --- FIGURA: diferencias contra el oficial (en absoluto las curvas se superponen) -------------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
otros = [(n, col) for n, _, col in MODELOS if n != "oficial"]
ancho = 0.8 / len(otros)
for j, (nombre, col) in enumerate(otros):
    for ax, k, f in ((axs[0], "psnr", 1), (axs[1], "nrmse", 100)):
        v = [(cache[f"{nombre}|{s}"][k] - cache[f"oficial|{s}"][k]) * f
             if f"{nombre}|{s}" in cache and f"oficial|{s}" in cache else np.nan for s in SIGMAS]
        x = np.arange(len(SIGMAS)) + (j - (len(otros) - 1) / 2) * ancho
        ax.bar(x, v, width=ancho * 0.9, color=col, alpha=0.6 if nombre.endswith("s1") else 1,
               label=nombre, hatch="//" if nombre == "v4" else None, edgecolor="white")
etq = [f"σ = {s}\nSNR {snr_in[s]:.1f} dB" if s in snr_in else f"σ = {s}" for s in SIGMAS]
for ax, t in zip(axs, ("PSNR − PSNR del oficial (dB; más es mejor)",
                       "NRMSE − NRMSE del oficial (puntos %; menos es mejor)")):
    ax.set_xticks(range(len(SIGMAS)), etq); ax.set_title(t, fontsize=13)
    ax.axhline(0, color=C["oficial"], lw=1.2, ls=(0, (2, 2)))
    ax.grid(axis="x", visible=False)
axs[1].legend(frameon=False, loc="upper left")
fig.suptitle("Fuera del σ de entrenamiento (sin re-entrenar), las finales empatan con el oficial; "
             "v4 queda ~0.06 dB abajo", x=0.01, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout()
guardar(fig, "h1_metricas_por_sigma")
print((SALIDA / "metricas_sigma.md").read_text())
