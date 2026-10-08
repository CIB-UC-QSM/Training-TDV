"""Imágenes denoisadas por cada modelo, con su PSNR, sobre imágenes de CBSD68 a σ = 25.

    uv run python docs/presentacion/denoise.py            # figura `img_denoise` + recortes sueltos

Misma convención que `evaluate.py`: ruido con semilla fija y `model(z, z)[-1]`. CPU a propósito
(las GPUs entrenan); con 6 hilos tarda ~1 min por modelo e imagen.
"""
import json
import sys

import numpy as np
from PIL import Image

from estilo import RAIZ, RUNS, OFICIAL, SALIDA, C, plt, guardar, env_cpu

torch = env_cpu()
sys.path.insert(0, str(RAIZ))
from evaluate import load_vnet  # noqa: E402
from ddr.data import add_noise, psnr  # noqa: E402

SIGMA = 25.0
SEMILLA = 1234
# (nombre, archivo de CBSD68, recorte (fila, col, lado) para el zoom)
IMAGENES = [("castillo", "102061", (150, 250, 110)), ("cebras", "253027", (60, 120, 110))]
MODELOS = [  # (etiqueta, checkpoint, color del marco)
    ("checkpoint oficial", OFICIAL, C["oficial"]),
    ("v4 · it 68 000 (sep)", RUNS / "replica-v4/checkpoint.pth", C["v4"]),
    ("final · semilla 0", RUNS / "t8delta-100k-s0/best.pth", C["final"]),
    ("final · semilla 1", RUNS / "t8delta-100k-s1/best.pth", C["final"]),
    ("fallida · referencia-100k", RUNS / "referencia-100k/best.pth", C["fallida"]),
]


def carga_img(nombre):
    p = next((RAIZ / "datasets/bsds500/images").glob(f"*/{nombre}.jpg"))
    x = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(x.transpose(2, 0, 1)[None].copy())


@torch.no_grad()
def denoisa(modelo, z):
    # Sin recortar a [0, 1], igual que evaluate.py:48: la PSNR se mide sobre la salida cruda.
    return modelo(z, z)[-1]


def a_img(t):
    return t[0].numpy().transpose(1, 2, 0).clip(0, 1)


def etiqueta(n, p):
    """A las finales les agrega la iteración del checkpoint: hoy en vuelo, mañana la final."""
    if not n.startswith("final"):
        return n
    it = torch.load(p, map_location="cpu", weights_only=False).get("iteration")
    return f"{n} · it {it // 1000}k" if it else n


resultados = {}
modelos = [(etiqueta(n, p), load_vnet(str(p))[0], c) for n, p, c in MODELOS]
filas = []
for k, (etq, nombre, rec) in enumerate(IMAGENES):
    y = carga_img(nombre)
    gen = torch.Generator().manual_seed(SEMILLA + k)
    z = add_noise(y, SIGMA, gen)
    fila = [("original", a_img(y), None, C["texto"]),
            (f"con ruido σ = 25", a_img(z.clamp(0, 1)), psnr(z, y), C["texto"])]
    for n, m, c in modelos:
        x = denoisa(m, z)
        fila.append((n, a_img(x), psnr(x, y), c))
        print(f"  {nombre} {n:28s} {fila[-1][2]:.3f} dB", flush=True)
    resultados[nombre] = {f[0]: f[2] for f in fila[1:]}
    filas.append((etq, rec, fila))

# --- FIGURA: imagen completa arriba, recorte ampliado abajo; una figura por imagen -------------
for etq, (f0, c0, lado), fila in filas:
    fig, axs = plt.subplots(2, len(fila), figsize=(3.1 * len(fila), 6.2),
                            gridspec_kw=dict(hspace=0.08, wspace=0.04))
    for c, (n, im, p, col) in enumerate(fila):
        a, b = axs[0, c], axs[1, c]
        a.imshow(im); b.imshow(im[f0:f0 + lado, c0:c0 + lado], interpolation="nearest")
        a.add_patch(plt.Rectangle((c0, f0), lado, lado, fill=False, ec="#f4d03f", lw=1.2))
        for ax in (a, b):
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(col); s.set_linewidth(2.5 if c >= 2 else 0.8)
        a.set_title(n.replace(" · ", "\n", 1), fontsize=12, color="#2b2a27", loc="center")
        b.set_xlabel("" if p is None else f"{p:.2f} dB", fontsize=14, color="#2b2a27",
                     fontweight="bold")
    guardar(fig, f"img_denoise_{etq}")
json.dump(resultados, open(SALIDA / "img_denoise.json", "w"), indent=1)
