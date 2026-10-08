"""Eigenfunciones no lineales de R (como la Fig. del paper TDV, ../tdv/ddr/eigenfunctions.py).

Minimiza ||grad_R(x) - Λ(x) x||² con descenso acelerado y proyección a [0,1], partiendo de un
recorte de imagen. Uso: python eigen.py <ckpt> <tag> <img> <out_dir> [iters]
"""
import sys, time
import numpy as np, torch
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, ".")
from evaluate import load_vnet

torch.set_num_threads(4)
ckpt, tag, img, out = sys.argv[1:5]
iters = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
dev = "cuda" if torch.cuda.is_available() else "cpu"
tau, crop, pad = 0.01, 96, 24

vn, _ = load_vnet(ckpt)
R = vn.R.double().to(dev)
for p in R.parameters():
    p.requires_grad_(False)

im = np.asarray(Image.open(img).convert("RGB"), dtype=np.float64) / 255
h, w, _ = im.shape
im = im[h // 2 - crop // 2: h // 2 + crop // 2, w // 2 - crop // 2: w // 2 + crop // 2]
x = torch.from_numpy(im.transpose(2, 0, 1)[None].copy()).to(dev)
mask = np.zeros((crop + 2 * pad,) * 2)
mask[pad // 2:-pad // 2, pad // 2:-pad // 2] = 1
mask = torch.from_numpy(gaussian_filter(mask, pad // 4)[None, None]).to(dev)
x = torch.nn.functional.pad(x, (pad,) * 4, "replicate") * mask


def rayleigh(x):
    g = R.grad(x)
    lam = (g * x).sum() / (x ** 2).sum()
    return ((g - lam * x) ** 2).mean(), lam


snaps, log = {}, []
x_old = x.clone()
t0 = time.time()
for j in range(iters + 1):
    xb = (x + (j - 1) / (j + 2) * (x - x_old)).detach().requires_grad_(True)
    x_old = x.clone()
    l, lam = rayleigh(xb)
    l.backward()
    x = torch.clamp(xb.data - tau * xb.grad, 0, 1)
    if j % 50 == 0:
        log.append((j, l.item(), lam.item()))
        print(tag, j, f"{l.item():.4g}", f"Λ={lam.item():.4g}", f"{time.time()-t0:.0f}s", flush=True)
    if j in (0, iters // 10, iters // 3, iters):
        snaps[j] = x[0, :, pad:-pad, pad:-pad].cpu().numpy().astype(np.float32).transpose(1, 2, 0)

np.savez_compressed(f"{out}/eig_{tag}.npz", start=im.astype(np.float32), **{f"it{k}": v for k, v in snaps.items()},
         log=np.array(log))
