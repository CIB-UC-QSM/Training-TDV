"""Un ejemplo de denoising de la red 3D (14 880) para la Fig. 1 del abstract (28-10): un recorte de
128³ de un T1 de validación, ruido gaussiano de SNR 25 por el pico (σ = 1/25, la convención de
`--val-snr`) con semilla fija, y los cortes centrales de la limpia, la ruidosa y la salida.

    CUDA_VISIBLE_DEVICES=1 ../../../.venv/bin/python herramientas_abstract/ejemplo_denoising.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

RAMA = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(RAMA))
from evaluate import load_vnet  # noqa: E402

ESTADO = RAMA / "runs/ciego-epocas-l32-20k/estado-014880.pth"
T1 = Path("/home/santicien/Documents/Training-TDV/datasets/ldm100k3d/images/val/sub-004207.npy")
SALIDA = RAMA / "runs/abstract-ismrm/28-10-ejemplo-denoising.npz"
LADO, SEMILLA = 128, 1234
SNR = float(__import__("os").environ.get("SNR", "25"))   # por el pico


def main() -> None:
    v = np.load(T1).astype(np.float32)
    c = [s // 2 for s in v.shape]
    rec = tuple(slice(max(ci - LADO // 2, 0), max(ci - LADO // 2, 0) + LADO) for ci in c)
    limpia = v[rec]
    ruido = np.random.default_rng(SEMILLA).normal(0, 1 / SNR, limpia.shape).astype(np.float32)
    ruidosa = limpia + ruido
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo, _ = load_vnet(ESTADO, device=dev)
    x = torch.from_numpy(ruidosa)[None, None].to(dev)
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        salida = modelo(x, x)[-1].float()[0, 0].cpu().numpy()
    psnr = [float(20 * np.log10(1 / np.sqrt(np.mean((a - limpia) ** 2)))) for a in (ruidosa, salida)]
    cortes = {}
    for nombre, a in (("limpia", limpia), ("ruidosa", ruidosa), ("salida", salida)):
        for eje in range(3):
            cortes[f"{nombre}_{eje}"] = np.take(a, a.shape[eje] // 2, axis=eje)
    salida_npz = SALIDA if SNR == 25 else SALIDA.with_name(f"28-10-ejemplo-denoising-snr{SNR:g}.npz")
    np.savez_compressed(salida_npz, psnr_ruidosa=psnr[0], psnr_salida=psnr[1], **cortes)
    print(salida_npz, "PSNR del recorte entero: ruidosa %.2f dB, salida %.2f dB" % tuple(psnr))


if __name__ == "__main__":
    main()
