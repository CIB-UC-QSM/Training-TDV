"""PSNR de un checkpoint TDV sobre imágenes completas.

    python evaluate.py checkpoints/Oficiales/tdv3-3-25-f32-color.pth --file-list datasets/bsds500/cbsd68.txt
    python evaluate.py runs/replica/best.pth --sigma 50

Para sigma distinto del de entrenamiento se aplica el reescalado del paper (sec. 5.2): se lleva la
entrada al nivel de ruido de referencia y se devuelve la salida a su escala. Es una aproximación de
dominio, no una identidad: la red NO es escala-equivariante —bias-free no alcanza— y el esquema
completo se desvía 15-28 % de VNet(cz) = c·VNet(z). Medido el 2026-09-06, está en la bitácora.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ddr.data import ImageDataset2D, add_noise, psnr
from ddr.model import VNet2D

__all__ = ["load_vnet", "evaluate_denoising"]


def load_vnet(path, device="cpu", efficient: bool = False) -> tuple[VNet2D, dict]:
    """Carga un checkpoint (oficial o propio). `strict=True` a propósito: si el port divergiera
    en nombres o formas, queremos que falle ruidosamente."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = VNet2D(ck["config"], efficient=efficient)
    model.load_state_dict(ck["model"], strict=True)
    return model.to(device).eval(), ck["config"]


@torch.no_grad()
def evaluate_denoising(model, dataset, sigma=25.0, sigma_ref=25.0, device="cpu",
                       seed=1234, data_range=1.0, noise_scale=255.0) -> dict:
    """PSNR promedio sobre `dataset`, con una realización de ruido fija por imagen."""
    scale = sigma_ref / sigma
    rows = []
    for i in range(len(dataset)):
        clean, name = dataset[i]
        clean = clean.unsqueeze(0).to(device)
        gen = torch.Generator(device="cpu").manual_seed(seed + i)
        noisy = add_noise(clean.cpu(), sigma, gen, noise_scale).to(device)

        denoised = model(noisy * scale, noisy * scale)[-1] / scale
        rows.append({
            "name": name,
            "psnr_noisy": psnr(noisy, clean, data_range),
            "psnr_denoised": psnr(denoised, clean, data_range),
        })

    return {
        "sigma": sigma,
        "num_images": len(rows),
        "psnr_noisy": float(np.mean([r["psnr_noisy"] for r in rows])),
        "psnr_denoised": float(np.mean([r["psnr_denoised"] for r in rows])),
        "per_image": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="PSNR de un checkpoint TDV en denoising gaussiano.")
    p.add_argument("checkpoint")
    p.add_argument("--data", default="datasets/bsds500", help="raíz del dataset")
    p.add_argument("--split", default="val")
    p.add_argument("--file-list", help="subconjunto canónico, p.ej. datasets/bsds500/cbsd68.txt")
    p.add_argument("--sigma", type=float, default=25.0)
    p.add_argument("--sigma-ref", type=float, default=25.0, help="sigma con el que se entrenó")
    p.add_argument("--gray", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", help="escribir el reporte JSON acá")
    args = p.parse_args()

    names = None
    if args.file_list:
        names = [l.strip() for l in Path(args.file_list).read_text().splitlines() if l.strip()]

    dataset = ImageDataset2D.from_root(args.data, args.split,
                                       channels=1 if args.gray else 3, names=names)
    model, config = load_vnet(args.checkpoint, args.device)
    report = evaluate_denoising(model, dataset, args.sigma, args.sigma_ref, args.device)

    print(f"checkpoint : {args.checkpoint}")
    print(f"conjunto   : {args.split} ({report['num_images']} imágenes)")
    print(f"sigma      : {args.sigma}   S={config['S']}")
    print(f"PSNR ruido : {report['psnr_noisy']:.2f} dB")
    print(f"PSNR salida: {report['psnr_denoised']:.2f} dB")

    if args.output:
        report["checkpoint"] = str(args.checkpoint)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"reporte    : {args.output}")


if __name__ == "__main__":
    main()
