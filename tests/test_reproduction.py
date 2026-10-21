"""El checkpoint oficial debe cargar en este port y reproducir los números publicados.

Es el test que sustituye a *confiar* en el port: verifica de punta a punta que la arquitectura,
el padding, las adjuntas, el antialias y el esquema desenrollado son equivalentes al original.

Se salta si no están el checkpoint o BSDS500 (no se versionan).
"""

from pathlib import Path

import pytest
import torch

from ddr.data import ImageDataset2D
from evaluate import evaluate_denoising, load_vnet

CHECKPOINTS = Path("checkpoints/Oficiales")
BSDS = Path("datasets/bsds500")
CBSD68 = BSDS / "cbsd68.txt"

# Valores publicados en Kobler et al. (2020), Tablas 2 y 3.
PUBLISHED = {
    ("color", 15): 34.12,
    ("color", 25): 31.53,
    ("color", 50): 28.26,
    ("gray", 25): 29.37,
}

# Se mira `images/val` y no la carpeta: `datasets/bsds500/` existe en cualquier clon, porque su
# README y `cbsd68.txt` se versionan, y sin las imágenes el test fallaría en vez de saltearse.
pytestmark = pytest.mark.skipif(
    not (CHECKPOINTS.is_dir() and (BSDS / "images" / "val").is_dir() and CBSD68.is_file()),
    reason="requiere los checkpoints oficiales, las imágenes de BSDS500 y datasets/bsds500/cbsd68.txt",
)


@pytest.fixture(scope="module")
def file_list() -> list[str]:
    return [line.strip() for line in CBSD68.read_text().splitlines() if line.strip()]


def test_official_config_matches_documented_values():
    """La config del checkpoint oficial en color es la del paper: S = 10, 32 features, tres
    bloques y tres escalas."""
    checkpoint = torch.load(CHECKPOINTS / "tdv3-3-25-f32-color.pth", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    assert config["S"] == 10
    assert config["R"]["config"] == {
        "in_channels": 3,
        "out_channels": 1,
        "num_features": 32,
        "num_scales": 3,
        "num_mb": 3,
        "multiplier": 1,
        "efficient": True,
    }
    assert config["D"]["config"]["use_prox"] is True
    # T aprendido: coincide con el T* = 0.0247 reportado para color
    assert checkpoint["model"]["T"].item() == pytest.approx(0.0247268, abs=1e-6)


def test_gray_checkpoint_confirms_paper_typo():
    """El paper dice T* = 0.172 para gris; el checkpoint dice 0.0172 (errata del paper)."""
    checkpoint = torch.load(CHECKPOINTS / "tdv3-3-25-f32-gray.pth", map_location="cpu", weights_only=False)
    assert checkpoint["model"]["T"].item() == pytest.approx(0.0172227, abs=1e-6)


def test_loads_strictly():
    """`strict=True`: ninguna clave sobra ni falta respecto del checkpoint oficial."""
    model, config = load_vnet(CHECKPOINTS / "tdv3-3-25-f32-color.pth")
    assert sum(p.numel() for p in model.parameters()) == 387_970  # el conteo del paper


@pytest.mark.parametrize("color,sigma", [("color", 15), ("color", 25), ("color", 50), ("gray", 25)])
def test_reproduces_published_psnr(color, sigma, file_list):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_vnet(CHECKPOINTS / f"tdv3-3-25-f32-{color}.pth", device=device)
    dataset = ImageDataset2D.from_root(
        BSDS, split="val", channels=3 if color == "color" else 1, names=file_list
    )

    report = evaluate_denoising(model, dataset, sigma=sigma, sigma_ref=25.0, device=device)

    expected = PUBLISHED[(color, sigma)]
    assert report["psnr_denoised"] == pytest.approx(expected, abs=0.02), (
        f"{color} sigma={sigma}: obtenido {report['psnr_denoised']:.3f} dB, publicado {expected} dB"
    )
