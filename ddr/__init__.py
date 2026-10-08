"""TDV 2D: el regularizador, el esquema desenrollado y los datos.

Port de `../tdv/ddr/` (2D, PyTorch nativo, sin `optox`). El checkpoint oficial carga con
`strict=True` y reproduce los cuatro números publicados — ver tests/test_reproduction.py.
"""

from .conv import Conv2d, ConvScale2d, ConvScaleTranspose2d
from .data import ImageDataset2D, PatchDataset2D, add_noise, psnr
from .model import VNet2D
from .tdv import TotalDeepVariation2D, student_t

__all__ = [
    "Conv2d", "ConvScale2d", "ConvScaleTranspose2d",
    "TotalDeepVariation2D", "student_t",
    "VNet2D",
    "PatchDataset2D", "ImageDataset2D", "add_noise", "psnr",
]
