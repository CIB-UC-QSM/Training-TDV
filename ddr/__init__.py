"""TDV 2D y 3D: el regularizador, el esquema desenrollado y los datos.

Port de `../tdv/ddr/` (2D, PyTorch nativo, sin `optox`). El checkpoint oficial carga con
`strict=True` y reproduce los cuatro números publicados — ver tests/test_reproduction.py.
Lo 3D (`conv3d.py`, `tdv3d.py`, `VNet3D`, `data3d.py`) es un espejo del 2D con un eje más. No
hay checkpoint oficial contra el que compararlo: lo sostienen tests/test_conv3d.py y
test_gradient3d.py.
"""

from .conv import Conv2d, ConvScale2d, ConvScaleTranspose2d
from .conv3d import Conv3d, ConvScale3d, ConvScaleTranspose3d
from .data import ImageDataset2D, PatchDataset2D, add_noise, psnr
from .data3d import PatchDataset3D, VolumeDataset3D, add_noise_ciego
from .model import VNet2D, VNet3D
from .tdv import TotalDeepVariation2D, student_t
from .tdv3d import TotalDeepVariation3D

__all__ = [
    "Conv2d", "ConvScale2d", "ConvScaleTranspose2d",
    "Conv3d", "ConvScale3d", "ConvScaleTranspose3d",
    "TotalDeepVariation2D", "TotalDeepVariation3D", "student_t",
    "VNet2D", "VNet3D",
    "PatchDataset2D", "ImageDataset2D", "add_noise", "psnr",
    "PatchDataset3D", "VolumeDataset3D", "add_noise_ciego",
]
