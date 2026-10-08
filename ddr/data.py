"""Datos 2D: parches, augmentación, ruido y PSNR.

Agnóstico del dominio — trabaja con arreglos `[C, H, W]` en float32. Para otro dataset alcanza con
registrar un cargador en `IMAGE_LOADERS` y ajustar `channels` / `noise_scale` en `train.py`.

Ojo con la escala: `T` y `lambda` se aprenden *para* un rango de datos concreto. Entrenar en [0,1]
y aplicar a datos en ppm sin normalizar es el problema que en `../tdv` obliga a calibrar `alpha`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

__all__ = [
    "IMAGE_LOADERS",
    "PatchDataset2D",
    "ImageDataset2D",
    "load_image",
    "list_images",
    "add_noise",
    "psnr",
]


# --- CARGADORES ------------------------------------------------------------------------------
# Punto de extensión para un dataset nuevo: IMAGE_LOADERS[".fits"] = mi_lector

def _load_pil(path: Path, channels: int) -> np.ndarray:
    """PNG/JPG de 8 bits, normalizados a [0, 1]."""
    mode = {1: "L", 3: "RGB"}.get(channels)
    if mode is None:
        raise ValueError(f"PIL no maneja {channels} canales; registrá un cargador propio")
    a = np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0
    if a.ndim == 2:
        a = a[:, :, None]
    return np.ascontiguousarray(a.transpose(2, 0, 1))


def _load_numpy(path: Path, channels: int) -> np.ndarray:
    """`.npy` / `.npz`, sin normalizar: se asume que ya vienen en la escala deseada."""
    a = np.load(path)
    if hasattr(a, "files"):
        a = a[a.files[0]]
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2:
        a = a[None]
    if a.shape[0] != channels:
        raise ValueError(f"{path.name}: {a.shape[0]} canales, se esperaban {channels}")
    return np.ascontiguousarray(a)


IMAGE_LOADERS: dict[str, Callable[[Path, int], np.ndarray]] = {
    ".png": _load_pil, ".jpg": _load_pil, ".jpeg": _load_pil,
    ".npy": _load_numpy, ".npz": _load_numpy,
}


def load_image(path: str | Path, channels: int = 3) -> np.ndarray:
    """Lee una imagen como float32 con forma [C, H, W]."""
    path = Path(path)
    loader = IMAGE_LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"sin cargador para {path.suffix!r}; registralo en IMAGE_LOADERS")
    return loader(path, channels)


def list_images(root: str | Path, splits: Sequence[str]) -> list[Path]:
    """Archivos de `<root>/images/<split>/`, en orden estable.

    Nota sobre BSDS500: reorganizó BSDS300, y **CBSD68 vive en `val`**. Por eso las 400 imágenes
    de entrenamiento son `train + test`, y `val` se reserva entero. Entrenar sobre `val` sería
    fuga de datos y volvería incomparable el 31.53 dB.
    """
    root = Path(root)
    files: list[Path] = []
    for split in splits:
        directory = root / "images" / split
        if not directory.is_dir():
            raise FileNotFoundError(f"no existe el split {split!r} en {root / 'images'}")
        files.extend(sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_LOADERS))
    if not files:
        raise FileNotFoundError(f"sin imágenes en {root} para {tuple(splits)}")
    return files


# --- RUIDO Y MÉTRICA -------------------------------------------------------------------------

def add_noise(y: torch.Tensor, sigma: float, generator=None, noise_scale: float = 255.0):
    """Ruido gaussiano de desviación `sigma/noise_scale`. Con 255, `sigma` va en niveles de 8 bits,
    que es como lo reporta el paper. Se re-muestrea en cada iteración, nunca se precomputa."""
    noise = torch.randn(y.shape, generator=generator, dtype=y.dtype, device=y.device)
    return y + (sigma / noise_scale) * noise


def psnr(x, y, data_range: float = 1.0) -> float:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    mse = float(np.mean((np.asarray(x) - np.asarray(y)) ** 2))
    return float("inf") if mse == 0.0 else float(20 * np.log10(data_range / np.sqrt(mse)))


# --- DATASETS --------------------------------------------------------------------------------

class PatchDataset2D(Dataset):
    """Parches aleatorios con flips y rotaciones de 90° (la augmentación del paper).

    `__len__` es `length`, no el número de imágenes: es un generador de parches con longitud
    declarada, y eso es lo que fija el número de iteraciones. El ruido NO se agrega acá.
    """

    def __init__(self, files, patch_size=93, channels=3, length=100_000, seed=0, augment=True):
        self.patch_size, self.length, self.seed, self.augment = patch_size, length, seed, augment
        # BSDS400 son 400 imágenes de ~481x321: entran holgadas en RAM.
        self.images = [load_image(f, channels) for f in files]

    @classmethod
    def from_root(cls, root, splits, **kw):
        return cls(list_images(root, splits), **kw)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        # Semilla por índice: el parche número i es siempre el mismo, sin sincronizar estado
        # entre los workers del DataLoader.
        rng = np.random.default_rng((self.seed, index))
        img = self.images[rng.integers(len(self.images))]

        _, h, w = img.shape
        size = self.patch_size
        if h < size or w < size:
            raise ValueError(f"imagen {h}x{w} menor que el parche {size}")
        top, left = int(rng.integers(h - size + 1)), int(rng.integers(w - size + 1))
        patch = img[:, top:top + size, left:left + size]

        if self.augment:
            if rng.random() < 0.5:
                patch = patch[:, ::-1, :]
            if rng.random() < 0.5:
                patch = patch[:, :, ::-1]
            k = int(rng.integers(4))
            if k:
                patch = np.rot90(patch, k=k, axes=(1, 2))

        return torch.from_numpy(np.ascontiguousarray(patch))


class ImageDataset2D(Dataset):
    """Imágenes completas para evaluación. `names` restringe al subconjunto canónico (CBSD68)."""

    def __init__(self, files, channels=3, names: Sequence[str] | None = None):
        files = [Path(f) for f in files]
        if names is not None:
            wanted = {Path(n).stem for n in names}
            files = [p for p in files if p.stem in wanted]
            missing = wanted - {p.stem for p in files}
            if missing:
                raise FileNotFoundError(f"faltan {len(missing)}: {sorted(missing)[:5]}")
        self.files = files
        self.images = [load_image(p, channels) for p in files]

    @classmethod
    def from_root(cls, root, split, channels=3, names=None):
        return cls(list_images(root, (split,)), channels=channels, names=names)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return torch.from_numpy(self.images[index]), self.files[index].stem
