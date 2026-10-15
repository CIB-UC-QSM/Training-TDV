"""Datos 3D: volúmenes, parches cúbicos con las 48 simetrías del cubo y ruido ciego.

Espejo de `ddr/data.py` para arreglos `[D, H, W]`. Los volúmenes son los `.npy` en float16 y en
[0, 1] que escribe `datasets/cache_ldm100k3d.py`; los parches salen en float32 `[1, P, P, P]`, así
que un lote apilado es `[B, 1, P, P, P]`. `add_noise` y `psnr` se reusan de `ddr/data.py`: no
dependen de la dimensión.

Con `mascaras`, cada parche es un cubo entero dentro del cerebro, a MARGEN_CEREBRO vóxeles o más
de su borde: las máscaras son las de `datasets/mascaras_ldm100k3d.py`, alineadas vóxel a vóxel con
los volúmenes cacheados.

Ojo con la reanudación: el parche número `i` depende sólo de `(seed, offset + i)`. Reanudar en la
iteración k es construir el dataset con `offset = k * lote`, sin guardar estado del muestreo.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import add_noise, list_images, psnr

__all__ = [
    "ORIENTACIONES",
    "PatchDataset3D",
    "VolumeDataset3D",
    "esquinas_validas",
    "load_volume",
    "orienta",
    "add_noise",
    "add_noise_ciego",
    "psnr",
]


# --- TEJIDO Y RECORTES -----------------------------------------------------------------------
# Medidos sobre 200 volúmenes de LDM_100k normalizados con `scripts/mide_tejido_ldm100k3d.py`
# (crudo en docs/mediciones/tejido-ldm100k3d.txt); los números, en el bloque final.
UMBRAL_TEJIDO = 0.05   # el de `datasets/cache_ldm100k.py`: sobre el 5 % de la norma, es tejido
MIN_TEJIDO = 0.35      # el `--min-cerebro` de los cortes 2D: rechaza el 6.5 % de los parches 48^3
INTENTOS = 12          # el de `elige_corte` en 2D: en el peor volumen, P(fallan los 12) = 3e-11
EVAL_RECORTE = 128     # entra en la caja del tejido de los 200: su mínimo por eje es 144, 195, 144

# --- CUBOS DE CEREBRO ------------------------------------------------------------------------
# Medido el 2026-10-09 sobre 24 sujetos (docs/bitacora.md, «Cubos 100 % cerebro»; crudo en
# docs/mediciones/cobertura-cubos-cerebro.txt y verificacion-cubos-cerebro.txt): bloque final.
MARGEN_CEREBRO = 1     # cubos 48^3 que tocan lo excluido por la de con TTA: 0.3 % con 0, 0.0 con 1


# --- CARGA -----------------------------------------------------------------------------------

def _carga(path: str | Path, dtype: type) -> np.ndarray:
    a = np.load(path)
    if a.ndim != 3:
        raise ValueError(f"{Path(path).name}: forma {a.shape}, se esperaba [D, H, W]")
    return a.astype(dtype, copy=False)


def load_volume(path: str | Path) -> np.ndarray:
    """Lee un volumen `.npy` (float16 en disco) como float32 `[D, H, W]`."""
    return _carga(path, np.float32)


def _filtra(files: Sequence[str | Path], names: Sequence[str] | None) -> list[Path]:
    """Restringe `files` a los de `names` (por nombre sin extensión), como `ImageDataset2D`."""
    files = [Path(f) for f in files]
    if names is None:
        return files
    wanted = {Path(n).stem for n in names}
    files = [p for p in files if p.stem in wanted]
    missing = wanted - {p.stem for p in files}
    if missing:
        raise FileNotFoundError(f"faltan {len(missing)}: {sorted(missing)[:5]}")
    return files


def _recorte_centrado(tejido: np.ndarray, r: int) -> tuple[slice, slice, slice]:
    """Ventana de `r`^3 centrada en la caja de `tejido`, corrida hacia adentro si se sale."""
    ventana = []
    for eje, n in enumerate(tejido.shape):
        idx = np.flatnonzero(tejido.any(axis=tuple(a for a in range(3) if a != eje)))
        centro = (int(idx[0]) + int(idx[-1]) + 1) // 2 if idx.size else n // 2
        inicio = min(max(centro - r // 2, 0), n - r)
        ventana.append(slice(inicio, inicio + r))
    return tuple(ventana)


# --- ESQUINAS --------------------------------------------------------------------------------

def _sumas_cubo(m: np.ndarray, q: int) -> np.ndarray:
    """Suma de `m` en cada cubo `q`^3, indexada por su esquina: forma `n - q + 1` por eje."""
    c = np.zeros(tuple(n + 1 for n in m.shape), np.int64)
    c[1:, 1:, 1:] = m.astype(np.int64).cumsum(0).cumsum(1).cumsum(2)
    return (c[q:, q:, q:] - c[:-q, q:, q:] - c[q:, :-q, q:] - c[q:, q:, :-q]
            + c[:-q, :-q, q:] + c[:-q, q:, :-q] + c[q:, :-q, :-q] - c[:-q, :-q, :-q])


def esquinas_validas(mascara: np.ndarray, patch: int,
                     margen: int = MARGEN_CEREBRO) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Esquinas de los parches `patch`^3 que quedan a `margen` vóxeles o más del borde de
    `mascara`: las de los cubos `(patch + 2 margen)`^3 que caen enteros adentro. Devuelve
    `(índices planos, grilla)`, con la grilla de todas las esquinas del volumen, `n - patch + 1`
    por eje, y los índices ordenados, en uint32. El borde del volumen cuenta como afuera.
    """
    mascara = np.asarray(mascara, dtype=bool)
    grilla = tuple(n - patch + 1 for n in mascara.shape)
    tipo = np.uint32 if np.prod(grilla, dtype=np.int64) <= 2 ** 32 else np.int64
    q = patch + 2 * margen
    # Las sumas, sólo en la caja de la máscara: 2.7x más rápido que en todo el volumen (0.027
    # contra 0.072 s en un cerebro, docs/mediciones/mascaras-val-ldm100k3d.txt).
    caja = []
    for eje in range(3):
        idx = np.flatnonzero(mascara.any(axis=tuple(a for a in range(3) if a != eje)))
        if idx.size < q:
            return np.empty(0, tipo), grilla
        caja.append(slice(int(idx[0]), int(idx[-1]) + 1))
    d, h, w = np.nonzero(_sumas_cubo(mascara[tuple(caja)], q) == q ** 3)
    # El cubo con margen arranca `margen` antes que el parche.
    esquinas = np.ravel_multi_index((d + caja[0].start + margen, h + caja[1].start + margen,
                                     w + caja[2].start + margen), grilla)
    return esquinas.astype(tipo), grilla


# --- ORIENTACIONES ---------------------------------------------------------------------------
# 6 permutaciones de ejes por 8 combinaciones de flips; la 0 es la identidad. Es el análogo de los
# flips y rotaciones de 90° del paper, que en 2D recorren las 8 simetrías del cuadrado.
ORIENTACIONES: list[tuple[tuple[int, ...], tuple[bool, ...]]] = [
    (perm, flips)
    for perm in itertools.permutations(range(3))
    for flips in itertools.product((False, True), repeat=3)
]


def orienta(vol: np.ndarray, k: int) -> np.ndarray:
    """La k-ésima de `ORIENTACIONES` sobre los tres últimos ejes: permuta y después invierte.

    Devuelve una vista: `[1, P, P, P]` conserva el eje de canales.
    """
    perm, flips = ORIENTACIONES[k]
    n = vol.ndim - 3
    vol = vol.transpose(*range(n), *(n + p for p in perm))
    ejes = tuple(n + e for e, f in enumerate(flips) if f)
    return np.flip(vol, ejes) if ejes else vol


# --- RUIDO CIEGO -----------------------------------------------------------------------------

def add_noise_ciego(y: torch.Tensor, sigma_min: float, sigma_max: float, generator=None,
                    noise_scale: float = 255.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Ruido gaussiano con sigma ~ U(sigma_min, sigma_max) por muestra (eje 0), en las unidades de
    `add_noise`. Devuelve `(ruidosa, sigmas)`.

    Del mismo generador salen primero los sigmas y después el ruido: con `sigma_min == sigma_max`
    el ruido NO es el de `add_noise` con ese sigma, porque el sorteo de los sigmas va antes.
    """
    if sigma_min > sigma_max:
        raise ValueError(f"sigma_min {sigma_min} > sigma_max {sigma_max}")
    n = y.shape[0]
    u = torch.rand(n, generator=generator, dtype=y.dtype, device=y.device)
    sigmas = sigma_min + (sigma_max - sigma_min) * u
    noise = torch.randn(y.shape, generator=generator, dtype=y.dtype, device=y.device)
    escala = (sigmas / noise_scale).view(n, *([1] * (y.ndim - 1)))
    return y + escala * noise, sigmas


# --- DATASETS --------------------------------------------------------------------------------

class PatchDataset3D(Dataset):
    """Parches cúbicos al azar con una de las 48 orientaciones: la augmentación del paper, en 3D.

    `__len__` es `length`, como en `PatchDataset2D`: un generador de parches con longitud
    declarada. El ruido no se agrega acá. `signo` y `ganancia` no son del paper: vienen apagados.

    `mascaras`, una por volumen y en su orden (bool, de su misma forma), cambia el criterio: en vez
    del filtro de tejido, el parche sale de las esquinas de `esquinas_validas`, y un volumen sin
    ninguna queda fuera del sorteo. Las máscaras se usan al construir y no se guardan: con un
    generador, cada una se lee y se suelta (2 000 juntas serían ~10 GB).
    """

    def __init__(self, files, patch_size=48, length=100_000, seed=0, offset=0, augment=True,
                 min_tejido=MIN_TEJIDO, umbral_tejido=UMBRAL_TEJIDO, intentos=INTENTOS,
                 signo=False, ganancia=1.0, mascaras: Iterable[np.ndarray] | None = None,
                 margen_cerebro=MARGEN_CEREBRO):
        if intentos < 1:
            raise ValueError(f"intentos = {intentos}: hace falta al menos uno")
        if ganancia < 1.0:
            raise ValueError(f"ganancia = {ganancia}: el factor se sortea en [1/g, g], g >= 1")
        self.patch_size, self.length, self.seed, self.offset = patch_size, length, seed, offset
        self.augment, self.intentos = augment, intentos
        self.min_tejido, self.umbral_tejido = min_tejido, umbral_tejido
        self.signo, self.ganancia = signo, ganancia
        self.files = [Path(f) for f in files]
        # float16 en RAM, la mitad que float32: 9.9 MB de mediana por volumen de LDM_100k
        # recortado (docs/mediciones/tejido-ldm100k3d.txt).
        self.volumes = [_carga(f, np.float16) for f in self.files]
        for f, v in zip(self.files, self.volumes):
            if min(v.shape) < patch_size:
                raise ValueError(f"{f.name}: volumen {v.shape} menor que el parche {patch_size}")
        self.margen_cerebro = margen_cerebro
        self.esquinas = self.grillas = self.con_esquinas = None
        if mascaras is not None:
            self._indexa(mascaras)

    def _indexa(self, mascaras: Iterable[np.ndarray]) -> None:
        """Las esquinas válidas de cada volumen, y los volúmenes que tienen alguna."""
        self.esquinas, self.grillas = [], []
        for k, m in enumerate(mascaras):
            if k == len(self.files):
                raise ValueError(f"más máscaras que volúmenes ({len(self.files)})")
            m = np.asarray(m)
            if m.dtype != bool or m.shape != self.volumes[k].shape:
                raise ValueError(f"{self.files[k].name}: máscara {m.dtype} {m.shape}, se "
                                 f"esperaba bool {self.volumes[k].shape}, la forma del volumen")
            esquinas, grilla = esquinas_validas(m, self.patch_size, self.margen_cerebro)
            self.esquinas.append(esquinas)
            self.grillas.append(grilla)
        if len(self.esquinas) != len(self.files):
            raise ValueError(f"{len(self.esquinas)} máscaras para {len(self.files)} volúmenes")
        self.con_esquinas = [k for k, e in enumerate(self.esquinas) if e.size]
        lado = self.patch_size + 2 * self.margen_cerebro
        if not self.con_esquinas:
            raise ValueError(f"ningún volumen de {len(self.files)} tiene un cubo de {lado}^3 "
                             f"(parche {self.patch_size}^3 y margen {self.margen_cerebro}) entero "
                             "dentro de su máscara")
        sin = [f.name for f, e in zip(self.files, self.esquinas) if not e.size]
        if sin:
            warnings.warn(f"{len(sin)} de {len(self.files)} volúmenes sin un cubo de {lado}^3 "
                          f"dentro de su máscara quedan fuera del sorteo: {sin[:5]}", stacklevel=3)

    @classmethod
    def from_root(cls, root, splits, **kw):
        return cls(list_images(root, splits), **kw)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        # Semilla por índice GLOBAL: el parche `offset + index` es siempre el mismo, sin estado
        # compartido entre workers ni entre una corrida y su reanudación.
        rng = np.random.default_rng((self.seed, self.offset + index))
        size = self.patch_size
        if self.esquinas is None:
            vol = self.volumes[rng.integers(len(self.volumes))]
            # Tope de intentos: si ninguno pasa queda el último, así que nunca cuelga.
            for _ in range(self.intentos):
                d, h, w = (int(rng.integers(n - size + 1)) for n in vol.shape)
                patch = vol[d:d + size, h:h + size, w:w + size]
                if (patch > self.umbral_tejido).mean() >= self.min_tejido:
                    break
        else:
            # Sin intentos ni filtro de tejido: toda esquina de la lista es un cubo de cerebro.
            k = self.con_esquinas[rng.integers(len(self.con_esquinas))]
            esquinas = self.esquinas[k]
            plano = int(esquinas[rng.integers(len(esquinas))])
            d, h, w = (int(i) for i in np.unravel_index(plano, self.grillas[k]))
            patch = self.volumes[k][d:d + size, h:h + size, w:w + size]

        if self.augment:
            patch = orienta(patch, int(rng.integers(len(ORIENTACIONES))))
        patch = np.ascontiguousarray(patch, dtype=np.float32)

        # La intensidad va al final: prenderla no cambia qué parche ni qué orientación salen.
        if self.signo and rng.random() < 0.5:
            patch *= -1.0
        if self.ganancia > 1.0:
            log_g = float(np.log(self.ganancia))
            patch *= float(np.exp(rng.uniform(-log_g, log_g)))
        return torch.from_numpy(patch)[None]


class VolumeDataset3D(Dataset):
    """Volúmenes de evaluación: un recorte FIJO de `recorte`^3 centrado en la caja del tejido.

    `names` restringe a la lista de evaluación, como en `ImageDataset2D`. Un volumen menor que el
    recorte en algún eje es un error, no se rellena: el PSNR se mediría sobre ceros inventados.
    Con `mascaras`, la carpeta de las máscaras de cerebro (`<sujeto>.npy`, alineadas con el
    volumen), `cerebro` guarda cada una recortada con la misma ventana; sin ella, None. Una que
    falta es un error: quien la pasa ya verificó que están todas.
    """

    def __init__(self, files, recorte=EVAL_RECORTE, umbral_tejido=UMBRAL_TEJIDO,
                 names: Sequence[str] | None = None, mascaras: str | Path | None = None):
        self.files = _filtra(files, names)
        self.recorte = recorte
        self.recortes = []
        self.cerebro = None if mascaras is None else []
        for f in self.files:
            vol = load_volume(f)
            if min(vol.shape) < recorte:
                raise ValueError(f"{f.name}: volumen {vol.shape} menor que el recorte {recorte}")
            ventana = _recorte_centrado(vol > umbral_tejido, recorte)
            self.recortes.append(np.ascontiguousarray(vol[ventana]))
            if mascaras is not None:
                m = _carga(Path(mascaras) / f.name, bool)
                if m.shape != vol.shape:
                    raise ValueError(f"{f.name}: máscara {m.shape} y volumen {vol.shape}")
                self.cerebro.append(np.ascontiguousarray(m[ventana]))

    @classmethod
    def from_root(cls, root, split, names=None, **kw):
        return cls(list_images(root, (split,)), names=names, **kw)

    def __len__(self) -> int:
        return len(self.recortes)

    def __getitem__(self, index: int):
        return torch.from_numpy(self.recortes[index])[None], self.files[index].stem

'''
# --- TEJIDO Y RECORTES -----------------------------------------------------------------------
# Un vóxel es tejido si supera UMBRAL_TEJIDO = 0.05 de la norma, la misma definición que
# `datasets/cache_ldm100k.py` usa para elegir cortes y `datasets/cache_ldm100k3d.py` para recortar.
# Medido el 2026-10-08 sobre 200 volúmenes de LDM_100k sorteados (semilla 20261008), normalizados
# por el percentil 99.5 y recortados a la caja del tejido con MARGEN = 4 (el de la cache), con
# `scripts/mide_tejido_ldm100k3d.py`; el crudo, en docs/mediciones/tejido-ldm100k3d.txt:
#   - el 38–51 % de los vóxeles crudos (p1–p99) vale exactamente 0, y fuera de la caja el p99
#     normalizado es 0.0098: lo que queda afuera es aire.
#   - la caja mide 144–160 x 195–223 x 144–154 y toca el borde del campo en el eje 2 en los 200:
#     con cráneo, la cabeza llena el volumen, y el recorte ahorra poco (9.9 MB de mediana en
#     float16 contra 10.9 MB enteros). 2 000 volúmenes son ~19 GB en RAM; 8 000, ~78 GB: no
#     entran en los 62 GB de la máquina.
#   - un parche 48^3 sorteado uniforme tiene una fracción de tejido de mediana 0.90, p10 0.44 y
#     p1 0.12. MIN_TEJIDO = 0.35 rechaza el 6.5 % (el peor volumen, el 13.2 %); con INTENTOS = 12
#     la probabilidad de que fallen todos es ≤ 3e-11 por parche, y entonces sale el último.
# El filtro no es del paper: en BSDS todo parche tiene contenido, y acá un tercio del cubo puede
# ser aire. Es el análogo del 35 % que los cortes 2D ya exigían.
# EVAL_RECORTE = 128: 2.1 M vóxeles; a 2.26 µs/vóxel (`rapido`, forward + backward, cota del
# forward solo) son ≤ 4.8 s por volumen. Entra en la caja de los 200 medidos, así que en ellos el
# recorte centrado no toca el margen ni el borde del campo.

---------------------------------------------------------------------------------------------------

# --- CUBOS DE CEREBRO ------------------------------------------------------------------------
# LDM_100k no está skull-stripped: con el filtro de tejido, de los parches 48^3 que pasan sólo el
# 4.0 % [2.8–6.2] cae entero en el cerebro, y el parche medio es 60 % cerebro; el resto es cráneo,
# cuero cabelludo, aire y LCR externo (docs/bitacora.md, «Cubos 100 % cerebro», n = 24). Con
# `mascaras` el parche sale entero del cerebro. La máscara es deepbet ∩ HD-BET sin TTA, con huecos
# rellenos (por qué la intersección, en `datasets/mascaras_ldm100k3d.py`).
# Una esquina c es válida si el cubo (P + 2·MARGEN_CEREBRO)^3 que arranca en c − MARGEN_CEREBRO
# cae entero en la máscara, con el borde del volumen como afuera: el parche queda a
# MARGEN_CEREBRO vóxeles o más del borde del cerebro. Se cuenta con una suma acumulada 3D sobre la
# caja de la máscara, como `sumas_cubo` de scripts/cobertura_cubos_cerebro.py, y
# `tests/test_data3d.py` lo compara con la definición, esquina por esquina.
# MARGEN_CEREBRO = 1 porque el borde de la máscara es incierto en un vóxel: HD-BET sin TTA y con
# TTA difieren en una cáscara de 1–1.4 mm (p99 de la distancia, mediana sobre 24). Con margen 0,
# el 0.3 % [0.1–1.7] de los cubos 48^3 de esta máscara toca algo que la versión con TTA deja
# afuera; con 1, el 0.0 % [0.0–0.3]. Con 2 se pierde otro 22 % de esquinas y la franja cubierta
# baja de 18.5 a 8.6 % (docs/mediciones/cobertura-cubos-cerebro.txt).
# Qué cubre, con e = 1 y la máscara con TTA (n = 24, mismo crudo): a 48^3, el 66.3 % [60.5–71.8]
# del cerebro y el 18.5 % [14.5–22.7] de la franja cortical (≤ 6 mm del borde, el 27.8 % del
# cerebro), con ~60 000 esquinas por cerebro; a 32^3, el 80.1 % y el 38.1 %, con ~250 000. Los
# cubos que entran se juntan en el centro: con 32^3, sólo el 0.25 % de lo que ve el entrenamiento
# cae en la franja. Sin TTA hay 9 % menos esquinas a 48^3 (54 850 de mediana; 54 854 en los 12 de
# evaluación, docs/mediciones/mascaras-val-ldm100k3d.txt).
# Un volumen sin esquinas queda fuera del sorteo, con un aviso. Las esquinas quedan como índices
# planos uint32 (~0.2 MB por volumen a 48^3) y la máscara se descarta; leerla y contarlas cuesta
# 0.06 s por volumen (medido el 2026-10-12 sobre los 12 de evaluación, CPU).

---------------------------------------------------------------------------------------------------

# --- ORIENTACIONES ---------------------------------------------------------------------------
# Las 48 forman el grupo de simetrías del cubo (octaédrico completo, con reflexiones): componer dos
# da otra de la lista, y un sorteo uniforme de k recorre el grupo de manera uniforme, igual que en
# 2D las 16 combinaciones de dos flips y cuatro rotaciones cubren cada una de las 8 dos veces. Los
# volúmenes son isotrópicos (1 mm), así que permutar ejes no estira nada. `orienta` permuta y
# después invierte; el orden inverso daría el mismo grupo con otra numeración.

---------------------------------------------------------------------------------------------------

# --- DATASETS --------------------------------------------------------------------------------
# Orden de los sorteos de `PatchDataset3D.__getitem__`, del generador `(seed, offset + index)`:
#   1. el volumen;  2. tres enteros por intento (esquina d, h, w);  3. la orientación, si
#   `augment`;  4. el signo, si `signo`;  5. el factor log-uniforme en [1/g, g], si `ganancia` > 1.
# Con `mascaras`, 1 y 2 son otros: el volumen, uniforme entre los que tienen esquinas, y un
# índice de su lista de esquinas válidas; 3 a 5 siguen igual y en el mismo orden. Sin `mascaras`
# el flujo es el de antes de que existieran, bit a bit: `tests/test_data3d.py` lo fija con el
# sha256 de 50 parches en tres configuraciones, calculado con el código anterior.
# Los aumentos de intensidad van al final para que prenderlos no cambie la geometría: un brazo con
# `signo` ve los mismos parches que su control, multiplicados por ±1. Que la semilla sea del índice
# global hace que reanudar sea pasar `offset = it * lote` y que un worker no dependa de otro.
# `VolumeDataset3D` guarda los recortes en float32 (12 de 128^3 son 100 MB) y la caja del tejido se
# recalcula sobre el volumen cacheado, que ya está recortado a la caja con margen: su centro es el
# de la cabeza, no el del campo de visión. Con `mascaras`, la de cerebro sale de la MISMA ventana,
# que sigue saliendo del tejido y no de la máscara: el recorte es el mismo con y sin ella, y los
# PSNR de antes no cambian (test). Es la región que ve el entrenamiento con --mascara-cerebro, que
# no ve cráneo ni aire, mientras el recorte entero sí los tiene.

---------------------------------------------------------------------------------------------------

# --- RUIDO CIEGO -----------------------------------------------------------------------------
# Un sigma por parche, uniforme en [sigma_min, sigma_max], en niveles de 8 bits como `add_noise`:
# U(0, 55) es la convención de DnCNN-B. Busca un regularizador para todo el rango en vez de uno
# por sigma: es la mejora candidata que apunta a quitar el reescalado de `evaluate.py --sigma-ref`
# y el `alpha` del solver ADMM-QSM (AGENTS.md, «Mejoras candidatas»), sin medir todavía. Como
# `add_noise`, se sortea en CPU con el generador de la corrida.
'''