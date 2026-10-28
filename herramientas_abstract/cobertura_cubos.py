"""Qué parte del cerebro ven los cubos de entrenamiento: con `--mascara-cerebro`, cada cubo de 48³
cae entero dentro de la máscara, así que la franja del borde casi no entra. Para la Fig. 1 del
abstract (28-10): la cobertura de un volumen de validación y su corte axial central.

    ../../../.venv/bin/python herramientas_abstract/cobertura_cubos.py [sujeto]

Centro válido: el cubo de 48³ centrado ahí queda a MARGEN vóxeles o más del borde, es decir, la
caja de (48 + 2·MARGEN)³ cae entera en la máscara (`MARGEN_CEREBRO` de `ddr/data3d.py`, el de la
corrida). Cubierto: algún cubo válido lo contiene. «Franja»: a ≤ 6 mm del borde (1 mm por vóxel).
En los 12 de validación (28-10): con margen 1, el 65.3 % del cerebro y el 19.5 % de la franja; la
bitácora da 66.3 y 18.5 % sobre otros 24 sujetos («Datos: LDM_100k en 3D»).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

DATOS = Path("/home/santicien/Documents/Training-TDV/datasets/ldm100k3d")
SALIDA = Path(__file__).absolute().parents[1] / "runs/abstract-ismrm"
LADO = 48
MARGEN = 1
FRANJA_MM = 6


def centros(m: np.ndarray, lado: int = LADO, margen: int = MARGEN) -> np.ndarray:
    """Los centros de los cubos válidos: la caja de (lado + 2·margen)³ centrada ahí, entera adentro."""
    q = lado + 2 * margen
    return ndimage.uniform_filter(m.astype(np.float32), size=q, mode="constant") > 1 - 0.5 / q ** 3


def cobertura(m: np.ndarray, lado: int = LADO, margen: int = MARGEN) -> np.ndarray:
    return (ndimage.uniform_filter(centros(m, lado, margen).astype(np.float32), size=lado,
                                   mode="constant") > 0.5 / lado ** 3)


def main() -> None:
    sujeto = sys.argv[1] if len(sys.argv) > 1 else "sub-004207"
    t1 = np.load(DATOS / f"images/val/{sujeto}.npy").astype(np.float32)
    m = np.load(DATOS / f"mascaras/val/{sujeto}.npy").astype(bool)
    cub = cobertura(m) & m
    dist = ndimage.distance_transform_edt(m)
    franja = m & (dist <= FRANJA_MM)
    res = {"sujeto": sujeto, "cerebro_cubierto": float(cub[m].mean()),
           "franja_cubierta": float(cub[franja].mean()), "franja_mm": FRANJA_MM, "lado": LADO}
    SALIDA.mkdir(parents=True, exist_ok=True)
    (SALIDA / f"28-10-cobertura-{sujeto}.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(SALIDA / f"28-10-cobertura-{sujeto}.npz", t1=t1.astype(np.float16),
                        mascara=m, cubierto=cub, centros=centros(m))
    print(res)


if __name__ == "__main__":
    main()
