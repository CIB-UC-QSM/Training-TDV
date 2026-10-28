"""¿Deja la red 2D (tríos axiales) un error que cambia de corte a corte? (28-10, abstract ISMRM)

    ../../../.venv/bin/python herramientas_abstract/anisotropia_error.py <sujeto> <etiqueta=npz:clave> ...

Para cada mapa, la energía media de la diferencia finita del error e = M·χ − χ_GT a lo largo de
cada eje (pares de vóxeles dentro de la máscara erosionada 4 vóxeles, para no medir el borde), y
la razón entre el eje de los cortes axiales (el último: el que la red 2D recorre de a tríos) y la
media de los otros dos. Una razón mayor que la de los métodos 3D indica error de corte a corte.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

RAMA = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(RAMA), str(RAMA / "herramientas_abstract")]
import sujetos  # noqa: E402


def main() -> None:
    nombre = sys.argv[1]
    datos = sujetos.carga_cosmos() if nombre == "COSMOS" else sujetos.carga_sim2()
    gt, M = datos["chi"], datos["mascara"]
    me = ndimage.binary_erosion(datos["mascara_eval"], ndimage.generate_binary_structure(3, 1), 4)
    for arg in sys.argv[2:]:
        etiqueta, fuente = arg.split("=", 1)
        npz, clave = fuente.split(":", 1)
        e = np.load(RAMA / npz)[clave].astype(np.float64) * M - gt
        energia = []
        for eje in range(3):
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[eje], b[eje] = slice(1, None), slice(None, -1)
            par = me[tuple(a)] & me[tuple(b)]
            energia.append(float(np.mean((e[tuple(a)] - e[tuple(b)])[par] ** 2)))
        razon = energia[2] / np.mean(energia[:2])
        print(f"{nombre} {etiqueta:8s} |Δe|² por eje (1e-4 ppm²): "
              + " ".join(f"{1e4 * v:.3f}" for v in energia) + f" | razón axial/otros {razon:.3f}")


if __name__ == "__main__":
    main()
