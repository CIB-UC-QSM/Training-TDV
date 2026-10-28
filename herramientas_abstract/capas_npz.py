"""Error por capa de profundidad de un χ guardado en npz contra los de las líneas base (28-10).

    ../../../.venv/bin/python herramientas_abstract/capas_npz.py <sujeto> <npz:clave> <base.json:clave1/clave2/...> ...

Imprime NRMSE y XSIM con la máscara entera y erosionada 4 vóxeles, y el cambio del RMSE por capa
(100·(RMSE_χ / RMSE_base − 1)) contra cada base, con `por_profundidad` y `erosiones` de
`grid_tdv2d.py`.
"""
import json
import sys
from pathlib import Path

import numpy as np

RAMA = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(RAMA), str(RAMA / "herramientas_abstract")]
import sujetos  # noqa: E402
from grid_tdv2d import erosiones, por_profundidad  # noqa: E402


def main() -> None:
    nombre = sys.argv[1]
    datos = sujetos.carga_cosmos() if nombre == "COSMOS" else sujetos.carga_sim2()
    npz, clave = sys.argv[2].split(":", 1)
    chi = np.load(npz)[clave].astype(np.float32) * datos["mascara"]
    p = por_profundidad(chi, datos["chi"], erosiones(datos["mascara_eval"]))
    print(f"{nombre}: NRMSE {p['nrmse'][0]:.2f} XSIM {p['xsim'][0]:.4f} | sin borde "
          f"{p['nrmse'][4]:.2f} {p['xsim'][4]:.4f}")
    r = np.asarray(p["rmse_capa"])
    for arg in sys.argv[3:]:
        ruta, claves = arg.split(":", 1)
        v = json.loads((RAMA / ruta).read_text())
        for c in claves.split("|"):
            v = v[c]
        print(f"  contra {ruta.split('/')[-1]}: " + " ".join(f"{x:+.1f}" for x in 100 * (r / np.asarray(v) - 1)))


if __name__ == "__main__":
    main()
