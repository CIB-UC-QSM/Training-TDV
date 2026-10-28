"""¿Crece la ventaja de la 3D sobre las líneas base al sacar el borde? (28-10, abstract ISMRM)

    ../../../.venv/bin/python herramientas_abstract/erosion_ventaja.py

NRMSE (%) de cada método en la máscara de evaluación erosionada e vóxeles (6-conexa; e = 0 es la
máscara entera), en t = 100 y a sus mejores parámetros, y la ventaja de la 3D (base − 3D, en
puntos) en cada e. La 3D no lineal (300 iteraciones, de DIRECTOR) se mide de su npz en Sim2.
"""
import json
import sys
from pathlib import Path

import numpy as np

RAMA = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(RAMA), str(RAMA / "herramientas_abstract")]
import sujetos  # noqa: E402
from grid_tdv2d import erosiones, por_profundidad  # noqa: E402

E = (0, 2, 4, 6, 8)
FUENTES = {
    "COSMOS": {
        "TV": ("runs/abstract-ismrm/28-10-tv-fansi-COSMOS-fino-iso.json", ["COSMOS/0.0001/300", "profundidad", "100", "nrmse"]),
        "2D": ("runs/abstract-ismrm/28-10-grid-tdv2d-COSMOS.json", ["configs", "0.0426/0.00568", "COSMOS", "profundidad", "nrmse"]),
        "3D": ("runs/grid-mu-alfa/28-10-abstract.json", ["14880/COSMOS/0.01512_0.008", "profundidad", "nrmse"]),
    },
    "Sim2": {
        "TV": ("runs/abstract-ismrm/28-10-tv-fansi-Sim2-fino.json", ["Sim2/0.0001/300", "profundidad", "100", "nrmse"]),
        "2D": ("runs/abstract-ismrm/28-10-grid-tdv2d-Sim2-orto.json", ["configs", "0.0426/0.01136", "Sim2", "profundidad", "nrmse"]),
        "3D": ("runs/grid-mu-alfa/28-10-abstract.json", ["14880/Sim2/0.0423_0.005674", "profundidad", "nrmse"]),
    },
}
NL = {"Sim2": ("/home/santicien/Documents/Training-TDV/runs/grid-mu-alfa/28-10-300-nl_chis.npz",
               "nl/Sim2/0.0102_0.00055/it300")}


def lee(ruta: str, claves: list) -> np.ndarray:
    v = json.loads((RAMA / ruta).read_text())
    for c in claves:
        v = v[c]
    return np.asarray(v, dtype=float)


def main() -> None:
    for s, fuentes in FUENTES.items():
        n = {m: lee(*f) for m, f in fuentes.items()}
        if s in NL:
            d = sujetos.carga_sim2()
            chi = np.load(NL[s][0])[NL[s][1]].astype(np.float32) * d["mascara"]
            n["3D-nl"] = np.asarray(por_profundidad(chi, d["chi"], erosiones(d["mascara_eval"]))["nrmse"])
        print(s, "e:", E)
        for m, v in n.items():
            print(f"  {m:6s} NRMSE", " ".join(f"{v[e]:6.2f}" for e in E))
        for m3 in [m for m in n if m.startswith("3D")]:
            for b in ("TV", "2D"):
                print(f"  {b} − {m3}:", " ".join(f"{n[b][e] - n[m3][e]:+6.2f}" for e in E))


if __name__ == "__main__":
    main()
