"""Rehace las figuras del abstract con el término de datos no lineal (28-10): la 3D no lineal a 300
iteraciones (χ de DIRECTOR), TV no lineal isótropo a su mejor punto en t = 300 (`tv_fansi.py` con
`NL=1 ISO=1`) y la 2D lineal a 100 iteraciones, a sus mejores puntos medidos.

    ../../../.venv/bin/python herramientas_abstract/rehacer_nl.py [sólo COSMOS|Sim2]

La Fig. 4 es un solo χ sin el borde, la 3D no lineal en Sim2 (`fig_erosion.py`). Escribe las
especificaciones `espec/*-nlfinal.json`, llama a los `fig_*.py` y deja las figuras en
`runs/abstract-ismrm/*_nlfinal.png`. Imprime el punto elegido de TV y sus métricas.
"""
import json
import subprocess
import sys
from pathlib import Path

RAMA = Path(__file__).absolute().parents[1]
SAL = RAMA / "runs/abstract-ismrm"
ESP = RAMA / "herramientas_abstract/espec"
PY = str(RAMA / "../../../.venv/bin/python")
D3 = {"COSMOS": ("/home/santicien/Documents/Training-TDV/runs/grid-mu-alfa/28-10-300-nl-cosmos_chis.npz",
                 "nl/COSMOS/0.0031_0.00054/it300"),
      "Sim2": ("/home/santicien/Documents/Training-TDV/runs/grid-mu-alfa/28-10-300-nl_chis.npz",
               "nl/Sim2/0.0102_0.00055/it300")}
D2 = {"COSMOS": ("runs/abstract-ismrm/28-10-chi-tdv2d-COSMOS-0.0426_0.00568.npz",
                 "runs/abstract-ismrm/28-10-grid-tdv2d-COSMOS.json", "0.0426/0.00568", "linear · 100 it"),
      "Sim2": ("runs/abstract-ismrm/28-10-chi-tdv2d-Sim2-0.0426_0.01136.npz",
               "runs/abstract-ismrm/28-10-grid-tdv2d-Sim2-orto.json", "0.0426/0.01136", "linear · 100 it")}


def mejor_tv(s: str) -> tuple:
    r = json.loads((SAL / f"28-10-tv-fansi-{s}-fino-iso-nl.json").read_text())
    clave = min(r, key=lambda k: r[k]["nrmse"][min(299, len(r[k]["nrmse"]) - 1)])
    _, a, k = clave.split("/")
    v = r[clave]
    p = v["profundidad"]["300"]
    print(f"TV no lineal {s}: α {a}, k {k} | t300 NRMSE {v['nrmse'][299]:.2f} XSIM "
          f"{v['xsim']['300']:.4f} | sin borde {p['nrmse'][4]:.2f} {p['xsim'][4]:.4f} | "
          f"puntos {len(r)}", flush=True)
    return (f"runs/abstract-ismrm/28-10-chi-tvisonl-{s}-{float(a):g}_{float(k):g}.npz", clave, a, k)


def corre(script: str, espec: dict, nombre: str) -> None:
    ruta = ESP / f"{nombre}-nlfinal.json"
    ruta.write_text(json.dumps(espec, indent=1, ensure_ascii=False))
    subprocess.run(["taskset", "-c", "16-31", PY, str(RAMA / "herramientas_abstract" / script),
                    str(ruta), str(SAL / f"{nombre}_nlfinal.png")], check=True)


def main() -> None:
    solo = sys.argv[1] if len(sys.argv) > 1 else None
    sujetos = [solo] if solo else ["COSMOS", "Sim2"]
    tv = {s: mejor_tv(s) for s in sujetos}
    rois = {"COSMOS": json.loads((ESP / "fig2v3.json").read_text())["rois"],
            "Sim2": json.loads((ESP / "fig3v3.json").read_text())["rois"]}
    for s in sujetos:
        npz_tv, _, a, k = tv[s]
        metodos = [
            {"nombre": "Reference", "npz": "referencia", "clave": ""},
            {"nombre": "TV", "npz": npz_tv, "clave": "chi300",
             "params": f"nonlinear · 300 it · λ {float(a):g}"},
            {"nombre": "2D TDV", "npz": D2[s][0], "clave": "chi", "params": D2[s][3]},
            {"nombre": "3D TDV (ours)", "npz": D3[s][0], "clave": D3[s][1],
             "params": "nonlinear · 300 it"},
        ]
        corre("fig_comparacion.py", {"sujeto": s, "titulo": s, "metodos": metodos, "rois": rois[s],
                                     "error_plano": 0, "dpi": 200},
              "fig2" if s == "COSMOS" else "fig3")
    if solo:
        return
    errores = {s: [{"nombre": "TV", "npz": tv[s][0], "clave": "chi300"},
                   {"nombre": "2D TDV", "npz": D2[s][0], "clave": "chi"},
                   {"nombre": "3D TDV (ours)", "npz": D3[s][0], "clave": D3[s][1]}]
               for s in sujetos}
    corre("fig_errores.py", {"metodos": errores, "rango": 0.05, "dpi": 200}, "fig5")
    subprocess.run(["taskset", "-c", "16-31", PY, str(RAMA / "herramientas_abstract/fig_erosion.py"),
                    f"{D3['Sim2'][0]}:{D3['Sim2'][1]}", str(SAL / "fig4_nlfinal.png")], check=True)
    corre("fig_preview.py", {"sujeto": "Sim2", "roi": {"plano": 0, "fila": 97, "col": 34, "lado": 30},
                             "dpi": 150, "metodos": [
                                 {"nombre": "Reference", "npz": "referencia", "clave": ""},
                                 {"nombre": "TV", "npz": tv["Sim2"][0], "clave": "chi300"},
                                 {"nombre": "2D TDV", "npz": D2["Sim2"][0], "clave": "chi"},
                                 {"nombre": "3D TDV", "npz": D3["Sim2"][0], "clave": D3["Sim2"][1]}]},
          "previewB")


if __name__ == "__main__":
    main()
