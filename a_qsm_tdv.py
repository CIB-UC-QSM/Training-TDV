# Hecho con IA
"""Convierte un `.pth` de `train3d.py` (TDV 3D de `ddr/`) al `checkpoint.pt` + `report.json` que
`_load_regularizer` de ../QSM-TDV carga con `strict=True`.

    python a_qsm_tdv.py runs/<corrida>/best_qsm.pth salida/   # salida/checkpoint.pt y report.json
    python a_qsm_tdv.py best.pth salida/ --tau-data 0.5       # el tau del dato, recalibrado para QSM

Ojo: la conversión NO es exacta. Los nombres y los pesos se mapean uno a uno, pero la bajada de
escala no es el mismo operador en el borde del volumen: `ddr/conv3d.py` desenfoca el KERNEL
(3^3 -> 7^3) y convoluciona sobre la entrada espejada 3 vóxeles; QSM-TDV convoluciona el 3^3 sobre
la entrada espejada 1 vóxel y después desenfoca la SALIDA espejada 2 vóxeles
(`operators/convolution.py:211-223`). En el interior dan lo mismo; en una banda junto a cada cara,
no: ∇R difiere 1.2–3.6 % en las caras del volumen (bitácora, «Hacia QSM-TDV»). Ningún re-pesado
lo arregla: igualar el interior fija el kernel aprendido, y con ese kernel el borde difiere.

`use_amp` queda en False: con True, QSM-TDV corre la red en float16 dentro de CUDA, y la nuestra se
entrenó en bf16 (`rapido`) o fp32; float16 es otro rango (máx. 65 504) y nadie lo midió.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

__all__ = ["clave_qsm", "convierte"]

# El binomial [1,4,6,4,1]/16 de los dos lados: idéntico al `blur` de `ddr/conv3d.py:190-193` y al
# `kernel` de `BinomialDownsample3d` (`operators/convolution.py:143-145`), repetido por canal.
BINOMIAL = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def clave_qsm(clave: str, num_scales: int) -> str | None:
    """Nombre en `TDVEnergy3D` de una clave de `TotalDeepVariation3D` (sin el prefijo `R.`).
    `None` para los buffers `blur`, que QSM-TDV guarda con otro nombre y forma."""
    if clave == "K1.weight":
        return "analysis.weight"
    if clave == "KN.weight":
        return "energy_head.weight"
    if m := re.fullmatch(r"mb\.(\d+)\.mb\.(\d+)\.(\d+)\.(conv[12])\.weight", clave):
        macro, escala, cual, conv = m.groups()
        escala, cual = int(escala), int(cual)
        if escala == num_scales - 1:
            bloque = "coarsest_block"
        else:
            bloque = f"{('pre_blocks', 'post_blocks')[cual]}.{escala}"
        return f"macro_blocks.{macro}.{bloque}.{conv}.weight"
    if m := re.fullmatch(r"mb\.(\d+)\.conv_(down|up)\.(\d+)\.weight", clave):
        macro, sentido, escala = m.groups()
        return f"macro_blocks.{macro}.{sentido}_operators.{escala}.learned.weight"
    if re.fullmatch(r"mb\.\d+\.conv_(down|up)\.\d+\.blur", clave):
        return None
    raise KeyError(f"clave sin traducción: {clave}")


def convierte(estado_R: dict[str, torch.Tensor], config_R: dict) -> dict[str, torch.Tensor]:
    """state_dict de `TDVEnergy3D` a partir del de `TotalDeepVariation3D`."""
    if config_R.get("multiplier", 1) != 1 or config_R.get("in_channels", 1) != 1:
        raise ValueError("QSM-TDV tiene un solo canal de entrada y los mismos features en todas "
                         f"las escalas; la config pide {config_R}")
    num_scales = config_R.get("num_scales", 3)
    m = config_R["num_features"]
    b3 = torch.einsum("i,j,k->ijk", BINOMIAL, BINOMIAL, BINOMIAL)
    salida: dict[str, torch.Tensor] = {}
    for clave, valor in estado_R.items():
        nueva = clave_qsm(clave, num_scales)
        if nueva is None:
            # El blur del checkpoint tiene que ser el binomial: si no, el modelo es otro.
            if not torch.equal(valor.reshape(5, 5, 5).float(), b3):
                raise ValueError(f"{clave} no es el binomial [1,4,6,4,1]/16")
            continue
        salida[nueva] = valor.detach().clone().float()
    for macro in range(config_R["num_mb"]):
        for sentido in ("down", "up"):
            for escala in range(num_scales - 1):
                salida[f"macro_blocks.{macro}.{sentido}_operators.{escala}.antialias.kernel"] = (
                    b3[None, None].repeat(m, 1, 1, 1, 1).clone())
    return salida


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("pth", type=Path, help="best.pth, last.pth o estado-<it>.pth de train3d.py")
    p.add_argument("salida", type=Path, help="directorio: checkpoint.pt y report.json")
    p.add_argument("--tau-data", type=float, default=None,
                   help="tau del término de datos de QSM; sin él, lambda/S del denoising, que no "
                        "se traslada (otro operador, otras unidades)")
    args = p.parse_args()

    ck = torch.load(args.pth, map_location="cpu", weights_only=False)
    config, modelo = ck["config"], ck["model"]
    if config["R"].get("type") != "tdv3d":
        raise ValueError(f"no es un checkpoint 3D: R.type = {config['R'].get('type')!r}")
    estado_R = {k.removeprefix("R."): v for k, v in modelo.items() if k.startswith("R.")}
    regularizador = convierte(estado_R, config["R"]["config"])

    S = config["S"]
    T, lmbda = float(modelo["T"]), float(modelo["lmbda"])
    tau_R = T / S
    tau_D = args.tau_data if args.tau_data is not None else lmbda / S
    args.salida.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": {f"regularizer.{k}": v for k, v in regularizador.items()},
        "config": {"features": config["R"]["config"]["num_features"],
                   "macro_blocks": config["R"]["config"]["num_mb"],
                   "use_amp": False, "S": S, "T": T, "lambda": lmbda,
                   "origen": str(args.pth.resolve()), "iteration": ck.get("iteration")},
    }, args.salida / "checkpoint.pt")
    (args.salida / "report.json").write_text(json.dumps({
        "taus": {"regularizer": tau_R, "data": tau_D},
        "nota": ("tau regularizer = T/S del denoising (ddr: x - (T/S) grad R y despues prox). "
                 "tau data = " + ("--tau-data" if args.tau_data is not None else
                                  "lambda/S del denoising: NO se traslada a QSM, recalibrar")),
        "conversion": "inexacta en una banda junto al borde: grad R difiere 1.2-3.6 % en las caras",
    }, indent=2) + "\n")
    print(f"{len(regularizador)} tensores -> {args.salida / 'checkpoint.pt'} | T={T:.5f} "
          f"lambda={lmbda:.5f} S={S} | taus: regularizer={tau_R:.6g} data={tau_D:.6g}")


if __name__ == "__main__":
    main()
