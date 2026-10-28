"""Los dos sujetos del abstract con las máscaras de sus desafíos, en el ADMM y en la evaluación.

COSMOS: `msk.mat` del desafío 2016 (`qsm3d.carga_cosmos(..., mascara="desafio")`). Sim2: la de
evaluación del QSM Challenge 2.0, `Sim2Snr1/MaskBrainExtracted.nii.gz` (Zenodo 4559541, Stage 2),
en la grilla de `sim2.npz`; `mask_final` es esa misma dilatada hasta 3 vóxeles, donde el χ de
referencia no generó el campo (mensaje de la sesión DIRECTOR, 28-10). El peso es el de
`qsm3d.prepara_datos`: la magnitud (√W² de `sim2.npz`) en la máscara, normalizada a media 1 en ella
y al cuadrado.
"""
from pathlib import Path

import nibabel as nib
import numpy as np

RAMA = Path(__file__).absolute().parents[1]
MASCARA_SIM2 = Path("/home/santicien/Documents/Training-TDV/datasets/qsm_challenge2/datos/"
                    "Sim2Snr1/MaskBrainExtracted.nii.gz")


def carga_cosmos() -> dict:
    import sys
    sys.path.insert(0, str(RAMA))
    from ddr import qsm3d
    return qsm3d.carga_cosmos(RAMA / "datasets/qsm_cosmos", mascara="desafio")


def carga_sim2() -> dict:
    z = np.load(RAMA / "datasets/qsm_sim2/sim2.npz")
    M = np.asarray(nib.load(MASCARA_SIM2).dataobj) > 0
    if M.shape != z["phase"].shape or (M & ~z["mascara"].astype(bool)).any():
        raise ValueError("MaskBrainExtracted no cae en la grilla de sim2.npz")
    phase, kernel = z["phase"].astype(np.float32), z["kernel"].astype(np.float32)
    W = np.sqrt(z["W2"].astype(np.float32)) * M
    W /= W[M].mean()
    W2 = W * W
    return {"phase": phase, "kernel": kernel, "K2": kernel * kernel, "W2": W2, "Wy": W2 * phase,
            "mascara": M, "chi": z["chi"].astype(np.float32), "mascara_eval": M}


def carga_sujetos() -> dict:
    return {"COSMOS": carga_cosmos(), "Sim2": carga_sim2()}
