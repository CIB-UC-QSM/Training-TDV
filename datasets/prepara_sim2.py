"""Sim2, el segundo sujeto de la evaluación QSM, en `datasets/qsm_sim2/sim2.npz` (lo lee
`carga_sim2` de `ddr/qsm3d.py`; `train3d.py --qsm-sim2 datasets/qsm_sim2`). Se corre una vez:

    taskset -c 16-31 .venv/bin/python datasets/prepara_sim2.py [--origen ../tdv/data_mat]

De `--origen` lee `Sim2.mat` (`phase` [176, 208, 208]), `Sim2ChiGT.nii` ([164, 205, 205], sin
ceros), `sim2_w.mat` (`w`) y `mask_final.mat`, y de `datasets/qsm_cosmos/params.mat` el valor del
kernel de COSMOS en k = 0. No hay kernel ni magnitud: cada pieza se arma y se verifica así:
- **Recorte de la fase:** el desplazamiento que maximiza la correlación, en la máscara, entre la
  fase recortada y el modelo directo D∗χ_GT.
- **Kernel:** el dipolo 1/3 − k_b0²/|k|² en la grilla de χ_GT, con el centro en el índice 0 y en
  k = 0 el valor de COSMOS; el eje de B0, el de mayor correlación con la fase.
- **Unidades:** la pendiente de la fase sobre D∗χ_GT en la máscara; si se aparta de 1 más de un
  5 %, la fase se divide por ella.
- **Pesos:** W2 = (w·M normalizado a media 1 en M)², como `prepara_datos` hace con la magnitud.
El informe de cada paso va a `datasets/qsm_sim2/informe.json`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.io

RAIZ = Path(os.path.abspath(__file__)).parent.parent
DESTINO = RAIZ / "datasets/qsm_sim2"


def dipolo(forma, eje_b0: int, valor_en_cero: float) -> np.ndarray:
    k = np.meshgrid(*(np.fft.fftfreq(n) for n in forma), indexing="ij")
    k2 = sum(ki ** 2 for ki in k)
    k2[0, 0, 0] = 1.0
    D = 1 / 3 - k[eje_b0] ** 2 / k2
    D[0, 0, 0] = valor_en_cero
    return D.astype(np.float32)


def campo(chi: np.ndarray, D: np.ndarray) -> np.ndarray:
    return np.real(np.fft.ifftn(D * np.fft.fftn(chi))).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--origen", type=Path, default=RAIZ.parent / "tdv/data_mat")
    datos = p.parse_args().origen
    phase_grande = scipy.io.loadmat(datos / "Sim2.mat")["phase"].astype(np.float32)
    chi = np.asarray(nib.load(datos / "Sim2ChiGT.nii").dataobj, dtype=np.float32)
    w = scipy.io.loadmat(datos / "sim2_w.mat")["w"].astype(np.float32)
    M = scipy.io.loadmat(datos / "mask_final.mat")["mask_final"].astype(bool)
    en_cero = float(scipy.io.loadmat(RAIZ / "datasets/qsm_cosmos/params.mat")["kernel"][0, 0, 0])
    forma = chi.shape
    informe = {"origen": str(datos), "forma_phase": list(phase_grande.shape), "forma": list(forma),
               "kernel_cosmos_en_0": en_cero}

    sobra = [a - b for a, b in zip(phase_grande.shape, forma)]
    mejor = None
    for eje in range(3):
        f = campo(chi, dipolo(forma, eje, en_cero))[M]
        for d0 in range(sobra[0] + 1):
            for d1 in range(sobra[1] + 1):
                for d2 in range(sobra[2] + 1):
                    c = float(np.corrcoef(phase_grande[d0:d0 + forma[0], d1:d1 + forma[1],
                                                       d2:d2 + forma[2]][M], f)[0, 1])
                    if mejor is None or c > mejor[0]:
                        mejor = (c, eje, (d0, d1, d2))
    corr, eje_b0, desp = mejor
    phase = phase_grande[desp[0]:desp[0] + forma[0], desp[1]:desp[1] + forma[1],
                         desp[2]:desp[2] + forma[2]]
    D = dipolo(forma, eje_b0, en_cero)
    f = campo(chi, D)
    pendiente = float(np.dot(phase[M], f[M]) / np.dot(f[M], f[M]))
    informe |= {"eje_b0": eje_b0, "desplazamiento": list(desp), "correlacion": corr,
                "pendiente_fase_sobre_modelo": pendiente}
    if abs(pendiente - 1) > 0.05:
        phase = phase / pendiente
        informe["fase_dividida_por"] = pendiente
    W = w * M
    W = W / W[M].mean()
    informe |= {"voxeles_mascara": int(M.sum()),
                "rms_chi_en_mascara": float(np.sqrt(np.mean(chi[M].astype(np.float64) ** 2)))}
    DESTINO.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DESTINO / "sim2.npz", phase=phase.astype(np.float32), kernel=D,
                        W2=(W * W).astype(np.float32), mascara=M, chi=(chi * M).astype(np.float32))
    (DESTINO / "informe.json").write_text(json.dumps(informe, indent=1))
    print(json.dumps(informe, indent=1))


if __name__ == "__main__":
    main()
