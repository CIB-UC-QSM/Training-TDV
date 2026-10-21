"""Evaluación QSM del regularizador: ADMM plug-and-play con TDV como denoiser, sobre COSMOS.

    min_chi  ½‖W(D chi − phase)‖² + R(chi),   con z2 = D chi y z_i = chi como variables partidas

Es el protocolo de `../tdv/wTDV_QSM_torch_3d.py` (el pipeline QSM del autor) generalizado a k
denoisers: k = 3 es su pseudo-3D (la red 2D en los tres ejes, un z_i y un dual s_i por eje) y k = 1
es el 3D, una red sobre el volumen entero. Lo que sigue la corrida es el NRMSE (%) contra
`chi_cosmos` dentro de su máscara, el `rmse` de `../tdv/ddr/utils.py:91`.

Con las máscaras del autor (las de su solver): la entrada y la salida de cada denoiser van por la
máscara del cerebro en cada iteración, z_i = M·D_i(alpha·M·(chi + s_i))/alpha, y el chi final es
M·chi. Afuera no hay datos (W² = 0) y el TDV amplifica.

Ojo con la escala: la red se entrenó en [0, 1] y chi está en ppm (±0.1). `alpha` lleva la
desviación estándar de la primera reconstrucción a TARGET_STD antes de cada denoiser y la deshace a
la salida: es la calibración del autor, que existe porque la red conoce un solo sigma.

`estabilidad` mide el número de estabilidad del paso explícito del esquema desenrollado,

    (T/S)·lambda_max(hess_R(x)) < 2

en un punto x: en `punto_qsm3d`, la entrada del primer denoiser del ADMM, o en uno de denoising.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import torch
from scipy import ndimage

from .data3d import _recorte_centrado

__all__ = [
    "MU", "MU2", "TARGET_STD", "MAX_ITER", "TOL", "CORTE",
    "ITERS_ESTABILIDAD", "SEMILLA_ESTABILIDAD", "UMBRAL_ESTABILIDAD", "LADO_ESTABILIDAD",
    "prepara_datos",
    "carga_cosmos",
    "nrmse_pct",
    "admm_wtdv",
    "denoiser_3d",
    "punto_qsm3d",
    "estabilidad",
    "cortes_centrales",
    "lamina_qsm",
    "lamina_denoising",
]

Denoiser = Callable[[torch.Tensor], torch.Tensor]

# --- PROTOCOLO DEL AUTOR ---------------------------------------------------------------------
# Todos de `../tdv/wTDV_QSM_torch_3d.py` (con máscaras), por línea. Con ellos, las máscaras y su red
# 2D color en los tres ejes el autor reporta 25.94 % de NRMSE en las 100 iteraciones, sin corte
# (`../tdv/README.md`, «Script pseudo-3D»; sin máscaras eran 26.89 % en 26). No tocar: es la
# referencia.
MU = 0.008                   # :25, el `mu` del oficial en PARAMS, "del grid search del 2D"
MU2 = 1.0                    # :52, peso del split del término de datos
TARGET_STD = 0.06            # :25, desviación de alpha·chi_0 en la máscara, la del oficial
MAX_ITER = 100               # :46
TOL = 1.15                   # :47 y :193: por debajo, el ADMM se da por convergido
CORTE = 1.30                 # :48 y :179 (`finUpdate`): convergido, corta si el update lo pasa

# --- MÁSCARA DEL AUTOR -----------------------------------------------------------------------
# El skull-stripping de `wTDV_QSM_torch_3d.py:36-45`, con sus números.
UMBRAL_SOLIDO = 0.05         # :36, sobre la magnitud cruda
FRACCION_MEDIA = 0.05        # :39, el umbral dinámico es esta fracción de la media en el cerebro
ITER_CIERRE = 2              # :42, closing con un cubo 3x3x3
SIGMA_SUAVIZADO = 1.0        # :45, gaussiana sobre la máscara binaria...
UMBRAL_SUAVIZADO = 0.5       # :45, ...umbralizada otra vez: la "máscara dura"

# --- ESTABILIDAD -----------------------------------------------------------------------------
# Los dos primeros, los del script que midió el número en los checkpoints 2D («el script 2D»): con
# ellos, lo de acá se compara con lo de allá. El porqué de cada uno, en docs/porque-del-codigo.md,
# ESTABILIDAD (de ddr/qsm3d.py).
ITERS_ESTABILIDAD = 40       # su ITERS: productos Hessiano-vector de la iteración de potencia
SEMILLA_ESTABILIDAD = 11     # su generador del vector de arranque
UMBRAL_ESTABILIDAD = 2.0     # derivado: |1 − (T/S)·lambda| < 1 sólo si 0 < (T/S)·lambda < 2
LADO_ESTABILIDAD = 96        # el `lado` de su `punto_qsm`, ahora un cubo

# --- LÁMINAS ---------------------------------------------------------------------------------
RANGO_QSM = (-0.1, 0.1)      # el `rango` de imshow_3d en wTDV_QSM_torch_3d.py:222, en ppm
RANGO_DENOISING = (0.0, 1.0) # el rango de los datos de entrenamiento (DATA_RANGE de train.py)
# Derivado: el ruido de sigma = 25/255 = 0.098 queda en 0.49, medio rango, y un error de 30 dB
# (RMSE 0.032) en 0.16: el residuo del denoiser se ve sin saturar el del ruido.
AMPLIACION_ERROR = 5.0
# Un ángulo por corte, en la convención de `rotation_by_permutarion` (`_gira`). Mirados el
# 2026-10-12 sobre chi_cosmos y sobre sub-004207 de ldm100k3d/val: los tres cortes, derechos.
ORIENTACION_COSMOS = (-90, -90, 90)  # los `angles` de imshow_3d en wTDV_QSM_torch_3d.py:222
ORIENTACION_T1 = (-90, -90, -90)     # con 90 el axial sale espejado; con 0, acostado
SEPARACION = 2               # cosmético: píxeles blancos entre cortes y entre filas


# --- DATOS -----------------------------------------------------------------------------------

def prepara_datos(phase: np.ndarray, kernel: np.ndarray, magnitud: np.ndarray,
                  chi: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """El preprocesamiento de `wTDV_QSM_torch_3d.py:21,36-51`, operación por operación.

    `kernel` es el dipolo en k-space con el centro en el índice 0 (sin `ifftshift`, :21).
    Devuelve float32 salvo las máscaras: `W2` es la magnitud enmascarada, normalizada a media 1
    en el cerebro y al cuadrado (el `weight_np` del script desde :48), `Wy` = W2·phase, `K2` =
    conj(D)·D, `mascara` la máscara refinada de :45 (la de alpha) y, si hay `chi`, `mascara_eval`
    = chi != 0 (:26), la del NRMSE.
    """
    phase = np.asarray(phase, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    weight = np.asarray(magnitud, dtype=np.float32)
    K2 = np.conj(kernel) * kernel

    brain_solid = ndimage.binary_fill_holes(weight > UMBRAL_SOLIDO)
    mean_brain = np.mean(weight[brain_solid])
    brain_clean = brain_solid & (weight > FRACCION_MEDIA * mean_brain)
    brain_closed = ndimage.binary_closing(brain_clean, structure=np.ones((3, 3, 3)),
                                          iterations=ITER_CIERRE)
    brain_final = ndimage.binary_fill_holes(brain_closed)
    refined_mask = (ndimage.gaussian_filter(brain_final.astype(np.float32), sigma=SIGMA_SUAVIZADO)
                    > UMBRAL_SUAVIZADO)

    # Las mismas tres operaciones en float32 que :46-48, en el mismo orden: la normalización
    # va antes del cuadrado.
    weight = weight * refined_mask
    weight /= weight[refined_mask].mean()
    weight *= weight

    datos = {"phase": phase, "kernel": kernel, "K2": K2, "W2": weight, "Wy": weight * phase,
             "mascara": refined_mask}
    if chi is not None:
        datos["chi"] = np.asarray(chi, dtype=np.float32)
        datos["mascara_eval"] = datos["chi"] != 0
    return datos


def carga_cosmos(directorio: str | Path) -> dict[str, np.ndarray]:
    """`params.mat` y `chi_cosmos.mat` de `directorio` (el `data_mat/` del autor), preparados como
    en el script. Los 160^3 entran en ~150 MB."""
    directorio = Path(directorio)
    params = scipy.io.loadmat(directorio / "params.mat")
    chi = scipy.io.loadmat(directorio / "chi_cosmos.mat")["chi_cosmos"]
    return prepara_datos(params["phase_use"], params["kernel"], params["magn_use"], chi)


def nrmse_pct(pred: np.ndarray, gt: np.ndarray, mascara: np.ndarray | None = None) -> float:
    """100·‖pred − gt‖/‖gt‖ sobre `mascara`: el `rmse` de `../tdv/ddr/utils.py:91`, que pese al
    nombre es un error relativo, no una raíz de media."""
    if mascara is None:
        mascara = np.ones_like(pred, dtype=bool)
    return float(100 * (np.linalg.norm(pred[mascara] - gt[mascara]) / np.linalg.norm(gt[mascara])))


# --- ADMM ------------------------------------------------------------------------------------

def admm_wtdv(denoisers: Sequence[Denoiser], datos: dict[str, np.ndarray], *, mu: float = MU,
              mu2: float = MU2, target_std: float = TARGET_STD, max_iter: int = MAX_ITER,
              tol: float = TOL, corte: float = CORTE, subidas_para_cortar: int = 0,
              device: str | torch.device,
              al_iterar: Callable[[int, float, float], None] | None = None) -> dict[str, Any]:
    """El bucle de `wTDV_QSM_torch_3d.py:166-230` (con máscaras) con k = len(denoisers) splits del
    prior.

    Cada denoiser es v -> z sobre un volumen float32 `[Z, Y, X]` en `device`; el escalado por
    alpha va acá, como `process_axis` lo hace por fuera de la red, y la máscara M =
    `datos["mascara"]` a la entrada y a la salida (:200-214): z_i = M·D_i(alpha·M·v_i)/alpha. El
    orden de `denoisers` es el de la suma del numerador: el (x, y, z) del script es
    `[perm (0,1,2), perm (1,0,2), perm (2,0,1)]`.

    `chi` es M·chi, el mapa final del script (:236). `nrmse` y `x_update` tienen una entrada por
    x-update, índice t como el `Iter` del script; la última es la de `chi`, que en un corte el
    script no imprime. El NRMSE de cada iteración se mide sobre M·chi, el mismo mapa que el final
    (el script imprime las iteraciones sin M), y el update sobre el volumen entero, como el
    script. `iteraciones` = len(nrmse), el `iter` = t+1 que guarda el script. `motivo`: "corte",
    "max_iter", "no_finito" o "empeora". `al_iterar(t, x_update, nrmse)`, si se da, se llama en
    cada x-update. Sin `chi` en `datos`, el NRMSE es NaN.

    Con `subidas_para_cortar` = n > 0, el ADMM para cuando el NRMSE sube n iteraciones seguidas
    (motivo "empeora") y, pare como pare (salvo con "no_finito"), devuelve el chi de menor NRMSE
    que vio: `nrmse_final` es el mínimo de lo recorrido. La trayectoria guarda también los NRMSE
    que hicieron parar. Mira la referencia: sirve para comparar checkpoints, no para un sujeto
    sin chi.
    """
    if subidas_para_cortar and datos.get("chi") is None:
        raise ValueError("subidas_para_cortar mira el NRMSE: necesita `chi` en `datos`")
    t0 = time.perf_counter()
    k = len(denoisers)
    kernel, K2, weight, Wy, mask = (torch.from_numpy(np.ascontiguousarray(datos[c])).to(device)
                                    for c in ("kernel", "K2", "W2", "Wy", "mascara"))
    chi_gt, mascara_eval = datos.get("chi"), datos.get("mascara_eval")
    mask_f = mask.float()

    N = kernel.shape
    z = [torch.zeros(N, dtype=torch.float32, device=device) for _ in range(k)]
    s = [torch.zeros(N, dtype=torch.float32, device=device) for _ in range(k)]
    qsm = torch.zeros(N, dtype=torch.float32, device=device)
    weight_mu2 = weight + mu2
    denominator = mu2 * K2 + mu
    z2 = Wy / weight_mu2
    s2 = torch.zeros(N, dtype=torch.float32, device=device)

    nrmse: list[float] = []
    x_update: list[float] = []
    converged, motivo, alpha = False, "max_iter", 1.0
    subidas, mejor_nrmse, mejor_qsm = 0, math.inf, None   # para `subidas_para_cortar`
    for t in range(max_iter):
        qsm_old = qsm.clone()

        # --- x-UPDATE: cerrado en Fourier ----------------------------------------------------
        fft_z2_s2 = torch.fft.fftn(z2 - s2)
        fft_z_s = [torch.fft.fftn(zi - si) for zi, si in zip(z, s)]
        suma = fft_z_s[0]
        for f in fft_z_s[1:]:      # de izquierda a derecha, como `fx + fy + fz` en :151
            suma = suma + f
        numerator = mu2 * kernel * fft_z2_s2 + (mu / k) * suma
        qsm = torch.real(torch.fft.ifftn(numerator / denominator)).to(torch.float32)

        if t == 0:
            qsm_std = torch.std(qsm[mask])
            alpha = target_std / qsm_std.item() if qsm_std.item() > 0 else 1.0

        # Sin enmascarar, como :189: es el criterio de parada del ADMM, no una métrica.
        x_update.append((100 * torch.sqrt(torch.mean((qsm - qsm_old) ** 2))
                         / torch.sqrt(torch.mean(qsm ** 2))).item())
        # Sobre M·chi, el mapa que se devuelve (:236).
        nrmse.append(nrmse_pct((qsm * mask_f).cpu().numpy(), chi_gt, mascara_eval)
                     if chi_gt is not None else math.nan)
        if al_iterar is not None:
            al_iterar(t, x_update[-1], nrmse[-1])

        # La parada por NRMSE: guarda el mejor chi y cuenta las subidas seguidas.
        if subidas_para_cortar:
            subidas = subidas + 1 if t > 0 and nrmse[-1] > nrmse[-2] else 0
            if nrmse[-1] < mejor_nrmse:
                mejor_nrmse, mejor_qsm = nrmse[-1], qsm.clone()
            if subidas >= subidas_para_cortar:
                motivo = "empeora"
                break
        # El corte de :179 mira el update de la iteración ANTERIOR: el chi recién calculado es el
        # final aunque su update no haya decidido nada.
        if converged and x_update[-2] > corte:
            motivo = "corte"
            break
        # Fuera del script, que seguiría hasta MAX_ITER con NaN: sólo pasa con una red muerta.
        if not math.isfinite(x_update[-1]):
            motivo = "no_finito"
            break
        if x_update[-1] < tol:
            converged = True

        FhDFx = torch.real(torch.fft.ifftn(kernel * torch.fft.fftn(qsm))).to(torch.float32)

        # --- PASO PROXIMAL: un denoiser por split, con la entrada y la salida en M (:200-214) --
        for i, denoiser in enumerate(denoisers):
            z[i] = denoiser(alpha * ((qsm + s[i]) * mask_f)) / alpha * mask_f
        for i in range(k):
            s[i] += qsm - z[i]

        # --- z2 Y SU DUAL: el término de datos pesado por W² ---------------------------------
        z2 = (Wy + mu2 * (FhDFx + s2)) / weight_mu2
        s2 += FhDFx - z2

    finitos = [(v, t) for t, v in enumerate(nrmse) if math.isfinite(v)]
    nrmse_min, it_min = min(finitos) if finitos else (math.nan, -1)
    nrmse_final = nrmse[-1]
    if mejor_qsm is not None and motivo != "no_finito":
        qsm, nrmse_final = mejor_qsm, mejor_nrmse
    return {
        "chi": (qsm * mask_f).cpu().numpy(), "nrmse": nrmse, "x_update": x_update, "alpha": alpha,
        "iteraciones": len(nrmse), "motivo": motivo, "nrmse_final": nrmse_final,
        "nrmse_min": nrmse_min, "it_min": it_min, "k": k,
        "segundos": time.perf_counter() - t0,
    }


def denoiser_3d(modelo: torch.nn.Module, *, autocast_dtype: torch.dtype | None = None) -> Denoiser:
    """`VNet3D` sobre el volumen entero de una vez, sin grad: v `[Z, Y, X]` -> x_S, como en
    `evaluate.evaluate_denoising3d`. El modelo tiene que estar en el device de v."""
    def denoiser(v: torch.Tensor) -> torch.Tensor:
        x = v.to(torch.float32)[None, None]
        with torch.no_grad(), torch.autocast(v.device.type, dtype=autocast_dtype,
                                             enabled=autocast_dtype is not None):
            # `clone`: x_S es una vista de los S+1 estados, que si no siguen vivos con ella.
            return modelo(x, x)[-1].float()[0, 0].clone()
    return denoiser


# --- ESTABILIDAD -----------------------------------------------------------------------------

def punto_qsm3d(datos: dict[str, np.ndarray], lado: int = LADO_ESTABILIDAD, *,
                device: str | torch.device = "cpu", mu: float = MU, mu2: float = MU2,
                target_std: float = TARGET_STD) -> np.ndarray:
    """La entrada del PRIMER denoiser de `admm_wtdv` con k = 1, alpha·M·chi_0 (cero fuera de la
    máscara M, como la ve el denoiser), recortada a un cubo `lado`^3 centrado en la caja de
    `datos["mascara"]` (la ventana de `VolumeDataset3D`), en float32 y en numpy.

    No se recalcula: se corre `admm_wtdv` una iteración con un denoiser que guarda lo que recibe,
    así que es la del ADMM por construcción y no puede divergir de ella. chi_0 sale sólo del
    término de datos, F⁻¹[mu2·D·F(W²·phase/(W² + mu2))/(mu2·D² + mu)], y no depende de la red.
    `ValueError` si `lado` no entra en el volumen.
    """
    forma = datos["mascara"].shape
    if not 1 <= lado <= min(forma):
        raise ValueError(f"lado {lado}: tiene que estar entre 1 y {min(forma)}, el menor eje de "
                         f"{forma}")
    entradas: list[torch.Tensor] = []

    def guarda(v: torch.Tensor) -> torch.Tensor:
        entradas.append(v.clone())
        return v

    admm_wtdv([guarda], datos, mu=mu, mu2=mu2, target_std=target_std, max_iter=1, device=device)
    if not entradas:
        raise ValueError("el ADMM no llegó al primer denoiser: chi_0 no es finito")
    v = entradas[0].cpu().numpy()
    return np.ascontiguousarray(v[_recorte_centrado(datos["mascara"], lado)])


@contextlib.contextmanager
def _sin_tf32(activo: bool) -> Iterator[None]:
    """Las convoluciones y los matmul en float32 IEEE mientras dura, y los flags como estaban al
    salir. `intermedio` prende TF32, que corta la mantisa a 10 bits (`rapido` no: es bf16 con fp32
    IEEE); en CPU no existe y no se toca."""
    if not activo:
        yield
        return
    conv, matmul = torch.backends.cudnn.conv, torch.backends.cuda.matmul
    previos = (conv.fp32_precision, matmul.fp32_precision)
    conv.fp32_precision = matmul.fp32_precision = "ieee"
    try:
        yield
    finally:
        conv.fp32_precision, matmul.fp32_precision = previos


def estabilidad(modelo: torch.nn.Module, x: torch.Tensor | np.ndarray, *,
                iters: int = ITERS_ESTABILIDAD,
                semilla: int = SEMILLA_ESTABILIDAD) -> dict[str, float | int]:
    """lambda_max(hess_R(x)) por iteración de potencia y el número del paso, (T/S)·lambda_max.

    `x` es un volumen `[D, H, W]` o un lote `[B, C, D, H, W]`. El producto Hessiano-vector es el
    EXACTO: la VJP por autograd de `R.grad` (la adjunta escrita a mano, que es diferenciable),
    H·v = J_grad(x)ᵀ·v, con el grafo armado una vez y recorrido `iters` veces. Va en el dtype de
    los pesos (float32 en la corrida, float64 en los tests), sin autocast y sin TF32, y llama al
    método de la clase, no al `R.grad` compilado que `train3d` pone en la instancia. No toca los
    pesos ni sus `.grad` (`autograd.grad` sólo respecto de x) ni el generador global: el vector
    de arranque sale de uno propio, con `semilla`.

    lambda_max es ‖Hv‖ del último v, como en el script 2D: el MÓDULO del autovalor dominante,
    al que la potencia converge desde abajo. `rayleigh`, vᵀHv, sólo da el signo
    (docs/porque-del-codigo.md, ESTABILIDAD de ddr/qsm3d.py).
    Devuelve lambda_max, rayleigh, paso, T, S, iteraciones, cambio_relativo (|Δlambda|/lambda en
    la última iteración: la convergencia, que subestima el error) y segundos.
    """
    t0 = time.perf_counter()
    R = modelo.R
    peso = next(R.parameters())
    x = torch.as_tensor(x)
    if x.ndim == 3:
        x = x[None, None]
    x = x.detach().to(device=peso.device, dtype=peso.dtype).requires_grad_(True)
    # Del método de la clase y no del atributo: `torch.compile` de train3d.py lo reemplaza en la
    # instancia, y el compilado sumaría una entrada de dynamo por forma y correría en bf16.
    grad_R = type(R).grad.__get__(R)
    generador = torch.Generator(device="cpu").manual_seed(semilla)
    v = torch.randn(x.shape, generator=generador, dtype=torch.float64)
    v = (v / v.norm()).to(device=x.device, dtype=x.dtype)

    lam, previo, rayleigh, hechas = math.nan, math.nan, math.nan, 0
    with (_sin_tf32(x.device.type == "cuda"), torch.enable_grad(),
          torch.autocast(x.device.type, enabled=False)):
        g = grad_R(x)
        for i in range(iters):
            (Hv,) = torch.autograd.grad(g, x, grad_outputs=v, retain_graph=i < iters - 1)
            hechas += 1
            # En float64: son sumas de ~10^6 productos, y el cambio relativo se lee hasta 1e-7.
            rayleigh = float((v.double() * Hv.double()).sum())
            previo, lam = lam, float(Hv.double().norm())
            if not (math.isfinite(lam) and lam > 0):
                break
            v = Hv / lam
    T = modelo.T.item()
    cambio = abs(lam - previo) / lam if lam > 0 else math.nan
    return {"lambda_max": lam, "rayleigh": rayleigh, "paso": T / modelo.S * lam, "T": T,
            "S": modelo.S, "iteraciones": hechas, "cambio_relativo": cambio,
            "segundos": time.perf_counter() - t0}


# --- LÁMINAS ---------------------------------------------------------------------------------

def _gira(corte: np.ndarray, angulo: int) -> np.ndarray:
    """`rotation_by_permutarion` de `../tdv/ddr/utils.py:9`: -90 es `np.rot90`, 90 sólo traspone
    (rota y espeja)."""
    if angulo == 0:
        return corte
    if angulo == 90:
        return corte.T
    if angulo == -90:
        return corte.T[::-1, :]
    # Su rama de 180 descarta una columna (`im[:, :-1]`): no se copia.
    raise ValueError(f"ángulo {angulo}: sólo 0, 90 y -90")


def _a_uint8(im: np.ndarray, rango: tuple[float, float]) -> np.ndarray:
    lo, hi = rango
    return np.round(np.clip((im - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)


def _junta(piezas: list[np.ndarray], eje: int) -> np.ndarray:
    """Pega `piezas` a lo largo de `eje` con SEPARACION blanca; el otro eje se rellena en negro."""
    otro = 1 - eje
    largo = max(p.shape[otro] for p in piezas)
    salida = []
    for i, p in enumerate(piezas):
        relleno = [(0, 0), (0, 0)]
        relleno[otro] = (0, largo - p.shape[otro])
        salida.append(np.pad(p, relleno))
        if i < len(piezas) - 1:
            forma = [0, 0]
            forma[eje], forma[otro] = SEPARACION, largo
            salida.append(np.full(forma, 255, dtype=np.uint8))
    return np.concatenate(salida, axis=eje)


def _volumen(v: Any) -> np.ndarray:
    if isinstance(v, torch.Tensor):
        # `.float()` antes de numpy, que no tiene bf16: una salida de autocast levantaría TypeError.
        v = v.detach().float().cpu()
    a = np.asarray(v, dtype=np.float32).squeeze()
    if a.ndim != 3:
        raise ValueError(f"se esperaba un volumen [D, H, W], llegó {a.shape}")
    return a


def cortes_centrales(vol: Any, rango: tuple[float, float],
                     orientacion: tuple[int, int, int] = ORIENTACION_COSMOS) -> np.ndarray:
    """Los tres cortes centrales de `vol` lado a lado, en uint8 `[alto, ancho]`: los de
    `imshow_3d` (`../tdv/ddr/utils.py:20`), sin matplotlib. `rango` va a 0..255 y lo de afuera
    satura."""
    vol = _volumen(vol)
    D, H, W = vol.shape
    cortes = (vol[D // 2], vol[:, H // 2], vol[:, :, W // 2])
    return _junta([_a_uint8(_gira(c, a), rango) for c, a in zip(cortes, orientacion)], eje=1)


def lamina_qsm(chi: Any, gt: Any, mascara: Any, rango: tuple[float, float] = RANGO_QSM,
               orientacion: tuple[int, int, int] = ORIENTACION_COSMOS) -> np.ndarray:
    """Filas GT | reconstrucción | error, las tres en `mascara` y en el mismo `rango`: el error a
    la escala de chi, no ampliado."""
    chi, gt = _volumen(chi), _volumen(gt)
    m = _volumen(mascara) > 0
    filas = (gt * m, chi * m, (chi - gt) * m)
    return _junta([cortes_centrales(f, rango, orientacion) for f in filas], eje=0)


def lamina_denoising(limpia: Any, ruidosa: Any, salida: Any,
                     rango: tuple[float, float] = RANGO_DENOISING,
                     ampliacion: float = AMPLIACION_ERROR,
                     orientacion: tuple[int, int, int] = ORIENTACION_T1) -> np.ndarray:
    """Filas limpia | ruidosa | salida | ampliacion·|salida − limpia|, todas en `rango`."""
    limpia, ruidosa, salida = _volumen(limpia), _volumen(ruidosa), _volumen(salida)
    error = rango[0] + ampliacion * np.abs(salida - limpia)
    return _junta([cortes_centrales(f, rango, orientacion)
                   for f in (limpia, ruidosa, salida, error)], eje=0)

'''
# --- PROTOCOLO DEL AUTOR ---------------------------------------------------------------------
# El ADMM parte chi en z2 = D chi (datos) y en k copias z_i = chi (prior), con duales escalados:
#   chi = F⁻¹[(mu2·D·F(z2 − s2) + (mu/k)·Σ F(z_i − s_i)) / (mu2·D² + mu)]       (D real)
#   z_i = M·D_i(alpha·M·(chi + s_i))/alpha,                 s_i += chi − z_i
#   z2  = (W²·phase + mu2·(D chi + s2))/(W² + mu2),         s2 += D chi − z2
# y devuelve M·chi. M es la máscara dura de alpha (`mascara`, `refined_mask` en :69), no la del
# NRMSE (chi_cosmos != 0). Las máscaras son las del solver de `../tdv` (2026-10-18), iguales en su
# 2D y su 3D. Con la entrada en M, el prox no depende de lo de afuera, y con la salida en M el
# prior vale 0 donde no hay datos (W² = 0), que es donde el TDV amplifica (comentarios de :199 y
# :213).
# Cada split del prior pesa mu/k, así que juntos pesan mu: k = 1 y k = 3 con tres denoisers
# iguales son el mismo ADMM salvo redondeo (2.5e-8 sobre un chi de 0.1, medido en el sintético
# del test, antes de las máscaras).
# `alpha` se fija en t = 0 con chi_0, que sale sólo del término de datos (los z_i valen 0): no
# depende de la red. En COSMOS da 4.8591, el mismo que los logs del autor.
# El update relativo, sobre el volumen entero, baja, toca un mínimo y vuelve a subir: el ADMM se da
# por convergido al bajar de TOL y corta en la primera iteración cuyo update ANTERIOR pasa de
# CORTE. El chi final es el de esa iteración. TOL = 1.15 y CORTE = 1.30 los ajustó el autor con la
# red 2D oficial y sólo la salida enmascarada (corte en la 38, 25.94 %); con la entrada
# también en M el update de esa red baja sólo a 1.24 y no corta (`../tdv/README.md`). Para otra red
# los umbrales cambian: en el 2D, el grid del autor eligió tolUpdate de 0.3 a 1.5 para las cinco
# redes (`../tdv/wTDV_QSM_torch.py:24-28`). Los de la red 3D se calibran con un grid después del
# entrenamiento (pendiente); mientras, la corrida no depende de que corte: `best_qsm.pth` y
# --meseta miran el NRMSE mínimo de la trayectoria (docs/porque-del-codigo.md, EVALUACIÓN QSM y
# MESETA de train3d.py).
# El resto de las notas de este archivo, en docs/porque-del-codigo.md, «## ddr/qsm3d.py».
'''
