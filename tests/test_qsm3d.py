"""`ddr/qsm3d.py` contra el script del autor (`../tdv/wTDV_QSM_torch_3d.py`), transcripto acá.

El ADMM se compara sobre un problema sintético de 20^3 (dipolo calculado, magnitud y chi de
juguete) con denoisers de juguete deterministas y NO lineales: con uno lineal, D(alpha·v)/alpha =
D(v) y un alpha mal aplicado pasaría. Son tres distintos, uno por eje, para que confundir el z o
el dual de un split con el de otro tampoco pase.
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.io
import torch
from scipy import ndimage

import train
from ddr import qsm3d
from ddr.model import VNet3D

N = 20


def busca_cosmos():
    """El `data_mat/` de `../tdv`, hermano del checkout: desde un worktree está más arriba."""
    for p in Path(__file__).resolve().parents:
        if (p.parent / "tdv" / "data_mat" / "params.mat").is_file():
            return p.parent / "tdv" / "data_mat"
    return None


DIR_COSMOS = busca_cosmos()


# --- TRANSCRIPCIONES DEL AUTOR ---------------------------------------------------------------

def rmse_del_autor(pred, gt, mask=None):
    """`../tdv/ddr/utils.py:91-99`, textual."""
    if mask is None:
        mask = np.ones_like(pred, dtype=bool)
    rmse_val = np.linalg.norm(pred[mask] - gt[mask]) / np.linalg.norm(gt[mask])
    return 100 * rmse_val


def rotation_by_permutarion(im, angle):
    """`../tdv/ddr/utils.py:9-18`, textual."""
    if angle == 90:
        im = im.T
    elif angle == -90:
        im = im.T
        im = im[::-1, :]
        return im
    elif angle == 180:
        im = im[:, :-1]
    return im


def bucle_del_script(datos, process_x, process_y, process_z, maxOuterIter, tolUpdate=1.15,
                     finUpdate=1.30):
    """El bucle del solver 3D de `../tdv` (`wTDV_QSM_torch_3d.py:85-236`), con sus nombres.
    `process_axis(v, ..., alpha, ...)` pasa a ser `process(v * alpha) / alpha`, que es lo que hace
    con el volumen.
    Además de lo que imprime el script (update y RMSE sin máscara) guarda el mapa de cada
    iteración, que el test enmascara para comparar el NRMSE de `admm_wtdv`, y devuelve el mapa
    final enmascarado, como :236."""
    device = "cpu"
    kernel = torch.from_numpy(datos["kernel"])
    K2 = torch.from_numpy(datos["K2"])
    weight = torch.from_numpy(datos["W2"])
    Wy = torch.from_numpy(datos["Wy"])
    mask_torch = torch.from_numpy(datos["mascara"])
    mask_f = mask_torch.float()
    chi_cosmos, cosmos_mask = datos["chi"], datos["mascara_eval"]
    N = kernel.shape
    mu = 0.008
    mu2 = 1.0

    zx = torch.zeros(N, dtype=torch.float32, device=device)
    sx = torch.zeros(N, dtype=torch.float32, device=device)
    zy = torch.zeros(N, dtype=torch.float32, device=device)
    sy = torch.zeros(N, dtype=torch.float32, device=device)
    zz = torch.zeros(N, dtype=torch.float32, device=device)
    sz = torch.zeros(N, dtype=torch.float32, device=device)
    qsm = torch.zeros(N, dtype=torch.float32, device=device)
    weight_mu2 = weight + mu2
    denominator = mu2 * K2 + mu
    z2 = Wy / weight_mu2
    s2 = torch.zeros(N, dtype=torch.float32, device=device)

    historia, mapas = [], []
    converged = False
    for t in range(0, maxOuterIter):
        qsm_old = qsm.clone()

        fft_z2_s2 = torch.fft.fftn(z2 - s2)
        fft_zx_sx = torch.fft.fftn(zx - sx)
        fft_zy_sy = torch.fft.fftn(zy - sy)
        fft_zz_sz = torch.fft.fftn(zz - sz)

        numerator = mu2 * kernel * fft_z2_s2 + (mu / 3.0) * (fft_zx_sx + fft_zy_sy + fft_zz_sz)
        qsm = torch.real(torch.fft.ifftn(numerator / denominator)).to(torch.float32)

        if converged and x_update.item()>finUpdate:
            break

        if t == 0:
            target_std = 0.06
            qsm_std = torch.std(qsm[mask_torch])
            alpha = target_std / qsm_std.item() if qsm_std.item() > 0 else 1.0

        x_update = 100*torch.sqrt(torch.mean((qsm-qsm_old) ** 2))/torch.sqrt(torch.mean((qsm) ** 2))
        current_rmse = rmse_del_autor(qsm.cpu().numpy(), chi_cosmos, mask=cosmos_mask)
        historia.append((x_update.item(), current_rmse))
        mapas.append(qsm.clone())

        if x_update.item() < tolUpdate:
            converged = True

        FhDFx = torch.real(torch.fft.ifftn(kernel*torch.fft.fftn(qsm))).to(torch.float32)

        vx = (qsm + sx) * mask_f
        vy = (qsm + sy) * mask_f
        vz = (qsm + sz) * mask_f
        zz = process_z(vz * alpha) / alpha
        zy = process_y(vy * alpha) / alpha
        zx = process_x(vx * alpha) / alpha

        zx, zy, zz = zx * mask_f, zy * mask_f, zz * mask_f

        sx += qsm - zx
        sy += qsm - zy
        sz += qsm - zz

        z2 = (Wy + mu2*(FhDFx+s2)) / weight_mu2
        s2 += FhDFx - z2

    return (qsm * mask_f).numpy(), historia, alpha, t, mapas


# --- PROBLEMA Y DENOISERS DE JUGUETE ---------------------------------------------------------

def dipolo(n):
    """1/3 − k_x²/|k|², B0 en el último eje y el centro en el índice 0, como el de params.mat."""
    k = np.fft.fftfreq(n)
    kz, ky, kx = np.meshgrid(k, k, k, indexing="ij")
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k2[0, 0, 0] = 1.0
    D = 1 / 3 - kx ** 2 / k2
    D[0, 0, 0] = 0.0
    return D.astype(np.float32)


def problema_sintetico():
    rng = np.random.default_rng(0)
    c = np.mgrid[:N, :N, :N] - (N - 1) / 2
    cabeza = (c[0] / 8) ** 2 + (c[1] / 7) ** 2 + (c[2] / 6) ** 2 <= 1
    magnitud = np.where(cabeza, 1 + 0.2 * rng.standard_normal((N,) * 3), 0.01)
    chi = 0.05 * ndimage.gaussian_filter(rng.standard_normal((N,) * 3), 1.5) / 0.1 * cabeza
    D = dipolo(N)
    phase = np.real(np.fft.ifftn(D * np.fft.fftn(chi))) + 0.002 * rng.standard_normal((N,) * 3)
    datos = qsm3d.prepara_datos(phase, D, magnitud, chi)
    assert 0.1 < datos["mascara"].mean() < 0.5 and datos["mascara_eval"].sum() > 100
    return datos


@pytest.fixture(scope="module")
def sintetico():
    return problema_sintetico()


def juguete(eje, umbral=0.02):
    """Media móvil de 3 vóxeles a lo largo de `eje` más el residuo encogido por `umbral`."""
    def D(v):
        suave = (v.roll(1, eje) + v + v.roll(-1, eje)) / 3
        r = v - suave
        return suave + torch.sign(r) * torch.clamp(r.abs() - umbral, min=0)
    return D


JUGUETES = [juguete(0), juguete(1), juguete(2)]


# --- EQUIVALENCIA CON EL SCRIPT --------------------------------------------------------------

def test_admm_k3_es_el_script(sintetico):
    """Seis iteraciones sin corte: mismo chi (enmascarado), mismos updates, el NRMSE de cada
    iteración sobre su mapa enmascarado y mismo alpha, bit a bit (mismas operaciones en el mismo
    orden)."""
    chi, historia, alpha, _, mapas = bucle_del_script(sintetico, *JUGUETES, maxOuterIter=6)
    r = qsm3d.admm_wtdv(JUGUETES, sintetico, max_iter=6, device="cpu")
    mascara = torch.from_numpy(sintetico["mascara"]).float()

    assert r["motivo"] == "max_iter" and r["iteraciones"] == 6 and r["k"] == 3
    assert r["alpha"] == alpha
    assert np.array_equal(r["chi"], chi)
    assert r["x_update"] == [h[0] for h in historia]
    assert r["nrmse"] == [rmse_del_autor((m * mascara).numpy(), sintetico["chi"],
                                         sintetico["mascara_eval"]) for m in mapas]
    assert r["nrmse_final"] == r["nrmse"][-1]
    assert r["nrmse_min"] == min(r["nrmse"]) and r["nrmse"][r["it_min"]] == r["nrmse_min"]


def test_corte_mira_el_update_anterior(sintetico):
    """Con `tol` entre el update 2 y el 3 y `corte` debajo del 3, el script se da por convergido en
    t = 3 y corta en t = 4, con el chi de t = 4: no en t = 3, que es lo que daría mirar el update
    recién calculado."""
    sin_corte = qsm3d.admm_wtdv(JUGUETES, sintetico, max_iter=8, device="cpu")
    u = sin_corte["x_update"]
    assert all(a > b for a, b in zip(u[:5], u[1:5])), "el sintético dejó de bajar: elegir otro t"
    tol, corte = (u[2] + u[3]) / 2, u[3] / 2

    chi, historia, _, t_final, _ = bucle_del_script(sintetico, *JUGUETES, maxOuterIter=8,
                                                    tolUpdate=tol, finUpdate=corte)
    r = qsm3d.admm_wtdv(JUGUETES, sintetico, max_iter=8, tol=tol, corte=corte, device="cpu")

    assert t_final == 4 and len(historia) == 4
    assert r["motivo"] == "corte" and r["iteraciones"] == 5
    assert np.array_equal(r["chi"], chi)
    assert r["x_update"][:4] == [h[0] for h in historia]
    assert r["nrmse_final"] == pytest.approx(
        rmse_del_autor(chi, sintetico["chi"], sintetico["mascara_eval"]), rel=1e-6)


def test_sin_umbral_recorre_todas_las_iteraciones(sintetico):
    """Con `tol` = -inf, lo que usa la evaluación de train3d.py, el ADMM nunca se da por convergido
    y recorre sus `max_iter` aunque los umbrales de arriba corten: el mismo chi y las mismas
    trayectorias con cualquier `corte`."""
    base = qsm3d.admm_wtdv(JUGUETES, sintetico, max_iter=8, tol=-np.inf, device="cpu")
    u = base["x_update"]
    tol, corte = (u[2] + u[3]) / 2, u[3] / 2
    assert qsm3d.admm_wtdv(JUGUETES, sintetico, max_iter=8, tol=tol, corte=corte,
                           device="cpu")["motivo"] == "corte"
    r = qsm3d.admm_wtdv(JUGUETES, sintetico, max_iter=8, tol=-np.inf, corte=corte, device="cpu")
    assert base["motivo"] == r["motivo"] == "max_iter" and r["iteraciones"] == 8
    assert np.array_equal(r["chi"], base["chi"])
    assert r["x_update"] == u and r["nrmse"] == base["nrmse"]


def falla_en(malas):
    """El juguete del eje 1, salvo en las llamadas (desde 1) de `malas`, donde devuelve ceros. La
    llamada c es el prox de la iteración c - 1, así que mueve el NRMSE de la iteración c."""
    llamadas = []

    def D(v):
        llamadas.append(1)
        return torch.zeros_like(v) if len(llamadas) in malas else juguete(1)(v)
    return D


def admm_sin_umbral(denoiser, sintetico, max_iter, subidas=0):
    return qsm3d.admm_wtdv([denoiser], sintetico, max_iter=max_iter, tol=-np.inf,
                           subidas_para_cortar=subidas, device="cpu")


def test_para_tras_dos_subidas_con_el_mejor_chi(sintetico):
    """Desde la llamada 4 el denoiser se rompe: el NRMSE sube, baja una vez y vuelve a subir. Con
    `subidas_para_cortar` = 2, el ADMM para en la segunda subida seguida y devuelve el chi de menor
    NRMSE que vio, que no es el de dos iteraciones antes: el mismo, bit a bit, que el de un ADMM
    cortado en esa iteración sin la regla. La trayectoria guarda también las subidas."""
    malas = set(range(4, 99))
    n = admm_sin_umbral(falla_en(malas), sintetico, 10)["nrmse"]
    sube = [t > 0 and n[t] > n[t - 1] for t in range(len(n))]
    t_para = next(t for t in range(1, len(n)) if sube[t] and sube[t - 1])
    t_mejor = min(range(t_para + 1), key=n.__getitem__)
    assert t_mejor < t_para - 2, "el sintético cambió: el caso ya no distingue «el mejor» de «t-2»"

    r = admm_sin_umbral(falla_en(malas), sintetico, 10, subidas=2)
    hasta = admm_sin_umbral(falla_en(malas), sintetico, t_mejor + 1)
    assert r["motivo"] == "empeora" and r["iteraciones"] == t_para + 1
    assert r["nrmse"] == n[:t_para + 1]
    assert np.array_equal(r["chi"], hasta["chi"])
    assert r["nrmse_final"] == r["nrmse_min"] == n[t_mejor] and r["it_min"] == t_mejor


def test_una_subida_sola_no_corta(sintetico):
    """Una sola llamada mala: el NRMSE sube una iteración y vuelve a bajar. Con dos subidas para
    cortar, el ADMM sigue hasta el techo, igual que sin la regla."""
    base = admm_sin_umbral(falla_en({4}), sintetico, 10)
    n = base["nrmse"]
    assert sum(t > 0 and n[t] > n[t - 1] for t in range(len(n))) == 1 and n[-1] == min(n)
    r = admm_sin_umbral(falla_en({4}), sintetico, 10, subidas=2)
    assert r["motivo"] == "max_iter" and r["iteraciones"] == 10
    assert np.array_equal(r["chi"], base["chi"]) and r["nrmse"] == n and r["nrmse_final"] == n[-1]


def test_en_el_techo_devuelve_el_mejor_chi(sintetico):
    """Si la única subida es la última iteración, el ADMM llega al techo sin cortar, pero devuelve
    el chi de la anterior, el mejor: `nrmse_final` es siempre el mínimo de lo recorrido."""
    n = admm_sin_umbral(falla_en({7}), sintetico, 8)["nrmse"]
    assert n[-1] > n[-2] and all(b < a for a, b in zip(n[:-1], n[1:-1]))
    r = admm_sin_umbral(falla_en({7}), sintetico, 8, subidas=2)
    hasta = admm_sin_umbral(falla_en({7}), sintetico, 7)
    assert r["motivo"] == "max_iter" and r["iteraciones"] == 8
    assert np.array_equal(r["chi"], hasta["chi"]) and r["nrmse_final"] == n[-2]


def test_la_regla_no_toca_un_admm_que_siempre_baja(sintetico):
    """Si el NRMSE baja en cada iteración, la regla no actúa: el mismo ADMM, hasta el techo."""
    base = qsm3d.admm_wtdv(JUGUETES, sintetico, max_iter=8, tol=-np.inf, device="cpu")
    assert all(b < a for a, b in zip(base["nrmse"], base["nrmse"][1:])), "el sintético dejó de bajar"
    for subidas in (1, 2):
        r = qsm3d.admm_wtdv(JUGUETES, sintetico, max_iter=8, tol=-np.inf,
                            subidas_para_cortar=subidas, device="cpu")
        assert r["motivo"] == base["motivo"] == "max_iter" and r["iteraciones"] == 8
        assert np.array_equal(r["chi"], base["chi"]) and r["nrmse"] == base["nrmse"]
        assert r["nrmse_final"] == r["nrmse_min"] == base["nrmse"][-1]


def test_la_regla_pide_la_referencia(sintetico):
    datos = {k: v for k, v in sintetico.items() if k not in ("chi", "mascara_eval")}
    with pytest.raises(ValueError, match="chi"):
        qsm3d.admm_wtdv(JUGUETES, datos, max_iter=2, subidas_para_cortar=2, device="cpu")


def test_k1_es_k3_con_el_mismo_denoiser(sintetico):
    """Tres splits iguales son uno solo: (mu/3)·3F = mu·F salvo redondeo."""
    D = juguete(1)
    uno = qsm3d.admm_wtdv([D], sintetico, max_iter=6, device="cpu")
    tres = qsm3d.admm_wtdv([D, D, D], sintetico, max_iter=6, device="cpu")
    assert uno["k"] == 1 and tres["k"] == 3
    np.testing.assert_allclose(uno["chi"], tres["chi"], rtol=0,
                               atol=1e-6 * np.abs(tres["chi"]).max())
    np.testing.assert_allclose(uno["nrmse"], tres["nrmse"], rtol=1e-5)


def test_red_muerta_corta_sin_recorrer_max_iter(sintetico):
    r = qsm3d.admm_wtdv([lambda v: torch.full_like(v, float("nan"))], sintetico, max_iter=50,
                        device="cpu")
    assert r["motivo"] == "no_finito" and r["iteraciones"] == 2
    assert np.isnan(r["nrmse_final"]) and r["it_min"] == 0


def test_sin_verdad_de_terreno(sintetico):
    datos = {k: v for k, v in sintetico.items() if k not in ("chi", "mascara_eval")}
    r = qsm3d.admm_wtdv(JUGUETES, datos, max_iter=2, device="cpu")
    assert np.isnan(r["nrmse"]).all() and r["it_min"] == -1 and np.isfinite(r["chi"]).all()


# --- DATOS REALES ----------------------------------------------------------------------------

@pytest.mark.skipif(DIR_COSMOS is None, reason="sin ../tdv/data_mat")
def test_carga_cosmos_es_el_script():
    """`wTDV_QSM_torch_3d.py:19-51` sobre los .mat reales, textual: igualdad exacta."""
    phase_np = scipy.io.loadmat(DIR_COSMOS / 'params.mat')['phase_use'].astype(np.float32)
    kernel_np = scipy.io.loadmat(DIR_COSMOS / 'params.mat')['kernel'].astype(np.float32)
    K2_np = np.conj(kernel_np)*kernel_np
    weight_np = scipy.io.loadmat(DIR_COSMOS / 'params.mat')['magn_use'].astype(np.float32)
    chi_cosmos = scipy.io.loadmat(DIR_COSMOS / 'chi_cosmos.mat')['chi_cosmos'].astype(np.float32)
    cosmos_mask = chi_cosmos != 0
    brain_solid = weight_np > 0.05
    brain_solid = ndimage.binary_fill_holes(brain_solid)
    mean_brain = np.mean(weight_np[brain_solid])
    umbral_dinamico = 0.05 * mean_brain
    mag_threshold = weight_np > umbral_dinamico
    brain_clean = brain_solid & mag_threshold
    brain_closed = ndimage.binary_closing(brain_clean, structure=np.ones((3,3,3)), iterations=2)
    brain_final = ndimage.binary_fill_holes(brain_closed)
    refined_mask = ndimage.gaussian_filter(brain_final.astype(np.float32), sigma=1.0) > 0.5
    weight_np = weight_np * refined_mask
    weight_np /= weight_np[refined_mask].mean()
    weight_np *= weight_np
    Wy_np = weight_np * phase_np

    datos = qsm3d.carga_cosmos(DIR_COSMOS)
    esperado = {"phase": phase_np, "kernel": kernel_np, "K2": K2_np, "W2": weight_np,
                "Wy": Wy_np, "mascara": refined_mask, "chi": chi_cosmos,
                "mascara_eval": cosmos_mask}
    assert datos.keys() == esperado.keys()
    for clave, valor in esperado.items():
        assert datos[clave].dtype == valor.dtype, clave
        assert np.array_equal(datos[clave], valor), clave
    assert datos["phase"].shape == (160, 160, 160)


# --- DENOISER 3D -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vnet3d():
    config = train.vnet_config(train.T_INIT, channels=1)
    config["R"]["type"] = "tdv3d"
    config["R"]["config"]["num_features"] = 4
    torch.manual_seed(0)
    return VNet3D(config).eval()


def test_denoiser_3d_es_el_modelo_sobre_el_volumen(vnet3d):
    v = torch.rand(12, 13, 14)
    z = qsm3d.denoiser_3d(vnet3d)(v)
    with torch.no_grad():
        esperado = vnet3d(v[None, None], v[None, None])[-1][0, 0]
    assert z.shape == v.shape and z.dtype == torch.float32 and not z.requires_grad
    assert torch.equal(z, esperado)
    assert not torch.equal(z, v)


def test_denoiser_3d_en_bf16_devuelve_float32(vnet3d):
    v = torch.rand(12, 12, 12)
    z = qsm3d.denoiser_3d(vnet3d, autocast_dtype=torch.bfloat16)(v)
    assert z.dtype == torch.float32 and torch.isfinite(z).all()


def test_admm_k1_con_el_denoiser_3d(sintetico, vnet3d):
    r = qsm3d.admm_wtdv([qsm3d.denoiser_3d(vnet3d)], sintetico, max_iter=2, device="cpu")
    assert r["iteraciones"] == 2 and np.isfinite(r["nrmse"]).all()


# --- ESTABILIDAD -----------------------------------------------------------------------------

def vnet3d_chica(semilla):
    config = train.vnet_config(train.T_INIT, channels=1)
    config["R"]["type"] = "tdv3d"
    config["R"]["config"]["num_features"] = 4
    torch.manual_seed(semilla)
    return VNet3D(config)


@pytest.mark.parametrize("semilla, amplitud", [(0, 1.0), (1, 1.0), (3, 0.1)])
def test_estabilidad_es_el_autovalor_dominante_de_la_hessiana(semilla, amplitud):
    """En float64 sobre 6^3, la potencia contra `eigvalsh` de la Hessiana de la ENERGÍA armada
    entera con autograd, sin pasar por `R.grad`: lambda_max es el MÓDULO del autovalor dominante, y
    `rayleigh` su signo. Con la semilla 0 el dominante es negativo (-5.35) y con la 1 positivo
    (3.61); con 200 iteraciones la potencia llega a ~1e-15. Con la 3 a amplitud 0.1 hay dos de
    signo opuesto casi empatados (-2.6731 y 2.6705): lambda_max queda entre los dos módulos y el
    cociente de Rayleigh no converge a ninguno (0.57 con 200 iteraciones, 0.97 con 40), que es
    por qué el número no es el cociente de Rayleigh (medido el 2026-10-12)."""
    modelo = vnet3d_chica(semilla).double()
    x = amplitud * torch.rand(1, 1, 6, 6, 6, dtype=torch.float64,
                              generator=torch.Generator().manual_seed(semilla))
    H = torch.autograd.functional.hessian(lambda y: modelo.R.energy(y).sum(), x)
    H = H.reshape(6 ** 3, 6 ** 3)
    assert torch.allclose(H, H.T, rtol=0, atol=1e-12)
    w = torch.linalg.eigvalsh(H)
    dominante = w[w.abs().argmax()].item()
    assert (dominante < 0) == (semilla in (0, 3)), "el caso negativo dejó de estar cubierto"

    r = qsm3d.estabilidad(modelo, x, iters=200)
    assert (r["T"], r["S"], r["iteraciones"]) == (modelo.T.item(), modelo.S, 200)
    assert r["paso"] == r["T"] / r["S"] * r["lambda_max"]
    if semilla == 3:
        assert w.max().item() / abs(dominante) > 0.998, "el empate dejó de estar cubierto"
        assert w.max().item() * (1 - 1e-9) <= r["lambda_max"] <= abs(dominante) * (1 + 1e-9)
        assert abs(r["rayleigh"]) < 0.5 * abs(dominante)
        return
    assert r["lambda_max"] == pytest.approx(abs(dominante), rel=1e-9)
    assert r["rayleigh"] == pytest.approx(dominante, rel=1e-9)
    assert r["cambio_relativo"] < 1e-12


def test_estabilidad_no_toca_el_modelo_ni_los_generadores():
    """Con los `.grad` que deja una iteración de entrenamiento y un `R.grad` en la instancia, como
    el compilado de train3d.py, que no se tiene que llamar: los pesos, los `.grad` y el generador
    global quedan como estaban, y dentro de un autocast bf16 sin grad da lo mismo bit a bit, porque
    lo apaga. Un volumen `[D, H, W]` en numpy es el mismo punto."""
    modelo = vnet3d_chica(1)
    x = torch.rand(1, 1, 8, 8, 8, generator=torch.Generator().manual_seed(2))
    modelo(x, x)[-1].pow(2).sum().backward()
    pesos = {k: v.clone() for k, v in modelo.state_dict().items()}
    grads = {k: p.grad.clone() for k, p in modelo.named_parameters()}
    rng = torch.get_rng_state()

    def compilado(*args, **kw):
        raise AssertionError("usó el R.grad de la instancia")
    modelo.R.grad = compilado

    a = qsm3d.estabilidad(modelo, x)
    with torch.autocast("cpu", dtype=torch.bfloat16), torch.no_grad():
        b = qsm3d.estabilidad(modelo, x)
    c = qsm3d.estabilidad(modelo, x[0, 0].numpy())
    assert a["iteraciones"] == qsm3d.ITERS_ESTABILIDAD == 40
    for otro in (b, c):
        assert all(a[k] == otro[k] for k in a if k != "segundos")
    assert all(torch.equal(v, modelo.state_dict()[k]) for k, v in pesos.items())
    assert all(torch.equal(grads[k], p.grad) for k, p in modelo.named_parameters())
    assert torch.equal(rng, torch.get_rng_state())
    assert not x.requires_grad


def test_sin_tf32_apaga_y_restaura_los_flags():
    """En GPU, `estabilidad` corre con TF32 apagado y deja los flags como estaban, también si
    levanta. Sin GPU se prueba el contexto solo: los flags existen igual."""
    previos = (torch.backends.cudnn.conv.fp32_precision, torch.backends.cuda.matmul.fp32_precision)
    try:
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        with pytest.raises(RuntimeError, match="adentro"):
            with qsm3d._sin_tf32(True):
                assert torch.backends.cudnn.conv.fp32_precision == "ieee"
                assert torch.backends.cuda.matmul.fp32_precision == "ieee"
                raise RuntimeError("adentro")
        assert torch.backends.cudnn.conv.fp32_precision == "tf32"
        assert torch.backends.cuda.matmul.fp32_precision == "tf32"
        with qsm3d._sin_tf32(False):
            assert torch.backends.cudnn.conv.fp32_precision == "tf32"
    finally:
        torch.backends.cudnn.conv.fp32_precision, torch.backends.cuda.matmul.fp32_precision = (
            previos)


def test_el_denoiser_ve_solo_el_cerebro(sintetico):
    """Lo que recibe cada denoiser vale 0 fuera de `mascara` en todas las iteraciones, y el chi
    final también, aunque adentro no lo sea. Que la salida vaya por la máscara lo prueba
    `test_admm_k3_es_el_script`, bit a bit contra el script."""
    entradas = []

    def guarda(v):
        entradas.append(v.clone())
        return JUGUETES[0](v)
    r = qsm3d.admm_wtdv([guarda], sintetico, max_iter=4, device="cpu")
    afuera = ~sintetico["mascara"]
    assert len(entradas) == 4 and all((e.numpy()[afuera] == 0).all() for e in entradas)
    assert all(np.abs(e.numpy()[~afuera]).max() > 0 for e in entradas)
    assert (r["chi"][afuera] == 0).all() and np.abs(r["chi"][~afuera]).max() > 0


def test_punto_qsm3d_es_la_entrada_del_primer_denoiser(sintetico):
    """El volumen entero es, bit a bit, lo que recibe el primer denoiser de un ADMM de cuatro
    iteraciones, y un cubo es su ventana centrada en la caja de la máscara. Contra la fórmula
    cerrada en float64, M·chi_0 con chi_0 = F⁻¹[mu2·D·F(W²·phase/(W² + mu2))/(mu2·D² + mu)] y alpha
    con la desviación con n − 1 de `torch.std`: la del ADMM, no la `np.std` del script 2D."""
    entradas = []

    def guarda(v):
        entradas.append(v.clone())
        return JUGUETES[0](v)
    qsm3d.admm_wtdv([guarda], sintetico, max_iter=4, device="cpu")
    primera = entradas[0].numpy()
    assert len(entradas) == 4 and not np.array_equal(primera, entradas[1].numpy())

    entero = qsm3d.punto_qsm3d(sintetico, N)
    assert entero.dtype == np.float32
    assert np.array_equal(entero.view(np.uint32), primera.view(np.uint32))
    mascara = sintetico["mascara"]
    for lado in (8, 13):
        ventana = []
        for eje in range(3):
            idx = np.flatnonzero(mascara.any(axis=tuple(a for a in range(3) if a != eje)))
            inicio = (idx[0] + idx[-1] + 1) // 2 - lado // 2
            assert 0 <= inicio and inicio + lado <= N     # adentro: no hace falta correrla
            ventana.append(slice(inicio, inicio + lado))
        assert np.array_equal(qsm3d.punto_qsm3d(sintetico, lado), entero[tuple(ventana)])

    D, K2 = sintetico["kernel"].astype(np.float64), sintetico["K2"].astype(np.float64)
    z2 = sintetico["Wy"].astype(np.float64) / (sintetico["W2"].astype(np.float64) + qsm3d.MU2)
    chi_0 = np.real(np.fft.ifftn(qsm3d.MU2 * D * np.fft.fftn(z2) / (qsm3d.MU2 * K2 + qsm3d.MU)))
    alpha = qsm3d.TARGET_STD / chi_0[mascara].std(ddof=1)
    np.testing.assert_allclose(entero, alpha * chi_0 * mascara, rtol=0,
                               atol=1e-5 * np.abs(alpha * chi_0).max())
    assert entero[mascara].std(ddof=1) == pytest.approx(qsm3d.TARGET_STD, rel=1e-5)
    with pytest.raises(ValueError, match="lado"):
        qsm3d.punto_qsm3d(sintetico, N + 1)


# --- NRMSE -----------------------------------------------------------------------------------

def test_metricas_de_fansi_en_los_extremos():
    """HFEN y XSIM como `compute_hfen.m` y `compute_xsim.m` de FANSI: un mapa contra sí mismo da
    0 y 1, el LoG suma cero, y el HFEN no cambia con la escala de los dos mapas."""
    rng = np.random.default_rng(2)
    gt = ndimage.gaussian_filter(rng.standard_normal((24, 26, 28)), 2).astype(np.float32)
    m = np.ones(gt.shape, dtype=bool)
    m[:3] = False
    pred = gt + 0.1 * gt.std() * rng.standard_normal(gt.shape).astype(np.float32)
    assert abs(qsm3d._log_fansi().sum()) < 1e-12 and qsm3d._log_fansi().shape == (15, 15, 15)
    assert qsm3d.hfen_pct(gt, gt, m) == 0.0
    assert qsm3d.xsim(gt, gt, m) == pytest.approx(1.0)
    assert qsm3d.hfen_pct(3 * pred, 3 * gt, m) == pytest.approx(qsm3d.hfen_pct(pred, gt, m))
    assert 0 < qsm3d.hfen_pct(pred, gt, m) and qsm3d.xsim(pred, gt, m) < 1


def test_residuo_y_cono():
    """El residuo es 0 si la fase es exactamente D∗χ, y el factor del cono de un error blanco
    ronda 1."""
    rng = np.random.default_rng(3)
    k = np.meshgrid(*(np.fft.fftfreq(N),) * 3, indexing="ij")
    k2 = sum(ki ** 2 for ki in k)
    k2[0, 0, 0] = 1.0
    D = (1 / 3 - k[2] ** 2 / k2).astype(np.float32)
    D[0, 0, 0] = 0.0
    chi = ndimage.gaussian_filter(rng.standard_normal((N,) * 3), 1).astype(np.float32)
    M = np.ones(chi.shape, dtype=bool)
    datos = {"phase": np.real(np.fft.ifftn(D * np.fft.fftn(chi))).astype(np.float32),
             "kernel": D, "W2": np.ones_like(chi), "mascara": M, "chi": chi, "mascara_eval": M}
    assert qsm3d.residuo_dato_pct(chi, datos) == pytest.approx(0.0, abs=1e-4)
    blanco = chi + rng.standard_normal(chi.shape).astype(np.float32)
    assert qsm3d.factor_cono(blanco, chi, datos) == pytest.approx(1.0, abs=0.15)


def test_nrmse_pct_es_el_rmse_del_autor():
    rng = np.random.default_rng(1)
    gt = rng.standard_normal((9, 10, 11)).astype(np.float32)
    pred = gt + 0.3 * rng.standard_normal(gt.shape).astype(np.float32)
    mascara = rng.random(gt.shape) > 0.4
    assert qsm3d.nrmse_pct(pred, gt, mascara) == rmse_del_autor(pred, gt, mask=mascara)
    assert qsm3d.nrmse_pct(pred, gt) == rmse_del_autor(pred, gt)
    assert qsm3d.nrmse_pct(gt, gt, mascara) == 0.0
    assert qsm3d.nrmse_pct(np.zeros_like(gt), gt, mascara) == pytest.approx(100.0)
    assert qsm3d.nrmse_pct(2 * gt, gt, mascara) == pytest.approx(100.0)
    # Fuera de la máscara no cuenta.
    otro = pred.copy()
    otro[~mascara] = 1e3
    assert qsm3d.nrmse_pct(otro, gt, mascara) == qsm3d.nrmse_pct(pred, gt, mascara)


# --- LÁMINAS ---------------------------------------------------------------------------------

def test_cortes_centrales_son_los_de_imshow_3d():
    rng = np.random.default_rng(2)
    vol = rng.uniform(-0.15, 0.15, (10, 12, 14)).astype(np.float32)
    im = qsm3d.cortes_centrales(vol, (-0.1, 0.1), (-90, -90, 90))
    D, H, W = vol.shape
    cortes = [rotation_by_permutarion(vol[D // 2, :, :], -90),
              rotation_by_permutarion(vol[:, H // 2, :], -90),
              rotation_by_permutarion(vol[:, :, W // 2], 90)]
    alto = max(c.shape[0] for c in cortes)
    assert im.dtype == np.uint8
    assert im.shape == (alto, sum(c.shape[1] for c in cortes) + 2 * qsm3d.SEPARACION)
    x = 0
    for c in cortes:
        esperado = np.round(np.clip((c + 0.1) / 0.2, 0, 1) * 255).astype(np.uint8)
        assert np.array_equal(im[:c.shape[0], x:x + c.shape[1]], esperado)
        x += c.shape[1] + qsm3d.SEPARACION


def test_rango_satura_en_los_extremos():
    vol = np.zeros((6, 6, 6), dtype=np.float32)
    s = qsm3d.SEPARACION
    sin_separacion = lambda im: np.delete(im, [6 + j for j in range(s)]
                                          + [12 + s + j for j in range(s)], axis=1)
    assert (sin_separacion(qsm3d.cortes_centrales(vol - 1, (-0.1, 0.1))) == 0).all()
    assert (sin_separacion(qsm3d.cortes_centrales(vol + 1, (-0.1, 0.1))) == 255).all()
    assert (sin_separacion(qsm3d.cortes_centrales(vol, (-0.1, 0.1))) == 128).all()


def test_lamina_qsm():
    rng = np.random.default_rng(3)
    gt = rng.uniform(-0.1, 0.1, (16, 16, 16)).astype(np.float32)
    mascara = gt > -0.05
    gt = gt * mascara
    lam = qsm3d.lamina_qsm(gt, gt, mascara)
    fila = qsm3d.cortes_centrales(gt, qsm3d.RANGO_QSM)
    assert lam.dtype == np.uint8 and lam.ndim == 2
    assert lam.shape == (3 * fila.shape[0] + 2 * qsm3d.SEPARACION, fila.shape[1])
    # Reconstrucción perfecta: la fila del error es el gris del cero.
    error = lam[2 * (fila.shape[0] + qsm3d.SEPARACION):]
    assert np.array_equal(lam[:fila.shape[0]], fila)
    assert ((error == 128) | (error == 255)).all() and (error == 128).any()


def test_lamina_denoising():
    rng = np.random.default_rng(4)
    limpia = rng.random((1, 1, 16, 16, 16)).astype(np.float32)
    ruidosa = torch.from_numpy(limpia + 0.1 * rng.standard_normal(limpia.shape).astype(np.float32))
    lam = qsm3d.lamina_denoising(limpia, ruidosa, limpia)
    fila = qsm3d.cortes_centrales(limpia, qsm3d.RANGO_DENOISING, qsm3d.ORIENTACION_T1)
    assert lam.dtype == np.uint8
    assert lam.shape == (4 * fila.shape[0] + 3 * qsm3d.SEPARACION, fila.shape[1])
    # Salida igual a la limpia: el |error| es negro.
    error = lam[3 * (fila.shape[0] + qsm3d.SEPARACION):]
    assert ((error == 0) | (error == 255)).all() and (error == 0).any()
    with pytest.raises(ValueError):
        qsm3d.lamina_denoising(limpia[0, 0, 0], limpia, limpia)
    # Una salida en bf16 (autocast de `rapido`): numpy no tiene bf16 y `np.asarray` levantaba.
    en_bf16 = qsm3d.lamina_denoising(limpia, ruidosa, torch.from_numpy(limpia).bfloat16())
    assert en_bf16.dtype == np.uint8 and en_bf16.shape == lam.shape
