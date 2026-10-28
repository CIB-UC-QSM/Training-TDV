"""La red 2D oficial de TDV (VLOGroup, color, BSDS400) como prior del mismo ADMM que la red 3D, con
el mismo protocolo del abstract ISMRM (28-10): la máscara del desafío en COSMOS, MaskBrainExtracted (la del desafío 2.0, `sujetos.py`) en
Sim2, término de datos lineal con el peso del autor, μ₂ = 1, 100 iteraciones sin corte.

    cd .claude/worktrees/abstract-ismrm
    CUDA_VISIBLE_DEVICES=1 taskset -c 16-31 ../../../.venv/bin/python \
        herramientas_abstract/grid_tdv2d.py {2d|pseudo3d} [sujeto]

"2d" es TDV-QSM (Milovic et al., ISMRM 2021): tríos de cortes axiales consecutivos como canales
RGB, cada corte promediado sobre los tríos que lo contienen (`process_axis` de `../tdv/ddr/utils.py`,
eje (2, 0, 1)). "pseudo3d" es el de `../tdv/wTDV_QSM_torch_3d.py`: lo mismo en los tres ejes, un
split por eje (k = 3), en el orden de su suma. La red recibe α·M·(χ + s), como la 3D.

Grid: la diagonal (k = μ/α fijo) por el punto ajustado de la oficial en `../tdv` (target 0.06,
μ 0.008) y la ortogonal por el mejor s (media de los dos sujetos, NRMSE en t = 100), como
`scripts/grid_msk_27_10.py`. NRMSE en cada iteración, XSIM cada 25, NRMSE y XSIM por erosión de
la máscara de evaluación (0…12) en t = 100, y los segundos. JSON tras cada configuración; χ en
t = 100 de cada una en un npz (float16).
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

RAMA = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(RAMA))
from ddr import qsm3d  # noqa: E402
sys.path.insert(0, str(RAMA / "herramientas_abstract"))
import sujetos  # noqa: E402
from evaluate import load_vnet  # noqa: E402

assert "desafio" in qsm3d.MASCARAS_COSMOS, f"{qsm3d.__file__} no trae la máscara del desafío"
OFICIAL = RAMA / "checkpoints/Oficiales/tdv3-3-25-f32-color.pth"
SALIDAS = RAMA / "runs/abstract-ismrm"
CENTRO = (0.06, 0.008)                     # (target, μ) del 2D oficial en ../tdv/README.md
DIAGONAL = (0.5, 0.71, 1.0, 1.41, 2.0)
ORTOGONAL = (0.5, 0.71, 1.41, 2.0)
MAX_ITER = 100
CADA_XSIM = 25
EROSIONES = 13
LOTE = int(__import__("os").environ.get("LOTE", "128"))
# El orden de la suma de `wTDV_QSM_torch_3d.py`: (x, y, z) = perm (0,1,2), (1,0,2), (2,0,1).
EJES = {"2d": [(2, 0, 1)], "pseudo3d": [(0, 1, 2), (1, 0, 2), (2, 0, 1)]}


def carga_sujetos() -> dict:
    return sujetos.carga_sujetos()


def denoiser_eje(vn, perm: tuple, amp=torch.bfloat16):
    """`process_axis` con α = 1 (la entrada ya viene escalada): tríos de cortes a lo largo del
    primer eje de `perm`, cada corte promediado sobre los tríos que lo contienen."""
    inversa = [0, 0, 0]
    for i, p in enumerate(perm):
        inversa[p] = i

    def denoiser(v: torch.Tensor) -> torch.Tensor:
        vp = v.permute(*perm).contiguous()
        D = vp.shape[0]
        acum = torch.zeros_like(vp)
        cuenta = torch.zeros((D, 1, 1), device=v.device)
        for d in (0, 1, 2):
            cuenta[d:D - 2 + d] += 1
        trios = vp.unfold(0, 3, 1).permute(0, 3, 1, 2)            # (D − 2, 3, H, W)
        for b0 in range(0, D - 2, LOTE):
            b1 = min(b0 + LOTE, D - 2)
            lote = trios[b0:b1].contiguous().to(memory_format=torch.channels_last)
            with torch.no_grad(), torch.autocast("cuda", dtype=amp):
                out = vn(lote, lote)[-1].float()
            for d in (0, 1, 2):
                acum[b0 + d:b1 + d] += out[:, d]
        return (acum / cuenta).permute(*inversa).contiguous()
    return denoiser


def erosiones(m: np.ndarray) -> list[np.ndarray]:
    eros = [m]
    for _ in range(EROSIONES - 1):
        eros.append(ndimage.binary_erosion(eros[-1]))
    return eros


def xsim_mapa(pred, gt, m) -> np.ndarray:
    """El mapa de `qsm3d.xsim`, para promediarlo en otra máscara (de `grid_msk_27_10.py`)."""
    p, g = np.where(m, pred, 0.0).astype(np.float64), np.where(m, gt, 0.0).astype(np.float64)

    def f(v):
        return ndimage.gaussian_filter(v, 1.5, mode="nearest", truncate=2.0)
    c1, c2 = 0.01 ** 2, 0.001 ** 2
    m1, m2 = f(p), f(g)
    s1, s2, s12 = f(p * p) - m1 * m1, f(g * g) - m2 * m2, f(p * g) - m1 * m2
    return ((2 * m1 * m2 + c1) * (2 * s12 + c2)) / ((m1 * m1 + m2 * m2 + c1) * (s1 + s2 + c2))


def por_profundidad(chi, gt, eros) -> dict:
    mapa = xsim_mapa(chi, gt, eros[0])
    capas = [eros[e] & ~eros[e + 1] for e in range(EROSIONES - 1)] + [eros[-1]]
    return {"nrmse": [qsm3d.nrmse_pct(chi, gt, m) for m in eros],
            "xsim": [float(mapa[m].mean()) for m in eros],
            "rmse_capa": [float(np.sqrt(np.mean((chi[c] - gt[c]) ** 2))) for c in capas]}


def admm(denoisers, datos, *, mu: float, target: float, device) -> dict:
    """El bucle de `qsm3d.admm_wtdv` con k splits, sin corte, con el NRMSE en cada iteración y el
    XSIM cada CADA_XSIM."""
    kernel, K2, W2, Wy, mask = (torch.from_numpy(np.ascontiguousarray(datos[c])).to(device)
                                for c in ("kernel", "K2", "W2", "Wy", "mascara"))
    gt, me = datos["chi"], datos["mascara_eval"]
    mf = mask.float()
    N, k = kernel.shape, len(denoisers)
    z = [torch.zeros(N, device=device) for _ in range(k)]
    s = [torch.zeros(N, device=device) for _ in range(k)]
    qsm = torch.zeros(N, device=device)
    W2mu2 = W2 + qsm3d.MU2
    den = qsm3d.MU2 * K2 + mu
    z2, s2 = Wy / W2mu2, torch.zeros(N, device=device)
    nrmse, upd, xs = [], [], {}
    alpha, chi, t0 = 1.0, None, time.time()
    for t in range(MAX_ITER):
        viejo = qsm.clone()
        suma = sum(torch.fft.fftn(zi - si) for zi, si in zip(z, s))
        qsm = torch.real(torch.fft.ifftn((qsm3d.MU2 * kernel * torch.fft.fftn(z2 - s2)
                                          + (mu / k) * suma) / den)).float()
        if t == 0:
            alpha = target / torch.std(qsm[mask]).item()
        upd.append((100 * torch.sqrt(torch.mean((qsm - viejo) ** 2))
                    / torch.sqrt(torch.mean(qsm ** 2))).item())
        chi = (qsm * mf).cpu().numpy()
        nrmse.append(qsm3d.nrmse_pct(chi, gt, me))
        if (t + 1) % CADA_XSIM == 0:
            xs[str(t + 1)] = qsm3d.xsim(chi, gt, me)
        if not math.isfinite(upd[-1]):
            break
        FhDFx = torch.real(torch.fft.ifftn(kernel * torch.fft.fftn(qsm))).float()
        for i, d in enumerate(denoisers):
            z[i] = d(alpha * ((qsm + s[i]) * mf)) / alpha * mf
        for i in range(k):
            s[i] += qsm - z[i]
        z2 = (Wy + qsm3d.MU2 * (FhDFx + s2)) / W2mu2
        s2 += FhDFx - z2
    torch.cuda.synchronize()
    seg = time.time() - t0
    return {"alpha": alpha, "k": mu / alpha, "nrmse": nrmse, "update": upd, "xsim": xs,
            "chi": chi, "segundos": seg, "s_por_iter": seg / len(nrmse)}


def verifica(denoisers, datos, device) -> float:
    """4 iteraciones contra `qsm3d.admm_wtdv`: la diferencia máxima de NRMSE (pp)."""
    ref = qsm3d.admm_wtdv(denoisers, datos, device=device, mu=CENTRO[1], target_std=CENTRO[0],
                          max_iter=4, tol=-math.inf)
    global MAX_ITER
    guardado, MAX_ITER = MAX_ITER, 4
    mio = admm(denoisers, datos, mu=CENTRO[1], target=CENTRO[0], device=device)
    MAX_ITER = guardado
    return float(np.max(np.abs(np.array(ref["nrmse"]) - np.array(mio["nrmse"]))))


def main() -> None:
    """Argumentos: modo, [sujeto o "-" para los dos], [configuraciones "t/μ,t/μ" en lugar del
    grid]. Cada χ va a su propio npz (float32)."""
    modo = sys.argv[1]
    solo = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    explicitas = ([tuple(float(x) for x in c.split("/")) for c in sys.argv[3].split(",")]
                  if len(sys.argv) > 3 else None)
    SALIDAS.mkdir(parents=True, exist_ok=True)
    etiqueta = (f"-{solo}" if solo else "") + __import__("os").environ.get("SUFIJO", "")
    salida = SALIDAS / f"28-10-grid-tdv{modo}{etiqueta}.json"
    res = json.loads(salida.read_text()) if salida.exists() else {"modo": modo}
    torch.backends.cudnn.benchmark = __import__("os").environ.get("BENCH") == "1"
    device = torch.device("cuda")
    vn, _ = load_vnet(OFICIAL, device=device)
    vn = vn.to(memory_format=torch.channels_last)
    dens = [denoiser_eje(vn, p) for p in EJES[modo]]
    sujetos = carga_sujetos()
    if solo:
        sujetos = {solo: sujetos[solo]}
    eros = {n: erosiones(d["mascara_eval"]) for n, d in sujetos.items()}
    if "verificacion_pp" not in res:
        res["verificacion_pp"] = verifica(dens, carga_sujetos()["COSMOS"], device)
        print(f"verificación contra admm_wtdv: {res['verificacion_pp']:.2e} pp", flush=True)


    def corre(etapa: str, target: float, mu: float) -> dict:
        clave = f"{target:.4g}/{mu:.4g}"
        fila = res.setdefault("configs", {}).setdefault(clave, {"target": target, "mu": mu,
                                                               "etapa": etapa})
        for nombre, datos in sujetos.items():
            if nombre in fila:
                continue
            with torch.no_grad():
                r = admm(dens, datos, mu=mu, target=target, device=device)
            chi = r.pop("chi")
            fila[nombre] = {**r, "nrmse_100": r["nrmse"][-1], "xsim_100": r["xsim"].get("100"),
                            "profundidad": por_profundidad(chi, datos["chi"], eros[nombre])}
            torch.cuda.empty_cache()
            salida.write_text(json.dumps(res, indent=1))
            np.savez(SALIDAS / f"28-10-chi-tdv{modo}-{nombre}-{target:.4g}_{mu:.4g}.npz", chi=chi)
            e = fila[nombre]
            print(f"{etapa:12s} {clave:14s} {nombre:6s} k={e['k']:.4g} | t=100: NRMSE "
                  f"{e['nrmse_100']:.2f} %, XSIM {e['xsim_100']:.4f}, update {e['update'][-1]:.2f}"
                  f" % | e=4: {e['profundidad']['nrmse'][4]:.2f} % {e['profundidad']['xsim'][4]:.4f}"
                  f" | {e['s_por_iter']:.2f} s/it", flush=True)
        return fila

    if explicitas:
        for t, mu in explicitas:
            corre("punto", t, mu)
        res["listo"] = True
        salida.write_text(json.dumps(res, indent=1))
        print("listo", flush=True)
        return
    t_c, mu_c = CENTRO
    diag = {s: corre("diagonal", t_c * s, mu_c * s) for s in DIAGONAL}
    media = {s: np.mean([f[n]["nrmse_100"] for n in sujetos]) for s, f in diag.items()}
    s_mejor = min(media, key=media.get)
    res["s_mejor"] = s_mejor
    print(f"diagonal: mejor s = {s_mejor} (media {media[s_mejor]:.2f} %)", flush=True)
    for f in ORTOGONAL:
        corre(f"ortogonal@{s_mejor}", t_c * s_mejor * f, mu_c * s_mejor / f)
    res["listo"] = True
    salida.write_text(json.dumps(res, indent=1))
    print("listo", flush=True)


if __name__ == "__main__":
    main()
