"""FANSI-TV lineal pesado (el `wTV.m` de FANSI) sobre COSMOS y Sim2, con el término de datos, las
máscaras y la evaluación del ADMM de la red 3D (`qsm3d.admm_wtdv`), para el abstract ISMRM (28-10).

    cd .claude/worktrees/abstract-ismrm
    taskset -c 16-31 ../../../.venv/bin/python herramientas_abstract/tv_fansi.py [hilos] [sujeto]

El TV de FANSI (Milovic et al., MRM 2018): split z = ∇χ (diferencias hacia adelante, periódicas)
con umbral suave por componente (anisótropo) y peso μ₁; split z2 = D∗χ con el peso W² del autor
(la magnitud normalizada a media 1 en la máscara, al cuadrado) y μ₂ = 1, el mismo z2 que el ADMM de
la red. La red ve M·(χ + s); el TV no tiene máscara adentro, como `wTV.m` (afuera W² = 0). El mapa
que se evalúa es M·χ, como el de la red; NRMSE en cada iteración sobre `mascara_eval` y XSIM en
los puntos de PUNTOS. Grid α_TV × k = μ₁/α_TV (la forma μ = k·α del supervisor, la de FANSI), con
TOPE iteraciones; JSON tras cada configuración, y se saltea lo hecho.

`ISO=1` usa el umbral isótropo; `NL=1`, el término de datos no lineal de FANSI (el z2 de
`qsm3d._z2_nolineal`, la fase en radianes con el `FASE_POR_PPM` de cada sujeto y el peso con máximo
1, como `admm_wtdv(datos_modelo="nolineal")`). Cada configuración deja un npz con `chi` (t = 100) y
`chi300` (t = TOPE).
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

RAMA = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(RAMA))
from ddr import qsm3d  # noqa: E402
sys.path.insert(0, str(RAMA / "herramientas_abstract"))
import sujetos  # noqa: E402

assert "desafio" in qsm3d.MASCARAS_COSMOS, f"{qsm3d.__file__} no trae la máscara del desafío"
SALIDA = RAMA / "runs/abstract-ismrm/28-10-tv-fansi.json"
ALFAS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
KS = (10.0, 30.0, 100.0, 300.0)
TOPE = 300
PUNTOS = (100, 200, 300)
MU2 = 1.0
ISOTROPO = __import__("os").environ.get("ISO") == "1"   # TV isótropo en lugar del de `wTV.m`
NOLINEAL = __import__("os").environ.get("NL") == "1"  # el z2 de `nlTV.m` (`qsm3d._z2_nolineal`)
ESCALAS = {"COSMOS": qsm3d.FASE_POR_PPM_COSMOS, "Sim2": qsm3d.FASE_POR_PPM_SIM2}


def carga_sujetos() -> dict:
    return sujetos.carga_sujetos()


def _grad(x: torch.Tensor) -> list[torch.Tensor]:
    return [torch.roll(x, -1, eje) - x for eje in range(3)]          # x[i+1] − x[i]


def _grad_t(g: list[torch.Tensor]) -> torch.Tensor:
    return sum(torch.roll(gi, 1, eje) - gi for eje, gi in enumerate(g))   # el adjunto de _grad


def admm_tv(datos: dict, alfa_tv: float, mu1: float, *, tope: int = TOPE,
            puntos: tuple = PUNTOS, guarda: tuple = ()) -> dict:
    """wTV lineal: min ½‖W(D∗χ − φ)‖² + α_TV‖∇χ‖₁ por ADMM. Devuelve las trayectorias y, en cada
    t de `guarda`, M·χ."""
    t0 = time.perf_counter()
    kernel = torch.from_numpy(np.ascontiguousarray(datos["kernel"]))
    W2 = torch.from_numpy(np.ascontiguousarray(datos["W2"]))
    Wy = torch.from_numpy(np.ascontiguousarray(datos["Wy"]))
    escala = 1.0
    if NOLINEAL:
        # Como `qsm3d.admm_wtdv(datos_modelo="nolineal")`: fase en radianes, peso con máximo 1;
        # α_TV y μ₁ quedan en esas unidades, y el mapa vuelve a ppm.
        escala = float(datos["escala"])
        fase = torch.from_numpy(np.ascontiguousarray(datos["phase"])) * escala
        W2 = W2 / W2.max()
        Wy = W2 * fase
    Mf = torch.from_numpy(datos["mascara"]).float()
    gt, Me = datos["chi"], datos["mascara_eval"]
    N = kernel.shape
    EtE = torch.zeros(N)
    for eje, n in enumerate(N):
        forma = [1, 1, 1]
        forma[eje] = n
        EtE = EtE + (2 - 2 * torch.cos(2 * math.pi * torch.arange(n) / n)).reshape(forma)
    denominador = mu1 * EtE + MU2 * kernel * kernel
    denominador.view(-1)[0] = 1.0          # la continua: numerador y denominador son 0
    x = torch.zeros(N)
    z = [torch.zeros(N) for _ in range(3)]
    s = [torch.zeros(N) for _ in range(3)]
    z2 = Wy / (W2 + MU2)
    s2 = torch.zeros(N)
    umbral = alfa_tv / mu1
    nrmse, update, xsims, mapas = [], [], {}, {}
    for t in range(1, tope + 1):
        x_viejo = x
        num = (mu1 * torch.fft.fftn(_grad_t([zi - si for zi, si in zip(z, s)]))
               + MU2 * kernel * torch.fft.fftn(z2 - s2))
        x = torch.real(torch.fft.ifftn(num / denominador)).float()
        update.append(float(100 * torch.sqrt(torch.mean((x - x_viejo) ** 2))
                            / torch.sqrt(torch.mean(x ** 2))))
        chi = (x * Mf).numpy() / escala
        nrmse.append(qsm3d.nrmse_pct(chi, gt, Me))
        if t in puntos:
            xsims[t] = qsm3d.xsim(chi, gt, Me)
        if t in guarda:
            mapas[t] = chi.copy()
        if not math.isfinite(nrmse[-1]):
            break
        gx = _grad(x)
        g = [gi + si for gi, si in zip(gx, s)]
        if ISOTROPO:            # umbral sobre la norma del gradiente en cada vóxel
            fac = torch.clamp(1 - umbral / torch.sqrt(sum(gi * gi for gi in g)).clamp_min(1e-12),
                              min=0)
            z = [gi * fac for gi in g]
        else:                   # umbral suave por componente, como `wTV.m`
            z = [torch.sign(gi) * torch.clamp(gi.abs() - umbral, min=0) for gi in g]
        Dx = torch.real(torch.fft.ifftn(kernel * torch.fft.fftn(x))).float()
        if NOLINEAL:
            z2 = qsm3d._z2_nolineal(MU2 * (Dx + s2), fase, W2, MU2)
        else:
            z2 = (Wy + MU2 * (Dx + s2)) / (W2 + MU2)
        s = [si + gi - zi for si, gi, zi in zip(s, gx, z)]
        s2 = s2 + Dx - z2
    i_min = int(np.nanargmin(nrmse))
    return {"alfa_tv": alfa_tv, "mu1": mu1, "k": mu1 / alfa_tv, "nrmse": nrmse,
            "update": update, "xsim": {str(k): v for k, v in xsims.items()},
            "nrmse_min": nrmse[i_min], "t_min": i_min + 1, "segundos": time.perf_counter() - t0,
            "mapas": mapas}


def main() -> None:
    """Argumentos: hilos, [sujeto], [α "a1,a2"], [k "k1,k2"]. Con α y k dados (el grid fino),
    guarda además M·χ en t = 100 en un npz por configuración y el error por erosión y por capa
    en t = 100 y en el final, con las funciones de `grid_tdv2d.py`."""
    hilos = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    solo = sys.argv[2] if len(sys.argv) > 2 else None
    fino = len(sys.argv) > 4
    alfas = [float(x) for x in sys.argv[3].split(",")] if fino else ALFAS
    ks = [float(x) for x in sys.argv[4].split(",")] if fino else KS
    torch.set_num_threads(hilos)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    salida = SALIDA.with_name(SALIDA.stem + (f"-{solo}" if solo else "")
                              + ("-fino" if fino else "") + ("-iso" if ISOTROPO else "")
                              + ("-nl" if NOLINEAL else "") + ".json")
    hecho = json.loads(salida.read_text()) if salida.exists() else {}
    sujetos = carga_sujetos()
    if fino:
        from grid_tdv2d import erosiones, por_profundidad
    for nombre, datos in sujetos.items():
        if solo and nombre != solo:
            continue
        datos["escala"] = ESCALAS[nombre]
        eros = erosiones(datos["mascara_eval"]) if fino else None
        for k in ks:
            for a in alfas:
                clave = f"{nombre}/{a:g}/{k:g}"
                if clave in hecho:
                    continue
                r = admm_tv(datos, a, k * a, guarda=(100, TOPE) if fino else ())
                mapas = r.pop("mapas")
                if fino:
                    r["profundidad"] = {str(t): por_profundidad(m, datos["chi"], eros)
                                        for t, m in mapas.items()}
                    np.savez(SALIDA.parent / f"28-10-chi-tv{'iso' if ISOTROPO else ''}{'nl' if NOLINEAL else ''}-{nombre}-{a:g}_{k:g}.npz",
                             chi=mapas[100], chi300=mapas[TOPE])
                hecho[clave] = r
                salida.write_text(json.dumps(hecho))
                print(f"{clave}: NRMSE t100 {r['nrmse'][min(99, len(r['nrmse']) - 1)]:.2f} final {r['nrmse'][-1]:.2f} "
                      f"min {r['nrmse_min']:.2f} (t={r['t_min']}) XSIM {r['xsim']} "
                      f"{r['segundos']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
