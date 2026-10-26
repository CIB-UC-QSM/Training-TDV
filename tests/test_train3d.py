"""El entrenador 3D: que la receta sea la de `train.py`, que el lr salga de un calendario puro y
que una corrida reanudada desde su estado sea, bit a bit, la corrida seguida.

Las corridas van en subprocesos, en CPU y en `referencia`, sobre volúmenes sintéticos en
`tmp_path` y con la red achicada por banderas de test (`--features 4 --patch 16 --eval-recorte
16`); que esas banderas no cambien los defaults lo verifica `test_los_defaults_son_la_receta`. Un
proceso por corrida, como en el uso real: la reanudación no hereda nada de la memoria del otro.
"""

import glob
import importlib.util
import inspect
import itertools
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.io
import torch
from scipy import ndimage
from torch.nn.utils import get_total_norm

import train
import train3d
from ddr import qsm3d, seguimiento
from ddr.data3d import VolumeDataset3D, add_noise, psnr
from ddr.model import VNet3D
from evaluate import evaluate_denoising3d

REPO = Path(__file__).resolve().parents[1]
# Con la CPU compartida, los hilos por defecto de torch vuelven estas redes chicas ~30x más lentas
# (medido el 2026-10-08). Todas las corridas usan los mismos, así que la comparación no depende de
# cuántos sean.
ENTORNO = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4"}
N = 12                                   # iteraciones de la corrida seguida
K = 6                                    # el corte de la rama
EVALUA = 3                               # divide a los dos cortes: la última de `k` no sobra
LOGUEA = 7                               # no los divide: el max|g| del log cruza el corte
HITOS = ["--hitos", "3", "6", "9"]       # uno antes de cada corte, uno en la 6 y otro en la 9
# Disyuntor forzado: ventana de 1 y razón 1 rechazan todo lote más alto que el último aceptado, y
# dos seguidos rebobinan al único respaldo (el de la it 1). Así hay rebobinados a los dos lados del
# corte con pasos aceptados entre medio, que es donde vive el alias de `rebobina()`.
FORZADO = ["--guard", "1", "--guard-ventana", "1", "--guard-seguidos", "2", "--guard-limpias",
           "1", "--guard-respaldo", "1000"]
VARIANTES = {
    # Respaldo cada 4 iteraciones: el de la 7 cuenta limpias de los dos lados del corte en la 6.
    # Con la ventana de 2 el relativo se arma en la 2, y el respaldo se refresca desde ahí.
    "receta": HITOS + ["--guard-respaldo", "4", "--guard-ventana", "2"],
    "disyuntor": HITOS + FORZADO + ["--sigma-max", "55", "--aug-signo", "--aug-ganancia", "2"],
}
# (variante, corte). El disyuntor se corta en la 6 justo tras un rebobinado (con el `rebobina()` de
# hoy, Adam y el respaldo comparten tensores) y en la 9 con una racha de rechazos a medias.
CASOS = [("receta", 6), ("disyuntor", 6), ("disyuntor", 9)]


def volumen(rng, forma):
    """Una «cabeza»: textura suave en [0.2, 1] dentro de un elipsoide, aire afuera."""
    ejes = np.meshgrid(*(np.linspace(-1, 1, n) for n in forma), indexing="ij")
    cabeza = sum(e ** 2 for e in ejes) < 0.8
    textura = ndimage.gaussian_filter(rng.standard_normal(forma), 1.5)
    textura = (textura - textura.min()) / (textura.max() - textura.min())
    return np.where(cabeza, 0.2 + 0.8 * textura, 0.0)


def arma_datos(raiz):
    """El layout de `datasets/ldm100k3d`: `images/<split>/*.npy` en float16 y sus listas."""
    rng = np.random.default_rng(0)
    for split, n in (("train", 3), ("val", 2)):
        carpeta = raiz / "images" / split
        carpeta.mkdir(parents=True)
        nombres = [f"sub-{split}{i}" for i in range(n)]
        for nombre in nombres:
            # Ejes distintos: una transpuesta confundida no pasa.
            np.save(carpeta / f"{nombre}.npy", volumen(rng, (24, 26, 28)).astype(np.float16))
        (raiz / f"lista_{split}.txt").write_text("".join(f"{x}\n" for x in nombres))
    # Sin W&B y sin evaluación QSM: las prenden sólo los tests de abajo que las miran.
    return ["--data-root", str(raiz), "--train-list", str(raiz / "lista_train.txt"),
            "--eval-list", str(raiz / "lista_val.txt"), "--mode", "referencia", "--sin-compilar",
            "--workers", "0", "--features", "4", "--patch", "16", "--eval-recorte", "16",
            "--lote", "2", "--micro-lote", "1", "--log-every", str(LOGUEA),
            "--eval-every", str(EVALUA), "--iterations", str(N), "--seguimiento", "no",
            "--qsm-datos", ""]


def lanza(base, salida, *banderas, codigo=None, entorno=ENTORNO):
    """`train3d.py` en un subproceso; con `codigo`, ese programa (`python -c`) con las banderas."""
    programa = ["-c", codigo] if codigo else [str(REPO / "train3d.py")]
    return subprocess.Popen([sys.executable, *programa, *base, "--out", str(salida), *banderas],
                            cwd=REPO, env=entorno, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def espera(*procesos, codigo=0):
    """Espera a todos y devuelve `(stdout, stderr)` de cada uno."""
    salidas = []
    for p in procesos:
        out, err = p.communicate()
        assert p.returncode == codigo, f"código {p.returncode}\n{out[-2000:]}\n{err[-3000:]}"
        salidas.append((out, err))
    return salidas


def log(*salidas):
    """Las líneas de iteración, de rechazo y de evaluación, sin el reloj ni el total pedido."""
    return [re.sub(r" \(\d+\.\dh restantes\)|/\d+\]", "", linea)
            for out in salidas for linea in out.splitlines()
            if linea.startswith(("[", "    eval"))]


def estado(carpeta, it):
    return torch.load(carpeta / f"estado-{it:06d}.pth", map_location="cpu", weights_only=False)


def traza(carpeta):
    return np.load(carpeta / "traza.npy")


def bit_a_bit(a, b):
    """`np.array_equal` sobre los bits: un NaN es igual a otro NaN sólo si es el mismo patrón."""
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def igual(a, b):
    """Igualdad exacta y recursiva de dos estados: tensores con `torch.equal`, la traza bit a bit,
    NaN con NaN."""
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a, b))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and bit_a_bit(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(igual(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(map(igual, a, b))
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return a == b


@pytest.fixture(scope="module")
def corridas(tmp_path_factory):
    """Todas las corridas del módulo, lanzadas en paralelo por etapas.

    Por variante, `seguida` hace N iteraciones de un tirón. Por caso, `reanudada` hace k, guarda su
    estado, y otro proceso sigue desde ahí hasta N en el mismo `--out`. Además, una rama con otro
    calendario desde una copia suelta del estado K de la seguida del disyuntor (sin la `traza.npy`
    de su directorio), tres reanudaciones que se tienen que rechazar, una que cambia el micro-lote
    (como al seguir en otra tarjeta), una corrida con el lote en un solo micro-lote, otra de una
    iteración con un parche por lote y otra con `--micro-lote auto`.
    """
    raiz = tmp_path_factory.mktemp("train3d")
    base = arma_datos(raiz / "datos")
    seguida = {v: raiz / v / "seguida" for v in VARIANTES}
    reanudada = {(v, k): raiz / v / f"reanudada-{k}" for v, k in CASOS}

    # La seguida del disyuntor guarda su estado en cada iteración: el test del respaldo los lee.
    salidas = espera(
        *[lanza(base, seguida[v], *VARIANTES[v],
                "--guardar-cada", "1" if v == "disyuntor" else str(N)) for v in VARIANTES],
        *[lanza(base, reanudada[v, k], *VARIANTES[v], "--iterations", str(k),
                "--guardar-cada", str(k)) for v, k in CASOS],
        lanza(base, raiz / "un-micro-lote", "--micro-lote", "2", "--iterations", "2"),
        lanza(base, raiz / "un-parche", "--lote", "1", "--iterations", "1"),
        lanza(base, raiz / "auto", "--lote", "4", "--micro-lote", "auto", "--vram-gb", "1.5",
              "--iterations", "1"))
    logs = {v: log(out) for v, (out, _) in zip(VARIANTES, salidas)}
    primeras = {c: out for c, (out, _) in zip(CASOS, salidas[len(VARIANTES):])}
    auto = salidas[-1][0]

    continua_mala, suelto = raiz / "continua-mala", raiz / "estado-suelto"
    otra_tarjeta = raiz / "otra-tarjeta"
    for carpeta in (continua_mala, suelto, otra_tarjeta):
        carpeta.mkdir()
    shutil.copy(reanudada["receta", K] / f"estado-{K:06d}.pth", continua_mala)
    shutil.copy(reanudada["receta", K] / f"estado-{K:06d}.pth", otra_tarjeta)
    shutil.copy(seguida["disyuntor"] / f"estado-{K:06d}.pth", suelto)
    origen = str(suelto / f"estado-{K:06d}.pth")
    sin_hitos = VARIANTES["disyuntor"][len(HITOS):]
    salidas = espera(
        *[lanza(base, reanudada[v, k], *VARIANTES[v], "--guardar-cada", str(N),
                "--reanudar", str(reanudada[v, k] / f"estado-{k:06d}.pth")) for v, k in CASOS],
        lanza(base, raiz / "rama", *sin_hitos, "--hitos", "2", "--reanudar", origen,
              "--guardar-cada", "0"),
        lanza(base, otra_tarjeta, *VARIANTES["receta"], "--micro-lote", "2", "--guardar-cada",
              str(N), "--reanudar", str(otra_tarjeta / f"estado-{K:06d}.pth")))
    for c, (out, _) in zip(CASOS, salidas):
        logs[c] = log(primeras[c], out)
    cambia_micro_lote = salidas[-1][0]
    # Las tres se rechazan antes de escribir nada: dos pueden compartir `--out`.
    *rechazos, lote_malo = espera(
        lanza(base, raiz / "rama-mala", *VARIANTES["disyuntor"], "--sigma", "30",
              "--reanudar", origen),
        lanza(base, continua_mala, *VARIANTES["receta"], "--hitos", "3", "6", "4",
              "--reanudar", str(continua_mala / f"estado-{K:06d}.pth")),
        lanza(base, continua_mala, *VARIANTES["receta"], "--lote", "4",
              "--reanudar", str(continua_mala / f"estado-{K:06d}.pth")),
        codigo=2)
    return {"seguida": seguida, "reanudada": reanudada, "raiz": raiz, "logs": logs,
            "rechazos": [err for _, err in rechazos], "base": base, "auto": auto,
            "cambia_micro_lote": cambia_micro_lote, "lote_malo": lote_malo[1]}


# --- RECETA ----------------------------------------------------------------------------------

def test_la_receta_es_la_de_train():
    """Lo que no cambia con la dimensión se importa, no se copia: tocar `train.py` mueve ambos. La
    razón, la cota y el respaldo del disyuntor sí cambian en 3D
    (`test_los_defaults_son_la_receta`)."""
    for nombre in ("LR", "BETAS", "SIGMA", "STEPS", "FEATURES", "SCALES", "MACRO_BLOCKS",
                   "DELTA_ADAM", "T_INIT", "LAMBDA_INIT",
                   "GUARD_VENTANA", "GUARD_SEGUIDOS", "GUARD_LIMPIAS", "MODES",
                   "NOISE_SCALE", "DATA_RANGE", "rechaza_lote", "respalda", "rebobina"):
        assert getattr(train3d, nombre) is getattr(train, nombre), nombre
    assert (train3d.LR, train3d.BETAS, train3d.STEPS) == (4e-4, (0.9, 0.999), 10)
    # El número de parches del paper, no su contenido: 4.3x los valores
    # (docs/porque-del-codigo.md, RECETA 3D).
    assert train3d.LOTE == train.BATCH_SIZE == 32
    # La VNet es la de `train.vnet_config` en gris: sólo cambia el tipo de R.
    esperada = train.vnet_config(train.T_INIT, 1)
    config = train3d.vnet_config3d(train.T_INIT)
    assert config["R"].pop("type") == "tdv3d"
    esperada["R"].pop("type")
    assert config == esperada


def test_t_init_respeta_la_regla_2delta():
    """Que los defaults 3D despejen `2·DELTA_ADAM`, como `tests/test_recipe.py::test_regla_2delta`.
    La «regla 2-δ» está retirada (bitácora, «Lo que dejó el 2D»): no es un piso. Queda
    como tripwire de procedencia —que nadie baje el `T_init` del 3D en silencio— y no se renombra
    para no romper referencias."""
    assert train3d.T_INIT >= 2 * train3d.DELTA_ADAM
    assert train3d.opciones().parse_args(["--out", "x"]).t_init == train3d.T_INIT


def test_los_defaults_son_la_receta():
    """Las banderas de test (`--features`, `--eval-recorte`, `--patch`, `--lote`...) existen para
    achicar la red en CPU; si alguna cambiara su default, la corrida real sería otra."""
    a = train3d.opciones().parse_args(["--out", "x"])
    assert (a.patch, a.lote, a.micro_lote, a.vram_gb) == (48, 32, "auto", 96.0)
    assert a.workers == 2 and not a.sonda_vram
    # No es receta (está en LIBRES), pero el costo derivado es el de 96^3
    # (docs/porque-del-codigo.md, ESTABILIDAD).
    assert a.estabilidad_lado == qsm3d.LADO_ESTABILIDAD == 96
    assert a.features == train.FEATURES == 32
    assert a.eval_recorte == train3d.EVAL_RECORTE == 128
    assert a.iterations == train3d.ITERACIONES_TANDA
    # En 3D `referencia` cuesta ~21x y a 48^3 x 4 no entra en 16 GB.
    assert a.mode == "rapido"
    assert a.hitos is None and not a.sin_compilar
    assert a.sigma_max == 0.0 and not a.aug_signo and a.aug_ganancia == 1.0
    assert not a.mascara_cerebro
    # El disyuntor 3D: razón 10, sin cota absoluta y respaldo cada 50 (bitácora, «Qué pasó con
    # los estallidos»); la ventana, la racha y las limpias son las del 2D.
    assert (a.guard, a.guard_abs, a.guard_respaldo) == (
        train3d.GUARD_3D, math.inf, train3d.GUARD_RESPALDO_3D) == (10, math.inf, 50)
    assert (a.guard_ventana, a.guard_seguidos, a.guard_limpias) == (
        train.GUARD_VENTANA, train.GUARD_SEGUIDOS, train.GUARD_LIMPIAS)
    modelo = VNet3D(train3d.vnet_config3d(a.t_init, a.features))
    assert sum(p.numel() for p in modelo.R.parameters()) == 1_162_112


def test_el_respaldo_se_certifica_con_el_relativo_y_la_racha_repetida_es_trabada(capsys):
    """Sin el relativo armado, «aceptada» sólo quiere decir «finita»: el respaldo se queda en el del
    arranque en vez de copiar una red que puede estar rota (`ciego-epocas-l32` respaldó en la 253
    un estado de después del estallido). Armado, se refresca cada --guard-respaldo. Dos
    rebobinados seguidos al mismo respaldo son una corrida trabada: lo dicen el log y W&B."""
    modelo = torch.nn.Linear(2, 1)
    opt = torch.optim.Adam(modelo.parameters())
    perdida = torch.tensor(1.0)

    def corrida_con(guard, ventana_llena, iteraciones=199):
        args = train3d.opciones().parse_args(["--out", "x", "--guard", str(guard)])
        piezas = SimpleNamespace(args=args, model=modelo, opt=opt, run=None)
        corrida = train3d.corrida_nueva(1)
        corrida.escala = [1.0] * (args.guard_ventana if ventana_llena else 1)
        for it in range(1, iteraciones + 1):
            train3d.aplica_disyuntor(piezas, corrida, it, False, 1.0, 1.0, 1.0, perdida)
        return piezas, corrida

    assert corrida_con(math.inf, True)[1].it_respaldo == 3      # sin relativo: el del arranque
    assert corrida_con(10, False)[1].it_respaldo == 3           # relativo sin armar
    piezas, corrida = corrida_con(10, True)
    assert corrida.it_respaldo == 153                           # 3, 53, 103 y 153

    def racha(desde):
        for it in range(desde, desde + piezas.args.guard_seguidos):
            train3d.aplica_disyuntor(piezas, corrida, it, True, 1e9, 1.0, 1.0, perdida)
        return desde + piezas.args.guard_seguidos

    it = racha(200)                      # primer rebobinado a la 153
    it = racha(it)                       # segundo al mismo respaldo: trabada
    for it in range(it, it + 60):        # aceptadas: refresca el respaldo
        train3d.aplica_disyuntor(piezas, corrida, it, False, 1.0, 1.0, 1.0, perdida)
    racha(it + 1)                        # rebobina al respaldo nuevo: no es trabada
    log = capsys.readouterr().out
    assert log.count("REBOBINADO") == 3 and log.count("TRABADA") == 1
    assert "TRABADA: dos rebobinados seguidos al respaldo de la it 153" in log
    assert log.count("ALERTA: rebobinado") == 3


def test_el_disyuntor_se_puede_cambiar_al_reanudar():
    """El disyuntor no toca la aritmética mientras no rechaza: una corrida lanzada sin él se puede
    reanudar protegida."""
    assert {"guard", "guard_abs", "guard_respaldo"} <= train3d.LIBRES


# --- MEMORIA ---------------------------------------------------------------------------------

# La grilla de costo medida en la 5070 Ti el 2026-10-08 (forward + backward con `R.grad`
# compilado; bitácora, «Costo y memoria»): (modo, parche, micro-lote, checkpointing) -> pico de
# `max_memory_allocated` en GiB. Falta la fila sin `torch.compile`, que el modelo no cubre. El
# borrador que la midió checkpointeaba los S pasos: lo que en `train3d` hace la entrada con
# gradiente (--guard-amp).
GRILLA = {
    ("rapido", 32, 1, False): 1.49, ("rapido", 32, 2, False): 2.66,
    ("rapido", 32, 4, False): 5.21, ("rapido", 48, 1, False): 4.14,
    ("rapido", 48, 2, False): 7.69, ("rapido", 48, 4, False): 14.72,
    ("rapido", 64, 1, False): 8.78, ("referencia", 32, 1, False): 2.48,
    ("referencia", 32, 2, False): 4.77, ("referencia", 48, 1, False): 7.60,
    ("intermedio", 48, 1, False): 7.55, ("rapido", 96, 1, True): 3.33,
    ("rapido", 64, 4, True): 4.11, ("referencia", 64, 2, True): 3.56,
}
# Las que dieron OOM en la misma tarjeta.
NO_ENTRARON = [("rapido", 64, 2), ("rapido", 64, 4), ("rapido", 80, 1), ("rapido", 80, 2),
               ("rapido", 80, 4), ("referencia", 48, 2), ("referencia", 64, 1),
               ("referencia", 64, 2), ("intermedio", 64, 1)]
# La memoria de la 5070 Ti según torch, que es la que lee `train3d`: «total: 16585719808» en el
# OOM del log de `s1-plain` (2026-09-01), 15.45 GiB. nvidia-smi da 16 303 MiB (15.92 GiB) porque
# cuenta también lo que reserva el driver.
GIB_5070TI = 16_585_719_808 / 2 ** 30
# El pico de la corrida de humo, evaluación incluida (2026-10-08): `train3d` en `rapido`,
# 48^3 x 4, --checkpointing y sin --guard-amp.
HUMO = 3.59


def test_el_modelo_de_memoria_reproduce_la_grilla():
    """A ±10 % de lo medido a 48^3 y 64^3 en `rapido`, donde se ajustó. En toda la grilla no
    subestima más que MARGEN, que es lo que el margen promete (a 32^3 x 4 subestima 8 %), y todo
    lo que no entró en la 5070 Ti lo predice por encima de su memoria. Tampoco subestima la
    corrida de humo, con el paso 1 sin checkpoint: 3.53 contra 3.59. Con las constantes solas,
    sin ese paso, predecía 2.11."""
    for (modo, parche, micro, ckpt), medido in GRILLA.items():
        predicho = train3d.vram_predicha(micro, parche, modo, ckpt, entrada_con_grad=True)
        if modo == "rapido" and parche in (48, 64):
            assert predicho == pytest.approx(medido, rel=0.10), (modo, parche, micro, ckpt)
        assert predicho >= (1 - train3d.MARGEN) * medido, (modo, parche, micro, ckpt)
    for modo, parche, micro in NO_ENTRARON:
        assert train3d.vram_predicha(micro, parche, modo, False) > GIB_5070TI, (modo, parche)
    assert train3d.vram_predicha(8, 48, "rapido", False) == pytest.approx(28.83, abs=0.01)
    assert train3d.vram_predicha(4, 48, "rapido", True) >= (1 - train3d.MARGEN) * HUMO


def test_sin_guard_amp_el_paso_1_va_sin_checkpoint():
    """La premisa de lo que `vram_predicha` suma con checkpointing: `VNet2D.forward` checkpointea
    un paso sólo si su x lleva gradiente, y x_0 lo lleva sólo con --guard-amp. Sin la bandera,
    autograd guarda fuera del checkpoint un paso medio de los S de un backward sin checkpointing
    (medido el 2026-10-09: de 1.00 a 1.04 pasos, de 16^3 a 24^3); con ella, casi nada. Si el paso
    1 pasara a checkpointearse, esto falla y la suma sobra. No prueba que en GPU compilado ese paso
    valga POR_VOXEL / S: eso lo mide la sonda."""
    def guardado(efficient, con_grad):
        torch.manual_seed(0)
        model = VNet3D(train3d.vnet_config3d(train.T_INIT, 4), efficient=efficient)
        x = torch.rand(1, 1, 16, 16, 16).requires_grad_(con_grad)
        vistos = {}

        def anota(tensor):
            vistos[tensor.untyped_storage().data_ptr()] = tensor.untyped_storage().nbytes()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(anota, lambda tensor: tensor):
            model(x, x)[-1].sum().backward()
        return sum(vistos.values())

    sin = guardado(False, False)
    paso_1, ninguno = guardado(True, False), guardado(True, True)
    assert 0.8 < (paso_1 - ninguno) / (sin / train3d.STEPS) < 1.25
    assert ninguno < 0.15 * paso_1


def test_micro_lote_auto():
    """Los casos de la especificación. A 48^3 en 48 GiB entran 12, y 32 se parte en 11 + 11 + 10:
    las tres pasadas de 12 con 39.4 GiB en vez de 42.9. Una porción de 48 GB decimales, 44.7 GiB,
    da lo mismo; la 5070 Ti, 3 (diez trozos de 3 y uno de 2: once pasadas y no dieciséis). Con
    checkpointing y sin --guard-amp el paso 1 va sin checkpoint: en la 5070 Ti entran 18 y no 32
    (23.93 GiB predichos contra 13.90 útiles), así que dos pasadas de 16, y a 64^3 en 48 GiB, 16
    (los 32, 55.89). En la de destino, 96 GB en GiB o en GB decimales, entran 24 y los 32 piden
    113.5: dos pasadas, 16 + 16."""
    auto = train3d.micro_lote_auto
    assert auto(32, 48, "rapido", False, 96) == 16
    assert auto(32, 48, "rapido", False, 96e9 / 2 ** 30) == 16
    assert auto(32, 48, "rapido", False, 48) == 11
    assert auto(32, 48, "rapido", False, 48e9 / 2 ** 30) == 11
    assert auto(32, 48, "rapido", False, 16) == 3
    assert auto(32, 48, "rapido", False, GIB_5070TI) == 3
    assert auto(32, 64, "rapido", False, 48) == 5
    assert auto(32, 48, "rapido", True, 48) == 32
    assert auto(32, 48, "rapido", True, GIB_5070TI) == 16
    assert auto(32, 48, "rapido", True, GIB_5070TI, entrada_con_grad=True) == 32
    assert auto(32, 64, "rapido", True, 48) == 16
    assert auto(32, 64, "rapido", True, 48, entrada_con_grad=True) == 32
    assert auto(24, 48, "rapido", False, 48) == 12
    with pytest.raises(ValueError, match="ni un parche de 48"):
        auto(32, 48, "rapido", False, 4)


def test_la_particion_es_casi_igual_y_suma_el_lote():
    """Para todo micro-lote m de todo lote: ceil(lote/m) trozos que suman el lote, difieren a lo
    sumo en uno, no pasan de m y van de mayor a menor; con un divisor, todos iguales. El mayor
    trozo da la misma partición: es el que `main` guarda como micro-lote efectivo."""
    for lote in range(1, 50):
        for m in range(1, lote + 1):
            t = train3d.particion(lote, m)
            assert sum(t) == lote and len(t) == -(-lote // m)
            assert max(t) - min(t) <= 1 and max(t) <= m and t == sorted(t, reverse=True)
            assert lote % m or t == [m] * (lote // m)
            assert train3d.particion(lote, t[0]) == t
        assert train3d.tamanos_utiles(lote) == sorted({train3d.particion(lote, m)[0]
                                                       for m in range(1, lote + 1)})
    assert train3d.particion(32, 12) == [11, 11, 10]
    assert train3d.tamanos_utiles(32) == [1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 32]


def test_micro_lote_auto_deja_la_menor_cantidad_de_trozos():
    """Sobre una grilla de lotes, parches, modos y presupuestos: un micro-lote que entra con el
    margen, que es el mayor trozo de su partición, y ningún otro que entre deja menos trozos; si
    no devuelve nada es porque ni 1 entra."""
    for lote in (1, 2, 8, 12, 24, 32, 48):
        for parche in (16, 32, 48, 64, 96):
            for (modo, ckpt), con_grad in itertools.product(train3d.POR_VOXEL, (False, True)):
                for presupuesto in (2, 8, 16, 24, 44.7, 48, 80, 96):
                    util = presupuesto * (1 - train3d.MARGEN)
                    entra = [m for m in range(1, lote + 1)
                             if train3d.vram_predicha(m, parche, modo, ckpt, con_grad) <= util]
                    try:
                        n = train3d.micro_lote_auto(lote, parche, modo, ckpt, presupuesto,
                                                    con_grad)
                    except ValueError:
                        assert not entra
                        continue
                    trozos = len(train3d.particion(lote, n))
                    assert n in entra and train3d.particion(lote, n)[0] == n
                    assert trozos == min(len(train3d.particion(lote, m)) for m in entra)


@pytest.mark.parametrize("banderas, mensaje", [
    (["--lote", "32", "--micro-lote", "40"], "--micro-lote 40 es mayor que --lote 32"),
    (["--lote", "0"], "--lote 0: tiene que ser al menos 1"),
    (["--micro-lote", "0"], "tiene que ser al menos 1"),
    (["--meseta", "-1"], "no pueden ser negativos"),
    (["--meseta", "2", "--meseta-desde", "-5"], "no pueden ser negativos"),
    (["--meseta", "2", "--qsm-datos", ""], "--meseta 2 decide con el NRMSE QSM"),
    (["--meseta-desde", "10000"], "necesitan --meseta N"),
    (["--meseta-psnr"], "necesitan --meseta N"),
    (["--qsm-max-iter", "0"], "--qsm-max-iter 0: tiene que ser al menos 1"),
    (["--sigma-log"], "--sigma-log pide --sigma-max > 0"),
    (["--sigma-log", "--sigma-max", "55"], "--sigma-log pide --sigma-max > 0"),
    (["--fraccion-limpias", "1"], "--fraccion-limpias 1: tiene que estar en [0, 1)"),
    (["--qsm-target-std", "0"], "--qsm-target-std 0.0: tiene que ser positivo y finito"),
    (["--qsm-target-std", "nan"], "--qsm-target-std nan: tiene que ser positivo y finito"),
    (["--qsm-target-std", "0.06", "0.06"], "--qsm-target-std [0.06, 0.06]: hay targets repetidos"),
    (["--val-snr", "25", "0"], "cada SNR tiene que ser positivo y finito"),
    (["--micro-lote", "ocho"], "un entero o `auto`"),
    (["--patch", "96", "--vram-gb", "2"], "ni un parche de 96^3 entra en 2.00 GiB"),
    # Con checkpointing, el paso 1 cuenta sin --guard-amp y no con: `main` le pasa la bandera.
    (["--patch", "96", "--checkpointing", "--vram-gb", "6"], "predice 6.44 GiB contra 5.40"),
    (["--patch", "96", "--checkpointing", "--guard-amp", "100", "--vram-gb", "3.5"],
     "predice 3.62 GiB contra 3.15"),
])
def test_un_micro_lote_imposible_es_un_error(banderas, mensaje, tmp_path, capsys, monkeypatch):
    """Antes de tocar `--out`: un micro-lote mayor que el lote o que no es un entero positivo, un
    lote vacío, una meseta negativa o sin datos QSM, y `auto` cuando ni un parche entra."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as salida:
        train3d.main(["--out", str(tmp_path / "x"), *banderas])
    assert salida.value.code == 2
    assert mensaje in capsys.readouterr().err
    assert not (tmp_path / "x").exists()


def test_sin_gpu_y_compilado_no_entrena_en_cpu(tmp_path, capsys, monkeypatch):
    """Sin GPU visible (un driver viejo, por ejemplo) y sin --sin-compilar, sale con 2 antes de
    tocar `--out`, en vez de entrenar compilado en CPU sin avisar."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # `main` llega a sembrar torch y a fijar los flags de cuDNN: se reponen al salir.
    for modulo, flag in ((torch.backends.cudnn, "benchmark"),
                         (torch.backends.cudnn.conv, "fp32_precision"),
                         (torch.backends.cuda.matmul, "fp32_precision")):
        monkeypatch.setattr(modulo, flag, getattr(modulo, flag))
    rng = torch.get_rng_state()
    try:
        with pytest.raises(SystemExit) as salida:
            train3d.main(["--out", str(tmp_path / "x")])
    finally:
        torch.set_rng_state(rng)
    assert salida.value.code == 2
    error = capsys.readouterr().err
    assert "torch no ve ninguna GPU" in error and "--sin-compilar" in error
    assert not (tmp_path / "x").exists()


def test_la_sonda_sin_gpu_termina_con_un_mensaje_claro(tmp_path, capsys, monkeypatch):
    """La sonda mide una GPU y en CPU no se puede probar. Lo que sí: sin GPU no entrena ni escribe
    en `--out`, y dice por qué."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as salida:
        train3d.main(["--out", str(tmp_path / "sonda"), "--sonda-vram"])
    assert salida.value.code == 2
    error = capsys.readouterr().err
    assert "--sonda-vram" in error and "GPU" in error and "tarjeta de destino" in error
    assert not (tmp_path / "sonda").exists()


def test_auto_registra_el_micro_lote_efectivo(corridas):
    """`auto` se resuelve antes de arrancar: `config.json`, el estado y el log llevan el número.
    En CPU el presupuesto es --vram-gb: 1.5 GiB dejan 1.35 útiles, y a 16^3 en `referencia` entran
    2 parches (1.18 GiB predichos) y no 4 (1.76). `config.json` lleva además el commit del código,
    también sin W&B (la corrida va con --seguimiento no)."""
    carpeta = corridas["raiz"] / "auto"
    config = json.loads((carpeta / "config.json").read_text())
    assert config["micro_lote"] == 2 and config["commit"] == train3d.commit_git()
    assert estado(carpeta, 1)["args"]["micro_lote"] == 2
    assert ("dispositivo: cpu, sin memoria que medir | presupuesto 1.50 GiB, 1.35 útiles | "
            "micro-lote 2 (auto), pico predicho 1.18 GiB") in corridas["auto"]
    assert "lote 4 = 2x2 de 16^3" in corridas["auto"]
    assert "(predicho 1.18)" in corridas["auto"]
    hecho = json.loads((carpeta / "estado.json").read_text())
    assert hecho["vram_predicha_gb"] == 1.18 and hecho["vram_pico_gb"] == 0.0


def test_reanudar_con_otro_micro_lote_avisa_y_sigue(corridas):
    """Seguir en otra tarjeta: el estado K de la receta, de micro-lote 1, sigue con 2. Está
    permitido y avisa. Hereda la traza bit a bit, y después sigue a la corrida seguida salvo por el
    orden de suma: mismos parches y, con sigma fijo y 16^3 = 256 x 16 valores, mismo ruido. Si el
    ruido cambiara con la partición, la pérdida se iría lejos y esto no pasaría. Cambiar --lote,
    en cambio, es otra receta y se rechaza."""
    carpeta = corridas["raiz"] / "otra-tarjeta"
    t, seguida = traza(carpeta), traza(corridas["seguida"]["receta"])
    assert bit_a_bit(t[:K], seguida[:K])
    # Medido el 2026-10-09: a lo sumo 7.8e-7 relativo, en ∂L/∂λ; T y λ, idénticos.
    np.testing.assert_allclose(t[K:, :6], seguida[K:, :6], rtol=1e-5)
    np.testing.assert_array_equal(t[:, 11:], seguida[:, 11:])
    assert bit_a_bit(t[K:, 10], t[K:, 5])            # un solo micro-lote: la 10 es la 5
    assert estado(carpeta, N)["args"]["micro_lote"] == 2
    aviso = [l for l in corridas["cambia_micro_lote"].splitlines() if l.startswith("AVISO")]
    assert len(aviso) == 1 and "el micro-lote pasa de 1 a 2" in aviso[0]
    assert "ruido" not in aviso[0]
    assert "cambian ['lote']" in corridas["lote_malo"]


# --- MÁSCARA DE CEREBRO ----------------------------------------------------------------------

def arma_mascaras(raiz):
    """Una por volumen de entrenamiento, como las deja `mascaras_ldm100k3d.py combina`: dos cajas
    (a 2 y a 1 vóxel del borde) y una vacía, que aloja 105, 315 y 0 cubos de 18^3 (16 + 2 · 1)."""
    carpeta = raiz / "mascaras" / "train"
    carpeta.mkdir(parents=True)
    for i, borde in enumerate((2, 1, None)):
        m = np.zeros((24, 26, 28), bool)
        if borde is not None:
            m[borde:-borde, borde:-borde, borde:-borde] = True
        np.save(carpeta / f"sub-train{i}.npy", m)


def test_mascara_cerebro(corridas, tmp_path):
    """Con la bandera, `train3d` lee una máscara por volumen de entrenamiento, deja fuera del
    sorteo al que no aloja ningún cubo, lo dice en el log y la registra en `config.json`, y elige
    `best.pth` por el PSNR en el cerebro. Sin las máscaras de entrenamiento o sin las de
    validación se niega antes de tocar `--out` y dice cómo generarlas. Es receta: reanudar con
    otro valor se rechaza, en los dos sentidos."""
    base = arma_datos(tmp_path / "datos")
    arma_mascaras(tmp_path / "datos")
    arma_mascaras_val(tmp_path / "datos")
    sin_mascaras = arma_datos(tmp_path / "sin")
    sin_val = arma_datos(tmp_path / "sin-val")
    arma_mascaras(tmp_path / "sin-val")
    con, receta = tmp_path / "con", corridas["reanudada"]["receta", K]
    procesos = [
        lanza(base, con, "--mascara-cerebro", "--iterations", "2", "--eval-every", "1",
              "--guardar-cada", "2"),
        lanza(sin_mascaras, tmp_path / "falta", "--mascara-cerebro"),
        lanza(corridas["base"], tmp_path / "prende", "--mascara-cerebro",
              *VARIANTES["receta"], "--reanudar", str(receta / f"estado-{K:06d}.pth")),
        lanza(sin_val, tmp_path / "falta-val", "--mascara-cerebro")]
    (salida, error), = espera(procesos[0])
    (_, falta), (_, prende), (_, falta_val) = espera(*procesos[1:], codigo=2)
    historia, final = json.loads((con / "history.json").read_text()), estado(con, 2)
    assert len(historia) == 2 and all("psnr_cerebro" in h for h in historia)
    assert final["best_psnr"] == max(h["psnr_cerebro"] for h in historia)
    assert final["mejor"]["iteration"] == max(historia, key=lambda h: h["psnr_cerebro"])[
        "iteration"]
    assert "faltan 2 de 2 máscaras de validación" in falta_val
    assert not (tmp_path / "falta-val").exists()
    assert ("máscara de cerebro: 2 de 3 volúmenes con cubos de 16^3 a 1 vóxel o más del borde; "
            "mediana 105 esquinas por volumen") in salida
    assert "1 de 3 volúmenes sin un cubo de 18^3" in error and "sub-train2.npy" in error
    assert json.loads((con / "config.json").read_text())["mascara_cerebro"] is True
    assert estado(con, 2)["args"]["mascara_cerebro"] is True
    assert traza(con).shape == (2, train3d.COLUMNAS)

    assert "faltan 3 de 3 máscaras" in falta and "mascaras_ldm100k3d.py combina" in falta
    assert not (tmp_path / "falta").exists()
    assert "cambian ['mascara_cerebro']" in prende
    (_, apaga), = espera(lanza(base, con, "--iterations", "4",
                               "--reanudar", str(con / "estado-000002.pth")), codigo=2)
    assert "cambian ['mascara_cerebro']" in apaga


# --- CALENDARIO DEL LR -----------------------------------------------------------------------

def test_lr_en_reproduce_multisteplr():
    """Iteración por iteración contra lo que hace `train.py`: `MultiStepLR(gamma=0.5)` con
    `sched.step()` DESPUÉS del paso. Hitos al azar sobre 0..N, desordenados y repetidos."""
    azar = random.Random(0)
    n = 300
    for _ in range(20):
        hitos = [azar.randint(0, n) for _ in range(azar.randint(0, 6))]
        w = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.Adam([w], lr=train.LR, betas=train.BETAS)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=hitos, gamma=0.5)
        for it in range(1, n + 1):
            assert opt.param_groups[0]["lr"] == train3d.lr_en(it, hitos), (hitos, it)
            opt.step()
            sched.step()


def test_el_hito_cae_despues_de_su_iteracion():
    """El ejemplo de la especificación, que confirmaba el log de `t1-rapido-30k`."""
    assert train3d.lr_en(25_000, [25_000]) == 4e-4
    assert train3d.lr_en(25_001, [25_000]) == 2e-4
    assert train3d.lr_en(1, []) == train3d.lr_en(10 ** 6, []) == train.LR


def test_un_rebobinado_no_cambia_el_lr(corridas):
    """El respaldo de la it 1 trae el lr de 4e-4 junto con Adam, y los rebobinados vuelven a él
    pasado un hito. En `train.py` eso deshacía los halvings; acá la columna 11, leída del
    optimizador antes de cada paso, sigue el calendario en todas las iteraciones."""
    t = traza(corridas["seguida"]["disyuntor"])
    hitos = [int(h) for h in HITOS[1:]]
    rebobinados = np.flatnonzero(t[:, 9]) + 1
    # Que el caso exista: rebobinados a un respaldo anterior al primer hito, después de él.
    assert len(rebobinados) and all(it > hitos[0] for it in rebobinados)
    assert all(t[it - 1, 12] < hitos[0] for it in rebobinados)
    assert t[:, 11].tolist() == [train3d.lr_en(it, hitos) for it in range(1, N + 1)]


# --- TRAZA -----------------------------------------------------------------------------------

def test_la_traza_tiene_las_14_columnas(corridas):
    """Las diez de `train.py` en su lugar, y 10-13 al final: ‖g‖ del primer micro-lote, lr,
    iteración efectiva y sigma media del lote."""
    t = traza(corridas["seguida"]["receta"])
    assert t.shape == (N, train3d.COLUMNAS) == (N, 14)
    assert np.isnan(t[0, 6]) and np.isnan(t[:, 7]).all()      # ventana vacía; sin --guard-amp
    assert not t[:, 8:10].any()                                 # la receta no rechaza acá
    assert (t[:, 5] > 0).all() and (t[:, 10] > 0).all()
    assert t[:, 12].tolist() == list(range(1, N + 1))       # sin rebobinados, efectiva = it
    assert (t[:, 13] == train.SIGMA).all()
    ciega = traza(corridas["seguida"]["disyuntor"])[:, 13]
    assert ((ciega >= 0) & (ciega <= 55)).all() and len(set(ciega)) > 1


def test_la_traza_conserva_las_diez_columnas_de_train():
    """Los `scripts/` leen `traza.npy` por posición: las columnas nuevas van al final."""
    fuente = inspect.getsource(train3d.entrena)
    assert "traza[it - 1, :5] = torch.stack((perdida, model.T.detach(), model.T.grad," in fuente
    assert "traza[it - 1, 5:] = (gnorm, centro, amp, rechazado, rebobinado," in fuente
    assert "# loss T dT lambda dlambda |g| centro amp rechazado rebobinado |g|micro" in fuente


def test_la_iteracion_efectiva_descuenta_lo_rebobinado(corridas):
    """Con un único respaldo en la it 1, cada rebobinado devuelve la corrida a la iteración 1 de
    progreso, y entre rebobinados la efectiva avanza de a una."""
    t = traza(corridas["seguida"]["disyuntor"])
    efectiva, previa = t[:, 12], 0
    for i in range(len(t)):
        assert efectiva[i] == (1 if t[i, 9] else previa + 1), i + 1
        previa = efectiva[i]


def test_leer_el_gradiente_entre_backwards_no_cambia_un_bit():
    """La columna 10 lee `.grad` tras el primer backward. Por construcción no toca la aritmética:
    el backward siguiente acumula sobre el mismo tensor, haya o no una lectura en medio. Acá se
    ejecuta: dos micro-lotes acumulados, con y sin la lectura, dan el mismo gradiente bit a bit."""
    torch.manual_seed(0)
    modelo = VNet3D(train3d.vnet_config3d(train3d.T_INIT, 4))
    lotes = [torch.rand(1, 1, 12, 13, 14) for _ in range(2)]

    def acumula(lee):
        modelo.zero_grad(set_to_none=True)
        for i, x in enumerate(lotes):
            loss = (modelo(x, x)[-1] - x).pow(2).flatten(1).sum(1).mean()
            (loss / len(lotes)).backward()
            if lee and i == 0:
                get_total_norm([q.grad for q in modelo.parameters()]).item()
        return [q.grad.clone() for q in modelo.parameters()]

    for a, b in zip(acumula(False), acumula(True)):
        assert torch.equal(a, b)


def test_la_columna_10_es_el_primer_micro_lote_como_si_fuera_el_lote(corridas):
    """Con `--micro-lote` = `--lote` el primer micro-lote ES el lote: la medición y
    `clip_grad_norm_` leen el mismo gradiente con la misma norma. Y con dos micro-lotes de un
    parche, la de la iteración 1 es la `‖g‖` de una corrida de lote 1: mismo parche, mismo ruido y
    mismos pesos; escalar por 2 y dividir la pérdida por 2 son exactos en binario."""
    t = traza(corridas["raiz"] / "un-micro-lote")
    assert bit_a_bit(t[:, 10], t[:, 5])
    assert (t[:, 13] == train.SIGMA).all()
    dos = traza(corridas["seguida"]["receta"])
    assert (dos[:, 10] != dos[:, 5]).all()
    assert bit_a_bit(dos[:1, 10], traza(corridas["raiz"] / "un-parche")[:1, 5])


# --- REANUDACIÓN -----------------------------------------------------------------------------

@pytest.mark.parametrize("variante, corte", CASOS)
def test_reanudar_da_la_misma_corrida_bit_a_bit(corridas, variante, corte):
    """N iteraciones seguidas contra k + (N - k) en dos procesos, en `referencia` y CPU: la misma
    `traza.npy`, la misma `history.json`, el mismo log y el mismo estado final completo (pesos,
    Adam, ventana, respaldo, contadores y generadores). El disyuntor cruza el corte además con el
    ruido ciego, los dos aumentos y rebobinados al mismo respaldo antes y después."""
    a, b = corridas["seguida"][variante], corridas["reanudada"][variante, corte]
    assert bit_a_bit(traza(a), traza(b))
    assert json.loads((a / "history.json").read_text()) == json.loads(
        (b / "history.json").read_text())
    assert corridas["logs"][variante] == corridas["logs"][variante, corte]

    final_a, final_b = estado(a, N), estado(b, N)
    for clave in ("args", "horas"):          # las banderas de la reanudación y el reloj
        final_a.pop(clave), final_b.pop(clave)
    assert igual(final_a, final_b), [k for k in final_a if not igual(final_a[k], final_b[k])]

    for nombre in ("last.pth", "best.pth"):
        ck_a = torch.load(a / nombre, map_location="cpu", weights_only=False)
        ck_b = torch.load(b / nombre, map_location="cpu", weights_only=False)
        assert igual(ck_a, ck_b), nombre
    # Seguir en el mismo `--out` desde el último estado no descarta nada.
    assert not list(b.glob("*-descartada-*"))

    hecho_a = json.loads((a / "estado.json").read_text())
    hecho_b = json.loads((b / "estado.json").read_text())
    for clave in ("horas", "reanudada_de"):
        hecho_a.pop(clave), hecho_b.pop(clave)
    assert hecho_a == hecho_b

    # Que cada caso ejercite lo que dice: si no, un estado incompleto pasaría igual.
    t = traza(a)
    rebobinados = np.flatnonzero(t[:, 9]) + 1
    if variante == "receta":
        # Respaldos en la 3, la 7 y la 11: el de la 7 necesita las 6 limpias de antes del corte.
        assert not t[:, 8].any() and final_a["it_respaldo"] == 11
    elif corte == 6:
        # Rebobinado en el corte, paso aceptado después y otro rebobinado al mismo respaldo: si
        # el alias de Adam con el respaldo no sobreviviera al estado, volvería a otros pesos.
        despues = rebobinados[rebobinados > corte]
        assert corte in rebobinados and len(despues), rebobinados
        assert not t[corte:despues[0] - 1, 8].all(), "ningún paso de Adam entre los rebobinados"
        assert t[despues[0] - 1, 12] == t[corte - 1, 12] == 1          # el mismo respaldo
    else:
        # Un rechazo en el corte y el rebobinado en la siguiente: hace falta `seguidos` = 1.
        assert t[corte - 1, 8] and t[corte, 9] and not t[corte - 2, 8]


def test_guardar_el_estado_no_toca_la_corrida(corridas):
    """La seguida del disyuntor guarda en cada iteración y la reanudada sólo en el corte: su estado
    ahí es el mismo."""
    for corte in (6, 9):
        a = estado(corridas["seguida"]["disyuntor"], corte)
        b = estado(corridas["reanudada"]["disyuntor", corte], corte)
        for clave in ("args", "horas"):
            a.pop(clave), b.pop(clave)
        assert igual(a, b), [k for k in a if not igual(a[k], b[k])]


def test_una_rama_cambia_el_calendario_y_nada_mas(corridas):
    """Desde la K de la seguida, con otro `--out` y `--hitos 2`: hereda la traza hasta la K y sigue
    con el lr del calendario nuevo. Cambiar otra bandera (`--sigma`) es otra corrida y se rechaza,
    igual que cambiar `--hitos` siguiendo en el mismo `--out`.

    El estado se copió solo a otro directorio: la traza heredada viene del estado, no de un
    `traza.npy` vecino."""
    t, origen = traza(corridas["raiz"] / "rama"), traza(corridas["seguida"]["disyuntor"])
    assert not (corridas["raiz"] / "estado-suelto" / "traza.npy").exists()
    assert bit_a_bit(t[:K], origen[:K])
    assert t[K:, 11].tolist() == [train3d.lr_en(it, [2]) for it in range(K + 1, N + 1)]
    assert t[K:, 11].tolist() != origen[K:, 11].tolist()
    rama_mala, continua_mala = corridas["rechazos"]
    assert "cambian ['sigma']" in rama_mala
    assert "cambian ['hitos']" in continua_mala


def test_el_estado_lleva_la_traza_y_el_mejor(corridas):
    """Cada estado lleva la traza hasta su iteración y el payload de `best.pth` de ese momento:
    reanudar no depende de que los archivos de su directorio sigan enteros ni sean de su linaje."""
    carpeta = corridas["seguida"]["disyuntor"]
    t = traza(carpeta)
    for it in range(1, N + 1):
        e = estado(carpeta, it)
        assert bit_a_bit(e["traza"], t[:it]), it
        if it < EVALUA:
            assert e["mejor"] is None and e["best_psnr"] == 0.0
        else:
            assert e["mejor"]["psnr"] == e["best_psnr"] == max(h["psnr"] for h in e["history"])
            assert e["mejor"]["iteration"] <= it
    final = torch.load(carpeta / "best.pth", map_location="cpu", weights_only=False)
    assert igual(final, estado(carpeta, N)["mejor"])


def test_el_ultimo_estado_se_guarda_siempre(corridas):
    """Con `--iterations` que no es múltiplo de `--guardar-cada` (2 y 1 contra 10 000), la última
    deja su estado: una corrida terminada se extiende sin rehacer iteraciones."""
    raiz = corridas["raiz"]
    assert estado(raiz / "un-micro-lote", 2)["iteration"] == 2
    assert estado(raiz / "un-parche", 1)["iteration"] == 1
    assert not list((raiz / "rama").glob("estado-*.pth"))          # --guardar-cada 0


def test_reanudar_aparta_lo_del_linaje_descartado(corridas, tmp_path):
    """Seguir en el mismo `--out` desde un estado anterior al último descarta lo que vino después:
    `history.json` y `traza.npy` vuelven a las del estado, `last.pth` y un `best.pth` posteriores
    se apartan como `<nombre>-descartada-<it>.pth`, y `best.pth` se repone desde el estado. Con
    `--iterations K` no corre ninguna iteración: se ve sólo el arranque."""
    carpeta = tmp_path / "corrida"
    shutil.copytree(corridas["reanudada"]["receta", K], carpeta)
    # El `best.pth` que deja un linaje que mejoró después del estado y murió.
    posterior = torch.load(carpeta / "last.pth", map_location="cpu", weights_only=False)
    posterior["psnr"] = 1e3
    torch.save(posterior, carpeta / "best.pth")
    origen = estado(carpeta, K)
    espera(lanza(corridas["base"], carpeta, *VARIANTES["receta"], "--iterations", str(K),
                 "--reanudar", str(carpeta / f"estado-{K:06d}.pth")))

    assert json.loads((carpeta / "history.json").read_text()) == origen["history"]
    assert bit_a_bit(traza(carpeta), origen["traza"])
    apartado = torch.load(carpeta / f"best-descartada-{N:06d}.pth", map_location="cpu",
                          weights_only=False)
    assert apartado["psnr"] == 1e3
    assert igual(torch.load(carpeta / "best.pth", map_location="cpu", weights_only=False),
                 origen["mejor"])
    assert not (carpeta / "last.pth").exists()
    assert (carpeta / f"last-descartada-{N:06d}.pth").exists()
    hecho = json.loads((carpeta / "estado.json").read_text())
    assert hecho["mejor_psnr"] == origen["mejor"]["psnr"] and hecho["iteraciones_hechas"] == K


def test_reanudar_con_checkpoints_cortados(corridas, tmp_path):
    """Un `last.pth` y un `best.pth` cortados a mitad de `torch.save` (un disco lleno, un corte de
    luz) no traban `--reanudar`: se apartan como ilegibles y `best.pth` se repone desde el estado.
    Un apartado anterior con el mismo nombre se conserva."""
    carpeta = tmp_path / "corrida"
    shutil.copytree(corridas["reanudada"]["receta", K], carpeta)
    for nombre in ("last.pth", "best.pth"):
        datos = (carpeta / nombre).read_bytes()
        (carpeta / nombre).write_bytes(datos[:len(datos) // 2])
    previo = carpeta / "last-descartada-ilegible.pth"
    previo.write_bytes(b"de un corte anterior")
    origen = estado(carpeta, K)
    espera(lanza(corridas["base"], carpeta, *VARIANTES["receta"], "--iterations", str(K),
                 "--reanudar", str(carpeta / f"estado-{K:06d}.pth")))

    assert previo.read_bytes() == b"de un corte anterior"
    assert (carpeta / "last-descartada-ilegible-2.pth").exists()
    assert (carpeta / "best-descartada-ilegible.pth").exists()
    assert not (carpeta / "last.pth").exists()
    assert igual(torch.load(carpeta / "best.pth", map_location="cpu", weights_only=False),
                 origen["mejor"])


def test_aparta_no_pisa_ni_aparta_el_vigente(tmp_path):
    """Tres linajes descartados con un checkpoint en la misma evaluación dejan tres archivos, no
    uno. El vigente se queda, también con un PSNR NaN, que `!=` daría por distinto."""
    last = tmp_path / "last.pth"
    for psnr in (30.0, 31.0, 32.0):
        torch.save({"iteration": 12, "psnr": psnr}, last)
        train3d.aparta(last, {"iteration": 10, "psnr": 29.0})
    for sufijo, psnr in (("", 30.0), ("-2", 31.0), ("-3", 32.0)):
        assert torch.load(tmp_path / f"last-descartada-000012{sufijo}.pth")["psnr"] == psnr

    for psnr in (29.0, float("nan")):
        torch.save({"iteration": 10, "psnr": psnr}, last)
        train3d.aparta(last, {"iteration": 10, "psnr": psnr})
        assert last.exists()
    train3d.aparta(last, None)                    # una corrida nueva: nada es vigente
    assert not last.exists()
    assert math.isnan(torch.load(tmp_path / "last-descartada-000010.pth")["psnr"])


def test_un_corte_al_guardar_no_deja_un_pth_roto(tmp_path, monkeypatch):
    """`last.pth`, `best.pth`, `best_qsm.pth` y los estados pasan por `guarda_pth`: si `torch.save`
    muere a mitad, el archivo con el nombre bueno sigue siendo el anterior, entero."""
    destino = tmp_path / "last.pth"
    train3d.guarda_pth({"iteration": 1}, destino)

    def corta(objeto, f):
        Path(f).write_bytes(b"PK\x03\x04")       # el principio de un zip, sin el final
        raise OSError("disco lleno")

    monkeypatch.setattr(torch, "save", corta)
    with pytest.raises(OSError):
        train3d.guarda_pth({"iteration": 2}, destino)
    monkeypatch.undo()
    assert torch.load(destino)["iteration"] == 1
    # El único `torch.save` del archivo es el de `guarda_pth`.
    assert inspect.getsource(train3d).count("torch.save(") == 1
    assert "torch.save(" in inspect.getsource(train3d.guarda_pth)


# Muere en la primera evaluación, como una corrida que se mata a mitad del bucle.
MUERE = ("import sys, train3d\n"
         "def muere(*args, **kw):\n"
         "    raise SystemExit(3)\n"
         "train3d.evaluate_denoising3d = muere\n"
         "train3d.main(sys.argv[1:])\n")


def test_seguir_y_morir_no_queda_terminada(corridas, tmp_path):
    """La reanudada de la receta terminó en la N con `estado.json` «terminada». Extenderla en el
    mismo `--out` y morir antes de terminar no puede dejar ese archivo: su ausencia es lo que dice
    «en vuelo o matada», y los análisis de supervivencia cuentan con eso."""
    carpeta = tmp_path / "corrida"
    shutil.copytree(corridas["reanudada"]["receta", K], carpeta)
    assert json.loads((carpeta / "estado.json").read_text())["terminada"]
    muere = subprocess.Popen(
        [sys.executable, "-c", MUERE, *corridas["base"], "--out", str(carpeta),
         *VARIANTES["receta"], "--iterations", str(N + EVALUA),
         "--reanudar", str(carpeta / f"estado-{N:06d}.pth")],
        cwd=REPO, env=ENTORNO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    espera(muere, codigo=3)
    assert not (carpeta / "estado.json").exists()
    assert traza(carpeta).shape == (N, train3d.COLUMNAS)


def test_el_respaldo_del_estado_es_el_del_momento_en_que_se_tomo(corridas):
    """El respaldo que viaja en `estado-<it>.pth` tiene que ser el estado de la iteración en que se
    tomó. Si `rebobina()` no copiara, tras un rebobinado Adam adoptaría los tensores del respaldo,
    y el paso aceptado siguiente los mutaría (bitácora, «El disyuntor en 3D»)."""
    carpeta = corridas["seguida"]["disyuntor"]
    for it in range(1, N + 1):
        e = estado(carpeta, it)
        if e["respaldo"] is None:
            continue
        cuando = estado(carpeta, e["it_respaldo"])
        pesos, adam, ventana = e["respaldo"]
        assert igual(pesos, cuando["model"]), it
        assert igual(adam["state"], cuando["optimizer"]["state"]), it
        assert ventana == cuando["escala"], it


# --- BUCLE Y EVALUACIÓN ----------------------------------------------------------------------

def test_el_bucle_corta_solo_por_meseta():
    """Como en `train.py`, los análisis de supervivencia cuentan filas de la traza: sin --meseta el
    bucle no corta nunca. El único `break` es el de la meseta, que deja el corte escrito en
    `estado.json` (`test_la_meseta_parte_el_lr_y_corta`), y no hay `return`."""
    fuente = inspect.getsource(train3d.entrena)
    assert fuente.count("break") == 1 and "if corrida.corte:\n            break" in fuente
    assert "return" not in fuente
    assert "if not rechazado:\n            opt.step()" in fuente
    # El corte lo marca sólo la meseta, y toda corrida empieza sin él.
    assert inspect.getsource(train3d).count("corte = ") == 1
    assert "corrida.corte = it" in inspect.getsource(train3d.aplica_meseta)
    assert train3d.corrida_nueva(1).corte is None


def test_la_evaluacion_mide_todo_el_recorte_y_el_tejido(tmp_path):
    """Con `T = 0` y `lambda = 0` fijos el esquema devuelve la entrada exacta: el PSNR de la salida
    es el de la ruidosa, y el de tejido es el de los vóxeles LIMPIOS > 0.05, con el ruido de la
    semilla `seed + i`."""
    rng = np.random.default_rng(1)
    archivos = []
    for i in range(2):
        archivos.append(tmp_path / f"v{i}.npy")
        np.save(archivos[-1], volumen(rng, (20, 22, 24)).astype(np.float16))
    datos = VolumeDataset3D(archivos, recorte=16)
    config = train3d.vnet_config3d(0.0, 4)
    config["T_mode"], config["lambda_mode"] = "fixed", "fixed"
    config["lambda"]["init"] = 0.0
    modelo = VNet3D(config).eval()

    reporte = evaluate_denoising3d(modelo, datos, sigma=25.0, seed=7)
    assert reporte["num_volumenes"] == 2
    assert reporte["psnr_denoised"] == reporte["psnr_noisy"]
    assert reporte["psnr_denoised_tejido"] == reporte["psnr_noisy_tejido"]
    esperado = []
    for i in range(len(datos)):
        limpio = datos[i][0][None]
        ruidoso = add_noise(limpio, 25.0, torch.Generator().manual_seed(7 + i), 255.0)
        tejido = (limpio > 0.05).numpy()
        error = (ruidoso - limpio).numpy()[tejido]
        assert 0 < tejido.mean() < 1
        esperado.append(-10 * np.log10(np.mean(error.astype(np.float64) ** 2)))
    assert reporte["psnr_noisy_tejido"] == pytest.approx(np.mean(esperado), abs=1e-4)


# --- SEGUIMIENTO Y EVALUACIÓN QSM ------------------------------------------------------------

QSM_N = 6                                # iteraciones de las corridas con evaluación QSM
QSM_EVALUA = 3                           # validaciones en la 3 y la 6
# La estabilidad en 12^3: un cubo que corta los recortes de 16^3 y el COSMOS de 24^3.
QSM = ["--iterations", str(QSM_N), "--eval-every", str(QSM_EVALUA), "--log-every", "2",
       "--guardar-cada", str(QSM_EVALUA), "--qsm-max-iter", "3", "--estabilidad-lado", "12"]
ESTABILIDAD = {"estabilidad", "lambda_max", "estabilidad_cambio"}
# El ADMM y la estabilidad levantan un OOM, como el de 160^3 en una tarjeta llena.
FALLA_QSM = ("import sys, torch, train3d\n"
             "def falla(*args, **kw):\n"
             "    raise torch.cuda.OutOfMemoryError('CUDA out of memory (simulado)')\n"
             "train3d.admm_wtdv = falla\n"
             "train3d.estabilidad = falla\n"
             "train3d.main(sys.argv[1:])\n")
# Armar la lámina de QSM levanta: el ADMM terminó y sus métricas no se pueden perder.
FALLA_LAMINA = ("import sys, train3d\n"
                "def falla(*args, **kw):\n"
                "    raise RuntimeError('lámina rota (simulado)')\n"
                "train3d.lamina_qsm = falla\n"
                "train3d.main(sys.argv[1:])\n")
SIN_WANDB = ("import sys, train3d\n"
             "train3d.main(sys.argv[1:])\n"
             "assert 'wandb' not in sys.modules, 'importó wandb'\n")
HAY_WANDB = importlib.util.find_spec("wandb") is not None
# `best_qsm.pth` con el NRMSE QSM mínimo y el final, el motivo y la estabilidad forzados, una
# validación por iteración. La 1 y la 2 bajan el mínimo (la 2 sin cortar, con el final en 90 % y
# estabilidad 7: no eligen); la 3 tiene el menor final pero no el menor mínimo; la 4 es peor; la
# 5 no tiene ningún NRMSE finito; la 6 empata con la 2. Queda la 2.
PASOS = [1.0, 7.0, 0.5, 0.5, 0.5, 1.0]
MINIMOS = [5.0, 4.0, 4.5, 4.6, math.nan, 4.0]
NRMSE = [6.0, 90.0, 3.0, 4.7, math.nan, 4.1]
MOTIVOS = ["corte", "max_iter", "corte", "corte", "no_finito", "corte"]
MEJOR_QSM = 2
CLAVES_BEST_QSM = {"config", "model", "iteration", "psnr", "target_std", "nrmse_min_pct",
                   "iter_min", "nrmse_final_pct", "iteraciones", "motivo", "estabilidad"}
BUSCA_BEST_QSM = "best_qsm.pth: el de menor NRMSE QSM mínimo del ADMM, el de su mejor iteración"


# Dos targets (0.06 y 0.25), dos evaluaciones (la 3 y la 6), un ADMM por target: en la 3 gana
# 0.25 (4.0 contra 5.0) y en la 6 gana 0.06 (3.0, el menor de todos).
DOS_MINIMOS = [5.0, 4.0, 3.0, 6.0]
DOS_FINALES = [6.0, 5.0, 4.0, 7.0]


def forzada(pasos, nrmse, minimos=None, motivos=None):
    """`train3d` con la validación j (desde 0) midiendo `pasos[j]` en sus dos puntos de
    estabilidad y un ADMM que da `nrmse[j]` de NRMSE final y, si se pasan, `minimos[j]` de mínimo
    y `motivos[j]` de motivo; lo demás, lo que miden de verdad."""
    return ("import itertools, sys, train3d\n"
            "from math import inf, nan\n"
            f"pasos, nrmse, minimos, motivos = {pasos!r}, {nrmse!r}, {minimos!r}, {motivos!r}\n"
            "est, admm = train3d.estabilidad, train3d.admm_wtdv\n"
            "n_est, n_admm = itertools.count(), itertools.count()\n"
            "train3d.estabilidad = lambda *a, **k: {**est(*a, **k),\n"
            "                                       'paso': pasos[next(n_est) // 2]}\n"
            "def admm_forzado(*a, **k):\n"
            "    j = next(n_admm)\n"
            "    return {**admm(*a, **k), 'nrmse_final': nrmse[j],\n"
            "            **({} if minimos is None else {'nrmse_min': minimos[j]}),\n"
            "            **({} if motivos is None else {'motivo': motivos[j]})}\n"
            "train3d.admm_wtdv = admm_forzado\n"
            "train3d.main(sys.argv[1:])\n")


def arma_cosmos(carpeta, n=24):
    """Un COSMOS de juguete con las claves de `../tdv/data_mat`: dipolo calculado, una cabeza
    elipsoidal y la fase que el dipolo le da a su chi, más ruido."""
    from test_qsm3d import dipolo
    rng = np.random.default_rng(0)
    c = np.mgrid[:n, :n, :n] - (n - 1) / 2
    cabeza = (c[0] / 10) ** 2 + (c[1] / 9) ** 2 + (c[2] / 8) ** 2 <= 1
    magnitud = np.where(cabeza, 1 + 0.2 * rng.standard_normal((n,) * 3), 0.01)
    chi = 0.5 * ndimage.gaussian_filter(rng.standard_normal((n,) * 3), 1.5) * cabeza
    D = dipolo(n)
    fase = np.real(np.fft.ifftn(D * np.fft.fftn(chi))) + 0.002 * rng.standard_normal((n,) * 3)
    carpeta.mkdir(parents=True)
    scipy.io.savemat(carpeta / "params.mat", {"phase_use": fase, "kernel": D, "magn_use": magnitud})
    scipy.io.savemat(carpeta / "chi_cosmos.mat", {"chi_cosmos": chi})
    return carpeta


def arma_mascaras_val(raiz):
    """Una máscara de cerebro por volumen de evaluación: una caja a 3 vóxeles del borde."""
    carpeta = raiz / "mascaras" / "val"
    carpeta.mkdir(parents=True)
    for i in range(2):
        m = np.zeros((24, 26, 28), bool)
        m[3:-3, 3:-3, 3:-3] = True
        np.save(carpeta / f"sub-val{i}.npy", m)


def entorno_wandb(raiz):
    """Offline sin credencial, sin red y sin tocar el home, como el `entorno` de
    tests/test_seguimiento.py."""
    entorno = {k: v for k, v in ENTORNO.items()
               if k not in ("WANDB_API_KEY", "WANDB_IDENTITY_TOKEN_FILE", "WANDB_MODE",
                            "WANDB_BASE_URL")}
    entorno.update(NETRC=str(raiz / "no-hay-netrc"), WANDB_SILENT="true",
                   **{v: str(raiz / v.lower()) for v in ("WANDB_CONFIG_DIR", "WANDB_CACHE_DIR",
                                                         "WANDB_DATA_DIR")})
    return entorno


def qsm_jsonl(carpeta):
    return [json.loads(l) for l in (carpeta / "qsm.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def con_qsm(tmp_path_factory):
    """Las corridas de esta sección, en paralelo: sin evaluación QSM, sin ella ni la estabilidad,
    con ella sobre un COSMOS de juguete de 24^3, con un ADMM y una estabilidad que levantan, con
    una lámina QSM que levanta, con la carpeta de COSMOS incompleta y sin W&B, con los NRMSE, el
    motivo y la estabilidad forzados (con un mínimo finito o sin ninguno), con COSMOS y sin la
    estabilidad y, si wandb está instalado, offline. Después, la de QSM reanudada desde su estado
    de la 3, tres reanudaciones de la forzada que elige que miran `best_qsm.pth` y, con wandb, la
    offline extendida, reanudada desde la 3 y con una rama desde la 3."""
    raiz = tmp_path_factory.mktemp("qsm")
    base = arma_datos(raiz / "datos")
    arma_mascaras_val(raiz / "datos")
    cosmos = arma_cosmos(raiz / "cosmos")
    incompleta = raiz / "incompleta"
    incompleta.mkdir()
    shutil.copy(cosmos / "params.mat", incompleta)
    w = entorno_wandb(raiz / "wandb-global")
    con_wandb = ["--seguimiento", "offline"] if HAY_WANDB else []
    corridas = {
        "sin": lanza(base, raiz / "sin", *QSM),
        "sin-estabilidad": lanza(base, raiz / "sin-estabilidad", *QSM, "--estabilidad-lado", "0"),
        "con": lanza(base, raiz / "con", *QSM, "--qsm-datos", str(cosmos)),
        "falla": lanza(base, raiz / "falla", *QSM, "--qsm-datos", str(cosmos), *con_wandb,
                       codigo=FALLA_QSM, entorno=w),
        "lamina": lanza(base, raiz / "lamina", *QSM, "--qsm-datos", str(cosmos), *con_wandb,
                        codigo=FALLA_LAMINA, entorno=w),
        "incompleta": lanza(base, raiz / "incompleta-out", *QSM, "--qsm-datos", str(incompleta),
                            codigo=SIN_WANDB),
        "elige": lanza(base, raiz / "elige", *QSM, "--qsm-datos", str(cosmos),
                       "--eval-every", "1", "--guardar-cada", "1",
                       codigo=forzada(PASOS, NRMSE, minimos=MINIMOS, motivos=MOTIVOS)),
        "sin-minimo": lanza(base, raiz / "sin-minimo", *QSM, "--qsm-datos", str(cosmos),
                            codigo=forzada([0.5, 0.5], [math.nan] * 2, minimos=[math.nan] * 2,
                                           motivos=["no_finito"] * 2)),
        "qsm-sin-estabilidad": lanza(base, raiz / "qsm-sin-estabilidad", *QSM, "--qsm-datos",
                                     str(cosmos), "--estabilidad-lado", "0"),
        "dos-targets": lanza(base, raiz / "dos-targets", *QSM, "--qsm-datos", str(cosmos),
                             "--qsm-target-std", "0.06", "0.25", *con_wandb,
                             codigo=forzada([1.0, 1.0], DOS_FINALES, minimos=DOS_MINIMOS),
                             entorno=w),
    }
    # Validación cada 2 y QSM cada 4: en la 4 y en la última, no en la 2.
    con_w = [*QSM, "--eval-every", "2", "--seguimiento", "offline"]
    if HAY_WANDB:
        corridas["wandb"] = lanza(base, raiz / "wandb", *con_w, "--qsm-datos", str(cosmos),
                                  "--eval-qsm-cada", "4", entorno=w)
    salidas = dict(zip(corridas, espera(*corridas.values())))

    reanudada = raiz / "reanudada"
    shutil.copytree(raiz / "con", reanudada)
    segundas = {"reanudada": lanza(base, reanudada, *QSM, "--qsm-datos", str(cosmos),
                                   "--reanudar", str(reanudada / f"estado-{QSM_EVALUA:06d}.pth"))}
    # `best_qsm.pth` al reanudar, sin correr iteraciones: desde el estado de la 2 (el suyo), desde
    # el de la 1 sin COSMOS (uno ajeno, de después) y una rama desde el de la 6.
    for nombre, desde, extra in (("conserva", MEJOR_QSM, ["--qsm-datos", str(cosmos)]),
                                 ("aparta", 1, [])):
        shutil.copytree(raiz / "elige", raiz / nombre)
        segundas[nombre] = lanza(base, raiz / nombre, *QSM, *extra, "--iterations", str(desde),
                                 "--reanudar", str(raiz / nombre / f"estado-{desde:06d}.pth"))
    segundas["rama-qsm"] = lanza(base, raiz / "rama-qsm", *QSM, "--qsm-datos", str(cosmos),
                                 "--reanudar", str(raiz / "elige" / f"estado-{QSM_N:06d}.pth"))
    if HAY_WANDB:
        # Qué corrida de W&B abre `--reanudar`, sin QSM (`--qsm-datos` es libre): extender la
        # terminada, seguir desde un estado anterior a lo registrado y una rama.
        for nombre, desde, extra in (("extiende", QSM_N, ["--iterations", str(QSM_N + 2)]),
                                     ("corte", QSM_EVALUA, [])):
            shutil.copytree(raiz / "wandb", raiz / nombre)
            segundas[nombre] = lanza(base, raiz / nombre, *con_w, *extra, "--reanudar",
                                     str(raiz / nombre / f"estado-{desde:06d}.pth"), entorno=w)
        segundas["rama"] = lanza(base, raiz / "rama", *con_w, "--reanudar",
                                 str(raiz / "wandb" / f"estado-{QSM_EVALUA:06d}.pth"), entorno=w)
    salidas.update(zip(segundas, espera(*segundas.values())))
    return {"raiz": raiz, "salidas": salidas, "cosmos": cosmos}


def test_la_evaluacion_qsm_no_toca_el_entrenamiento(con_qsm):
    """Con la evaluación QSM prendida, en `referencia` y CPU, la traza, `history.json`, el log del
    entrenamiento y los pesos del estado final son bit a bit los de la corrida sin ella: no
    consume el generador del ruido ni el de los parches, y deja el modelo en modo train."""
    sin, con = con_qsm["raiz"] / "sin", con_qsm["raiz"] / "con"
    assert bit_a_bit(traza(sin), traza(con))
    assert json.loads((sin / "history.json").read_text()) == json.loads(
        (con / "history.json").read_text())
    assert log(con_qsm["salidas"]["sin"][0]) == log(con_qsm["salidas"]["con"][0])
    final_sin, final_con = estado(sin, QSM_N), estado(con, QSM_N)
    assert igual(final_sin["model"], final_con["model"])
    assert igual(final_sin["optimizer"], final_con["optimizer"])
    # Ni el generador global, que el bucle no usa y la traza no vería.
    for g in ("rng_cpu", "noise_gen"):
        assert torch.equal(final_sin[g], final_con[g]), g
    assert final_sin["qsm"] == [] and len(final_con["qsm"]) == 2


def test_la_estabilidad_no_toca_el_entrenamiento(con_qsm):
    """Con la estabilidad y sin ella (`--estabilidad-lado 0`), en `referencia` y CPU: la traza, el
    log del entrenamiento, los pesos, Adam y los generadores del estado final, bit a bit. La
    `history.json` es la misma salvo sus claves, que la corrida sin la medición no tiene."""
    sin, apagada = con_qsm["raiz"] / "sin", con_qsm["raiz"] / "sin-estabilidad"
    assert bit_a_bit(traza(sin), traza(apagada))
    assert log(con_qsm["salidas"]["sin"][0]) == log(con_qsm["salidas"]["sin-estabilidad"][0])
    final_sin, final_apagada = estado(sin, QSM_N), estado(apagada, QSM_N)
    for clave in ("model", "optimizer", "rng_cpu", "noise_gen"):
        assert igual(final_sin[clave], final_apagada[clave]), clave
    medida = json.loads((sin / "history.json").read_text())
    sin_medir = json.loads((apagada / "history.json").read_text())
    assert len(medida) == 2 and all(ESTABILIDAD <= set(h) for h in medida)
    assert [{k: v for k, v in h.items() if k not in ESTABILIDAD} for h in medida] == sin_medir
    salida = con_qsm["salidas"]["sin-estabilidad"][0]
    assert "| estabilidad apagada (--estabilidad-lado 0)" in salida
    assert "    estabilidad" not in salida


def test_la_estabilidad_mide_los_puntos_declarados(con_qsm):
    """Recalculada fuera de la corrida con los pesos de su estado final: en la validación, sobre
    el cubo central de 12^3 del primer recorte con el ruido de la validación; en QSM, sobre
    `punto_qsm3d` del COSMOS de juguete. Es lo que dejan `history.json` y `qsm.jsonl`, con el T
    de esa iteración, y el log lo dice en cada validación."""
    con, salida = con_qsm["raiz"] / "con", con_qsm["salidas"]["con"][0]
    final = estado(con, QSM_N)
    modelo = VNet3D(final["config"])
    modelo.load_state_dict(final["model"])
    datos = con_qsm["raiz"] / "datos"
    eval_set = VolumeDataset3D.from_root(datos, "val", recorte=16,
                                         names=train3d.lee_lista(datos / "lista_val.txt"))
    val = qsm3d.estabilidad(modelo, train3d.punto_validacion(eval_set, 12, train3d.SIGMA))
    qsm = qsm3d.estabilidad(modelo, qsm3d.punto_qsm3d(qsm3d.carga_cosmos(con_qsm["cosmos"]), 12))
    medidas = (json.loads((con / "history.json").read_text())[-1], qsm_jsonl(con)[-1])
    for medida, r in zip(medidas, (val, qsm)):
        assert medida["iteration"] == QSM_N
        assert medida["lambda_max"] == pytest.approx(r["lambda_max"], rel=1e-6)
        assert medida["estabilidad"] == pytest.approx(r["paso"], rel=1e-6)
        assert r["T"] == final["model"]["T"].item()
    assert "estabilidad con la cadencia de QSM, en 12^3 de validación y 12^3 de COSMOS" in salida
    for punto in ("val", "qsm"):
        assert salida.count(f"    estabilidad {punto}: (T/S)·λ_max = ") == 2, punto


def test_la_evaluacion_qsm_deja_sus_metricas(con_qsm):
    """Una línea de `qsm.jsonl` por validación, con lo que devuelve `admm_wtdv`; las mismas en el
    log, en `estado.json` (la última y la mejor) y en el estado, que las lleva al reanudar."""
    con, salida = con_qsm["raiz"] / "con", con_qsm["salidas"]["con"][0]
    filas = qsm_jsonl(con)
    assert [f["iteration"] for f in filas] == [QSM_EVALUA, QSM_N]
    for f in filas:
        assert set(f) == {"iteration", "efectiva", "target_std", "nrmse_final_pct",
                          "nrmse_min_pct", "iter_min", "iteraciones", "alpha", "segundos",
                          "motivo", "nrmse_trayectoria_pct", "update_trayectoria"} | ESTABILIDAD \
            | {"hfen_pct", "xsim", "residuo_dato_pct", "factor_cono", "chi0_nrmse_pct",
               "chi0_hfen_pct", "chi0_xsim", "chi0_residuo_dato_pct", "gt_residuo_dato_pct"}
        assert 0 < f["xsim"] <= 1 and f["hfen_pct"] > 0 and f["chi0_nrmse_pct"] > 0
        assert f["target_std"] == qsm3d.TARGET_STD == 0.06
        assert f["iteraciones"] == 3 and f["motivo"] == "max_iter"
        assert 0 < f["nrmse_min_pct"] <= f["nrmse_final_pct"] and f["alpha"] > 0
        # Las trayectorias: una entrada por iteración, la última es el final y el menor, el mínimo.
        trayectoria = f["nrmse_trayectoria_pct"]
        assert len(trayectoria) == len(f["update_trayectoria"]) == f["iteraciones"]
        assert trayectoria[-1] == f["nrmse_final_pct"]
        assert min(trayectoria) == f["nrmse_min_pct"] == trayectoria[f["iter_min"]]
    assert salida.count("    qsm NRMSE=") == 2 and "COSMOS 24x24x24" in salida
    hecho = json.loads((con / "estado.json").read_text())
    mejor = min(filas, key=lambda f: f["nrmse_final_pct"])
    assert (hecho["qsm_evaluaciones"], hecho["qsm_fallos"]) == (2, 0)
    assert hecho["qsm_nrmse_ultimo_pct"] == filas[-1]["nrmse_final_pct"]
    assert (hecho["qsm_nrmse_mejor_pct"], hecho["qsm_iteracion_mejor"]) == (
        mejor["nrmse_final_pct"], mejor["iteration"])
    assert estado(con, QSM_N)["qsm"] == filas
    sin = json.loads((con_qsm["raiz"] / "sin" / "estado.json").read_text())
    assert sin["qsm_evaluaciones"] == 0 and sin["qsm_nrmse_mejor_pct"] is None
    assert not (con_qsm["raiz"] / "sin" / "qsm.jsonl").exists()


def test_la_validacion_mide_nrmse_y_cerebro(con_qsm):
    """`history.json` lleva el NRMSE de denoising, la estabilidad (con `--eval-qsm-cada 0`, en
    todas) y, con las máscaras de los de evaluación, el PSNR en el cerebro; el log, también."""
    for h in json.loads((con_qsm["raiz"] / "con" / "history.json").read_text()):
        assert 0 < h["nrmse_pct"] and 0 < h["nrmse_pct_tejido"] and "psnr_cerebro" in h
        assert ESTABILIDAD <= set(h)
    salida = con_qsm["salidas"]["con"][0]
    assert "validación: cada 3 iteraciones y en la última | PSNR en cerebro sí" in salida
    assert ", en cerebro " in salida


def test_reanudar_con_evaluacion_qsm(con_qsm):
    """Desde el estado de la 3, en el mismo `--out`: la traza es la de la corrida seguida, y
    `qsm.jsonl` se reescribe desde el estado y sigue: la de la 3 y una nueva en la 6."""
    a, b = con_qsm["raiz"] / "con", con_qsm["raiz"] / "reanudada"
    assert bit_a_bit(traza(a), traza(b))
    sin_reloj = [{k: v for k, v in f.items() if k != "segundos"} for f in qsm_jsonl(a)]
    assert [{k: v for k, v in f.items() if k != "segundos"} for f in qsm_jsonl(b)] == sin_reloj
    assert qsm_jsonl(b)[0] == qsm_jsonl(a)[0]           # la de la 3 viene del estado, entera
    assert (a / "best_qsm.pth").exists() == (b / "best_qsm.pth").exists()
    if (a / "best_qsm.pth").exists():
        assert igual(*(torch.load(c / "best_qsm.pth", map_location="cpu", weights_only=False)
                       for c in (a, b)))


def best_qsm_esperado(carpeta):
    """El `best_qsm` que se deduce de `qsm.jsonl`: la fila sin fallo de menor NRMSE mínimo
    finito, la primera si empatan; o None."""
    filas = qsm_jsonl(carpeta) if (carpeta / "qsm.jsonl").exists() else []
    validas = [f for f in filas if "fallo" not in f and math.isfinite(f["nrmse_min_pct"])]
    mejor = min(validas, key=lambda f: f["nrmse_min_pct"], default=None)
    return mejor and {k: mejor.get(k) for k in train3d.RESUMEN_BEST_QSM}


def test_best_qsm_es_el_de_menor_nrmse_minimo(con_qsm):
    """Con los NRMSE, el motivo y la estabilidad forzados, `best_qsm.pth` es el estado de la 2,
    aunque su ADMM no cortó, su final sea 90 % y su estabilidad 7: no lo pisan ni el menor final
    (la 3), ni uno sin NRMSE finito (la 5), ni uno peor o empatado (la 4 y la 6). Lleva config,
    pesos, iteración y PSNR de esa validación más sus métricas QSM, sin el optimizer; el log lo
    dice cada vez que se escribe, y cada estado lleva el de su momento. La traza es la de la
    corrida sin QSM."""
    raiz, salida = con_qsm["raiz"], con_qsm["salidas"]["elige"][0]
    carpeta = raiz / "elige"
    assert bit_a_bit(traza(carpeta), traza(raiz / "sin"))
    filas = qsm_jsonl(carpeta)
    np.testing.assert_array_equal([f["nrmse_min_pct"] for f in filas], MINIMOS)
    np.testing.assert_array_equal([f["nrmse_final_pct"] for f in filas], NRMSE)
    assert [f["motivo"] for f in filas] == MOTIVOS
    assert [f["estabilidad"] for f in filas] == PASOS
    best = torch.load(carpeta / "best_qsm.pth", map_location="cpu", weights_only=False)
    assert set(best) == CLAVES_BEST_QSM
    origen, fila = estado(carpeta, MEJOR_QSM), filas[MEJOR_QSM - 1]
    assert igual(best["model"], origen["model"]) and best["config"] == origen["config"]
    h = origen["history"][-1]
    assert best["iteration"] == fila["iteration"] == h["iteration"] == MEJOR_QSM
    assert best["psnr"] == h["psnr"]
    metricas = ("nrmse_min_pct", "iter_min", "nrmse_final_pct", "iteraciones", "motivo",
                "estabilidad")
    assert {k: best[k] for k in metricas} == {k: fila[k] for k in metricas}
    for it in range(1, QSM_N + 1):
        assert estado(carpeta, it)["mejor_qsm"]["iteration"] == min(it, MEJOR_QSM), it
    assert igual(estado(carpeta, QSM_N)["mejor_qsm"], best)
    hecho = json.loads((carpeta / "estado.json").read_text())
    assert hecho["best_qsm"] == best_qsm_esperado(carpeta) == {
        k: fila[k] for k in train3d.RESUMEN_BEST_QSM}
    assert (hecho["best_qsm"]["nrmse_min_pct"], hecho["best_qsm"]["nrmse_final_pct"]) == (4.0, 90.0)
    assert hecho["qsm_nrmse_mejor_pct"] == 3.0          # el menor final de todas
    assert salida.count("    best_qsm.pth: NRMSE QSM ") == 2
    assert BUSCA_BEST_QSM + "\n" in salida


def test_sin_un_nrmse_finito_no_hay_best_qsm(con_qsm):
    """Sin un NRMSE mínimo finito en ninguna evaluación, con el ADMM y la estabilidad fallando y
    sin COSMOS, no hay `best_qsm.pth` y `estado.json` dice null; el arranque dice una vez qué
    busca o por qué no. Sin la estabilidad (`--estabilidad-lado 0`) se busca y se escribe igual:
    no elige."""
    raiz, salidas = con_qsm["raiz"], con_qsm["salidas"]
    for nombre, arranque in (("sin-minimo", BUSCA_BEST_QSM), ("falla", BUSCA_BEST_QSM),
                             ("sin", "best_qsm.pth: no se busca (sin COSMOS)")):
        carpeta, salida = raiz / nombre, salidas[nombre][0]
        assert not (carpeta / "best_qsm.pth").exists(), nombre
        assert json.loads((carpeta / "estado.json").read_text())["best_qsm"] is None, nombre
        assert estado(carpeta, QSM_N)["mejor_qsm"] is None, nombre
        assert [l for l in salida.splitlines() if "best_qsm.pth:" in l] == [arranque], nombre
    filas = qsm_jsonl(raiz / "sin-minimo")
    assert [f["motivo"] for f in filas] == ["no_finito"] * 2
    assert all(math.isnan(f["nrmse_min_pct"]) for f in filas)
    assert bit_a_bit(traza(raiz / "sin-minimo"), traza(raiz / "sin"))
    sin_estabilidad = raiz / "qsm-sin-estabilidad"
    assert BUSCA_BEST_QSM in salidas["qsm-sin-estabilidad"][0]
    assert len(qsm_jsonl(sin_estabilidad)) == 2 and (sin_estabilidad / "best_qsm.pth").exists()
    assert json.loads((sin_estabilidad / "estado.json").read_text())["best_qsm"][
        "estabilidad"] is None


def test_con_varios_targets_cuenta_el_de_menor_nrmse_minimo(con_qsm):
    """Con --qsm-target-std 0.06 0.25 el ADMM corre a los dos en cada evaluación: la fila de
    `qsm.jsonl` lleva arriba el de menor NRMSE mínimo y los dos en `por_target`, `best_qsm.pth` es
    el menor de todos con su target, el log dice cuál ganó y, offline, W&B lleva `qsm/*` del mejor
    con `qsm/mejor_target` y `qsm_t006/*`, `qsm_t025/*` de cada uno."""
    raiz, (salida, _) = con_qsm["raiz"], con_qsm["salidas"]["dos-targets"]
    carpeta = raiz / "dos-targets"
    filas = qsm_jsonl(carpeta)
    assert [f["target_std"] for f in filas] == [0.25, 0.06]
    assert [f["nrmse_min_pct"] for f in filas] == [4.0, 3.0]
    assert [f["nrmse_final_pct"] for f in filas] == [5.0, 4.0]
    for f, (a, b) in zip(filas, [(5.0, 4.0), (3.0, 6.0)]):
        assert set(f["por_target"]) == {"0.06", "0.25"}
        assert (f["por_target"]["0.06"]["nrmse_min_pct"],
                f["por_target"]["0.25"]["nrmse_min_pct"]) == (a, b)
    best = torch.load(carpeta / "best_qsm.pth", map_location="cpu", weights_only=False)
    assert (best["iteration"], best["target_std"], best["nrmse_min_pct"]) == (QSM_N, 0.06, 3.0)
    hecho = json.loads((carpeta / "estado.json").read_text())
    assert hecho["best_qsm"] == best_qsm_esperado(carpeta) and hecho["best_qsm"]["target_std"] == 0.06
    assert "con target_std 0.06, 0.25" in salida
    assert salida.count("    qsm 0.06 NRMSE=") == salida.count("    qsm 0.25 NRMSE=") == 2
    assert "    qsm: mejor target 0.25 (4.00 %)" in salida
    assert "    qsm: mejor target 0.06 (3.00 %)" in salida
    assert "target 0.06), el menor hasta ahora" in salida
    if HAY_WANDB:
        from test_seguimiento import historia
        rid = json.loads((carpeta / "seguimiento.json").read_text())["id"]
        h = historia(carpeta, rid)
        assert (h[QSM_EVALUA]["qsm/mejor_target"], h[QSM_N]["qsm/mejor_target"]) == (0.25, 0.06)
        assert (h[QSM_EVALUA]["qsm/nrmse_min_pct"], h[QSM_N]["qsm/nrmse_min_pct"]) == (4.0, 3.0)
        assert (h[QSM_EVALUA]["qsm_t006/nrmse_min_pct"], h[QSM_EVALUA]["qsm_t025/nrmse_min_pct"]) == (
            5.0, 4.0)
        assert (h[QSM_N]["qsm_t006/nrmse_min_pct"], h[QSM_N]["qsm_t025/nrmse_min_pct"]) == (3.0, 6.0)
        for paso in (QSM_EVALUA, QSM_N):
            assert {"qsm/update_final", "qsm_t006/update_final", "qsm_t025/update_final",
                    "qsm_t006/fallo", "qsm_t025/fallo"} <= set(h[paso]), paso


def test_best_qsm_sale_de_qsm_jsonl(con_qsm):
    """En toda corrida de esta sección, `best_qsm` de `estado.json` es el que se deduce de
    `qsm.jsonl`, y `best_qsm.pth` existe sólo si hay uno, con su iteración: también con la
    estabilidad que da de verdad el modelo de test, al reanudar y en las ramas."""
    carpetas = sorted(c for c in con_qsm["raiz"].iterdir() if (c / "estado.json").exists())
    assert len(carpetas) == (18 if HAY_WANDB else 14)
    for carpeta in carpetas:
        esperado = best_qsm_esperado(carpeta)
        assert json.loads((carpeta / "estado.json").read_text())["best_qsm"] == esperado, carpeta
        assert (carpeta / "best_qsm.pth").exists() == (esperado is not None), carpeta
        if esperado is not None:
            best = torch.load(carpeta / "best_qsm.pth", map_location="cpu", weights_only=False)
            assert {k: best[k] for k in esperado} == esperado, carpeta


def test_reanudar_repone_o_aparta_best_qsm(con_qsm):
    """Desde el estado de la 2, el `best_qsm.pth` del directorio es el suyo y se queda tal cual.
    Desde el de la 1, sin COSMOS, es de iteraciones que no están en la traza: se aparta como
    `best_qsm-descartada-000002.pth`, se repone el del estado y el arranque dice que lo hereda.
    Una rama desde la 6 hereda el del tronco."""
    raiz, salidas = con_qsm["raiz"], con_qsm["salidas"]
    carga = lambda ruta: torch.load(ruta, map_location="cpu", weights_only=False)
    tronco = raiz / "elige" / "best_qsm.pth"
    conserva, aparta, rama = raiz / "conserva", raiz / "aparta", raiz / "rama-qsm"
    assert (conserva / "best_qsm.pth").read_bytes() == tronco.read_bytes()
    assert not list(conserva.glob("best_qsm-descartada-*"))
    assert igual(carga(aparta / f"best_qsm-descartada-{MEJOR_QSM:06d}.pth"), carga(tronco))
    assert igual(carga(aparta / "best_qsm.pth"), estado(aparta, 1)["mejor_qsm"])
    primera = qsm_jsonl(raiz / "elige")[0]
    assert (primera["nrmse_min_pct"], primera["estabilidad"]) == (MINIMOS[0], PASOS[0])
    assert json.loads((aparta / "estado.json").read_text())["best_qsm"] == {
        k: primera[k] for k in train3d.RESUMEN_BEST_QSM}
    assert ("best_qsm.pth: no se busca (sin COSMOS) | hereda el del estado, de la 1"
            in salidas["aparta"][0])
    assert igual(carga(rama / "best_qsm.pth"), carga(tronco))
    assert f"| hereda el del estado, de la {MEJOR_QSM}\n" in salidas["rama-qsm"][0]


def test_un_fallo_de_la_evaluacion_qsm_no_mata_la_corrida(con_qsm):
    """Con un ADMM y una estabilidad que levantan `torch.cuda.OutOfMemoryError` en cada
    evaluación, la corrida termina, la traza es la de la corrida sin QSM, y cada fallo queda en
    el log, en `qsm.jsonl` (y el de la estabilidad de validación, en `history.json`), en
    `estado.json` y, offline, como `qsm/fallo` = 1 en W&B, sin métricas de estabilidad."""
    raiz, (salida, _) = con_qsm["raiz"], con_qsm["salidas"]["falla"]
    assert bit_a_bit(traza(raiz / "falla"), traza(raiz / "sin"))
    assert salida.count("    qsm FALLÓ (OutOfMemoryError: CUDA out of memory (simulado))") == 2
    for punto in ("val", "qsm"):
        assert salida.count(f"    estabilidad {punto} FALLÓ (OutOfMemoryError: CUDA out of "
                            "memory (simulado)); el entrenamiento sigue") == 2, punto
    filas = qsm_jsonl(raiz / "falla")
    assert [f["iteration"] for f in filas] == [QSM_EVALUA, QSM_N]
    assert all(f["fallo"].startswith("OutOfMemoryError") for f in filas)
    historia_json = json.loads((raiz / "falla" / "history.json").read_text())
    for fila in filas + historia_json:
        assert fila["estabilidad_fallo"].startswith("OutOfMemoryError")
        assert not ESTABILIDAD & set(fila)
    hecho = json.loads((raiz / "falla" / "estado.json").read_text())
    assert hecho["terminada"] and hecho["qsm_fallos"] == 2 and hecho["qsm_nrmse_mejor_pct"] is None
    if HAY_WANDB:
        from test_seguimiento import historia
        rid = json.loads((raiz / "falla" / "seguimiento.json").read_text())["id"]
        h = historia(raiz / "falla", rid)
        assert h[QSM_EVALUA]["qsm/fallo"] == h[QSM_N]["qsm/fallo"] == 1
        assert "qsm/nrmse_final_pct" not in h[QSM_N]
        assert not {"val/estabilidad", "qsm/estabilidad"} & set(h[QSM_N])


def test_una_lamina_qsm_rota_no_tira_sus_metricas(con_qsm):
    """Si armar la lámina de QSM levanta, se pierde la imagen y nada más: las métricas del ADMM
    quedan en `qsm.jsonl` (las de la corrida sin la falla), en `estado.json` y en W&B, sin
    `qsm/fallo`. Sin seguimiento la lámina ni se arma."""
    raiz, (salida, _) = con_qsm["raiz"], con_qsm["salidas"]["lamina"]
    assert bit_a_bit(traza(raiz / "lamina"), traza(raiz / "sin"))
    sin_reloj = [[{k: v for k, v in f.items() if k != "segundos"} for f in qsm_jsonl(raiz / c)]
                 for c in ("lamina", "con")]
    assert sin_reloj[0] == sin_reloj[1] and len(sin_reloj[0]) == 2
    assert json.loads((raiz / "lamina" / "estado.json").read_text())["qsm_fallos"] == 0
    rotas = salida.count("seguimiento: la lámina qsm/reconstruccion falló (RuntimeError: lámina")
    assert rotas == (2 if HAY_WANDB else 0) and "qsm FALLÓ" not in salida
    if HAY_WANDB:
        from test_seguimiento import historia
        rid = json.loads((raiz / "lamina" / "seguimiento.json").read_text())["id"]
        fila = historia(raiz / "lamina", rid)[QSM_N]
        assert fila["qsm/fallo"] == 0 and fila["qsm/nrmse_final_pct"] == qsm_jsonl(
            raiz / "lamina")[-1]["nrmse_final_pct"]
        assert "qsm/reconstruccion" not in fila and fila["val/denoising"]["_type"] == "image-file"


@pytest.mark.skipif(not HAY_WANDB, reason="sin wandb: el .venv principal no lo tiene")
def test_reanudar_abre_la_corrida_de_wandb_que_corresponde(con_qsm):
    """Extender una corrida terminada sigue su corrida de W&B, con los pasos nuevos detrás de los
    viejos. Reanudar desde un estado anterior a lo ya registrado (lo común tras un corte) abre otra
    con el mismo nombre y deja la vieja en `descartadas`. Una rama es otra, en el grupo del tronco
    y con su id de padre."""
    raiz, salidas = con_qsm["raiz"], con_qsm["salidas"]
    from test_seguimiento import historia, identidad
    tronco = identidad(raiz / "wandb")
    extiende, corte, rama = (identidad(raiz / c) for c in ("extiende", "corte", "rama"))
    assert extiende["id"] == tronco["id"] and extiende["paso"] == QSM_N + 2
    assert f"corrida {tronco['id']}, reabierta" in salidas["extiende"][0]
    assert sorted(historia(raiz / "extiende", tronco["id"])) == [2, 4, 6, 8]
    assert corte["id"] != tronco["id"] and corte["descartadas"] == [tronco["id"]]
    assert f"corrida {corte['id']}, nueva" in salidas["corte"][0]
    assert sorted(historia(raiz / "corte", corte["id"])) == [4, 6]
    assert rama["id"] not in (tronco["id"], corte["id"])
    assert (rama["grupo"], rama["padre_id"]) == ("wandb", tronco["id"])


def test_sin_datos_qsm_avisa_y_sigue_sin_wandb(con_qsm):
    """Con la carpeta de COSMOS sin `chi_cosmos.mat`, un aviso que dice qué falta y de dónde
    copiarlo, y la corrida termina sin evaluación QSM. Con `--seguimiento no` wandb ni se importa
    (el programa lo verifica al salir): corre igual en el .venv principal, que no lo tiene."""
    salida, carpeta = con_qsm["salidas"]["incompleta"][0], con_qsm["raiz"] / "incompleta-out"
    aviso = [l for l in salida.splitlines() if l.startswith("AVISO: evaluación QSM apagada")]
    assert len(aviso) == 1 and "falta chi_cosmos.mat" in aviso[0] and "../tdv/data_mat" in aviso[0]
    assert "| QSM apagada" in salida and "seguimiento: sin W&B (--seguimiento no)" in salida
    assert json.loads((carpeta / "estado.json").read_text())["terminada"]
    assert not (carpeta / "qsm.jsonl").exists() and not (carpeta / "wandb").exists()
    assert not (carpeta / "seguimiento.json").exists()


@pytest.mark.skipif(not HAY_WANDB, reason="sin wandb: el .venv principal no lo tiene")
def test_seguimiento_offline_de_una_corrida(con_qsm):
    """`--seguimiento offline` deja la corrida en `<out>/wandb` y su id en `seguimiento.json`.
    W&B tiene, por paso: `train/*` cada --log-every con los nombres de `metricas_traza` (las 14
    columnas), s/it y el max|g| del log; `val/*` y la lámina de denoising en cada validación;
    `qsm/*` y su lámina sólo donde toca (`--eval-qsm-cada 4`: la 4 y la última, no la 2). La
    config lleva el micro-lote efectivo y el commit; el resumen, lo de `estado.json`. Con W&B, la
    evaluación QSM y otro calendario de validación, la traza es la de la corrida sin nada."""
    from test_seguimiento import corrida, historia, resumen_de
    out = con_qsm["raiz"] / "wandb"
    assert bit_a_bit(traza(out), traza(con_qsm["raiz"] / "sin"))
    ident = json.loads((out / "seguimiento.json").read_text())
    assert ident["modo"] == "offline" and ident["paso"] == QSM_N
    assert glob.glob(str(out / "wandb" / f"offline-run-*-{ident['id']}"))
    assert "seguimiento: W&B offline (modo offline)" in con_qsm["salidas"]["wandb"][0]
    h = historia(out, ident["id"])
    assert sorted(h) == [2, 4, 6]
    for paso in (2, 4, 6):
        fila = h[paso]
        assert all(fila[k] == v for k, v in seguimiento.metricas_traza(traza(out), paso).items())
        assert fila["train/s_por_it"] > 0
        assert fila["train/grad_norm_max"] >= fila["train/grad_norm"]
        assert {"val/psnr", "val/psnr_tejido", "val/psnr_cerebro", "val/nrmse_pct",
                "val/nrmse_pct_tejido"} <= set(fila)
        assert fila["val/denoising"]["_type"] == "image-file"
        assert ("qsm/nrmse_final_pct" in fila) == (paso != 2), paso
        assert ("qsm/reconstruccion" in fila) == (paso != 2), paso
        # La estabilidad va con la cadencia de QSM, también la de validación.
        for clave in ("val/estabilidad", "val/lambda_max", "qsm/estabilidad", "qsm/lambda_max"):
            assert (clave in fila) == (paso != 2), (clave, paso)
    filas = qsm_jsonl(out)
    assert [f["iteration"] for f in filas] == [4, 6]
    assert h[6]["qsm/nrmse_final_pct"] == filas[-1]["nrmse_final_pct"] and h[6]["qsm/fallo"] == 0
    historia_json = json.loads((out / "history.json").read_text())
    assert [x["psnr"] for x in historia_json] == [h[p]["val/psnr"] for p in (2, 4, 6)]
    assert [ESTABILIDAD <= set(x) for x in historia_json] == [False, True, True]
    for paso, x, f in zip((4, 6), historia_json[1:], filas):
        assert (h[paso]["val/estabilidad"], h[paso]["val/lambda_max"]) == (
            x["estabilidad"], x["lambda_max"])
        assert (h[paso]["qsm/estabilidad"], h[paso]["qsm/lambda_max"]) == (
            f["estabilidad"], f["lambda_max"])
    _, config = corrida(out, ident["id"])
    assert config["micro_lote"] == 1 and "commit" in config and config["seguimiento"] == "offline"
    final = resumen_de(out, ident["id"])
    hecho = json.loads((out / "estado.json").read_text())
    assert final["mejor_psnr"] == hecho["mejor_psnr"]
    assert final["qsm_nrmse_mejor_pct"] == hecho["qsm_nrmse_mejor_pct"]
    assert final["val/psnr"] == {"max": hecho["mejor_psnr"]}


def test_reanudar_un_estado_sin_banderas_nuevas(corridas, tmp_path):
    """Un estado anterior a una bandera de receta (uno escrito sin `mascara_cerebro`), a las
    del seguimiento, la evaluación QSM y la estabilidad y a la clave
    `qsm` se reanuda:
    la bandera que falta vale su default. Pasarla prendida sí cambia la receta."""
    carpeta = tmp_path / "corrida"
    shutil.copytree(corridas["reanudada"]["receta", K], carpeta)
    ruta = carpeta / f"estado-{K:06d}.pth"
    viejo = estado(carpeta, K)
    for clave in ("mascara_cerebro", "seguimiento", "wandb_proyecto", "qsm_datos",
                  "eval_qsm_cada", "qsm_max_iter", "qsm_target_std", "estabilidad_lado",
                  "meseta_desde",
                  "meseta_psnr"):
        del viejo["args"][clave]
    del viejo["qsm"], viejo["mejor_qsm"]
    torch.save(viejo, ruta)
    (_, prende), = espera(lanza(corridas["base"], carpeta, *VARIANTES["receta"],
                                "--mascara-cerebro", "--reanudar", str(ruta)), codigo=2)
    assert "cambian ['mascara_cerebro']" in prende
    espera(lanza(corridas["base"], carpeta, *VARIANTES["receta"], "--iterations", str(K + 1),
                 "--reanudar", str(ruta)))
    assert bit_a_bit(traza(carpeta), traza(corridas["seguida"]["receta"])[:K + 1])
    assert json.loads((carpeta / "estado.json").read_text())["terminada"]


def test_la_mascara_de_cerebro_no_cambia_los_psnr_de_antes(tmp_path):
    """Con `mascaras`, el recorte es el mismo (la ventana sale del tejido) y la máscara sale de la
    misma ventana; los PSNR y NRMSE de antes no cambian, y se agregan los del cerebro. Una
    máscara de otra forma es un error."""
    rng = np.random.default_rng(1)
    archivos, carpeta = [], tmp_path / "mascaras"
    carpeta.mkdir()
    for i in range(2):
        vol = volumen(rng, (20, 22, 24)).astype(np.float16)
        archivos.append(tmp_path / f"v{i}.npy")
        np.save(archivos[-1], vol)
        np.save(carpeta / f"v{i}.npy", vol > 0.5)
    sin = VolumeDataset3D(archivos, recorte=16)
    con = VolumeDataset3D(archivos, recorte=16, mascaras=carpeta)
    assert sin.cerebro is None and len(con.cerebro) == 2
    for recorte, otro, cerebro in zip(sin.recortes, con.recortes, con.cerebro):
        assert np.array_equal(recorte, otro) and np.array_equal(cerebro, recorte > 0.5)

    torch.manual_seed(0)
    modelo = VNet3D(train3d.vnet_config3d(train3d.T_INIT, 4)).eval()
    a = evaluate_denoising3d(modelo, sin, seed=7)
    b = evaluate_denoising3d(modelo, con, seed=7, devuelve_primero=True)
    assert all(a[k] == b[k] for k in a if k != "per_volume")
    assert "psnr_denoised_cerebro" not in a and "primero" not in a
    limpia, ruidosa, salida = b["primero"]
    m = con.cerebro[0]
    assert np.array_equal(limpia, con.recortes[0])
    fila = b["per_volume"][0]
    assert fila["psnr_noisy_cerebro"] == pytest.approx(psnr(ruidosa[m], limpia[m]), abs=1e-6)
    assert fila["psnr_denoised_cerebro"] == pytest.approx(psnr(salida[m], limpia[m]), abs=1e-6)
    error = salida.astype(np.float64) - limpia
    assert fila["nrmse_denoised"] == pytest.approx(
        100 * np.linalg.norm(error) / np.linalg.norm(limpia.astype(np.float64)), rel=1e-9)

    np.save(carpeta / "v1.npy", np.ones((20, 22, 23), bool))
    with pytest.raises(ValueError, match="máscara"):
        VolumeDataset3D(archivos, recorte=16, mascaras=carpeta)


# --- TROZOS DESIGUALES, SOBREAJUSTE, MESETA Y W&B ONLINE -------------------------------------

# Imprime la huella de cada lote ruidoso que arma el bucle: el sobreajuste repite el mismo en cada
# iteración, y su primero es el de la corrida normal con la misma semilla.
HUELLAS = ("import hashlib, sys, train3d\n"
           "def con_huella(f):\n"
           "    def g(lote, *a, **k):\n"
           "        r = f(lote, *a, **k)\n"
           "        print('RUIDOSA', len(lote), hashlib.sha1(r.numpy().tobytes()).hexdigest(),\n"
           "              flush=True)\n"
           "        return r\n"
           "    return g\n"
           "train3d.add_noise = con_huella(train3d.add_noise)\n"
           "train3d.main(sys.argv[1:])\n")
# El NRMSE QSM mínimo de cada evaluación de la meseta: con paciencia 1 y 2 mitades, la 1 y la 2
# mejoran, la 3 y la 4 parten el lr (hitos 3 y 4), la 5 mejora y la 6 corta. La 7 y la 8 no llegan
# a correr. El final salta a contramano, como el de un ADMM que no converge: la meseta no lo mira.
NRMSE_MESETA = [5.0, 4.0, 4.5, 4.2, 3.9, 4.0, 9.0, 9.0]
FINAL_MESETA = [90.0, 140.0, 30.0, 20.0, 100.0, 10.0, 9.0, 9.0]
MESETA = ["--iterations", "8", "--eval-every", "1", "--guardar-cada", "1", "--meseta", "1",
          "--meseta-mitades", "2"]
# La meseta Y desde la 3, con el NRMSE mínimo y el PSNR forzados: la 1 y la 2 (1.0 %, que ganarían
# a todas) no cuentan y la 3 fija los dos mejores; en la 4 el NRMSE no mejora pero el PSNR sí, y no
# parte; en la 5 ninguno: lr/2 desde la 6; en la 6 mejora el NRMSE; en la 7 ninguno y, sin mitades,
# corta. La 8 no llega a correr.
NRMSE_Y = [1.0, 1.0, 5.0, 6.0, 6.0, 4.0, 4.5, 9.0]
PSNR_Y = [20.0, 20.0, 20.0, 21.0, 20.5, 20.0, 20.0, 20.0]
MESETA_Y = ["--iterations", "8", "--eval-every", "1", "--guardar-cada", "1", "--meseta", "1",
            "--meseta-mitades", "1", "--meseta-desde", "3", "--meseta-psnr"]
# W&B que no abre aunque haya credencial, como con la red caída al arrancar.
NO_ABRE = ("import sys, train3d\n"
           "train3d.seguimiento._hay_credencial = lambda: True\n"
           "train3d.seguimiento.abre = lambda *a, **k: None\n"
           "train3d.main(sys.argv[1:])\n")


def huellas(salida):
    """`(tamaño, huella)` de cada lote ruidoso, en orden."""
    return [tuple(l.split()[1:]) for l in salida.splitlines() if l.startswith("RUIDOSA")]


def forzada_psnr(minimos, psnrs):
    """`forzada` con el NRMSE mínimo de cada evaluación QSM en `minimos` y, además, el PSNR de cada
    validación en `psnrs`: el que elige `best.pth` y mira --meseta-psnr sin --mascara-cerebro."""
    n = len(minimos)
    return forzada([1.0] * n, [50.0] * n, minimos=minimos).replace(
        "train3d.main(sys.argv[1:])\n",
        f"psnrs = {psnrs!r}\n"
        "ev, n_ev = train3d.evaluate_denoising3d, itertools.count()\n"
        "train3d.evaluate_denoising3d = lambda *a, **k: {**ev(*a, **k),\n"
        "                                                'psnr_denoised': psnrs[next(n_ev)]}\n"
        "train3d.main(sys.argv[1:])\n")


@pytest.fixture(scope="module")
def nuevas(tmp_path_factory):
    """Las corridas de esta sección, en paralelo: el lote de 3 en trozos de 1, de 2 + 1 y de 3; el
    sobreajuste de ese lote; la meseta con el NRMSE forzado y la que W&B no abre. Después, la
    meseta reanudada desde su estado de la 3."""
    raiz = tmp_path_factory.mktemp("nuevas")
    base = arma_datos(raiz / "datos")
    cosmos = arma_cosmos(raiz / "cosmos")
    # Sin el punto de estabilidad, que también pasa por `add_noise`: las huellas son las del bucle.
    corto = ["--iterations", "4", "--eval-every", "2", "--lote", "3", "--estabilidad-lado", "0"]
    procesos = {f"trozos-{m}": lanza(base, raiz / f"trozos-{m}", *corto, "--micro-lote", str(m),
                                     codigo=HUELLAS) for m in (1, 2, 3)}
    procesos["sobreajuste"] = lanza(base, raiz / "sobreajuste", *corto, "--micro-lote", "3",
                                    "--sobreajuste", codigo=HUELLAS)
    procesos["meseta"] = lanza(base, raiz / "meseta", *QSM, "--qsm-datos", str(cosmos), *MESETA,
                               codigo=forzada([1.0] * 8, FINAL_MESETA, minimos=NRMSE_MESETA))
    procesos["meseta-y"] = lanza(base, raiz / "meseta-y", *QSM, "--qsm-datos", str(cosmos),
                                 *MESETA_Y, codigo=forzada_psnr(NRMSE_Y, PSNR_Y))
    no_abre = lanza(base, raiz / "no-abre", "--seguimiento", "online", codigo=NO_ABRE)
    salidas = dict(zip(procesos, (out for out, _ in espera(*procesos.values()))))
    (_, error_no_abre), = espera(no_abre, codigo=2)

    reanudada = raiz / "meseta-reanudada"
    reanudada.mkdir()
    shutil.copy(raiz / "meseta" / "estado-000003.pth", reanudada)
    (salidas["meseta-reanudada"], _), = espera(lanza(
        base, reanudada, *QSM, "--qsm-datos", str(cosmos), *MESETA,
        "--reanudar", str(reanudada / "estado-000003.pth"),
        codigo=forzada([1.0] * 5, FINAL_MESETA[3:], minimos=NRMSE_MESETA[3:])))
    return {"raiz": raiz, "salidas": salidas, "no_abre": error_no_abre}


def test_un_lote_en_trozos_desiguales_es_el_mismo_lote(nuevas):
    """El lote de 3 partido en 1 + 1 + 1, en 2 + 1 y entero: los mismos parches con el mismo ruido
    (16^3 es múltiplo de 16) y el mismo gradiente salvo el orden de suma. Lo que no depende de la
    partición, idéntico; la columna 10 es la del primer trozo, y con un solo trozo es la 5."""
    raiz, salidas = nuevas["raiz"], nuevas["salidas"]
    t = {m: traza(raiz / f"trozos-{m}") for m in (1, 2, 3)}
    for m in (1, 2):
        np.testing.assert_allclose(t[m][:, :6], t[3][:, :6], rtol=1e-5)
        np.testing.assert_array_equal(t[m][:, 11:], t[3][:, 11:])
    assert bit_a_bit(t[3][:, 10], t[3][:, 5]) and not np.array_equal(t[2][:, 10], t[2][:, 5])
    assert [h[0] for h in huellas(salidas["trozos-2"])] == ["2", "1"] * 4
    for m, trozos in ((1, "3x1"), (2, "2+1"), (3, "1x3")):
        assert f"lote 3 = {trozos} de 16^3" in salidas[f"trozos-{m}"]
        assert json.loads((raiz / f"trozos-{m}" / "config.json").read_text())["micro_lote"] == m


def test_el_sobreajuste_repite_el_mismo_lote_con_el_mismo_ruido(nuevas):
    """Cada iteración, el mismo lote ruidoso: el primero de la corrida normal con la misma semilla
    (los 3 volúmenes de `arma_datos` son los primeros 3). La primera pérdida es la de esa corrida,
    bit a bit, y después baja; va sin las cotas del disyuntor, y la validación lleva el PSNR del
    lote."""
    raiz, salidas = nuevas["raiz"], nuevas["salidas"]
    fijo, normal = huellas(salidas["sobreajuste"]), huellas(salidas["trozos-3"])
    assert len(fijo) == 4 and len(set(fijo)) == 1 and fijo[0] == normal[0]
    assert len(set(normal)) == 4
    t = traza(raiz / "sobreajuste")
    assert bit_a_bit(t[:1, 0], traza(raiz / "trozos-3")[:1, 0]) and t[-1, 0] < t[0, 0]
    assert (t[:, 8] == 0).all()
    config = json.loads((raiz / "sobreajuste" / "config.json").read_text())
    assert config["guard"] == config["guard_abs"] == math.inf
    historia = json.loads((raiz / "sobreajuste" / "history.json").read_text())
    for entrada in historia:
        mse = t[entrada["iteration"] - 1, 0] / 16 ** 3
        assert entrada["psnr_lote"] == pytest.approx(10 * math.log10(1 / mse), rel=1e-9)
    assert "psnr_lote" not in json.loads((raiz / "trozos-3" / "history.json").read_text())[0]
    assert "sin las cotas del disyuntor" in salidas["sobreajuste"]


def test_paso_meseta():
    """La regla, evaluación por evaluación. Con paciencia 1, toda evaluación que no mejora parte el
    lr hasta agotar las mitades, y la siguiente corta; con 2 hacen falta dos seguidas, y la cuenta
    vuelve a cero al mejorar y al partir. Un NRMSE que no es finito no cuenta: ni mejora ni suma."""
    m = train3d.meseta_nueva()
    acciones = [train3d.paso_meseta(m, x, it, 1, 2) for it, x in enumerate(NRMSE_MESETA[:6], 1)]
    assert acciones == ["mejora", "mejora", "mitad", "mitad", "mejora", "corte"]
    assert m == {"hitos": [3, 4], "sin_mejora": 0, "mejor": 3.9, "it_mejor": 5}
    m = train3d.meseta_nueva()
    acciones = [train3d.paso_meseta(m, x, it, 2, 1)
                for it, x in enumerate([5, 5.1, 4.9, 5, 5.2, 5.3, 5.4], 1)]
    assert acciones == ["mejora", "espera", "mejora", "espera", "mitad", "espera", "corte"]
    m = train3d.meseta_nueva()
    assert [train3d.paso_meseta(m, x, it, 1, 1) for it, x in
            enumerate([math.nan, math.inf, 3.0, math.nan, 3.5], 1)] == [
        "nula", "nula", "mejora", "nula", "mitad"]
    assert m == {"hitos": [5], "sin_mejora": 0, "mejor": 3.0, "it_mejor": 3}
    assert train3d.lr_en(5, m["hitos"]) == train3d.LR
    assert train3d.lr_en(6, m["hitos"]) == train3d.LR / 2


def test_paso_meseta_con_psnr():
    """La regla Y (--meseta-psnr): cada métrica lleva su mejor y su cuenta, y la mitad o el corte
    piden las dos en la paciencia. Con el NRMSE estancado desde la 1, sólo NRMSE partiría en la 3;
    con el PSNR haciendo récord en la 2 y la 3, espera hasta la 5. Al partir vuelven a cero las dos
    cuentas, y un empate no es récord en ninguna. Un PSNR no finito anula la evaluación."""
    nrmse = [5.0, 5.1, 5.2, 5.3, 5.0, 4.9, 5.0, 5.0]
    psnr = [30.0, 30.1, 30.2, 30.1, 30.2, 30.0, 30.0, 30.0]
    solo = train3d.meseta_nueva()
    assert [train3d.paso_meseta(solo, x, it, 2, 1) for it, x in enumerate(nrmse[:3], 1)] == [
        "mejora", "espera", "mitad"]
    m = train3d.meseta_nueva()
    acciones = [train3d.paso_meseta(m, x, it, 2, 1, psnr=y)
                for it, (x, y) in enumerate(zip(nrmse, psnr), 1)]
    assert acciones == ["mejora", "mejora", "mejora", "espera", "mitad", "mejora", "espera",
                        "corte"]
    assert m == {"hitos": [5], "sin_mejora": 0, "mejor": 4.9, "it_mejor": 6,
                 "mejor_psnr": 30.2, "it_mejor_psnr": 3, "sin_mejora_psnr": 0}
    m = train3d.meseta_nueva()
    assert [train3d.paso_meseta(m, 5.0, 1, 1, 1, psnr=math.nan),
            train3d.paso_meseta(m, 5.0, 2, 1, 1, psnr=30.0)] == ["nula", "mejora"]
    assert m["it_mejor"] == m["it_mejor_psnr"] == 2


def test_la_evaluacion_qsm_para_si_empeora_con_techo_de_25(monkeypatch):
    """`evalua_qsm` le pasa al ADMM `tol` = -inf (sin el corte por umbral del autor), la parada
    por NRMSE y el techo de --qsm-max-iter, 25 por defecto; devuelve el NRMSE y el update de cada
    iteración."""
    visto = {}

    def admm(denoisers, datos, **kw):
        visto.update(kw)
        return {"chi": np.zeros(1), "nrmse": [40.0, 30.0], "x_update": [100.0, 9.0],
                "alpha": 2.0, "iteraciones": 2, "motivo": "max_iter", "nrmse_final": 30.0,
                "nrmse_min": 30.0, "it_min": 1, "k": 1, "segundos": 0.1}

    monkeypatch.setattr(train3d, "admm_wtdv", admm)
    r = train3d.evalua_qsm(torch.nn.Linear(1, 1), {}, "cpu", None, 25)
    assert visto["max_iter"] == 25 and visto["tol"] == -math.inf
    assert visto["subidas_para_cortar"] == train3d.QSM_SUBIDAS == 2
    assert r["nrmse_trayectoria_pct"] == [40.0, 30.0] and r["update_trayectoria"] == [100.0, 9.0]
    assert train3d.QSM_MAX_ITER == 25
    assert train3d.opciones().get_default("qsm_max_iter") == 25


def test_qsm_target_std_llega_al_admm_y_al_punto_de_estabilidad(monkeypatch):
    """--qsm-target-std (0.06 por defecto, el TARGET_STD del oficial) es el `target_std` del ADMM
    de la evaluación y el del punto donde se mide la estabilidad en QSM."""
    visto = []

    def admm(denoisers, datos, **kw):
        visto.append(kw["target_std"])
        return {"chi": np.zeros(1), "nrmse": [30.0], "x_update": [1.0], "alpha": 2.0,
                "iteraciones": 1, "motivo": "max_iter", "nrmse_final": 30.0, "nrmse_min": 30.0,
                "it_min": 0, "k": 1, "segundos": 0.1}

    monkeypatch.setattr(train3d, "admm_wtdv", admm)
    train3d.evalua_qsm(torch.nn.Linear(1, 1), {}, "cpu", None, 25)
    train3d.evalua_qsm(torch.nn.Linear(1, 1), {}, "cpu", None, 25, 0.25)
    assert visto == [qsm3d.TARGET_STD, 0.25]

    def punto(datos, lado, **kw):
        visto.append(kw["target_std"])
        return np.zeros((lado,) * 3, dtype=np.float32)

    monkeypatch.setattr(train3d, "punto_qsm3d", punto)
    args = train3d.opciones().parse_args(["--out", "x", "--qsm-target-std", "0.25",
                                          "--estabilidad-lado", "4"])
    eval_set = [(torch.zeros(1, 8, 8, 8), "v")]
    train3d.arma_puntos_de_estabilidad(args, eval_set, {"mascara": np.ones((8, 8, 8), bool)},
                                       torch.device("cpu"))
    assert visto[-1] == args.qsm_target_std[0] == 0.25
    assert train3d.opciones().get_default("qsm_target_std") == [qsm3d.TARGET_STD] == [0.06]
    assert "qsm_target_std" in train3d.LIBRES


def test_la_meseta_parte_el_lr_y_corta(nuevas):
    """Con el NRMSE mínimo forzado (y el final a contramano): lr/2 desde la 4 y desde la 5, y en la
    6, sin mitades, el corte. La traza tiene 6 filas, el estado de la 6 se guarda, y `estado.json`
    dice terminada, con el corte y las iteraciones hechas. Reanudada desde la 3, la misma corrida
    bit a bit."""
    raiz, salidas = nuevas["raiz"], nuevas["salidas"]
    filas = qsm_jsonl(raiz / "meseta")
    assert [f["nrmse_min_pct"] for f in filas] == NRMSE_MESETA[:6]
    assert [f["nrmse_final_pct"] for f in filas] == FINAL_MESETA[:6]
    t = traza(raiz / "meseta")
    assert t.shape[0] == 6
    np.testing.assert_array_equal(t[:, 11], train3d.LR * np.array([1, 1, 1, 0.5, 0.25, 0.25]))
    hecho = json.loads((raiz / "meseta" / "estado.json").read_text())
    assert hecho["terminada"] and hecho["iteraciones_pedidas"] == 8
    assert hecho["iteraciones_hechas"] == 6
    assert hecho["corte"] == {"motivo": "meseta del NRMSE QSM", "iteracion": 6, "mitades": 2}
    assert hecho["meseta"]["hitos"] == [3, 4] and hecho["meseta"]["mejor"] == 3.9
    assert (raiz / "meseta" / "estado-000006.pth").exists()
    assert not (raiz / "meseta" / "estado-000007.pth").exists()
    assert len(qsm_jsonl(raiz / "meseta")) == 6
    lineas = [l for l in salidas["meseta"].splitlines() if "meseta:" in l]
    assert "lr/2 desde la 4, mitad 1 de 2" in lineas[1] and "desde la 5, mitad 2" in lineas[2]
    assert "la corrida se corta en la 6" in lineas[3]
    assert "cortada por meseta en la 6, tras 2 mitades" in salidas["meseta"]
    seguida, reanudada = raiz / "meseta", raiz / "meseta-reanudada"
    assert bit_a_bit(traza(reanudada), t)
    otra = json.loads((reanudada / "estado.json").read_text())
    assert all(otra[k] == hecho[k] for k in ("iteraciones_hechas", "corte", "meseta"))
    sin_reloj = [{k: v for k, v in f.items() if k != "segundos"} for f in qsm_jsonl(seguida)]
    assert [{k: v for k, v in f.items() if k != "segundos"}
            for f in qsm_jsonl(reanudada)] == sin_reloj


def test_la_meseta_y_mira_desde_una_iteracion(nuevas):
    """--meseta-desde 3 y --meseta-psnr con el NRMSE mínimo y el PSNR forzados (NRMSE_Y, PSNR_Y):
    las evaluaciones de la 1 y la 2 no cuentan, el PSNR que sube en la 4 frena la mitad, ninguno en
    la 5 la da (lr/2 desde la 6), y en la 7, sin mitades, corta. Sin --meseta-desde, el 1.0 % de
    la 1 habría partido el lr en la 2."""
    raiz, salida = nuevas["raiz"], nuevas["salidas"]["meseta-y"]
    t = traza(raiz / "meseta-y")
    assert t.shape[0] == 7
    np.testing.assert_array_equal(t[:, 11], train3d.LR * np.array([1, 1, 1, 1, 1, 0.5, 0.5]))
    hecho = json.loads((raiz / "meseta-y" / "estado.json").read_text())
    assert hecho["corte"] == {"motivo": "meseta del NRMSE QSM y del PSNR", "iteracion": 7,
                              "mitades": 1}
    assert hecho["meseta"] == {"hitos": [5], "sin_mejora": 0, "mejor": 4.0, "it_mejor": 6,
                               "mejor_psnr": 21.0, "it_mejor_psnr": 4, "sin_mejora_psnr": 0}
    assert [f["nrmse_min_pct"] for f in qsm_jsonl(raiz / "meseta-y")] == NRMSE_Y[:7]
    lineas = [l for l in salida.splitlines() if "meseta:" in l]
    assert "ni el PSNR de la validación sube del suyo" in lineas[0]
    assert "mira desde la 3" in lineas[0]
    assert ("sin bajar de 5.00 % (la 3) ni subir de 21.000 dB (la 4): lr/2 desde la 6, mitad 1 "
            "de 1") in lineas[1]
    assert "la corrida se corta en la 7" in lineas[2] and len(lineas) == 3


def test_online_sin_wandb_listo_sale_antes_de_entrenar(nuevas, tmp_path, capsys, monkeypatch):
    """Con el supervisor mirando W&B, `--seguimiento online` no arranca a ciegas. Sin credencial
    (o sin wandb instalado) sale con 2 antes de cargar los datos y sin tocar `--out`; si W&B no
    abre, sale con 2 antes de la primera iteración."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(seguimiento, "_hay_credencial", lambda: False)
    with pytest.raises(SystemExit) as salida:
        train3d.main(["--out", str(tmp_path / "x"), "--seguimiento", "online"])
    assert salida.value.code == 2
    error = capsys.readouterr().err
    assert "--seguimiento online: " in error and ("wandb login" in error or "uv sync" in error)
    assert not (tmp_path / "x").exists()
    assert "--seguimiento online: W&B no abrió" in nuevas["no_abre"]
    assert not (nuevas["raiz"] / "no-abre" / "traza.npy").exists() or \
        traza(nuevas["raiz"] / "no-abre").shape[0] == 0


def test_load_vnet_carga_un_checkpoint_3d(nuevas):
    """`evaluate.load_vnet` arma la VNet que dice el `type` del regularizador: un `.pth` de
    `train3d.py` carga como VNet3D con `strict=True`, con los pesos guardados. Los 2D siguen
    cargando como antes (tests/test_reproduction.py)."""
    from evaluate import load_vnet
    ruta = nuevas["raiz"] / "trozos-3" / "best.pth"
    modelo, config = load_vnet(ruta)
    guardado = torch.load(ruta, map_location="cpu", weights_only=False)["model"]
    assert isinstance(modelo, VNet3D) and config["R"]["type"] == "tdv3d"
    assert all(torch.equal(v, guardado[k]) for k, v in modelo.state_dict().items())


def test_falta_para_online(monkeypatch):
    """Lo que `train3d.py` pregunta antes de cargar los datos: sin wandb, `uv sync`; sin
    credencial, `wandb login`; con las dos cosas, nada."""
    if HAY_WANDB:
        monkeypatch.setattr(seguimiento, "_hay_credencial", lambda: False)
        assert "wandb login" in seguimiento.falta_para_online()
        monkeypatch.setattr(seguimiento, "_hay_credencial", lambda: True)
        assert seguimiento.falta_para_online() is None
    monkeypatch.setattr(seguimiento.importlib.util, "find_spec", lambda nombre: None)
    assert "uv sync" in seguimiento.falta_para_online()


def test_epocas_reanuda_bit_a_bit_y_cambia_el_sorteo(tmp_path):
    """--epocas con lote 1 y 3 volúmenes: 3 iteraciones por época. Cortada en la 5, a mitad de la
    segunda, y reanudada, repite bit a bit la seguida; y su traza no es la de sin épocas, porque
    cambia de qué volumen sale cada parche."""
    base = [*arma_datos(tmp_path / "datos"), "--lote", "1"]
    seguida, sin, cortada = (tmp_path / n for n in ("seguida", "sin", "cortada"))
    (out, _), *_ = espera(lanza(base, seguida, "--epocas", "--guardar-cada", str(N)),
                          lanza(base, sin, "--guardar-cada", str(N)),
                          lanza(base, cortada, "--epocas", "--iterations", "5",
                                "--guardar-cada", "5"))
    assert "épocas: 3 iteraciones por época" in out
    espera(lanza(base, cortada, "--epocas", "--guardar-cada", str(N),
                 "--reanudar", str(cortada / "estado-000005.pth")))
    assert bit_a_bit(traza(seguida), traza(cortada))
    assert not bit_a_bit(traza(seguida), traza(sin))
