"""Datos 3D: que el parche dependa sólo de su índice global, que el filtro de tejido, las máscaras
de cerebro y las 48 orientaciones hagan lo que dicen, y que la cache y las máscaras armen lo que
`ddr.data3d` lee.

Todo corre sobre datos sintéticos: arreglos chicos para `ddr.data3d`, y NIfTI chicos en `tmp_path`
para los tres subcomandos de `datasets/cache_ldm100k3d.py` y para `combina` de
`datasets/mascaras_ldm100k3d.py`, nunca sobre LDM_100k. Que el escaneo y la selección repitan bit
a bit a los scripts que los midieron se verificó aparte, sobre LDM_100k (bloque final de
`datasets/cache_ldm100k3d.py`), y lo mismo la alineación de las máscaras reales (bitácora,
«Datos: LDM_100k en 3D»).
"""

import hashlib
import importlib.util
import itertools
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from scipy import ndimage

from ddr.data3d import (ORIENTACIONES, PatchDataset3D, VolumeDataset3D, add_noise_ciego,
                        esquinas_validas, load_volume, orienta)

CACHE = Path(__file__).resolve().parents[1] / "datasets" / "cache_ldm100k3d.py"
_spec = importlib.util.spec_from_file_location("cache_ldm100k3d", CACHE)
cache = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cache)


def guarda(carpeta, nombre, vol):
    """Un volumen como lo deja la cache: `.npy` en float16."""
    f = carpeta / f"{nombre}.npy"
    np.save(f, np.asarray(vol).astype(np.float16))
    return f


def tejido(x):
    return float((x > 0.05).float().mean())


def lee(lista):
    return lista.read_text().split()


# --- ORIENTACIONES ---------------------------------------------------------------------------

def test_las_48_orientaciones_son_el_grupo_del_cubo():
    """Sobre un cubo sin simetrías, las 48 dan 48 arreglos distintos, la 0 es la identidad y
    orientar dos veces da otra de la lista: es un grupo, y sortear k uniforme lo recorre parejo."""
    assert len(ORIENTACIONES) == 48
    cubo = np.random.default_rng(0).random((4, 4, 4))
    vistas = [orienta(cubo, k) for k in range(48)]
    claves = {v.tobytes() for v in vistas}
    assert len(claves) == 48
    assert np.array_equal(vistas[0], cubo)
    for a in range(48):
        for b in range(48):
            assert orienta(vistas[a], b).tobytes() in claves
    # Sobre `[1, P, P, P]` el eje de canales no se mueve.
    assert np.array_equal(orienta(cubo[None], 37)[0], vistas[37])


# --- PARCHES ---------------------------------------------------------------------------------

def test_forma_y_dtype_del_parche(tmp_path):
    f = guarda(tmp_path, "v", np.random.default_rng(0).random((20, 24, 28)))
    ds = PatchDataset3D([f], patch_size=16, length=10)
    x = ds[3]
    assert x.shape == (1, 16, 16, 16) and x.dtype == torch.float32 and x.is_contiguous()
    assert ds.volumes[0].dtype == np.float16
    assert len(ds) == 10
    with pytest.raises(ValueError):
        PatchDataset3D([f], patch_size=21)


def test_el_parche_depende_solo_del_indice_global(tmp_path):
    """Reanudar en la iteración k es `offset = k * lote`: el parche i del dataset reanudado tiene
    que ser bit a bit el `k * lote + i` del original, con todos los sorteos prendidos."""
    files = [guarda(tmp_path, f"v{j}", np.random.default_rng(j).random((20, 20, 20)))
             for j in range(3)]
    kw = dict(patch_size=12, length=100, seed=7, signo=True, ganancia=2.0)
    corrida = PatchDataset3D(files, **kw)
    reanudada = PatchDataset3D(files, offset=40, **kw)
    for i in range(10):
        assert torch.equal(reanudada[i], corrida[40 + i])
    assert torch.equal(corrida[5], corrida[5])   # sin estado entre llamadas
    otra = PatchDataset3D(files, **{**kw, "seed": 8})
    assert not any(torch.equal(otra[i], corrida[i]) for i in range(10))


def test_el_parche_cumple_min_tejido_cuando_hay_tejido(tmp_path):
    """Un octavo del volumen es tejido: sin filtro casi ningún parche llega a la mitad, con filtro
    todos. La segunda mitad es la que prueba que el filtro hace algo."""
    vol = np.zeros((32, 32, 32))
    vol[:16, :16, :16] = 1.0
    f = guarda(tmp_path, "v", vol)
    con = PatchDataset3D([f], patch_size=16, length=200, min_tejido=0.5, intentos=1000)
    sin = PatchDataset3D([f], patch_size=16, length=200, min_tejido=0.0)
    assert min(tejido(con[i]) for i in range(200)) >= 0.5
    assert np.mean([tejido(sin[i]) < 0.5 for i in range(200)]) > 0.8


def test_con_los_intentos_agotados_devuelve_el_ultimo(tmp_path):
    """Sin tejido en ningún lado el bucle termina y devuelve el último intento. Replica el orden
    de los sorteos, que es el contrato: volumen, y después tres enteros por intento."""
    vol = (np.arange(20 ** 3).reshape(20, 20, 20) % 97) / 97 * 0.04   # todo bajo el umbral
    f = guarda(tmp_path, "v", vol)
    ds = PatchDataset3D([f], patch_size=8, seed=3, intentos=5, augment=False)
    x = ds[4]
    rng = np.random.default_rng((3, 4))
    rng.integers(1)
    for _ in range(5):
        d, h, w = (int(rng.integers(20 - 8 + 1)) for _ in range(3))
    esperado = np.load(f)[d:d + 8, h:h + 8, w:w + 8].astype(np.float32)
    assert np.array_equal(x[0].numpy(), esperado)


# Tres configuraciones que recorren los sorteos de `PatchDataset3D` sin máscara: el filtro de
# tejido con intentos de sobra, la reanudación con signo y ganancia, y los intentos agotados.
CONFIGS_HASH = {
    "receta": dict(patch_size=16, seed=0),
    "aumentos": dict(patch_size=12, seed=7, offset=40, signo=True, ganancia=2.0),
    "sin-augment": dict(patch_size=10, seed=3, offset=123, augment=False, min_tejido=0.6,
                        intentos=3),
}
# sha256 de los bytes float32 de los parches 0..49 de cada configuración, sobre los volúmenes de
# `volumenes_hash`. Calculados el 2026-10-12 con un `ddr/data3d.py` sin `mascaras` (numpy 2.5.2,
# torch 2.13.0).
HASHES = {
    "receta": "ba6f592d4616a6c08ec2b3c17505015ac1a51346b2062d6bd0e3ffaf3dd05a68",
    "aumentos": "dc2fadfcd9d1ed16cdb2cd1fa38f3936725313a651f6488423a606653b54809b",
    "sin-augment": "e248bac37e2457f5524b2ce600e8beda30a6ec4358b039b3061fac4a2879a13d",
}


def volumenes_hash(carpeta):
    """Tres elipsoides de ruido uniforme, con aire alrededor, en float16 y con ejes distintos."""
    files = []
    for j, forma in enumerate(((20, 22, 24), (24, 20, 22), (22, 24, 20))):
        rng = np.random.default_rng(100 + j)
        ejes = np.meshgrid(*(np.linspace(-1, 1, n) for n in forma), indexing="ij")
        cabeza = sum(e ** 2 for e in ejes) < 0.9
        files.append(guarda(carpeta, f"h{j}", np.where(cabeza, rng.random(forma), 0.0)))
    return files


def hashes_sin_mascara(carpeta):
    files = volumenes_hash(carpeta)
    hashes = {}
    for nombre, kw in CONFIGS_HASH.items():
        ds = PatchDataset3D(files, length=50, **kw)
        datos = b"".join(ds[i].numpy().tobytes() for i in range(50))
        hashes[nombre] = hashlib.sha256(datos).hexdigest()
    return hashes


def test_sin_mascara_los_parches_son_los_de_antes(tmp_path):
    """Sin `mascaras`, el dataset sortea lo mismo que antes de que existieran: mismos parches,
    bit a bit, con el generador consumido en el mismo orden."""
    assert hashes_sin_mascara(tmp_path) == HASHES


def test_signo_y_ganancia_no_cambian_la_geometria(tmp_path):
    """Los aumentos de intensidad se sortean al final: el parche aumentado es el mismo parche por
    un factor ±g con g en [1/2, 2], y aparecen los dos signos. Por default no hay ninguno."""
    files = [guarda(tmp_path, f"v{j}", 0.1 + np.random.default_rng(j).random((20, 20, 20)))
             for j in range(2)]
    base = PatchDataset3D(files, patch_size=12, seed=1)
    aum = PatchDataset3D(files, patch_size=12, seed=1, signo=True, ganancia=2.0)
    signos = set()
    for i in range(40):
        x, y = base[i], aum[i]
        assert x.min() >= 0
        factor = float((y * x).sum() / (x * x).sum())
        assert torch.allclose(y, factor * x, rtol=1e-5, atol=1e-6)
        assert 0.5 - 1e-6 <= abs(factor) <= 2.0 + 1e-6
        signos.add(factor > 0)
    assert signos == {True, False}


# --- CUBOS DE CEREBRO ------------------------------------------------------------------------

def bolas(forma, rng, n=3, radios=(2, 6)):
    """Una máscara con entrantes y bordes: la unión de `n` bolas de centro y radio al azar."""
    ejes = np.meshgrid(*(np.arange(k) for k in forma), indexing="ij")
    m = np.zeros(forma, bool)
    for _ in range(n):
        centro, radio = rng.uniform(0, forma), rng.uniform(*radios)
        m |= sum((e - c) ** 2 for e, c in zip(ejes, centro)) < radio ** 2
    return m


def esquinas_a_mano(m, p, e):
    """La definición, esquina por esquina: el cubo (p + 2e)^3 que rodea al parche, dentro del
    volumen y todo en la máscara."""
    grilla = tuple(n - p + 1 for n in m.shape)
    validas = []
    for c in itertools.product(*(range(n) for n in grilla)):
        lo, hi = [x - e for x in c], [x + p + e for x in c]
        if min(lo) >= 0 and all(b <= n for b, n in zip(hi, m.shape)) and \
                m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].all():
            validas.append(int(np.ravel_multi_index(c, grilla)))
    return validas


def test_esquinas_validas_contra_fuerza_bruta():
    """Sobre máscaras chicas de bolas, una llena (sólo pesa el borde del volumen) y una vacía, con
    varios parches y márgenes: la suma acumulada da exactamente las esquinas de la definición."""
    rng = np.random.default_rng(0)
    con_alguna = 0
    for forma in ((9, 10, 11), (12, 8, 10)):
        mascaras = [np.ones(forma, bool), np.zeros(forma, bool)]
        mascaras += [bolas(forma, rng) for _ in range(4)]
        for m, p, e in itertools.product(mascaras, (1, 2, 4, 6), (0, 1, 2)):
            esquinas, grilla = esquinas_validas(m, p, e)
            assert grilla == tuple(n - p + 1 for n in forma)
            assert esquinas.dtype == np.uint32
            assert esquinas.tolist() == esquinas_a_mano(m, p, e), (forma, p, e)
            con_alguna += bool(esquinas.size)
    assert con_alguna > 30
    # Llena, las esquinas son las de los parches a `e` del borde del volumen.
    esquinas, grilla = esquinas_validas(np.ones((10, 10, 10), bool), 4, 2)
    assert np.unravel_index(esquinas, grilla)[0].tolist() == sorted([2] * 9 + [3] * 9 + [4] * 9)


def volumenes_con_mascara(carpeta, rng, n=3, forma=(20, 22, 24), margen=1):
    """Volúmenes que valen 1 en la máscara erosionada `margen` vóxeles (cubo de 2 margen + 1), 0.5
    en el resto de la máscara y 0 afuera. La máscara es un elipsoide que aloja cubos de 8^3, con
    bolas pegadas que le dan entrantes."""
    files, mascaras = [], []
    ejes = np.meshgrid(*(np.linspace(-1, 1, k) for k in forma), indexing="ij")
    for j in range(n):
        m = (sum(e ** 2 for e in ejes) < 0.75) | bolas(forma, rng, n=4, radios=(3, 6))
        ero = ndimage.binary_erosion(m, np.ones((2 * margen + 1,) * 3), border_value=0)
        files.append(guarda(carpeta, f"m{j}", np.where(ero, 1.0, np.where(m, 0.5, 0.0))))
        mascaras.append(m)
    return files, mascaras


def test_con_mascara_el_parche_cae_entero_en_la_mascara_erosionada(tmp_path):
    """Con el margen por defecto, todo parche sorteado (en cualquier orientación) cae entero en la
    máscara erosionada: vale 1 en todos sus vóxeles. Sin máscara, el mismo dataset saca muchos
    con borde o aire, que es lo que prueba que el volumen de prueba discrimina."""
    files, mascaras = volumenes_con_mascara(tmp_path, np.random.default_rng(1))
    ds = PatchDataset3D(files, patch_size=6, length=300, seed=2, mascaras=mascaras)
    assert ds.con_esquinas == [0, 1, 2]
    for i in range(300):
        x = ds[i]
        assert x.shape == (1, 6, 6, 6) and bool((x == 1.0).all()), i
    sin = PatchDataset3D(files, patch_size=6, length=300, seed=2)
    assert sum(not bool((sin[i] == 1.0).all()) for i in range(300)) > 100


def test_con_mascara_se_sortean_todas_las_esquinas_validas(tmp_path):
    """Con un volumen que codifica la posición (enteros exactos en float16) y sin orientaciones,
    la esquina de cada parche se lee en su primer vóxel: las sorteadas son exactamente las válidas,
    sin corrimiento de `margen`."""
    forma = (10, 11, 12)
    m = bolas(forma, np.random.default_rng(3), n=3, radios=(4, 7))
    f = guarda(tmp_path, "pos", np.arange(np.prod(forma)).reshape(forma))
    esquinas, grilla = esquinas_validas(m, 3)
    assert 5 < esquinas.size < 200
    ds = PatchDataset3D([f], patch_size=3, length=4000, augment=False, mascaras=[m])
    vistas = set()
    for i in range(4000):
        x = ds[i][0].numpy().astype(np.int64)
        d, h, w = np.unravel_index(x[0, 0, 0], forma)
        assert np.array_equal(x, np.arange(np.prod(forma)).reshape(forma)[d:d + 3, h:h + 3,
                                                                          w:w + 3])
        vistas.add(int(np.ravel_multi_index((d, h, w), grilla)))
    assert vistas == set(esquinas.tolist())


def test_con_mascara_el_parche_depende_solo_del_indice_global(tmp_path):
    """Lo mismo que sin máscara: reanudar en `offset` da los mismos parches, también si las
    máscaras llegan de un generador (como las lee `train3d.py`, una por vez)."""
    files, mascaras = volumenes_con_mascara(tmp_path, np.random.default_rng(4))
    kw = dict(patch_size=6, length=100, seed=7, signo=True, ganancia=2.0)
    corrida = PatchDataset3D(files, mascaras=mascaras, **kw)
    reanudada = PatchDataset3D(files, offset=40, mascaras=(m for m in mascaras), **kw)
    for i in range(10):
        assert torch.equal(reanudada[i], corrida[40 + i])
    assert torch.equal(corrida[5], corrida[5])
    # Con máscara la geometría es otra: los mismos índices dan otros parches.
    sin = PatchDataset3D(files, **kw)
    assert not any(torch.equal(sin[i], corrida[i]) for i in range(10))


def test_volumenes_sin_esquinas(tmp_path):
    """Un volumen cuya máscara no aloja ningún cubo queda fuera del sorteo, con un aviso; si no
    queda ninguno, o las máscaras no corresponden a los volúmenes, es un error."""
    files, mascaras = volumenes_con_mascara(tmp_path, np.random.default_rng(5))
    chica = np.zeros_like(mascaras[1])
    chica[5:12, 5:12, 5:12] = True             # 7^3: no entra un cubo de 6 + 2
    np.save(files[1], np.full(chica.shape, 0.25, np.float16))
    with pytest.warns(UserWarning, match=r"1 de 3 volúmenes .* fuera del sorteo: \['m1.npy'\]"):
        ds = PatchDataset3D(files, patch_size=6, length=200, mascaras=[mascaras[0], chica,
                                                                       mascaras[2]])
    assert ds.con_esquinas == [0, 2] and ds.esquinas[1].size == 0
    assert all(bool((ds[i] == 1.0).all()) for i in range(200))
    with pytest.raises(ValueError, match="ningún volumen de 1"):
        PatchDataset3D(files[1:2], patch_size=6, mascaras=[chica])
    with pytest.raises(ValueError, match="2 máscaras para 3 volúmenes"):
        PatchDataset3D(files, patch_size=6, mascaras=mascaras[:2])
    with pytest.raises(ValueError, match="más máscaras que volúmenes"):
        PatchDataset3D(files[:2], patch_size=6, mascaras=mascaras)
    with pytest.raises(ValueError, match="se esperaba bool"):
        PatchDataset3D(files[:1], patch_size=6, mascaras=[mascaras[0][:, :, 1:]])
    with pytest.raises(ValueError, match="se esperaba bool"):
        PatchDataset3D(files[:1], patch_size=6, mascaras=[mascaras[0].astype(np.uint8)])


# --- EVALUACIÓN ------------------------------------------------------------------------------

def test_volume_dataset_recorta_centrado_en_la_caja_del_tejido(tmp_path):
    vol = np.zeros((40, 44, 48))
    vol[5:35, 8:40, 10:44] = 0.1 + 0.9 * np.random.default_rng(0).random((30, 32, 34))
    f = guarda(tmp_path, "sub-000001", vol)
    x, nombre = VolumeDataset3D([f], recorte=16)[0]
    # Centro de la caja (20, 24, 27): el recorte arranca en (12, 16, 19).
    esperado = np.load(f)[12:28, 16:32, 19:35].astype(np.float32)
    assert nombre == "sub-000001"
    assert x.shape == (1, 16, 16, 16) and x.dtype == torch.float32
    assert np.array_equal(x[0].numpy(), esperado)


def test_volume_dataset_se_corre_adentro_y_no_rellena(tmp_path):
    vol = np.zeros((20, 20, 20))
    vol[:4, :4, :4] = 1.0   # tejido en una esquina: el centro de la caja es (2, 2, 2)
    f, g = guarda(tmp_path, "a", vol), guarda(tmp_path, "b", vol)
    x, _ = VolumeDataset3D([f], recorte=12)[0]
    assert np.array_equal(x[0].numpy(), vol[:12, :12, :12])
    with pytest.raises(ValueError):
        VolumeDataset3D([f], recorte=24)
    assert [n for _, n in VolumeDataset3D([f, g], recorte=12, names=["b"])] == ["b"]
    with pytest.raises(FileNotFoundError):
        VolumeDataset3D([f], recorte=12, names=["c"])


def test_load_volume(tmp_path):
    f = guarda(tmp_path, "v", np.full((3, 4, 5), 0.5))
    v = load_volume(f)
    assert v.shape == (3, 4, 5) and v.dtype == np.float32 and (v == 0.5).all()
    np.save(tmp_path / "plano.npy", np.zeros((4, 5), np.float16))
    with pytest.raises(ValueError):
        load_volume(tmp_path / "plano.npy")


# --- RUIDO CIEGO -----------------------------------------------------------------------------

def test_add_noise_ciego_respeta_el_rango_de_sigma():
    y = torch.zeros(500, 1, 8, 8, 8)
    ruidosa, sigmas = add_noise_ciego(y, 5.0, 55.0, torch.Generator().manual_seed(0), 255.0)
    assert sigmas.shape == (500,) and ruidosa.shape == y.shape
    assert 5.0 <= sigmas.min() < 10.0 and 50.0 < sigmas.max() <= 55.0
    # Cada muestra con su sigma: la desviación sobre 512 vóxeles cae a ±20 % (el error es ~3 %).
    sd = ruidosa.flatten(1).std(1) * 255.0
    assert torch.allclose(sd / sigmas, torch.ones(500), rtol=0.2)
    # Del mismo generador, primero los sigmas y después el ruido.
    gen = torch.Generator().manual_seed(0)
    u = torch.rand(500, generator=gen)
    ruido = torch.randn(y.shape, generator=gen)
    assert torch.equal(sigmas, 5.0 + 50.0 * u)
    assert torch.allclose(ruidosa, (sigmas / 255.0).view(-1, 1, 1, 1, 1) * ruido)
    with pytest.raises(ValueError):
        add_noise_ciego(y, 10.0, 5.0)


# --- CACHE -----------------------------------------------------------------------------------

FORMA = (32, 40, 32)    # múltiplo de 8 en cada eje, como 160x224x160
N_SUJETOS = 40
FALTA, DEFORME, RARO = 5, 6, 7


def cabeza(forma, rng):
    """Elipsoide con fondo exactamente 0 (como LDM_100k), tamaño e intensidad propios."""
    z, y, x = np.meshgrid(*(np.linspace(-1, 1, n) for n in forma), indexing="ij")
    radio = 0.7 + 0.2 * rng.random(3)
    r = (z / radio[0]) ** 2 + (y / radio[1]) ** 2 + (x / radio[2]) ** 2
    vol = np.where(r < 1, 1 - 0.6 * r + 0.1 * rng.random(forma), 0.0)
    return vol * (800 + 400 * rng.random())


def corre(*args):
    r = subprocess.run([sys.executable, str(CACHE), *args], capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0, r.stderr
    return r.stdout


@pytest.fixture
def rawdata(tmp_path):
    """BIDS sintético: uno que falta, uno con otra forma y uno 30 veces más brillante."""
    raw = tmp_path / "rawdata"
    rng = np.random.default_rng(0)
    for i in range(N_SUJETOS):
        if i == FALTA:
            continue
        vol = cabeza(FORMA if i != DEFORME else (32, 40, 24), rng)
        if i == RARO:
            vol = vol * 30
        anat = raw / f"sub-{i:06d}" / "anat"
        anat.mkdir(parents=True)
        nib.save(nib.Nifti1Image(vol.astype(np.uint16), np.eye(4)),
                 anat / f"sub-{i:06d}_T1w.nii.gz")
    return raw


def test_cache_escanea_selecciona_y_cachea(rawdata, tmp_path):
    escaneo, salida = tmp_path / "escaneo", tmp_path / "ldm100k3d"
    forma = [str(n) for n in FORMA]

    # escanea: los archivos que `selecciona` lee, y la miniatura del volumen normalizado.
    corre("escanea", "--rawdata", str(rawdata), "--salida", str(escaneo), "--n", str(N_SUJETOS),
          "--forma", *forma, "--procesos", "2")
    mini = np.load(escaneo / "miniaturas.npy")
    e = np.load(escaneo / "escalares.npz")
    assert mini.shape == (N_SUJETOS, 4, 5, 4) and mini.dtype == np.float16
    assert e["escalares"].shape == (N_SUJETOS, len(e["nombres"]))
    assert np.flatnonzero(~e["ok"]).tolist() == [FALTA, DEFORME]
    crudo = np.asarray(nib.load(rawdata / "sub-000000/anat/sub-000000_T1w.nii.gz").dataobj)
    v = crudo.astype(np.float32)
    v = np.clip(v / float(np.percentile(v[v > 0], 99.5)), 0.0, 1.0)
    esperada = v.reshape(4, 8, 5, 8, 4, 8).mean((1, 3, 5)).astype(np.float16)
    assert np.array_equal(mini[0], esperada)

    # selecciona: evaluación y entrenamiento disjuntos, sin los ilegibles ni el atípico.
    args = ["--escaneo", str(escaneo), "--n", "8", "16", "--n-eval", "3", "--componentes", "6"]
    corre("selecciona", *args, "--salida", str(salida))
    evaluacion = lee(salida / "lista_eval.txt")
    train = {n: lee(salida / f"lista_train_{n}.txt") for n in (8, 16)}
    assert 1 <= len(evaluacion) <= 3
    for n, lista in train.items():
        assert len(lista) == len(set(lista)) == n
        assert not set(lista) & set(evaluacion)
        assert not {f"sub-{i:06d}" for i in (FALTA, DEFORME, RARO)} & set(lista + evaluacion)
    # El atípico no queda afuera por azar: lo saca el control de calidad.
    sel = np.load(salida / "seleccion.npz")
    assert RARO in sel["ids"][sel["fuera"]]
    # Regla `mezcla`: la primera mitad de cada lista es el k-center voraz, en su orden.
    ids, vivos, resto, orden = sel["ids"], sel["vivos"], sel["resto"], sel["orden_fps"]
    assert evaluacion == [f"sub-{ids[vivos[i]]:06d}" for i in sel["eval_loc"]]
    for n, lista in train.items():
        assert lista[:n // 2] == [f"sub-{ids[vivos[resto[j]]]:06d}" for j in orden[:n // 2]]
    # La semilla fija la selección: repetirla da las mismas listas.
    otra = tmp_path / "otra"
    corre("selecciona", *args, "--salida", str(otra))
    for f in ("lista_eval.txt", "lista_train_8.txt", "lista_train_16.txt"):
        assert (otra / f).read_bytes() == (salida / f).read_bytes()

    # cachea: la lista se nombra relativa a --salida, como en el uso real.
    for lista, split in (("lista_train_8.txt", "train"), ("lista_eval.txt", "val")):
        corre("cachea", "--lista", lista, "--split", split, "--rawdata", str(rawdata),
              "--salida", str(salida), "--procesos", "2")
    archivos = sorted((salida / "images" / "train").glob("*.npy"))
    assert [f.stem for f in archivos] == sorted(train[8])
    sujeto = archivos[0].stem
    crudo = np.asarray(nib.load(rawdata / sujeto / "anat" / f"{sujeto}_T1w.nii.gz").dataobj)
    v = crudo.astype(np.float32)
    v = np.clip(v / max(np.percentile(v[v > 0], 99.5), 1e-6), 0.0, 1.0)
    caja = []
    for eje in range(3):
        idx = np.flatnonzero((v > 0.05).any(axis=tuple(a for a in range(3) if a != eje)))
        caja.append(slice(max(idx[0] - cache.MARGEN, 0),
                          min(idx[-1] + 1 + cache.MARGEN, v.shape[eje])))
    guardado = np.load(archivos[0])
    assert guardado.dtype == np.float16 and guardado.ndim == 3
    assert np.array_equal(guardado, v[tuple(caja)].astype(np.float16))

    # Y `ddr.data3d` lee lo que la cache escribió.
    ds = PatchDataset3D.from_root(salida, ("train",), patch_size=16, length=8)
    assert len(ds.volumes) == 8 and ds[0].shape == (1, 16, 16, 16)
    ev = VolumeDataset3D.from_root(salida, "val", names=evaluacion, recorte=16)
    assert sorted(n for _, n in ev) == sorted(evaluacion)


# --- MÁSCARAS --------------------------------------------------------------------------------

MASCARAS = CACHE.with_name("mascaras_ldm100k3d.py")


def test_combina_alinea_las_mascaras_con_la_cache(rawdata, tmp_path):
    """`mascaras_ldm100k3d.py combina` sobre máscaras sintéticas en el espacio del NIfTI, una en
    `.npz` (como las deja `deepbet`) y otra en `.npy` suelto: escribe la intersección con huecos
    rellenos, recortada con la caja de la cache y de la forma del volumen cacheado, y
    `PatchDataset3D` la lee. Si un volumen cacheado no es el del NIfTI, ese sujeto falla, los
    demás se escriben y el comando termina con error. deepbet y HD-BET no corren en los tests."""
    salida, dir_d, dir_h = tmp_path / "ldm100k3d", tmp_path / "deepbet", tmp_path / "hdbet"
    for carpeta in (salida, dir_d, dir_h):
        carpeta.mkdir()
    sujetos = [f"sub-{i:06d}" for i in range(3)]
    (salida / "lista.txt").write_text("".join(f"{s}\n" for s in sujetos))
    corre("cachea", "--lista", "lista.txt", "--split", "train", "--rawdata", str(rawdata),
          "--salida", str(salida), "--procesos", "2")
    esperadas = {}
    for s in sujetos:
        crudo = np.asarray(nib.load(rawdata / s / "anat" / f"{s}_T1w.nii.gz").dataobj)
        a = crudo > 0.5 * crudo.max()
        b = np.roll(a, 1, axis=0)               # corrida: la intersección no es ninguna de las dos
        centro = tuple(n // 2 for n in b.shape)
        b[centro] = False                       # un hueco, que `combina` rellena
        np.savez_compressed(dir_d / f"{s}.npz", mascara=a)
        np.save(dir_h / f"{s}.npy", b)
        v = crudo.astype(np.float32)
        v = np.clip(v / max(np.percentile(v[v > 0], 99.5), 1e-6), 0.0, 1.0)
        caja = []
        for eje in range(3):
            idx = np.flatnonzero((v > 0.05).any(axis=tuple(e for e in range(3) if e != eje)))
            caja.append(slice(max(idx[0] - cache.MARGEN, 0),
                              min(idx[-1] + 1 + cache.MARGEN, v.shape[eje])))
        lleno = a & b
        lleno[centro] = True
        esperadas[s] = lleno[tuple(caja)]

    def combina():
        return subprocess.run(
            [sys.executable, str(MASCARAS), "combina", "--lista", "lista.txt", "--split", "train",
             "--salida", str(salida), "--rawdata", str(rawdata), "--deepbet", str(dir_d),
             "--hdbet", str(dir_h), "--procesos", "2"],
            capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})

    r = combina()
    assert r.returncode == 0, r.stderr
    assert "train: 3/3 máscaras" in r.stdout and "huecos: min 1  mediana 1  max 1" in r.stdout
    for s in sujetos:
        m = np.load(salida / "mascaras" / "train" / f"{s}.npy")
        cacheado = np.load(salida / "images" / "train" / f"{s}.npy")
        assert m.dtype == bool and m.shape == cacheado.shape
        assert np.array_equal(m, esperadas[s])
    archivos = [salida / "images" / "train" / f"{s}.npy" for s in sujetos]
    ds = PatchDataset3D(archivos, patch_size=6, length=4, mascaras=[
        np.load(salida / "mascaras" / "train" / f"{s}.npy") for s in sujetos])
    assert ds.con_esquinas == [0, 1, 2] and ds[0].shape == (1, 6, 6, 6)

    roto = salida / "images" / "train" / "sub-000001.npy"
    np.save(roto, np.zeros_like(np.load(roto)))
    (salida / "mascaras" / "train" / "sub-000001.npy").unlink()
    r = combina()
    assert r.returncode == 1
    assert "sub-000001: el .npy cacheado no es normaliza(NIfTI)[caja]" in r.stdout
    assert "train: 2/3 máscaras" in r.stdout
    assert not (salida / "mascaras" / "train" / "sub-000001.npy").exists()
