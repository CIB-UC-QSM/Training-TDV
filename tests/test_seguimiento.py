"""El seguimiento con W&B (`ddr/seguimiento.py`): que escriba la corrida, que la identidad
sobreviva a `--reanudar` y a las ramas, y que ningún error de wandb llegue al bucle.

Los que necesitan wandb lo piden con `pytest.importorskip`: en un entorno sin wandb la suite
tiene que seguir pasando. Todo offline dentro de `tmp_path`, sin red ni credencial: lo que
W&B guardó se lee del `.wandb` de cada corrida, no de lo que el módulo dice que hizo.
"""

import glob
import json
import random
import re
import struct
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from ddr import seguimiento
from ddr.data import psnr

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def dir_wandb(tmp_path_factory):
    return tmp_path_factory.mktemp("wandb-global")


@pytest.fixture
def entorno(monkeypatch, tmp_path, dir_wandb):
    """Sin credencial, sin red y sin tocar el home. Los directorios globales de wandb son los de
    la sesión: wandb los lee una vez por proceso."""
    for variable in ("WANDB_API_KEY", "WANDB_IDENTITY_TOKEN_FILE", "WANDB_MODE", "WANDB_BASE_URL"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("NETRC", str(tmp_path / "no-hay-netrc"))
    monkeypatch.setenv("WANDB_SILENT", "true")
    for variable in ("WANDB_CONFIG_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR"):
        monkeypatch.setenv(variable, str(dir_wandb / variable.lower()))
    return tmp_path


def segmentos(out, rid):
    """Los directorios offline de `rid`, en orden de inicio. Si dos empiezan en el mismo segundo,
    wandb le agrega `-1` al segundo."""
    base = Path(out) / "wandb"
    return sorted(glob.glob(str(base / f"offline-run-*-{rid}"))
                  + glob.glob(str(base / f"offline-run-*-{rid}-*")))


def _lee_wandb(ruta):
    """El log de LevelDB que escribe wandb 0.30: 7 bytes de cabecera de archivo, bloques de
    32 KiB y, por registro, crc, largo y tipo en 7 bytes."""
    pb = pytest.importorskip("wandb.proto.wandb_internal_pb2")
    salida, datos, i, parcial = [], Path(ruta).read_bytes(), 7, b""
    while i + 7 <= len(datos):
        if 32768 - i % 32768 < 7:                # relleno al final del bloque
            i += 32768 - i % 32768
            continue
        largo, tipo = struct.unpack("<HB", datos[i + 4:i + 7])
        parcial += datos[i + 7:i + 7 + largo]
        i += 7 + largo
        if tipo in (1, 4):                       # FULL o LAST
            registro = pb.Record()
            registro.ParseFromString(parcial)
            salida.append(registro)
            parcial = b""
    return salida


def registros(out, rid):
    """Los registros de los `.wandb` offline de `rid`, segmento por segmento. Cada segmento se
    lee hasta ver su `exit`, que es el último. Dos de las tres primeras corridas de este archivo,
    con la máquina cargada, no encontraron la fila del paso 1, y no volvió a pasar en 43 corridas
    ni en 30 cierres seguidos leídos al instante (2026-10-12). La causa no está identificada; si
    fuera que el proceso de wandb termina de escribir después de que `finish` vuelve, esto la
    cubre."""
    salida = []
    for segmento in segmentos(out, rid):
        ruta, limite = next(Path(segmento).glob("*.wandb")), time.monotonic() + 10
        while True:
            leidos = _lee_wandb(ruta)
            if (leidos and leidos[-1].WhichOneof("record_type") == "exit"
                    or time.monotonic() > limite):
                break
            time.sleep(0.05)
        salida += leidos
    return salida


def _pone(destino, item):
    """Un item de history o de summary, sin las claves internas de wandb. Un dict (una imagen)
    llega partido en claves anidadas: se rearma."""
    claves = [item.key] if item.key else list(item.nested_key)
    if claves[0].startswith("_"):
        return
    for clave in claves[:-1]:
        destino = destino.setdefault(clave, {})
    destino[claves[-1]] = json.loads(item.value_json)


def historia(out, rid):
    """paso -> {métrica: valor}, de todos los segmentos."""
    filas = {}
    for r in registros(out, rid):
        if r.WhichOneof("record_type") == "history":
            fila = filas.setdefault(r.history.step.num, {})
            for item in r.history.item:
                _pone(fila, item)
    return filas


def resumen_de(out, rid):
    final = {}
    for r in registros(out, rid):
        if r.WhichOneof("record_type") == "summary":
            for item in r.summary.update:
                _pone(final, item)
    return final


def corrida(out, rid):
    """(grupo, config) del registro `run` del primer segmento."""
    for r in registros(out, rid):
        if r.WhichOneof("record_type") == "run":
            return r.run.run_group, {c.key: json.loads(c.value_json) for c in r.run.config.update}


def identidad(out):
    return json.loads((Path(out) / "seguimiento.json").read_text())


def test_offline_escribe_la_corrida(entorno, capsys):
    pytest.importorskip("wandb")
    out = entorno / "tronco"
    run = seguimiento.abre(out, {"lote": 32, "clip": float("inf")}, nombre="tronco")
    assert run is not None
    seguimiento.registra(run, {"train/loss": 2.0, "train/T": 0.1}, 1)
    seguimiento.registra(run, {"train/loss": 1.0}, 2)
    seguimiento.registra(run, {"val/psnr": 30.5, "val/nrmse_pct": 7.5}, 2)   # misma fila
    seguimiento.registra_imagenes(run, {"val/denoising": np.zeros((8, 24), np.uint8)}, 2,
                                  leyendas={"val/denoising": "ruidoso | TDV | limpio"})
    seguimiento.alerta(run, "prueba", "offline no la manda, y no apaga nada")
    seguimiento.resumen(run, {"mejor_psnr": 30.5})
    seguimiento.cierra(run)

    ident = identidad(out)
    assert (ident["modo"], ident["paso"], ident["nombre"]) == ("offline", 2, "tronco")
    h = historia(out, ident["id"])
    assert h[1] == {"train/loss": 2.0, "train/T": 0.1}
    assert (h[2]["train/loss"], h[2]["val/psnr"], h[2]["val/nrmse_pct"]) == (1.0, 30.5, 7.5)
    assert h[2]["val/denoising"]["_type"] == "image-file"
    assert h[2]["val/denoising"]["caption"] == "ruidoso | TDV | limpio"
    assert glob.glob(str(out / "wandb" / "offline-run-*" / "files" / "media" / "images" / "val"
                         / "denoising_2_*.png"))
    final = resumen_de(out, ident["id"])
    assert final["mejor_psnr"] == 30.5
    assert final["val/psnr"] == {"max": 30.5} and final["val/nrmse_pct"] == {"min": 7.5}
    assert corrida(out, ident["id"])[1]["clip"] == float("inf")
    assert f"wandb sync {out}/wandb/offline-run-*-{ident['id']}*" in capsys.readouterr().out


def test_cada_iteracion_de_una_traza(entorno):
    """El uso del integrador: `metricas_traza` en cada iteración. 400 filas pasan los 32 KiB del
    primer bloque del `.wandb`, así que esto prueba también el lector de arriba."""
    pytest.importorskip("wandb")
    traza = np.random.default_rng(0).uniform(1, 2, (400, 14))
    traza[:, 8:10] = 0
    out = entorno / "out"
    run = seguimiento.abre(out, {})
    for it in range(1, 401):
        seguimiento.registra(run, seguimiento.metricas_traza(traza, it), it)
    seguimiento.cierra(run)
    h = historia(out, identidad(out)["id"])
    assert sorted(h) == list(range(1, 401))
    assert all(h[it] == pytest.approx(seguimiento.metricas_traza(traza, it), rel=1e-15)
               for it in (1, 200, 400))
    assert Path(next(Path(segmentos(out, identidad(out)["id"])[0]).glob("*.wandb"))).stat(
        ).st_size > 32768


def test_la_alerta_llega_y_si_falla_no_apaga_el_registro(capsys):
    """`alerta` le pasa título y texto a `run.alert`. Si falla, lo dice en el log y las métricas
    siguen: una alerta perdida no puede costar el registro del resto de la corrida."""
    class Falsa:
        def __init__(self):
            self.alertas, self.filas = [], []

        def alert(self, **campos):
            self.alertas.append(campos)

        def log(self, metricas, step):
            self.filas.append((step, metricas))

    run = Falsa()
    seguimiento.alerta(run, "rebobinado", "it 180")
    assert run.alertas == [{"title": "rebobinado", "text": "it 180", "level": "WARN"}]

    def sin_red(**campos):
        raise RuntimeError("sin red")
    run.alert = sin_red
    seguimiento.alerta(run, "TRABADA", "dos rebobinados al mismo respaldo")
    assert "la alerta «TRABADA» no salió (RuntimeError: sin red)" in capsys.readouterr().out
    seguimiento.registra(run, {"train/loss": 1.0}, 3)
    assert run.filas == [(3, {"train/loss": 1.0})]
    seguimiento.alerta(None, "sin seguimiento", "no hace nada")


def test_reanudar_en_el_mismo_out_sigue_la_misma_corrida(entorno):
    pytest.importorskip("wandb")
    out = entorno / "tronco"
    run = seguimiento.abre(out, {})
    seguimiento.registra(run, {"train/loss": 1.0}, 5)
    seguimiento.cierra(run)
    rid = identidad(out)["id"]

    run = seguimiento.abre(out, {}, padre=out / "estado-000005.pth", desde=5)
    assert run.id == rid
    seguimiento.registra(run, {"train/loss": 0.5}, 6)
    seguimiento.cierra(run)
    assert len(segmentos(out, rid)) == 2
    assert historia(out, rid) == {5: {"train/loss": 1.0}, 6: {"train/loss": 0.5}}
    assert identidad(out)["paso"] == 6 and identidad(out)["descartadas"] == []


def test_reanudar_desde_antes_de_lo_registrado_abre_otra(entorno):
    """W&B descartaría sin aviso los pasos 6 a 8 del linaje nuevo y guardaría los del viejo."""
    pytest.importorskip("wandb")
    out = entorno / "tronco"
    run = seguimiento.abre(out, {})
    for it in range(1, 9):
        seguimiento.registra(run, {"train/loss": float(it)}, it)
    seguimiento.cierra(run)
    vieja = identidad(out)["id"]

    run = seguimiento.abre(out, {}, padre=out / "estado-000005.pth", desde=5)
    nueva = run.id
    assert nueva != vieja
    seguimiento.registra(run, {"train/loss": -6.0}, 6)
    seguimiento.cierra(run)
    assert identidad(out)["descartadas"] == [vieja]
    assert historia(out, nueva) == {6: {"train/loss": -6.0}}
    assert historia(out, vieja)[8] == {"train/loss": 8.0}      # la vieja no se toca

    # Arrancar de cero en un `--out` usado es lo mismo.
    run = seguimiento.abre(out, {})
    seguimiento.cierra(run)
    assert run.id not in (vieja, nueva)
    assert identidad(out)["descartadas"] == [vieja, nueva]


def test_una_rama_es_otra_corrida_en_el_grupo_del_tronco(entorno):
    pytest.importorskip("wandb")
    tronco, rama = entorno / "tronco3d", entorno / "rama3d-10k"
    run = seguimiento.abre(tronco, {"hitos": []})
    seguimiento.registra(run, {"train/loss": 1.0}, 10)
    seguimiento.cierra(run)
    tid = identidad(tronco)["id"]

    padre = tronco / "estado-000010.pth"
    run = seguimiento.abre(rama, {"hitos": [10]}, padre=padre, desde=10)
    assert run.id != tid
    seguimiento.registra(run, {"train/loss": 0.9}, 11)
    seguimiento.cierra(run)
    ident = identidad(rama)
    assert (ident["grupo"], ident["padre"], ident["padre_id"]) == ("tronco3d", str(padre), tid)
    grupo, config = corrida(rama, ident["id"])
    assert grupo == "tronco3d"
    assert (config["padre"], config["padre_id"], config["hitos"]) == (str(padre), tid, [10])
    assert identidad(tronco)["id"] == tid                      # el tronco no cambia

    # La rama reanudada en su propio `--out` sigue siendo ella, en el mismo grupo.
    run = seguimiento.abre(rama, {"hitos": [10]}, padre=rama / "estado-000011.pth", desde=11)
    assert run.id == ident["id"]
    seguimiento.cierra(run)
    assert identidad(rama)["grupo"] == "tronco3d"


def test_otro_punto_de_rama_en_el_mismo_out_es_otra_corrida(entorno):
    """Reusar el `--out` de una rama desde otro estado del tronco, posterior a lo registrado: es
    otro linaje (`train3d.py` reescribe `--out` desde el estado), no la misma corrida aunque
    `desde` pase el último paso. Reintentar la MISMA rama sin haber registrado nada sí la sigue."""
    pytest.importorskip("wandb")
    tronco, rama = entorno / "tronco3d", entorno / "rama3d"
    run = seguimiento.abre(tronco, {})
    seguimiento.registra(run, {"train/loss": 1.0}, 20)
    seguimiento.cierra(run)

    primero = tronco / "estado-000010.pth"
    run = seguimiento.abre(rama, {}, padre=primero, desde=10)
    seguimiento.cierra(run)
    run = seguimiento.abre(rama, {}, padre=primero, desde=10)
    assert run.id == identidad(rama)["id"]
    seguimiento.registra(run, {"train/loss": 0.9}, 12)
    seguimiento.cierra(run)
    vieja = identidad(rama)["id"]

    segundo = tronco / "estado-000020.pth"
    run = seguimiento.abre(rama, {}, padre=segundo, desde=20)
    assert run.id != vieja
    seguimiento.registra(run, {"train/loss": 0.8}, 21)
    seguimiento.cierra(run)
    ident = identidad(rama)
    assert (ident["padre"], ident["descartadas"]) == (str(segundo), [vieja])
    assert corrida(rama, ident["id"])[1]["padre"] == str(segundo)
    assert historia(rama, ident["id"]) == {21: {"train/loss": 0.8}}
    assert historia(rama, vieja) == {12: {"train/loss": 0.9}}


def test_modo_no_es_no_op_y_no_importa_wandb(tmp_path):
    """En un proceso aparte, porque en éste otro test ya pudo importar wandb."""
    codigo = textwrap.dedent("""
        import os, sys
        from ddr import seguimiento as s
        r = s.abre(sys.argv[1], {"a": 1}, modo="no")
        s.registra(r, {"x": 1.0}, 1)
        s.registra_imagenes(r, {"i": None}, 1)
        s.resumen(r, {"y": 2})
        s.cierra(r)
        assert r is None
        os.environ["WANDB_MODE"] = "disabled"
        assert s.abre(sys.argv[1], {}, modo="auto") is None
        assert "wandb" not in sys.modules, "se importó wandb"
    """)
    subprocess.run([sys.executable, "-c", codigo, str(tmp_path / "out")], cwd=RAIZ, check=True)
    assert not (tmp_path / "out").exists()


def test_un_error_de_wandb_no_llega_al_bucle(entorno, monkeypatch, capsys):
    pytest.importorskip("wandb")
    run = seguimiento.abre(entorno / "out", {})
    llamadas = []

    def falla(self, *args, **kwargs):
        llamadas.append(1)
        raise RuntimeError("se cayó el servicio")

    monkeypatch.setattr(type(run), "log", falla)
    capsys.readouterr()
    seguimiento.registra(run, {"train/loss": 1.0}, 1)
    seguimiento.registra(run, {"train/loss": 1.0}, 2)
    seguimiento.registra_imagenes(run, {"val/denoising": np.zeros((4, 4), np.uint8)}, 2)
    seguimiento.resumen(run, {"mejor_psnr": 1.0})
    seguimiento.cierra(run)
    salida = capsys.readouterr().out
    assert len(llamadas) == 1, "después del primer error no se llama más a wandb"
    assert salida.count("falló") == 1 and "RuntimeError: se cayó el servicio" in salida
    monkeypatch.undo()
    run.finish()


def test_una_imagen_que_no_es_uint8_apaga_el_seguimiento(entorno, capsys):
    pytest.importorskip("wandb")
    run = seguimiento.abre(entorno / "out", {})
    seguimiento.registra_imagenes(run, {"val/denoising": np.zeros((4, 4), np.float32)}, 1)
    seguimiento.registra(run, {"train/loss": 1.0}, 2)
    seguimiento.cierra(run)                    # caída: no la cierra; la cierra wandb al salir
    run.finish()
    assert "float32, no uint8" in capsys.readouterr().out
    assert historia(entorno / "out", identidad(entorno / "out")["id"]) == {}


def test_abre_con_init_que_falla_devuelve_none(entorno, monkeypatch, capsys):
    wandb = pytest.importorskip("wandb")

    def falla(*args, **kwargs):
        raise wandb.errors.CommError("sin red")

    monkeypatch.setattr(wandb, "init", falla)
    assert seguimiento.abre(entorno / "out", {}) is None
    assert "wandb.init en modo offline falló (CommError: sin red)" in capsys.readouterr().out
    assert not (entorno / "out" / "seguimiento.json").exists()


def test_auto_cae_a_offline_si_online_no_arranca(entorno, monkeypatch):
    wandb = pytest.importorskip("wandb")
    original, modos = wandb.init, []

    def init(*args, mode, **kwargs):
        modos.append(mode)
        if mode == "online":
            raise wandb.errors.CommError("sin red")
        return original(*args, mode=mode, **kwargs)

    monkeypatch.setattr(wandb, "init", init)
    monkeypatch.setattr(seguimiento, "_hay_credencial", lambda: True)
    monkeypatch.setattr(seguimiento, "_wandb_tiene_credencial", lambda wandb: True)
    run = seguimiento.abre(entorno / "out", {})
    seguimiento.cierra(run)
    assert modos == ["online", "offline"] and identidad(entorno / "out")["modo"] == "offline"

    # `online` explícito sin credencial no llama a init: wandb preguntaría la clave.
    monkeypatch.setattr(seguimiento, "_hay_credencial", lambda: False)
    assert seguimiento.abre(entorno / "otro", {}, modo="online") is None
    assert modos == ["online", "offline"]


def test_online_sin_credencial_para_wandb_no_llama_a_init(entorno, monkeypatch, capsys):
    """`_hay_credencial` lee archivos y no ve el `base_url` del archivo de settings de wandb (el
    de `wandb login --host`): si dice que sí y wandb, para su host, no tiene clave, init online
    la pide por la terminal. Se le pregunta a wandb antes, sin prompt: auto sale offline sin
    probar online, y online explícito no llama a init."""
    wandb = pytest.importorskip("wandb")
    original, modos = wandb.init, []

    def init(*args, mode, **kwargs):
        modos.append(mode)
        return original(*args, mode=mode, **kwargs)

    monkeypatch.setattr(wandb, "init", init)
    monkeypatch.setattr(seguimiento, "_hay_credencial", lambda: True)
    run = seguimiento.abre(entorno / "out", {})
    seguimiento.cierra(run)
    assert modos == ["offline"] and identidad(entorno / "out")["modo"] == "offline"
    assert "wandb no encuentra credencial" in capsys.readouterr().out

    assert seguimiento.abre(entorno / "otro", {}, modo="online") is None
    assert modos == ["offline"]


def test_credencial(monkeypatch, tmp_path):
    for variable in ("WANDB_API_KEY", "WANDB_IDENTITY_TOKEN_FILE", "WANDB_BASE_URL"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("NETRC", str(tmp_path / "netrc"))
    assert not seguimiento._hay_credencial()
    (tmp_path / "netrc").write_text("machine otra.org login u password x\n")
    assert not seguimiento._hay_credencial()
    # Sin password wandb descarta la entrada y pregunta la clave: contarla colgaba `auto`.
    (tmp_path / "netrc").write_text("machine api.wandb.ai login user\n")
    assert not seguimiento._hay_credencial()
    (tmp_path / "netrc").write_text("default login anonymous password\n")
    assert not seguimiento._hay_credencial()
    (tmp_path / "netrc").write_text("machine api.wandb.ai login user password " + "0" * 40 + "\n")
    assert seguimiento._hay_credencial()
    monkeypatch.setenv("NETRC", str(tmp_path / "no-hay"))
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)
    assert seguimiento._hay_credencial()


def test_no_toca_los_generadores(entorno):
    """Prender el seguimiento no puede cambiar la aritmética de la corrida."""
    pytest.importorskip("wandb")
    antes = (torch.get_rng_state().clone(), np.random.get_state()[1].copy(), random.getstate())
    run = seguimiento.abre(entorno / "out", {"a": 1})
    seguimiento.registra(run, {"train/loss": 1.0}, 1)
    seguimiento.registra_imagenes(run, {"val/denoising": np.zeros((4, 4), np.uint8)}, 1)
    seguimiento.resumen(run, {"b": 2})
    seguimiento.cierra(run)
    assert torch.equal(antes[0], torch.get_rng_state())
    assert np.array_equal(antes[1], np.random.get_state()[1])
    assert antes[2] == random.getstate()


def test_metricas_traza():
    traza = np.arange(5 * 14, dtype=np.float64).reshape(5, 14) + 1.0
    traza[:, 6] = [np.nan, 10.0, 10.0, 20.0, 20.0]       # mediana: vacía en la iteración 1
    traza[:, 5] = [7.0, 30.0, 15.0, 40.0, 40.0]
    traza[:, 7] = np.nan                                   # sin --guard-amp
    traza[:, 8] = [0, 1, 0, 1, 0]
    traza[:, 9] = [0, 0, 0, 1, 0]
    m = seguimiento.metricas_traza(traza, 3)
    for c, nombre in seguimiento.COLUMNAS_TRAZA.items():
        if c != 7:
            assert m[nombre] == traza[2, c], nombre
    assert (m["train/rechazos"], m["train/rebobinados"], m["train/grad_razon"]) == (1, 0, 1.5)
    assert "train/amp" not in m
    assert seguimiento.metricas_traza(traza, 4)["train/rebobinados"] == 1
    primera = seguimiento.metricas_traza(traza, 1)
    assert "train/grad_mediana" not in primera and "train/grad_razon" not in primera
    traza[0, 0] = np.nan                                   # una muerte sí entra
    assert np.isnan(seguimiento.metricas_traza(traza, 1)["train/loss"])


def test_las_columnas_son_las_de_train3d():
    """Si `train3d.py` agrega una columna, `metricas_traza` la tiene que nombrar. Se lee el fuente
    y no se importa: el módulo arrastra los datos y el modelo."""
    fuente = (RAIZ / "train3d.py").read_text()
    columnas = int(re.search(r"^COLUMNAS = (\d+)", fuente, re.M).group(1))
    assert set(seguimiento.COLUMNAS_TRAZA) | {8, 9} == set(range(columnas))


def test_nrmse_y_psnr_son_la_misma_informacion_por_volumen():
    """NRMSE = 100 · R · 10^(−PSNR/20) / rms(y) con la misma máscara: en un volumen, la una es
    función de la otra, y val/nrmse_pct sólo cambia el promedio entre volúmenes. El NRMSE es el
    de `ddr/qsm3d.py`, que es el que va a val/nrmse_pct y a qsm/*."""
    qsm3d = pytest.importorskip("ddr.qsm3d")
    gen = np.random.default_rng(0)
    limpio = gen.uniform(0, 1, (24, 24, 24))
    estimado = limpio + 0.05 * gen.standard_normal(limpio.shape)
    for mascara in (np.ones(limpio.shape, bool), limpio > 0.3):
        y, x = limpio[mascara], estimado[mascara]
        esperado = 100 * 1.0 * 10 ** (-psnr(x, y, 1.0) / 20) / np.sqrt(np.mean(y ** 2))
        assert qsm3d.nrmse_pct(estimado, limpio, mascara) == pytest.approx(esperado, rel=1e-12)
