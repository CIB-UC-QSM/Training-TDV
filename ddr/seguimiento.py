"""Seguimiento de experimentos con Weights & Biases para `train3d.py`: una vista de la traza, no
otra fuente. Lo que cuenta sigue siendo `--out` (traza.npy, history.json, estado.json).

Invariante: el seguimiento nunca corta el entrenamiento. Cada llamada a wandb va protegida: el
primer error se imprime una vez, con su tipo y su mensaje, y la corrida sigue sin seguimiento.
`abre` tampoco levanta: si no puede, devuelve None, y con None todas las demás son no-op. Con
`modo="no"` wandb ni se importa, así que este módulo no lo exige instalado.

Identidad: el id de W&B vive en `<out>/seguimiento.json`. Reanudar en el mismo `--out` reabre la
misma corrida si no se registró nada después del estado (abajo, `desde`); una rama (otro `--out`)
es otra, en el grupo del tronco.

Ojo con `desde`: el paso de W&B es la iteración, y W&B descarta sin aviso un paso menor que el
último. Reanudar desde un estado anterior a lo ya registrado abre otra corrida en vez de mezclar
dos linajes en una; para eso `abre` necesita la iteración de la que se sigue. Como `train3d.py`
llama a `registra` cada `--log-every` y escribe los estados cada `--guardar-cada`, eso es lo
normal tras un corte: la misma corrida sólo se reabre al extender una terminada o si el corte cayó
justo en un estado.
"""

from __future__ import annotations

import importlib.util
import json
import math
import netrc
import os
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

__all__ = [
    "abre",
    "falta_para_online",
    "registra",
    "registra_imagenes",
    "resumen",
    "cierra",
    "metricas_traza",
]


# --- MODOS Y CREDENCIAL ----------------------------------------------------------------------
MODOS = ("auto", "online", "offline", "no")
PROYECTO = "training-tdv3d"       # supuesto: el nombre lo decide el autor, con la bandera
HOST = "https://api.wandb.ai"     # el de wandb 0.30 cuando WANDB_BASE_URL no dice otro
# Cuánto espera `wandb.init` antes de rendirse; el default de wandb es 90 s. Supuesto: online no
# se midió (no hay credencial en esta máquina); offline, la primera apertura del proceso tarda
# 0.29 s y las demás 0.046 (medido el 2026-10-12).
INIT_TIMEOUT = 30.0


def _hay_credencial() -> bool:
    """Lo que wandb 0.30 lee antes de preguntar (`sdk/lib/wbauth/authenticate.py`,
    `_use_system_auth`): WANDB_API_KEY, WANDB_IDENTITY_TOKEN_FILE o la entrada del host en el
    .netrc. Sólo lee archivos: ni red, ni prompt, ni wandb."""
    if os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_IDENTITY_TOKEN_FILE"):
        return True
    host = urlsplit(os.environ.get("WANDB_BASE_URL", HOST)).netloc
    ruta = Path(os.environ.get("NETRC", "~/.netrc")).expanduser()
    try:
        entrada = netrc.netrc(ruta).authenticators(host)
    except (OSError, netrc.NetrcParseError):
        return False
    # Sin password, wandb descarta la entrada ("with no API key", `wbauth/wbnetrc.py`) y pregunta:
    # contarla colgaba `auto` en el prompt (visto bajo una pty el 2026-10-12). `authenticators`
    # cae también en `default`.
    return entrada is not None and bool(entrada[2])


def _wandb_tiene_credencial(wandb) -> bool:
    """La misma pregunta, hecha a wandb sin dejarlo preguntar ni ir a la red: él resuelve su host
    (también el `base_url` de su archivo de settings, que `_hay_credencial` no ve)."""
    try:
        return bool(wandb.login(prompt=False, verify=False))
    except Exception:
        return False


def falta_para_online() -> str | None:
    """Qué falta para que `abre(modo="online")` abra, o None: wandb instalado y una credencial.
    Sin importar wandb ni ir a la red, para poder preguntarlo antes de que el DataLoader forkee."""
    if importlib.util.find_spec("wandb") is None:
        return "wandb no está instalado (`uv sync` en la raíz del repo)"
    if not _hay_credencial():
        return "no hay credencial de W&B (`uv run wandb login`, o WANDB_API_KEY)"
    return None


def _modo(modo: str, previo: str | None) -> tuple[str, str]:
    """El modo efectivo ('online', 'offline' o 'no') y su porqué, para el aviso. `previo` es el
    modo de la corrida que se reabre, si se reabre una."""
    if modo != "auto":
        return modo, f"modo {modo}"
    entorno = os.environ.get("WANDB_MODE", "").lower()
    if entorno in ("disabled", "offline", "dryrun", "online", "run"):
        efectivo = {"disabled": "no", "dryrun": "offline", "run": "online"}.get(entorno, entorno)
        return efectivo, f"WANDB_MODE={entorno}"
    if previo == "offline":
        return "offline", "la corrida que se reabre es offline"
    if not _hay_credencial():
        return "offline", "sin credencial: ni WANDB_API_KEY ni el host en ~/.netrc"
    return "online", "hay credencial"


# --- IDENTIDAD DE LA CORRIDA -----------------------------------------------------------------
ARCHIVO = "seguimiento.json"


def _lee_json(ruta: Path) -> dict | None:
    if not ruta.exists():
        return None
    try:
        return json.loads(ruta.read_text())
    except (OSError, ValueError) as e:
        print(f"seguimiento: {ruta} no se pudo leer ({type(e).__name__}: {e}); se abre otra "
              "corrida", flush=True)
        return None


def _escribe_json(ruta: Path, datos: dict) -> None:
    """A un temporal y después `replace`, como `guarda_traza` de `train3d.py`."""
    temporal = ruta.with_name(ruta.stem + ".tmp.json")
    temporal.write_text(json.dumps(datos, indent=2))
    temporal.replace(ruta)


def _identidad(out: Path, proyecto: str, nombre: str, grupo: str | None, padre,
               desde: int) -> tuple[dict, bool]:
    """Los datos de `seguimiento.json` para esta apertura, y si reabre la corrida que ya nombra.
    Los cuatro casos, en el bloque final."""
    previo = _lee_json(out / ARCHIVO)
    datos = {"id": None, "proyecto": proyecto, "nombre": nombre, "grupo": grupo, "modo": None,
             "padre": None, "padre_id": None, "paso": desde, "descartadas": []}
    rama = padre is not None and Path(padre).resolve().parent != out.resolve()
    # Una rama que arranca en un `out` usado sólo sigue su corrida si sale del mismo estado: desde
    # otro es otro linaje aunque `desde` pase lo registrado (`train3d.py` reescribe `out`).
    mismo_origen = not rama or (bool((previo or {}).get("padre"))
                                and Path(previo["padre"]).resolve() == Path(padre).resolve())
    if (previo and previo.get("proyecto") == proyecto and desde >= previo.get("paso", 0)
            and mismo_origen):
        return {**datos, **previo, "nombre": nombre, "grupo": grupo or previo.get("grupo")}, True
    if previo is not None:
        # El análogo de `aparta`: la corrida vieja queda en W&B y su id, acá.
        datos["descartadas"] = [i for i in previo.get("descartadas", []) + [previo.get("id")] if i]
        datos["grupo"] = grupo or previo.get("grupo")
    if rama:
        origen = Path(padre).resolve().parent
        tronco = _lee_json(origen / ARCHIVO) or {}
        datos["padre"], datos["padre_id"] = str(padre), tronco.get("id")
        datos["grupo"] = grupo or tronco.get("grupo") or tronco.get("nombre") or origen.name
    return datos, False


def abre(out_dir, config: dict | None = None, *, modo: str = "auto", proyecto: str = PROYECTO,
         nombre: str | None = None, grupo: str | None = None, padre=None, desde: int = 0):
    """Abre (o reabre) la corrida de W&B de `out_dir` y la devuelve; None si no hay seguimiento.

    `padre` es el `estado-<it>.pth` del que se reanuda (`--reanudar`), y `desde` su iteración; sin
    `--reanudar`, None y 0. `nombre` es por defecto el de `out_dir`. Imprime una línea con el modo,
    el id y, offline, cómo subirlo después.
    """
    if modo == "no":
        return None
    try:
        if modo not in MODOS:
            raise ValueError(f"modo {modo!r}, que no es ninguno de {MODOS}")
        out = Path(out_dir)
        nombre = nombre or out.resolve().name
        datos, reabre = _identidad(out, proyecto, nombre, grupo, padre, desde)
        efectivo, motivo = _modo(modo, datos.get("modo") if reabre else None)
        if efectivo == "no":
            return None
        if efectivo == "online" and not _hay_credencial():
            # wandb.init preguntaría la clave por la terminal, y bajo tmux esperaría para siempre.
            raise RuntimeError("online sin credencial: `wandb login` o WANDB_API_KEY")
        cfg = dict(config or {})
        if datos["padre"]:
            cfg.update(padre=datos["padre"], padre_id=datos["padre_id"])
        import wandb
        if efectivo == "online" and not _wandb_tiene_credencial(wandb):
            # `_hay_credencial` dijo que sí y wandb, para su host, no: init pediría la clave.
            if modo != "auto":
                raise RuntimeError("online sin credencial para el host de wandb: `wandb login`")
            efectivo, motivo = "offline", "wandb no encuentra credencial para su host"
        out.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"seguimiento: no se abre ({type(e).__name__}: {e}); la corrida sigue sin W&B",
              flush=True)
        return None

    intentos = [efectivo] + (["offline"] if modo == "auto" and efectivo == "online" else [])
    run = None
    for m in intentos:
        try:
            run = wandb.init(project=proyecto, name=nombre, group=datos["grupo"], id=datos["id"],
                             resume="allow" if reabre else None, dir=str(out), mode=m,
                             config=cfg, settings=wandb.Settings(init_timeout=INIT_TIMEOUT,
                                                                 console="off"))
            if m != efectivo:
                efectivo, motivo = m, "online no arrancó"
            break
        except Exception as e:
            print(f"seguimiento: wandb.init en modo {m} falló ({type(e).__name__}: {e}); "
                  + ("se prueba offline" if m != intentos[-1] else "la corrida sigue sin W&B"),
                  flush=True)
    if run is None:
        return None

    try:
        for clave, extremo in EXTREMOS.items():
            run.define_metric(clave, summary=extremo)
        datos.update(id=run.id, modo=efectivo)
        _escribe_json(out / ARCHIVO, datos)
        _CORRIDAS[id(run)] = {"json": out / ARCHIVO, "datos": datos, "caido": False}
        como = ("reabierta" if reabre else f"rama de {datos['padre']}" if datos["padre"]
                else "nueva")
        print(f"seguimiento: W&B {efectivo} ({motivo}) | corrida {run.id}, {como}"
              + (f" | grupo {datos['grupo']}" if datos["grupo"] else "")
              + (f" | se sube con `wandb sync {out}/wandb/offline-run-*-{run.id}*`"
                 if efectivo == "offline" else f" | {run.url}"), flush=True)
    except Exception as e:
        print(f"seguimiento: no se abre ({type(e).__name__}: {e}); la corrida sigue sin W&B",
              flush=True)
        _CORRIDAS.pop(id(run), None)
        try:
            run.finish()
        except Exception:
            pass
        return None
    return run


# --- REGISTRO --------------------------------------------------------------------------------
_CORRIDAS: dict[int, dict] = {}    # id(run) -> {"json": Path, "datos": dict, "caido": bool}


def _protegido(run, que: str, accion) -> None:
    """Corre `accion` si la corrida existe y no se cayó; el primer error la apaga para siempre."""
    if run is None:
        return
    estado = _CORRIDAS.setdefault(id(run), {"json": None, "datos": None, "caido": False})
    if estado["caido"]:
        return
    try:
        accion()
    except Exception as e:
        estado["caido"] = True
        print(f"seguimiento: {que} falló ({type(e).__name__}: {e}); W&B queda apagado el resto "
              "de la corrida y el entrenamiento sigue", flush=True)


def _avanza(run, paso: int) -> None:
    """Anota en `seguimiento.json` el último paso registrado: es lo que compara `desde`."""
    estado = _CORRIDAS.get(id(run))
    if estado and estado["json"] is not None and paso > estado["datos"]["paso"]:
        estado["datos"]["paso"] = paso
        _escribe_json(estado["json"], estado["datos"])


def registra(run, metricas: dict, paso: int) -> None:
    """Las `metricas` en el paso `paso`, que es la iteración. Varias llamadas con el mismo paso van
    a la misma fila; un paso menor que el último lo descarta W&B sin aviso."""
    def accion():
        run.log(dict(metricas), step=int(paso))
        _avanza(run, int(paso))
    _protegido(run, "registra", accion)


def registra_imagenes(run, imagenes: dict, paso: int, leyendas: dict | None = None) -> None:
    """Imágenes uint8, `[H, W]` o `[H, W, 3]`, en el paso `paso`. Otro dtype es un error, que
    apaga el seguimiento: wandb 0.30 pasa un float a uint8 con `astype`
    (`wandb/sdk/data_types/image.py:39`), y una imagen en [0, 1] saldría negra sin aviso. Las
    láminas de `ddr/qsm3d.py` ya salen en uint8, con un rango fijo."""
    def accion():
        import wandb
        lote = {}
        for clave, imagen in imagenes.items():
            a = np.asarray(imagen)
            if a.dtype != np.uint8:
                raise TypeError(f"la imagen {clave!r} es {a.dtype}, no uint8")
            lote[clave] = wandb.Image(a, caption=(leyendas or {}).get(clave))
        run.log(lote, step=int(paso))
        _avanza(run, int(paso))
    _protegido(run, "registra_imagenes", accion)


def alerta(run, titulo: str, texto: str) -> None:
    """Una alerta de W&B: llega por correo o Slack si la cuenta tiene encendidas las «Scriptable run
    alerts». Va fuera de `_protegido`: si falla (offline no las manda), avisa en el log y no apaga
    el registro de métricas."""
    if run is None or _CORRIDAS.get(id(run), {}).get("caido"):
        return
    try:
        run.alert(title=titulo, text=texto, level="WARN")
    except Exception as e:
        print(f"seguimiento: la alerta «{titulo}» no salió ({type(e).__name__}: {e})", flush=True)


def resumen(run, datos: dict) -> None:
    """Escribe `datos` en el resumen de la corrida (la tabla de corridas de W&B)."""
    _protegido(run, "resumen", lambda: run.summary.update(dict(datos)))


def cierra(run) -> None:
    """Cierra la corrida. Va después de escribir `estado.json`: online puede tardar en subir lo
    pendiente. Si el seguimiento se cayó, no la toca: la cierra wandb al salir el proceso."""
    if run is None:
        return
    _protegido(run, "cierra", lambda: run.finish())
    _CORRIDAS.pop(id(run), None)


# --- MÉTRICAS --------------------------------------------------------------------------------
# Columna de `traza.npy` -> nombre en W&B; las columnas, en docs/porque-del-codigo.md, TRAZA.
# La 8 y la 9, rechazado y rebobinado (0/1), entran acumuladas por `metricas_traza`.
COLUMNAS_TRAZA = {
    0: "train/loss", 1: "train/T", 2: "train/dT", 3: "train/lambda", 4: "train/dlambda",
    5: "train/grad_norm", 6: "train/grad_mediana", 7: "train/amp", 10: "train/grad_norm_micro",
    11: "train/lr", 12: "train/iteracion_efectiva", 13: "train/sigma_medio",
}
# El resumen de W&B guarda el último valor; de éstas guarda el extremo, que es el que se compara.
EXTREMOS = {
    "val/psnr": "max", "val/psnr_tejido": "max", "val/psnr_cerebro": "max",
    "val/nrmse_pct": "min", "val/nrmse_pct_tejido": "min",
    "qsm/nrmse_final_pct": "min", "qsm/nrmse_min_pct": "min",
    "val/nrmse_pct_cerebro": "min", "qsm/hfen_pct": "min", "qsm/xsim": "max",
    "qsm_sim2/nrmse_min_pct": "min", "qsm_sim2/hfen_pct": "min", "qsm_sim2/xsim": "max",
}


def metricas_traza(traza: np.ndarray, it: int) -> dict[str, float]:
    """Las métricas `train/*` de la iteración `it` (1-indexada) de una traza de `train3d.py`.

    `train/rechazos` y `train/rebobinados` acumulan las columnas 8 y 9 hasta `it`, como los
    contadores del log. `train/grad_razon` es ‖g‖ sobre la mediana de la ventana, lo que compara
    el disyuntor. Lo que es NaN por diseño no entra: `train/amp` sin --guard-amp, y la mediana y
    la razón con la ventana vacía. Un NaN de la pérdida, de `T` o de ‖g‖ sí: es una muerte.
    """
    fila = traza[it - 1]
    m = {nombre: float(fila[c]) for c, nombre in COLUMNAS_TRAZA.items()}
    m["train/rechazos"] = float(traza[:it, 8].sum())
    m["train/rebobinados"] = float(traza[:it, 9].sum())
    for clave in ("train/amp", "train/grad_mediana"):
        if not math.isfinite(m[clave]):
            del m[clave]
    if m.get("train/grad_mediana", 0.0) > 0:
        m["train/grad_razon"] = m["train/grad_norm"] / m["train/grad_mediana"]
    return m


'''
# --- MODOS Y CREDENCIAL ----------------------------------------------------------------------
# `auto`, el default, elige en este orden:
#   1. WANDB_MODE, si está: disabled es `no`, offline o dryrun es `offline`, online o run es
#      `online`. Es la convención de wandb, y es como un test de `train3d.py` lo apaga sin bandera.
#   2. Si se reabre una corrida offline, offline: sus segmentos se suben juntos y en orden con
#      `wandb sync`. Uno online en el medio llegaría al servidor antes que los offline
#      anteriores, con pasos mayores; qué hace el servidor con los menores que lleguen después
#      no está verificado.
#   3. Online si hay credencial (`_hay_credencial`); si no, offline.
#   4. Si `wandb.init` online falla (sin red, clave inválida, el timeout), reintenta offline.
# `online` explícito no cae a offline: sin credencial o si init falla, no hay seguimiento, con el
# aviso. `_hay_credencial` va antes de init porque wandb 0.30, online y sin credencial, pregunta
# la clave por la terminal (`wbauth/authenticate.py`, `_use_prompted_auth`) con el timeout de la
# configuración global, que por defecto es ninguno: bajo tmux la corrida esperaría para siempre
# antes de la primera iteración. No mira la `api_key` de un archivo de settings de wandb: con la
# clave sólo ahí, `auto` sale offline. Tampoco ve su `base_url` (el que deja `wandb login
# --host`): con él en otro host y el .netrc sólo para api.wandb.ai, `_hay_credencial` decía que sí
# y init se quedaba en el prompt; lo mismo con una entrada del .netrc sin password (las dos, vistas
# bajo una pty el 2026-10-12). Por eso, antes de init online, se le pregunta a wandb con
# `wandb.login(prompt=False, verify=False)`, que resuelve su host y sus fuentes sin prompt ni red;
# si dice que no, `auto` va offline sin probar online.
# Con un servidor que no contesta, init online tarda INIT_TIMEOUT más ~10 s (40.6 s medido) y
# cae a offline; con uno que contesta 401, 0.6 s.
# `console="off"`: el log de `train3d.py` (stdout, al journal o a tmux) es la fuente, y la
# captura de wandb lo duplicaría y envolvería stdout. Las métricas de sistema (GPU, CPU, memoria)
# quedan en su default, prendidas.
# Offline, la corrida vive en `<out>/wandb/offline-run-<fecha>-<id>/`, un directorio por cada vez
# que se abrió (con `-1`, `-2`... si dos empiezan en el mismo segundo); `wandb sync
# <out>/wandb/offline-run-*-<id>*` los sube en orden de inicio (lo dice `wandb sync --help` de
# 0.30; sin servidor, no verificado). Esta máquina no tiene credencial (2026-10-12): acá `auto` es
# offline.

---------------------------------------------------------------------------------------------------

# --- IDENTIDAD DE LA CORRIDA -----------------------------------------------------------------
# `<out>/seguimiento.json`: id, proyecto, nombre, grupo, modo, padre, padre_id, paso (el último
# registrado) y descartadas. Se escribe al abrir y en cada `registra` que avanza el paso, a un
# temporal y después `replace`. `abre` decide entre cuatro casos:
#   - sin json en `out`: una corrida nueva.
#   - json del mismo proyecto y `desde` >= paso: la MISMA corrida, con resume="allow"; los pasos
#     siguen. Es seguir tras un corte desde el último estado, o extender una terminada. Si `padre`
#     está en otro directorio (arranca una rama), además tiene que ser el `padre` del json: otro
#     punto de rama en el mismo `out` es otro linaje y cae en el caso siguiente.
#   - json y `desde` < paso: una corrida NUEVA, y el id viejo pasa a `descartadas`. Es el análogo
#     de `aparta` en `train3d.py`: lo registrado después de `desde` es de un linaje que no sigue.
#     Reabrir la misma no serviría: W&B descarta sin aviso un paso menor que el último (medido
#     offline, 2026-10-12), así que la corrida guardaría el tramo del linaje viejo y tiraría el
#     del nuevo hasta pasarlo, sin que nada se ponga rojo. Arrancar de cero (`desde` = 0) en un
#     `--out` usado cae acá.
#   - `padre` en otro directorio: una rama. Corrida nueva, con group = el grupo del tronco o, si
#     no tiene, su nombre (el de su json o el de su directorio), y config["padre"] = la ruta del
#     estado, config["padre_id"] = el id del tronco. Reanudar la rama en su propio `--out` es el
#     segundo caso: sigue siendo ella, en el mismo grupo.
# La corrida nueva no copia la historia previa: una rama empieza en `desde` + 1, y lo anterior
# está en la del tronco (padre_id); un linaje descartado, en la descartada. La historia entera
# está en `traza.npy`, que la rama hereda del tronco.
# Sin verificar (no hay servidor): que online, al reabrir, wandb acepte una config con valores
# cambiados (`reanudar`, `iterations`), y que los extremos del resumen sobrevivan a una reapertura.

---------------------------------------------------------------------------------------------------

# --- REGISTRO --------------------------------------------------------------------------------
# Toda llamada a wandb pasa por `_protegido`: atrapa `Exception` (no KeyboardInterrupt), imprime
# una línea con el tipo y el mensaje, y marca la corrida como caída: ninguna llamada más le llega,
# ni `cierra`. La cierra el `atexit` de wandb al salir el proceso. Así un servicio de wandb caído,
# un disco lleno o una imagen mal formada cuestan el seguimiento, nunca la corrida.
# Varias llamadas con el mismo paso van a la misma fila, y la fila se cierra cuando llega un paso
# mayor o en `cierra`: un corte pierde, como mucho, la última fila. La traza la tiene.
# Dónde va cada llamada en `train3d.py` (el porqué, en docs/porque-del-codigo.md, SEGUIMIENTO):
#   - `abre(out, vars(args) | commit, modo=..., padre=args.reanudar, desde=it0)`, en
#     `abre_seguimiento`, después de `iter(loader)`: los workers ya están creados, con
#     `persistent_workers`, y no se forkea un proceso con los hilos de wandb vivos (derivado; no
#     se vio fallar).
#   - cada --log-every, `registra(run, metricas_traza(traza, it) | extra, it)`, con extra =
#     train/grad_norm_max (el max|g| del log en el intervalo: un estallido, que es de una
#     iteración, no se pierde entre dos muestras), train/s_por_it y train/vram_pico_gib. Por
#     iteración también entraba: medido offline (2026-10-12, con load average 61 en 32 CPU),
#     0.185 ms de mediana por llamada y p99 0.335 ms, contra ~9 s de una iteración en la
#     5070 Ti; 1 337 B por fila de 15 métricas, 53 MB a 40 000 iteraciones (derivado), más
#     ~0.64 MB de métricas de sistema por hora de corrida (derivado: un registro de 1 kB cada
#     5.8 s). Un estado pesa ~51 MB en la 40 000 (docs/porque-del-codigo.md, RECETA 3D). Se
#     eligió --log-every porque la traza ya tiene todas las filas y W&B muestra lo mismo que el
#     log.
#   - en cada validación, `registra` con val/* y `registra_imagenes` con val/denoising; en cada
#     evaluación QSM, qsm/* y qsm/reconstruccion.
#   - al final, después de `estado.json`, `resumen(run, <estado.json>)` y `cierra(run)`.
# Las banderas de `train3d.py` (--seguimiento, --wandb-proyecto) están en LIBRES: reanudar un
# estado anterior a ellas o cambiar el modo no lo rechaza `--reanudar`. Sus tests, con `no`, no
# abren nada, y los que lo miran corren offline.
# Medido en tests/test_seguimiento.py: abrir, registrar métricas e imágenes y cerrar no toca el
# generador global de torch, el de numpy ni el de `random`: prender el seguimiento no cambia la
# aritmética de la corrida.

---------------------------------------------------------------------------------------------------

# --- MÉTRICAS --------------------------------------------------------------------------------
# Los nombres, para que el integrador use éstos y no otros:
#   train/loss (col. 0), train/T (1), train/dT (2, ∂L/∂T), train/lambda (3), train/dlambda (4),
#   train/grad_norm (5, ‖g‖ sin recortar), train/grad_mediana (6, de la ventana del disyuntor),
#   train/amp (7, sólo con --guard-amp), train/grad_norm_micro (10), train/lr (11),
#   train/iteracion_efectiva (12), train/sigma_medio (13), train/rechazos y train/rebobinados
#   (8 y 9 acumuladas), train/grad_razon (5 / 6), más train/grad_norm_max, train/s_por_it y
#   train/vram_pico_gib, que no son una fila de la traza. ‖g‖ se lee contra la mediana, nunca en
#   absoluto (AGENTS.md): por eso grad_razon.
#   val/psnr, val/psnr_tejido, val/psnr_cerebro (sólo con las máscaras de `mascaras/val`): los de
#   `evaluate_denoising3d`. val/nrmse_pct, val/nrmse_pct_tejido: el NRMSE por volumen, en todo el
#   recorte y en tejido, promediado entre volúmenes como el PSNR; `evaluate_denoising3d` lo
#   calcula con `nrmse_pct` de `ddr/qsm3d.py`, la misma definición, sin una segunda copia.
#   qsm/nrmse_final_pct, qsm/nrmse_min_pct, qsm/iter_min, qsm/iteraciones, qsm/alpha,
#   qsm/segundos: las claves nrmse_final, nrmse_min, it_min, iteraciones, alpha y segundos de lo
#   que devuelve `qsm3d.admm_wtdv`; `motivo` ("corte", "max_iter", "no_finito") es texto y va al
#   log, a `qsm.jsonl` y al resumen, no a la historia. qsm/fallo: 1 si la evaluación levantó, 0
#   si no.
#   Imágenes: val/denoising y qsm/reconstruccion, las láminas `lamina_denoising` y `lamina_qsm`
#   de `ddr/qsm3d.py`, en uint8 y con rango fijo ([0, 1] en T1, ±0.1 ppm en QSM).
# NRMSE = 100 · ‖x̂ − y‖ / ‖y‖, el `rmse` de `../tdv/ddr/utils.py:91-99` (26.89 % final en
# `../tdv/wTDV_QSM_torch_3d.py:244`). En un volumen y con la misma máscara no agrega información
# al PSNR: NRMSE = 100 · R · 10^(−PSNR/20) / rms(y), con R el rango de los datos (test). Lo que
# cambia es el promedio entre volúmenes, que con NRMSE pesa más a los de poca energía. En
# validación sirve para hablar en las unidades de QSM; en QSM es LA métrica, porque la
# susceptibilidad tiene signo y no tiene rango: el PSNR no está definido.
'''
