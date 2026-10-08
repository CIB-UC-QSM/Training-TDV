"""Copia `last.pth` de las corridas finales cada vez que cambia: estados cada 2 000 it hasta la 100k.

    nohup setsid uv run python docs/presentacion/vigia_instantaneas.py > /dev/null 2>&1 &

Por qué existe: `train.py` sólo guarda `best.pth` y `last.pth` y los pisa en cada evaluación, así
que sin esto los estados intermedios se pierden (los de la it 50 000–60 000 ya se perdieron; ver
`por_implementar.md`). Es SÓLO LECTURA sobre `runs/`: no toca `train.py`, ni el proceso, ni las GPUs.

Ojo con la escritura: `torch.save` no es atómico (train.py:347). Se copia recién cuando el
archivo lleva 20 s sin cambiar, y la copia se valida cargándola en CPU; si falla, se reintenta.
Termina solo cuando las dos corridas tienen `estado.json` y su última instantánea copiada.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
CORRIDAS = ["t8delta-100k-s0", "t8delta-100k-s1"]
DESTINO = RAIZ / "docs/presentacion/salida/ckpt/inst"
QUIETO_S, CADA_S = 20, 60
LOG = DESTINO / "vigia.log"


def anota(msg):
    DESTINO.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def copia(n):
    """Copia last.pth si es nuevo y está quieto. Devuelve la iteración copiada o None."""
    src = RAIZ / "runs" / n / "last.pth"
    if not src.exists() or time.time() - src.stat().st_mtime < QUIETO_S:
        return None
    tmp = DESTINO / n / ".tmp.pth"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    m0 = src.stat().st_mtime
    shutil.copy2(src, tmp)
    if src.stat().st_mtime != m0:          # lo reescribieron mientras se copiaba
        return None
    try:
        it = int(torch.load(tmp, map_location="cpu", weights_only=False)["iteration"])
    except Exception as e:                  # noqa: BLE001 — copia a medio escribir: reintentar
        anota(f"{n}: copia inválida ({type(e).__name__}), reintento")
        return None
    final = DESTINO / n / f"it{it:06d}.pth"
    if final.exists():
        tmp.unlink()
        return None
    tmp.rename(final)
    anota(f"{n}: it {it}")
    return it


def terminada(n):
    e = RAIZ / "runs" / n / "estado.json"
    return e.exists() and json.loads(e.read_text()).get("terminada")


anota(f"arranca, pid {os.getpid()}")
while True:
    for n in CORRIDAS:
        copia(n)
    listas = [terminada(n) and (DESTINO / n / "it100000.pth").exists() for n in CORRIDAS]
    if all(listas):
        anota("las dos terminaron y tienen la it 100 000: fin")
        sys.exit(0)
    time.sleep(CADA_S)
