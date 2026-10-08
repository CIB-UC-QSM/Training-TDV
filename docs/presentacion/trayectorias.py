"""Catálogo de estados guardados a lo largo de UNA misma trayectoria de entrenamiento.

Las corridas sólo guardan `best.pth` y `last.pth`, pero en `referencia` el entrenamiento es
bit-reproducible, así que dos corridas con la misma semilla, el mismo `T_init` y el mismo código
recorren el mismo camino hasta donde las dos llegan. Con eso se arma cada trayectoria:

- sana s1: `t8delta-100k-corte-s1` ES la final s1 hasta la it 18 000 (predicción 3 de la enmienda,
  verificada evaluación por evaluación, `docs/bitacora.md` ~3700).
- estallada: `referencia-100k` ES `lambdainit/B-seed0` reproducida bit a bit y extendida
  (`docs/bitacora.md`, Estado). Por eso B-seed0 da los estados de la it 1 000 y 4 000.
- primer disyuntor: mismo init y misma semilla que la estallada; es su control pareado.

El init (it 0) se reconstruye como lo hace `train.py:179-193`: `torch.manual_seed(seed)` y
`VNet2D(vnet_config(t_init))`, en CPU, sin nada que consuma el generador antes.
"""
from estilo import RUNS, OFICIAL, RAIZ, C

CKPT_INIT = RAIZ / "docs" / "presentacion" / "salida" / "ckpt"   # *.pth, gitignorados

# tag -> (etiqueta corta, iteración, checkpoint o ("init", semilla, t_init))
ESTADOS = {
    "init_s1": ("init", 0, ("init", 1, 0.1011928851)),
    "s1_18k": ("it 18k", 18_000, RUNS / "t8delta-100k-corte-s1/last.pth"),
    "s1_fin": ("it 100k (final)", None, RUNS / "t8delta-100k-s1/best.pth"),
    "init_s0": ("init", 0, ("init", 0, 0.1011928851)),
    "s0_16k": ("it 16k", 16_000, RUNS / "t8delta-100k-corte-s0/best.pth"),
    "s0_fin": ("it 64k (mejor)", None, RUNS / "t8delta-100k-s0/best.pth"),
    "init_ref": ("init", 0, ("init", 0, 0.1)),
    "ref_1k": ("it 1k", 1_000, RUNS / "lambdainit/B-seed0/best.pth"),
    "ref_4k": ("it 4k", 4_000, RUNS / "lambdainit/B-seed0/last.pth"),
    "ref_38k": ("it 38k", 38_000, RUNS / "referencia-100k/best.pth"),
    "ref_100k": ("it 100k", 100_000, RUNS / "referencia-100k/last.pth"),
    "dis_4k": ("it 4k", 4_000, RUNS / "disyuntor-100k/best.pth"),
    "dis_50k": ("it 50k", 50_000, RUNS / "disyuntor-100k/last.pth"),
    "r10_3k": ("it 3k", 3_000, RUNS / "precision/rapido-seed10/best.pth"),
    "r10_4k": ("it 4k", 4_000, RUNS / "precision/rapido-seed10/last.pth"),
    "v4_68k": ("it 68k", 68_000, RUNS / "replica-v4/checkpoint.pth"),
    "v4_100k": ("it 100k", 100_000, RUNS / "replica-v4/last.pth"),
    "oficial": ("oficial", None, OFICIAL),
}

# Estados cuyas eigenfunciones ya calcula otro lote con otro nombre: no se calculan dos veces.
ALIAS_EIG = {"s0_fin": "final_s0", "s1_fin": "final_s1", "ref_38k": "ref100k", "r10_4k": "estallada",
             "v4_68k": "v4", "oficial": "oficial"}

# Filas de las láminas: (título, color, [tags en orden temporal])
TRAYECTORIAS = [
    ("final s1 · sana (rebobinó 4 veces)", C["final"], ["init_s1", "s1_18k", "s1_fin"]),
    ("final s0 · sana", C["final"], ["init_s0", "s0_16k", "s0_fin"]),
    ("referencia-100k · estalla en la it 1 958", C["fallida"],
     ["init_ref", "ref_1k", "ref_4k", "ref_38k", "ref_100k"]),
    ("1er disyuntor · mismo init, se traba", C["fallida"], ["init_ref", "dis_4k", "dis_50k"]),
    ("rapido-seed10 · estalla en la it 3 019", C["fallida"], ["r10_3k", "r10_4k"]),
    ("v4 · estalla en la it 69 250", C["v4"], ["v4_68k", "v4_100k"]),
]


def ruta(tag):
    """Ruta del checkpoint del estado; los init se crean la primera vez (CPU, segundos)."""
    _, _, ck = ESTADOS[tag]
    if not (isinstance(ck, tuple) and ck[0] == "init"):
        return ck
    p = CKPT_INIT / f"{tag}.pth"
    if not p.exists():
        import os, sys
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        sys.path.insert(0, str(RAIZ))
        from ddr.model import VNet2D
        from train import vnet_config
        _, semilla, t_init = ck
        torch.manual_seed(semilla)
        m = VNet2D(vnet_config(t_init))
        CKPT_INIT.mkdir(parents=True, exist_ok=True)
        torch.save({"config": vnet_config(t_init), "model": m.state_dict(), "iteration": 0}, p)
    return p


if __name__ == "__main__":
    # Imprime los pares tag=checkpoint para eigen_extra.sh
    print(" ".join(f"{ALIAS_EIG.get(t, t)}={ruta(t)}" for t in ESTADOS if t != "oficial"
                   and t not in ("s0_fin", "s1_fin", "ref_38k", "v4_68k")))
