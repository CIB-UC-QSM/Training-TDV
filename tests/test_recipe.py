"""Invariantes de la receta: que los constantes de `train.py` sigan siendo los publicados, y que
`T_INIT` quede fuera de la banda donde un pico de gradiente puede llevar `T` a cero.

Estos tests no verifican el modelo, verifican **decisiones**. Existen porque `T_INIT` se deriva de
`LR` y `BETAS`: tocar cualquiera de los dos lo recalcula en silencio, y sin esta guardia podría
volver a caer donde murieron tres corridas.
"""

import math

import pytest

import train


def test_constantes_publicadas():
    """Sec. 5.1, textual: batch 32, 1e5 iteraciones, beta1=0.9, beta2=0.999, lr 4e-4 dividido a la
    mitad cada 25000. Más la arquitectura TDV33 leída de los .pth oficiales."""
    assert (train.BATCH_SIZE, train.ITERATIONS) == (32, 100_000)
    assert (train.LR, train.LR_HALVE_EVERY, train.BETAS) == (4e-4, 25_000, (0.9, 0.999))
    assert (train.PATCH_SIZE, train.SIGMA, train.STEPS) == (93, 25.0, 10)
    assert (train.FEATURES, train.SCALES, train.MACRO_BLOCKS) == (32, 3, 3)
    # El batch que fija el paper es el EFECTIVO, y `ACUMULACION` es división entera: con
    # MICRO_BATCH = 12 el batch real pasa a 24 y las líneas de arriba siguen pasando.
    assert train.ACUMULACION * train.MICRO_BATCH == train.BATCH_SIZE
    assert train.CHANNELS == 3          # gris construiría otro modelo sin romper nada más


def test_el_scheduler_divide_a_la_mitad():
    """«lr 4e-4 dividido a la mitad cada 25 000» son DOS cosas, y sólo el cada-cuánto estaba
    verificado: cambiar el `gamma` del `StepLR` a 0.1 dejaba los tests en verde."""
    import inspect
    fuente = inspect.getsource(train.main)
    assert "StepLR" in fuente
    assert "gamma=0.5" in fuente, "el paper divide a la mitad, no por otro factor"


def test_delta_adam_es_el_desplazamiento_publicado():
    """`DELTA_ADAM` se deriva de dos números de la sec. 5.1 y de nada más. Si alguien toca `LR` o
    `BETAS`, la regla 2-delta se recalcula sola y los dos tests de abajo lo atrapan."""
    assert train.DELTA_ADAM == train.LR / math.sqrt(1 - train.BETAS[1])
    assert abs(train.DELTA_ADAM - 0.012649) < 1e-6


def test_regla_2delta():
    """Que los DEFAULTS despejen `DELTA_ADAM` con margen. El nombre dice «regla 2-delta» y esa
    regla está retirada (bitácora, «Lo que dejó el 2D»): el factor 2 era una elección y
    nunca predijo nada —`tinit/B-larga` arrancó EN 2·delta y murió con `T = 0`, `1.1·delta` terminó
    las 100 000—. El test se queda porque lo que assertea sigue valiendo como tripwire de
    procedencia: que nadie baje los init en silencio. No renombrado para no romper referencias.

    Un pico de gradiente desplaza cualquier parámetro en `DELTA_ADAM`, cantidad que NO depende del
    tamaño del pico (AGENTS.md, «Lo que ya se sabe»), así que un escalar con pared en 0 que valga
    menos que eso lo manda al clamp un solo lote.
    `T_INIT = 0.001` mató tres corridas por esa vía.

    Es un PISO, no el valor: los dos escalares arrancan en 0.1, que lo cumple con margen. Y lo que
    el piso NO cubre: `lambda` no muere por un pico sino por deriva, y contra eso hace falta
    2.2*DELTA_ADAM. Vale como higiene de procedencia, no como corrección de un modo de fallo —
    medido el 2026-09-06, `lambda` cuesta 0.003 dB en todo ese rango."""
    for nombre, valor in (("T_INIT", train.T_INIT), ("LAMBDA_INIT", train.LAMBDA_INIT)):
        assert valor >= 2 * train.DELTA_ADAM, (
            f"{nombre}={valor:.5f} queda a menos de dos picos del clamp en cero "
            f"(DELTA_ADAM = {train.DELTA_ADAM:.5f})"
        )


def test_lambda_no_esta_atada_a_T():
    """`T` y `lambda` son dos escalares independientes, como en el `model.py` de los autores: sus
    únicos modos son `fixed` y `learned`, no hay modo atado. Los dos arrancan en 0.1 por medición
    (2026-09-07), pero cambiar uno no mueve al otro."""
    cfg = train.vnet_config(0.05)
    assert cfg["T"]["init"] == 0.05
    assert cfg["lambda"]["init"] == train.LAMBDA_INIT == 0.1


def test_multiplier_no_duplica_features():
    """Los checkpoints oficiales usan multiplier=1; el test unitario del repo original usa 2."""
    assert train.vnet_config(train.T_INIT)["R"]["config"]["multiplier"] == 1
