"""El disyuntor de lotes: que rechace el pico sin tocar la corrida sana.

Como `tests/test_recipe.py`, estos tests verifican **decisiones**, no el modelo. Cada número del
disyuntor está calibrado sobre las 205 829 iteraciones sanas de las 81 corridas del repo
(`scripts/calibrar_disyuntor.py`) más el post-mortem de `runs/disyuntor-100k`. Sin esta guardia,
aflojar la razón o subir la cota pasaría desapercibido y volvería a dejar entrar el lote que mete
la corrida en la cuenca mala.
"""

import copy
import math

import pytest
import torch

import train

INF = float("inf")
NADA: list[float] = []
# El transitorio del arranque, mediana sobre las 81 corridas: sube 26 -> 828 en doce iteraciones.
TRANSITORIO = [26, 29, 54, 82, 125, 179, 251, 350, 440, 527, 565, 828, 667, 738, 695, 790]


def corre(normas, ratio=train.GUARD_RATIO, ventana=train.GUARD_VENTANA, amp_max=INF, amps=None,
          abs_max=INF):
    """Pasa una secuencia de normas por el disyuntor y devuelve (rechazos, escala, centros)."""
    escala, rech, centros = [], [], []
    for i, g in enumerate(normas):
        a = INF if amps is None else amps[i]
        r, c = train.rechaza_lote(float(g), float(a), escala, ratio, abs_max, amp_max, ventana)
        rech.append(r)
        centros.append(c)
    return rech, escala, centros


def test_la_razon_queda_debajo_de_la_banda_de_entrada():
    """La razón ya NO sale de `beta2`. `1/sqrt(1-beta2) = 31.62` es el pico que duplica `v`, y con
    ése `runs/disyuntor-100k` entró a la cuenca mala: los tres lotes que la metieron iban a 8.97,
    7.94 y 7.70 veces la mediana, y los aceptó a los tres. La razón tiene que quedar debajo."""
    assert train.GUARD_RATIO < 7.70
    # Y no tan abajo: a razón 5 el corpus sano rechaza 318/205829 = 0.1545 %, que es el costo
    # medido. Por debajo de 3 no hay medición y el tramo sano llega a 130x la mediana.
    assert 3 <= train.GUARD_RATIO <= 7


def test_la_cota_absoluta_deja_pasar_todo_el_tramo_sano():
    """Separa «ya está muerta» de «viva», no previene: el máximo sano de las 81 corridas es 9 397
    y el mínimo de `runs/disyuntor-100k` desde que se trabó es 8.29e4, un hueco de 8.8x. La cota
    va en medio. Si bajara de 9 397 empezaría a rechazar iteraciones sanas del corpus."""
    assert 9_397 < train.GUARD_ABS < 82_880
    # Y el piso real, medido el 2026-10-05: las excursiones que se resuelven SOLAS llegan a 5 312
    # (`tinit-7p9delta-10k` ev.1) y 6 113 (`precision/rapido-seed2`, que terminó en 30.937 dB sin
    # disyuntor). Bajar la cota ahí abajo la haría rechazar excursiones que no hacían falta tocar.
    assert train.GUARD_ABS > 6_113


def test_el_rebobinado_no_se_dispara_en_sano():
    """A razón 5 el corpus sano llega a 18 rechazos seguidos (medido). El umbral tiene que quedar
    por encima, o una corrida sana rebobinaría sola y perdería progreso que estaba bien."""
    assert train.GUARD_SEGUIDOS > 18


def test_la_cota_absoluta_funciona_con_la_ventana_vacia():
    """Es su único trabajo que el relativo no puede hacer: el relativo está desarmado hasta que la
    ventana se llena, y ahí la absoluta es lo único que queda. Desde que la ventana viaja con el
    respaldo ése es su único tramo. Con la ventana vacía, el pico de 1e6 tiene que salir afuera."""
    rech, escala, _ = corre([1e6, 30.0, 31.0], ventana=100, abs_max=train.GUARD_ABS)
    assert rech == [True, False, False]
    # Y no entra en la escala, igual que cualquier otro rechazo.
    assert escala == [30.0, 31.0]


def test_el_rebobinado_devuelve_pesos_y_momentos_exactos():
    """Rebobinar tiene que deshacer el paso malo COMPLETO: si volvieran los pesos y no `m`/`v`,
    Adam seguiría empujando en la dirección que lo rompió."""
    modelo = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-2, betas=train.BETAS)
    for _ in range(3):
        opt.zero_grad()
        for q in modelo.parameters():
            q.grad = torch.full_like(q, 0.5)
        opt.step()

    respaldo = train.respalda(modelo, opt, [10.0, 11.0, 12.0])
    bueno = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
    estado_bueno = copy.deepcopy(opt.state_dict())

    # El paso que rompe: un gradiente enorme que mueve los pesos y envenena `m` y `v`.
    opt.zero_grad()
    for q in modelo.parameters():
        q.grad = torch.full_like(q, 1e6)
    opt.step()
    assert not torch.equal(modelo.weight.detach(), bueno["weight"])

    escala = [1.0, 2.0, 3.0]
    train.rebobina(modelo, opt, respaldo, escala)
    for k, v in modelo.state_dict().items():
        assert torch.equal(v, bueno[k]), k
    for i, q in enumerate(modelo.parameters()):
        adam, viejo = opt.state[q], estado_bueno["state"][i]
        assert torch.equal(adam["exp_avg"], viejo["exp_avg"])
        assert torch.equal(adam["exp_avg_sq"], viejo["exp_avg_sq"])
        assert adam["step"] == viejo["step"]
    # Y la ventana vuelve a la del respaldo, que es la que vale otra vez: borrarla dejaba el
    # relativo desarmado 100 iteraciones justo al salir del evento, y las dos primeras `‖g‖`
    # post-rebobinado medidas en `tinit-7p9delta-10k` —48.5 y 21.3— caen DENTRO de esa ventana.
    assert escala == [10.0, 11.0, 12.0]


def test_el_segundo_rebobinado_al_mismo_respaldo_tambien_devuelve_adam():
    """El test de arriba rebobina una vez. El lazo de `t1-guard-2delta-5k` empezó en el SEGUNDO
    rebobinado a la it 1 756: `opt.load_state_dict` adopta los tensores que recibe, y los pasos
    dados entre uno y otro habían mutado el respaldo (`docs/mediciones/alias-respaldo-adam.txt`)."""
    modelo = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-2, betas=train.BETAS)

    def paso(valor):
        opt.zero_grad()
        for q in modelo.parameters():
            q.grad = torch.full_like(q, valor)
        opt.step()

    for _ in range(3):
        paso(0.5)
    respaldo = train.respalda(modelo, opt, [10.0])
    estado_bueno = copy.deepcopy(opt.state_dict())

    for _ in range(2):
        paso(1e6)
        train.rebobina(modelo, opt, respaldo, [])
    for i, q in enumerate(modelo.parameters()):
        assert torch.equal(opt.state[q]["exp_avg"], estado_bueno["state"][i]["exp_avg"])
        assert torch.equal(opt.state[q]["exp_avg_sq"], estado_bueno["state"][i]["exp_avg_sq"])
        assert opt.state[q]["step"] == estado_bueno["state"][i]["step"]


def test_un_rebobinado_que_cruza_un_hito_no_deshace_el_halving():
    """El `state_dict` de Adam trae el lr del momento del respaldo. Restaurarlo tal cual dejaba el
    lr al doble del calendario hasta el final de la corrida (6-13 % de riesgo en `t1-rapido-30k`,
    `docs/mediciones/edad-del-respaldo.txt`)."""
    modelo = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(modelo.parameters(), lr=train.LR, betas=train.BETAS)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[3], gamma=0.5)
    respaldo = None
    for it in range(1, 6):
        opt.zero_grad()
        for q in modelo.parameters():
            q.grad = torch.full_like(q, 0.5)
        opt.step()
        sched.step()
        if it == 2:
            respaldo = train.respalda(modelo, opt, [])
    assert opt.param_groups[0]["lr"] == train.LR / 2
    train.rebobina(modelo, opt, respaldo, [])
    assert opt.param_groups[0]["lr"] == train.LR / 2


def test_la_ventana_se_arma_antes_del_evento_mas_temprano():
    """El disyuntor queda armado en la it `ventana+1`, y el evento más temprano de las 81 corridas
    es el de `lambdainit/B-seed5`, en la 289."""
    assert train.GUARD_VENTANA + 1 < 289


def test_desarmado_no_rechaza_el_transitorio_inicial():
    """Mientras la ventana no se llena la escala no es estimable, y el arranque sube 27x en doce
    iteraciones: una EMA arrancada en el primer valor rechazaría ahí (medido, 1.135 % contra
    0.064 %). Con la ventana, no rechaza nada."""
    rech, escala, _ = corre(TRANSITORIO)
    assert not any(rech)
    assert len(escala) == len(TRANSITORIO)


def test_rechaza_el_pico_y_no_lo_absorbe():
    """Lo que lo distingue del clip: el pico no entra en la escala. Si entrara, un solo lote
    subiría el umbral y el siguiente pasaría."""
    rech, escala, centros = corre([100.0] * 10 + [1e5, 1e5, 100.0], ventana=10)
    assert rech == [False] * 10 + [True, True, False]
    assert centros[-1] == pytest.approx(100.0), "un pico rechazado no puede mover la escala"
    assert max(escala) == 100.0


def test_la_ventana_no_crece_mas_alla_de_su_largo():
    rech, escala, _ = corre([50.0] * 500, ventana=100)
    assert not any(rech) and len(escala) == 100


def test_el_pico_medido_en_el_evento_queda_afuera():
    """Las cuatro iteraciones de `caza-adjunta/seed7` que llevan al estallido. Con la mediana sana
    de 72 la razón 5 pone el umbral en 360, así que corta la rampa COMPLETA desde la it 2696:
    tres iteraciones antes de que `‖g‖` cruce 1e4 en la 2699. A razón 31.62 el umbral valía 2278 y
    la primera, de 1816, pasaba."""
    rech, _, _ = corre([72.07] * 10 + [1816.0, 4127.0, 7294.0, 1.015e4], ventana=10)
    assert rech[10:] == [True, True, True, True]
    # Y la de 1816 pasaba con la razón vieja: es exactamente lo que se ganó al bajarla.
    vieja, _, _ = corre([72.07] * 10 + [1816.0], ratio=31.6228, ventana=10)
    assert vieja[10:] == [False]


def test_nan_e_inf_no_pasan_nunca():
    """Un solo paso con NaN vuelve NaN los 388k pesos, y de ahí no se vuelve."""
    rech, escala, _ = corre([float("nan"), INF, 1.0], ratio=INF, ventana=1)
    assert rech == [True, True, False]
    assert escala == [1.0]


def test_razon_infinita_es_la_corrida_de_antes():
    """El brazo de control. `--guard inf` no rechaza nunca, y como el disyuntor no toca la
    aritmética, control y tratamiento son bit a bit la misma corrida hasta el primer rechazo."""
    rech, _, _ = corre([1.0, 1e3, 1e9, 1e15], ratio=INF)
    assert not any(rech)


def test_la_prueba_de_amplificacion_es_independiente_de_la_escala():
    """`amp` tiene base sana estable —mediana 0.697 en las cinco trazas instrumentadas— así que su
    umbral es absoluto y no necesita ventana. En 14 018 iteraciones sanas dispara una sola vez: la
    it 2694 de `caza-adjunta/seed7`, que es el paso cuya omisión evitó el estallido."""
    rech, _, _ = corre([4592.0] * 3, ratio=INF, amp_max=100.0,
                       amps=[0.697, 217.053, float("nan")])
    # El NaN del final es lo que llega cuando no se instrumentó: no debe disparar nada.
    assert rech == [False, True, False]


def test_la_traza_conserva_las_seis_columnas_viejas():
    """Ocho `scripts/` leen `traza.npy` por posición y `[:, 5]` es `‖g‖` en los 81 registros ya
    guardados. Las columnas nuevas van al final o se rompe todo el análisis histórico."""
    import inspect
    fuente = inspect.getsource(train.main)
    assert "traza = np.zeros((args.iterations, 10)" in fuente
    assert "loss T dT lambda dlambda |g| centro amp rechazado rebobinado" in fuente
    assert "traza[it - 1, 5:] = (gnorm, centro, amp, rechazado, rebobinado)" in fuente
    # `amp` queda en la 7 en los DOS formatos de traza del repo —el de aquí y el de 30 columnas
    # de `measure_traza_adjunta.py`— para que `traza[:, 7]` no dependa del ancho.
    assert "`traza[:, 7]` significa lo mismo en los dos formatos" in fuente


def test_el_bucle_no_corta_nunca():
    """`scripts/veredicto_precision.py` separa corrida terminada de corrida en vuelo CONTANDO
    FILAS, y `scripts/hazard_fase1.py` usa `len(traza)` como tiempo de exposición. Un `break` por
    rechazos seguidos haría que una corrida cortada se contara como censurada viva, y el análisis
    de supervivencia se invertiría sin que nada se ponga rojo. El disyuntor rechaza; no corta."""
    import inspect
    fuente = inspect.getsource(train.main)
    assert "break" not in fuente
    assert "if not rechazado:\n            opt.step()" in fuente
    assert "sched.step()" in fuente, "el calendario del lr no se detiene por un lote rechazado"


def test_el_paso_rechazado_deja_adam_intacto():
    """El mecanismo entero depende de esto, así que se ejecuta y no se inspecciona: un pico que no
    da el paso no mueve el parámetro ni los momentos, y la corrida con el pico intercalado termina
    idéntica a la que no lo tuvo. `zero_grad` + `step` NO sería equivalente: decae las dos EMA y
    aplica un paso igual."""
    def entrena(normas):
        torch.manual_seed(0)
        w = torch.nn.Parameter(torch.ones(4))
        opt = torch.optim.Adam([w], lr=train.LR, betas=train.BETAS)
        escala = []
        for g in normas:
            opt.zero_grad(set_to_none=True)
            w.grad = torch.full((4,), g / 2.0)          # ‖g‖ = g sobre cuatro coordenadas
            r, _ = train.rechaza_lote(float(w.grad.norm()), INF, escala,
                                      train.GUARD_RATIO, INF, INF, 4)
            if not r:
                opt.step()
        return w.detach().clone(), copy.deepcopy(opt.state_dict()["state"][0])

    sanos = [1.0] * 5
    w1, s1 = entrena(sanos)
    w2, s2 = entrena(sanos[:4] + [1e6] + sanos[4:])     # un pico intercalado, que se rechaza
    assert torch.equal(w1, w2)
    for k in ("exp_avg", "exp_avg_sq"):
        assert torch.equal(s1[k], s2[k])
    assert int(s1["step"]) == int(s2["step"]) == 5       # el pico no consumió un paso de Adam

def test_la_forma_cerrada_de_pS_coincide_con_autograd():
    """`‖p_S‖` sale de la pérdida y no de `retain_grad`, porque `VNet2D.forward` devuelve los
    estados APILADOS en un solo tensor: `states[-1]` es una vista NUEVA en cada indexado, así que
    el hook se le pegaría a un temporal y `.grad` queda en `None` — costó una corrida del smoke
    test. De la pérdida sale `p_S = 2(x_S − clean)/MICRO_BATCH`, y acá se verifica contra
    autograd."""
    torch.manual_seed(0)
    m = 4
    xS = torch.randn(m, 3, 5, 5, requires_grad=True)
    clean = torch.randn(m, 3, 5, 5)
    (xS.float() - clean.float()).pow(2).flatten(1).sum(1).mean().backward()
    cerrada = 2 * (xS.detach() - clean).norm() / m
    assert torch.allclose(xS.grad.norm(), cerrada, rtol=1e-6)


def test_el_estado_apilado_no_retiene_gradiente():
    """La trampa, aislada: indexar dos veces el tensor apilado da dos nodos distintos, y el
    `retain_grad` del primero no ve nada del backward del segundo."""
    base = torch.randn(4, requires_grad=True)
    apilado = torch.stack([base, base * 2, base * 3])
    apilado[-1].retain_grad()                       # se le pega a un temporal
    apilado[-1].pow(2).sum().backward()
    with pytest.warns(UserWarning, match="not a leaf"):
        assert apilado[-1].grad is None             # y por eso hay que usar la forma cerrada


def test_el_respaldo_exige_limpias_seguidas():
    """Una sola iteración sana no alcanza para confiar en el estado: dentro de una excursión letal
    hay lotes ACEPTADOS —4 en el evento de `tinit-7p9delta-10k`, 2 en `disyuntor-100k`, hueco
    máximo 1 en los dos—, y un respaldo tomado ahí se lleva la cuenca mala. Medido: `t8delta-100k-s1`
    tomó uno en la it 6003 con 3 rechazos en las 30 previas y UNA limpia detrás."""
    assert train.GUARD_LIMPIAS > 1
    import inspect
    fuente = inspect.getsource(train.main)
    assert "limpias >= args.guard_limpias" in fuente
    # Y se resetea en cuanto hay un rechazo, o no serían seguidas.
    assert "seguidos, limpias = seguidos + 1, 0" in fuente


def test_el_respaldo_no_depende_de_multiplos_exactos():
    """El respaldo se toma sólo en iteración sana. Si además se exigiera `it % N == 0`, una corrida
    que se daña antes del primer múltiplo se queda sin respaldo y el rebobinado no puede disparar
    nunca: medido en la prueba de humo del 2026-10-04, 180 rechazos seguidos con el umbral en 5 y
    cero rebobinados. Se cuenta desde el último respaldo, y el primero se toma en cuanto hay una
    iteración sana."""
    import inspect
    fuente = inspect.getsource(train.main)
    assert "it % args.guard_respaldo" not in fuente
    assert "respaldo is None or it - it_respaldo >= args.guard_respaldo" in fuente
