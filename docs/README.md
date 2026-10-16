# Qué hay en `docs/`

Mapa de la carpeta al 2026-10-16: la fase 1 (BSDS) cerrada, y los cortes 2D de T1 y el entrenador
3D integrados en `main`. La regla para orientarse: **lo vivo se corrige y lo histórico se marca,
pero no se reescribe** (skill `afirmar`, paso 3 de «Retirar una afirmación publicada»; desde el
2026-10-08 las skills son locales).

| entrada | qué es | quién la escribe | vivo o histórico |
|---|---|---|---|
| [`bitacora.md`](bitacora.md) | el registro cronológico del proyecto —fase 1 en BSDS, cortes 2D de T1 y entrenador 3D—: cada corrida, pre-registro, veredicto y retirada, con la tabla `## Descartado`. Arriba, `## Estado` (historial congelado), `## Cómo retomar` e `## Inventario de corridas` | a mano | histórico, salvo `## Cómo retomar`, que se reescribe, y `## Descartado`, que suma filas. Pesa ~550 kB: se consulta con `grep -n` y `sed -n`, no entera |
| [`metricas.md`](metricas.md) | qué significa cada número que se mide y en qué rango es sano. La sección 0 explica sin fórmulas qué es una semilla, un brazo y el hazard | a mano | vivo |
| [`decisiones-3d-vs-2d.md`](decisiones-3d-vs-2d.md) | qué cambia el entrenamiento 3D (`train3d.py`) frente al 2D, ítem por ítem: mecánico, decisión distinta o nuevo; porqué, evidencia y default, con punteros a `897fb8b` | generado por agentes y verificado en parte, 2026-10-12 | foto de `897fb8b`: ante una duda manda el código |
| [`mediciones/`](mediciones/) | 72 archivos (71 `.txt` y un `.json`): la salida cruda de un script de medición o de veredicto, uno por pregunta. Es la evidencia que la bitácora cita cuando publica un número. Cómo leerlos, abajo | los `scripts/measure_*.py`, `veredicto_*.py` y compañía, hoy locales | histórico: no se edita. Si un número resulta mal, se retira en la bitácora |
| [`resultados/`](resultados/README.md) | una ficha por corrida de `runs/` —115—: configuración, PSNR por evaluación, `T`, `λ`, estado y rechazos. El índice, ordenado por mejor PSNR en CBSD68, lista 110: le faltan `rapido-100k-s1`, `t2delta-100k-s1`, `t1p1delta-rapido-100k`, `t1-rapido-30k` y `t1-guard-2delta-5k` (las dos últimas, de T1, no se evalúan en CBSD68) | `scripts/resumen.py`, local, leyendo `runs/`, también local. El índice es del 2026-10-07; las fichas llegan a corridas terminadas el 2026-10-11 | derivado. Es la única descripción versionada de cada corrida de `runs/`; de sus pesos se versionan sólo los tres de `checkpoints/Entrenados/` |
| [`presentacion/`](presentacion/README.md) | la presentación de cierre de la fase 1: guion, brief, scripts de figuras y `salida/` con lo generado. Trae dos documentos que valen fuera de ella: [`pipeline-explicado.txt`](presentacion/pipeline-explicado.txt), cómo se entrena TDV explicado desde cero (~60 kB), y [`conceptos_y_parametros.md`](presentacion/conceptos_y_parametros.md), el glosario, la receta, los valores del disyuntor y `T` y `λ` iniciales por familia de corridas | a mano el guion, el brief y esos dos documentos; `salida/`, los scripts de la carpeta, con `bash docs/presentacion/rehacer.sh` (CPU) | vivo; `salida/` es generada |
| [`avance/`](avance/README.md) | la presentación anterior, del 2026-10-02: `avance.html`, copia local y autocontenida de una página publicada, armada desde `plantilla.html` con las figuras de `salida/` | los scripts de la carpeta; el orden está en su README | histórico: `presentacion/` la reemplaza. Pero ver «Qué depende de qué» |
| [`papers/`](papers/INDEX.md) | la bibliografía: `INDEX.md` y una nota destilada por paper en `notes/`, que se lee antes que el PDF. Los PDF no se versionan | a mano; skill `papers` | vivo |

`pendientes.md` era la cola de corridas de la fase 1. Se eliminó el 2026-10-08, al cerrarla, y lo
que tenía abierto quedó listado en la entrada de ese día de la bitácora.

**Los hashes que citan la bitácora y `decisiones-3d-vs-2d.md`** (`6190bfd`, `897fb8b`, …) no
están en la historia pública: en GitHub, `main` arranca con dos commits comprimidos (`6bb5b88` y
`ff4aa3c`). Están en dos etiquetas locales, `historia-main` y `historia-3D-Training`, que no se
suben (bitácora, entrada «Integración de `3D-Training` en `main`»).

## Cómo están organizadas las mediciones

- **Un archivo por pregunta**, con la pregunta como nombre (`rebobinados-t1.txt`,
  `costo-tdv3d.txt`). No hay subcarpetas, y un archivo no se reemplaza por otro posterior: una
  pregunta nueva es un archivo nuevo.
- **La cabecera** dice, en la primera línea, la pregunta —a veces con la respuesta entre
  paréntesis, «(no)»— y, en la segunda, el script, la máquina y la fecha. Debajo va la salida
  cruda, sin editar.
- **Trece archivos de la fase 1 empiezan directamente con la salida**: `adam-respuesta-al-pico`,
  `adjunta-por-paso`, `cota-absoluta-226k`, `degeneracion-T-KN`, `escala-equivariancia`,
  `eventos-historicos`, `grad-T-vs-lambda`, `hazard-fase1`, `lambdainit`, `lipschitz-tinit`,
  `precision`, `t8delta-100k` y `tinit-2delta-50k`. Su script y su fecha están en la entrada de la
  bitácora que los cita (`grep -n <archivo> docs/bitacora.md`). `lipschitz-tinit.txt` y
  `adam-trayectorias-T.json` no los cita ningún archivo.
- **Los scripts no están en el repositorio**: `scripts/` es local desde el 2026-10-08, y las
  cabeceras nuevas lo marcan con «(local)». La salida queda versionada justamente para que el
  número siga siendo auditable sin el script.
- **Por tema** (orientativo; un archivo puede servir a más de uno):

| tema | archivos |
|---|---|
| estallidos, `T` y `λ` (BSDS) | `adam-respuesta-al-pico`, `adam-trayectorias-T.json`, `adjunta-por-paso`, `adjunta-sigma-max`, `transicion-adjunta`, `eventos-historicos`, `hazard-fase1`, `replay-ulp`, `replay-T-2delta-100k`, `lipschitz-tinit`, `grad-T-vs-lambda`, `gauge-lambda-T`, `gauge-4-corridas`, `degeneracion-T-KN`, `lambda-sensibilidad`, `lambda-replay-adam`, `lambdainit`, `escala-equivariancia` |
| tandas y veredictos (BSDS) | `tinit01-vs-control`, `tinit-10k`, `tinit-2delta-50k`, `t8delta-100k`, `t1p1delta-100k`, `modo-y-tinit-100k-s1`, `precision`, `rapido-vs-referencia-100k`, `eigen-rapido-vs-referencia`, `lambda-rapido-vs-referencia`, `calendario-lr` |
| disyuntor y rebobinado | `disyuntor-calibracion`, `disyuntor-v2`, `disyuntor-t1`, `cotas-disyuntor-dominio`, `cota-absoluta-226k`, `lazo-disyuntor-t1`, `alias-respaldo-adam`, `edad-del-respaldo`, `umbral-guard-escala`, `dispersion-g`, `rebobinados-t1`, `nivel-g-t1-vs-bsds` |
| tarjetas | `dos-gpu`, `velocidad-dos-tarjetas` |
| cortes 2D de T1 (`datasets/ldm100k`) | `t1-rapido-30k`, `escalares-t1`, `contraste-t1-ldm100k`, `variedad-ldm100k`, `campo-receptivo-tdv2d` |
| entrenamiento 3D (`train3d.py`, `datasets/ldm100k3d`) | `costo-tdv3d`, `arranque-g-3d`, `humo-train3d`, `tejido-ldm100k3d`, `seleccion-ldm100k3d`, `cobertura-cubos-cerebro`, `verificacion-cubos-cerebro`, `cruz-siete-cubos`, `mascaras-cerebro-ldm`, `mascaras-val-ldm100k3d`, `revision-mascara-cerebro`, `ventana-3d-12h` |
| QSM, estabilidad PnP y seguimiento | `qsm-prior-4corridas`, `qsm-eval-cosmos`, `qsm-eval-gpu`, `compat-qsm-tdv`, `compat-qsm-tdv-simetrico`, `convergencia-estabilidad`, `costo-estabilidad`, `dynamo-estabilidad`, `dynamo-entradas-qsm`, `traza-compilada-qsm`, `costo-seguimiento`, `seguimiento-cuelgues` |

## La diferencia entre `mediciones/`, `resultados/` y `avance/`

- **`mediciones/` se organiza por pregunta.** Cada archivo es la salida de un script que contestó
  algo puntual —el hazard, el `σ_max` del adjunto, si `λ` importa— y existe para que el número que
  la bitácora publica siga siendo auditable aunque el script desaparezca.
- **`resultados/` se organiza por corrida.** No contesta ninguna pregunta: describe cada corrida de
  `runs/` con el mismo formato, generado en bloque.
- **`avance/` no es evidencia, es comunicación**: una presentación vieja, como `presentacion/` es
  la nueva.

## Qué depende de qué

- **`presentacion/` usa código de `avance/`.** `eigen_lote.sh` y `eigen_extra.sh` —y por ellos
  `eigen_cola.sh` y `rehacer.sh eigen`— llaman a `docs/avance/eigen.py`, y las eigenfunciones del
  oficial, v4 y `referencia-100k` sobre la imagen 1 se copiaron de `avance/salida/`. Para borrar
  `avance/` hay que mover antes `eigen.py` y cambiar esas rutas.
- **Las dos `fig_sigma.py`** (`avance/` y `presentacion/`) leen `mediciones/adjunta-sigma-max.txt`.
- **Los scripts de `presentacion/` leen `runs/`** (local), `checkpoints/Oficiales/`, y cargan
  `evaluate.py`, `ddr/` y `train.py`. En un clon sin `runs/` no se regeneran.
- **`mediciones/` y `resultados/` se leen en cualquier clon pero se regeneran sólo en la máquina
  del autor**: salen de `scripts/`, local desde el 2026-10-08, y de `runs/`, que nunca se versionó.
- **Citan archivos de `mediciones/` por nombre** la bitácora, `AGENTS.md`, el `README.md` de la
  raíz, `metricas.md`, `decisiones-3d-vs-2d.md`, `train.py`, `train3d.py`, módulos de `ddr/` y de
  `datasets/`, `datasets/ldm100k3d/README.md`, varios archivos de `presentacion/` y de `avance/`
  (entre ellos las dos `fig_sigma.py`), una nota de `papers/`, una ficha de `resultados/` y
  docstrings de `tests/`. Renombrar uno rompe esas citas: antes, `grep -rn <archivo>`.
