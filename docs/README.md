# Qué hay en `docs/`

Mapa de la carpeta al 2026-10-08, con la fase 1 cerrada. La regla para orientarse: **lo vivo se
corrige y lo histórico se marca, pero no se reescribe** (skill `afirmar`, paso 3 de «Retirar una
afirmación publicada»; desde el 2026-10-08 las skills son locales).

| entrada | qué es | quién la escribe | vivo o histórico |
|---|---|---|---|
| [`bitacora.md`](bitacora.md) | el registro cronológico de la fase 1: cada corrida, pre-registro, veredicto y retirada, con la tabla `## Descartado`. Arriba, `## Estado` y `## Cómo retomar` | a mano | histórico, salvo esas dos secciones de arriba. Pesa ~290 kB: se consulta con `grep -n` y `sed -n`, no entera |
| [`metricas.md`](metricas.md) | qué significa cada número que se mide y en qué rango es sano. La sección 0 explica sin fórmulas qué es una semilla, un brazo y el hazard | a mano | vivo |
| [`mediciones/`](mediciones/) | 25 archivos: la salida cruda de un script de medición o de veredicto, uno por pregunta. Es la evidencia que la bitácora cita cuando publica un número | los `scripts/measure_*.py`, `veredicto_*.py` y compañía, hoy locales | histórico: no se edita. Si un número resulta mal, se retira en la bitácora |
| [`resultados/`](resultados/README.md) | una ficha por corrida de `runs/` —110, más un índice ordenado por mejor PSNR—: configuración, PSNR por evaluación, `T`, `λ`, estado y rechazos | `scripts/resumen.py`, local, leyendo `runs/`, también local. Última vez: 2026-10-07 | derivado. Es la única descripción versionada de cada corrida de `runs/`; de sus pesos se versionan sólo los tres de `checkpoints/Entrenados/` |
| [`presentacion/`](presentacion/README.md) | la presentación de cierre de la fase 1: guion, brief, scripts de figuras y `salida/` con lo generado. Trae dos documentos que valen fuera de ella: [`pipeline-explicado.txt`](presentacion/pipeline-explicado.txt), cómo se entrena TDV explicado desde cero (~60 kB), y [`conceptos_y_parametros.md`](presentacion/conceptos_y_parametros.md), el glosario, la receta, los valores del disyuntor y `T` y `λ` iniciales por familia de corridas | a mano el guion, el brief y esos dos documentos; `salida/`, los scripts de la carpeta, con `bash docs/presentacion/rehacer.sh` (CPU) | vivo; `salida/` es generada |
| [`avance/`](avance/README.md) | la presentación anterior, del 2026-10-02: `avance.html` es la copia local de una página publicada, armada desde `plantilla.html` con las figuras de `salida/` | los scripts de la carpeta; el orden está en su README | histórico: `presentacion/` la reemplaza. Pero ver «Qué depende de qué» |
| [`papers/`](papers/INDEX.md) | la bibliografía: `INDEX.md` y una nota destilada por paper en `notes/`, que se lee antes que el PDF. Los PDF no se versionan | a mano; skill `papers` | vivo |

`pendientes.md` era la cola de corridas de la fase 1. Se eliminó el 2026-10-08, al cerrarla, y lo
que tenía abierto quedó listado en la entrada de ese día de la bitácora.

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
- **La bitácora, `AGENTS.md`, el `README.md` de la raíz, `train.py`,
  `presentacion/pipeline-explicado.txt`, el README de `avance/`, una nota de `papers/` y dos
  docstrings de `tests/` citan archivos de `mediciones/` por nombre**, además de las dos
  `fig_sigma.py`. Renombrar uno rompe esas citas.
