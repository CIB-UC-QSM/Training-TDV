# El porqué del código

El porqué de cada constante y de cada decisión de `train3d.py`, `ddr/qsm3d.py` y la parte de
`train.py` que hereda el 3D, por secciones. El código las cita por nombre («MESETA», «MEMORIA»):
una sección se busca con `grep -n '^### <NOMBRE>' docs/porque-del-codigo.md`. El estado y las
decisiones de la corrida larga están en la [bitácora](bitacora.md).

## train3d.py

### BUCLE

El de `train.py`, repartido en funciones con nombre. `main` arma la corrida en pasos
(`revisa_banderas`, `elige_micro_lote`, `lee_estado`, `prepara_backend`, `arma_modelo`,
`arma_entrenamiento`, `arma_validacion`, `arma_puntos_de_estabilidad`, `prepara_salida`,
`imprime_arranque`, `abre_seguimiento`, `restaura_generadores`); lo que se arma una vez va en
`piezas` y las cuentas que cambian y viajan en el estado, en `corrida` (`corrida_nueva`). Después
llama a `entrena`, que en cada iteración:

- fija el lr con `lr_en` (CALENDARIO DEL LR);
- hace forward y backward del lote, micro-lote por micro-lote (`gradiente_del_lote`);
- deja que `rechaza_lote` decida si hay paso de Adam, re-proyecta (`project_`) y, con
  `aplica_disyuntor`, rebobina o refresca el respaldo;
- escribe una fila de traza; cada `--log-every`, la línea de log (`registra_progreso`); cada
  `--eval-every` y en la última, valida (`valida`: denoising y, con la cadencia de
  `--eval-qsm-cada`, estabilidad y QSM); cada `--guardar-cada`, en la última y en un corte,
  escribe el estado (`guarda_estado`).

Al final, `cierra` escribe `estado.json`. Respecto de `train.py`, cuatro diferencias:

1. Parches de `PATCH_SIZE`³ de `ddr.data3d.PatchDataset3D`, con al menos 35 % de tejido
   (`MIN_TEJIDO`; o, con `--mascara-cerebro`, cubos enteros dentro del cerebro) y una de las 48
   simetrías del cubo, partidos en ceil(`--lote` / `--micro-lote`) trozos casi iguales. El
   micro-lote lo elige la memoria (MEMORIA y MICRO-LOTES).
2. El lr no lo lleva un scheduler: `lr_en(it, hitos)` lo fija al principio de cada iteración, con
   los hitos de `--meseta` sumados (MESETA).
3. La cota absoluta del disyuntor escala con el parche (`cota_absoluta`).
4. `--guardar-cada` escribe el estado completo, y `--reanudar` sigue desde ahí bit a bit con el
   mismo micro-lote (REANUDACIÓN).

Y tres agregados que no tocan la aritmética (test): el seguimiento en W&B (SEGUIMIENTO), la
evaluación QSM (EVALUACIÓN QSM) y el número de estabilidad (ESTABILIDAD). Como en `train.py`, el
bucle no corta ni por NaN, ni por rechazos, ni por un fallo de W&B, del ADMM o de la estabilidad.
El único `break` de `entrena` es el de `--meseta`, apagada por defecto, y el corte queda escrito
en `estado.json` (`test_el_bucle_corta_solo_por_meseta`). Una corrida terminó si su `estado.json`
dice `"terminada": true`. `--sobreajuste` cambia los datos para la prueba de sobreajuste
(SOBREAJUSTE).

Una falla de la validación de denoising sí corta la corrida: `valida_denoising` no tiene `try`, a
diferencia de la estabilidad y el ADMM (bitácora, «Abierto»).

### RECETA 3D

Lo que no cambia se importa de `train.py`, como los mismos objetos: `LR`, `BETAS`, `SIGMA`,
`STEPS`, la arquitectura (tres macro-bloques de tres escalas), `T_INIT` = 8·δ_Adam,
`LAMBDA_INIT`, el disyuntor y los modos. También lo que acá no se usa (`SCALES`,
`MACRO_BLOCKS`, `DELTA_ADAM`), porque los tests verifican que la receta sea esa. Lo que cambia,
con su procedencia:

- **`PATCH_SIZE = 48`:** 48³ × 4 es el menor costo por vóxel de la grilla medida en `rapido` en
  la 5070 Ti (2026-10-08): 2.26 µs, contra 2.77 a 32³ × 4 y 2.36 a 64³ × 1; 64³ × 2 no entra en
  16 GB.
- **`LOTE = 32`:** el «mini batch size of 32» del paper (sec. 5.1), decisión del autor del
  2026-10-09.
- **`MICRO_LOTE = "auto"`:** el que parte `LOTE` en menos trozos y entra en `--vram-gb`
  (MICRO-LOTES).
- **`ITERACIONES_TANDA = 40 000`:** el tronco de un piloto de tronco y ramas que no se lanzó.
- **`GUARDAR_CADA = 10 000`:** los puntos de rama de ese piloto: 10 000, 20 000 y 30 000.

Y lo que hay detrás de cada uno:

- **`LOTE` es el mismo número de parches, no el mismo contenido:** 32 × 48³ = 3.54 M vóxeles por
  paso, 4.3× los 830 k valores de 32 × 93² × 3 en BSDS color. El paper no tiene versión 3D, así
  que ningún lote 3D es «el del paper»; igualar los valores de BSDS pediría ~8 parches (885 k).
  Cuál entrena mejor por hora no está medido: las columnas 5 y 10 de la traza dicen si 32 queda
  por encima de la escala de ruido del gradiente, donde cada parche de más compra menos.
- **El micro-lote es hardware, no receta:** con acumulación el gradiente es el mismo para
  cualquier partición, porque ninguna capa acopla muestras. `auto` da 16 en 96 GiB, 11 en 48 y 3
  en la 5070 Ti. En `referencia`, 48³ × 2 no entró en 16 GB: los tests y las corridas exactas
  usan `--micro-lote 1`.
- **El costo se lee en s/it, no en horas.** En la 5070 Ti, con micro-lote 2, la sonda del
  2026-10-13 dio 0.280–0.291 s por parche (8.96–9.30 s/it), y `ventana-3d-12h` corrió a
  9.17–9.24 s/it sin contar el ADMM ni la estabilidad. La grilla del 2026-10-08, sin el resto del
  bucle, daba 0.250 s por parche a micro-lote 4 y 0.265 a 2. En otra tarjeta lo mide la sonda, y
  N ≈ horas · 3 600 / (s/it), menos las validaciones.
- **Tamaño de un estado** (derivado): pesos, m y v de Adam tres veces (la corrida, el respaldo y
  el mejor), los pesos de `best_qsm.pth` (4.7 MB) y la traza hasta `it` (112 B por iteración:
  4.5 MB en la 40 000). Da ~51 MB en la 40 000; la bitácora cuenta ~47 MB por estado en la
  corrida larga.
- **`ITERACIONES_TANDA` y `GUARDAR_CADA`** son defaults de ese piloto: la corrida larga pasa los
  suyos por bandera (bitácora, «La corrida larga»).

`--mode` vale `rapido` por defecto, no `referencia` como en `train.py`: en 3D `referencia` cuesta
~21× por vóxel (47.9 contra 2.26 µs) y a 48³ × 2 no entró en 16 GB. Una corrida en `rapido` no
repite bit a bit, así que de ella se lee la tasa sobre semillas; una contra otra, sólo rama contra
tronco (mismos parches y ruido), sobre la dispersión del tronco (AGENTS.md, «Entrenamiento 3D»).

### MEMORIA

`vram_predicha` predice el pico de `torch.cuda.max_memory_allocated` del entrenamiento, sin la
evaluación, como `FIJA + n · P³ · POR_VOXEL[modo, checkpointing]`, con n el micro-lote. Se ajustó
a la grilla de la 5070 Ti (2026-10-08: forward + backward, `R.grad` compilado):

- **`FIJA = 0.61`:** la recta de 48³ en `rapido` por n = 1 y n = 4 (4.14 y 14.72 GB), en n = 0.
  Por mínimos cuadrados, con n = 2 (7.69), sería 0.625 y 3.525: no cambia ninguna elección.
- **`rapido`, 3.19e-5 GiB por vóxel:** la pendiente de esa recta (3.527 por parche de 48³).
  Predice 8.97 a 64³ × 1 contra 8.78 medidos, y a 32³ × 4 subestima 8 % (4.79 contra 5.21): por
  eso el `MARGEN`.
- **`intermedio` y `referencia`, 7.0e-5:** por encima de los cuatro medidos (`referencia` 32³ × 1
  y × 2, 2.48 y 4.77; 48³ × 1, 7.60; `intermedio` 48³ × 1, 7.55), que sobreestima 9 a 17 %: la
  pendiente desde `FIJA` da 5.7e-5 a 6.3e-5.
- **Con `--checkpointing`:** 3.4e-6 en `rapido` (64³ × 4: 4.11; 96³ × 1: 3.33, que sobreestima
  9 %) y 5.6e-6 en `referencia` (64³ × 2: 3.56); `intermedio` no se midió y usa el de
  `referencia`. La grilla checkpointeaba los S pasos, pero el forward de la VNet checkpointea un
  paso sólo si su x lleva gradiente: x_0 lo lleva con `--guard-amp`, y sin la bandera el paso 1
  guarda sus activaciones hasta el backward. `vram_predicha` le suma entonces
  `POR_VOXEL[modo, False] / S`, un paso de los S sin checkpointing (medido en CPU: 1.00 a 1.04
  pasos), 6.6e-6 en total en `rapido` y 1.26e-5 en `referencia`. Por eso `--checkpointing` baja
  la memoria ~5×, y ~9× sólo con `--guard-amp`.
- **`MARGEN = 0.10`** del presupuesto, supuesto: la evaluación (recortes de 128³ sin gradiente), la
  fragmentación del allocator y los workspace de cuDNN, que en otra tarjeta pueden ser otros. Con
  micro-lote 3 y validaciones se midieron 11.47 GiB contra 11.19 predichos (bitácora, «Costo y
  memoria»).

**El presupuesto** (`elige_micro_lote`) es min(`--vram-gb`, memoria de la tarjeta), en GiB como
los reporta torch, algo menos que la etiqueta: la 5070 Ti de «16 GB» da 15.45 (16 303 MiB en
`nvidia-smi`). Si es menor que la tarjeta, `set_per_process_memory_fraction` no deja a PyTorch
pasar de ahí: la tarjeta puede estar compartida, y mejor un OOM propio que pisar a otro proceso.
En CPU el presupuesto es `--vram-gb`, y `auto` elige lo mismo que en una tarjeta de ese tamaño.

**En la tarjeta de destino**, `VRAM_GB = 96` (RTX PRO 6000 Blackwell Max-Q, entera para el
autor): en ~95 GiB entrarían 24 y `auto` da 16, con 57.1 GiB predichos; 32 pediría 113.5. Si 16
es más rápido por parche que 8 no está medido: lo dice la sonda, y si no lo es, `--micro-lote 8`
(28.8 GiB) rinde igual. Con 48 GiB a 48³ en `rapido` entran 12 (42.9 de 43.2 útiles) y `auto` da
11: 32 = 11 + 11 + 10, las mismas tres pasadas que 12, con 39.4 GiB predichos. Lo que cuenta es
saturar el cómputo, no llenar la memoria: en la 5070 Ti, de 48³ × 2 a × 4 el costo por vóxel ya
bajaba sólo 6 % (2.39 a 2.26 µs). Con `--checkpointing` `auto` da 32 (23.9 GiB predichos), un solo
micro-lote: la columna 10 de la traza es entonces la 5 y no estima nada; para conservarla,
`--micro-lote 16`.

**`--sonda-vram` lo mide** (`sonda_vram`). Todo lo anterior se ajustó en 16 GB y se extrapola a
96: es derivado, no medido en la tarjeta de destino. La sonda corre `SONDA_ITERACIONES = 4` de
forward + backward con cada tamaño de `tamanos_utiles(--lote)` (para 32: 1 a 8, 11, 16 y 32), de
menor a mayor y compilado como para entrenar, hasta el primero cuyo pico pasa de los útiles o da
OOM; imprime predicho contra medido y lo deja en `sonda-vram.json`, sin escribir nada más ni
entrenar. Si lo medido admite más que `auto`, se pasa `--micro-lote` explícito. Imprime además
los segundos por parche de las tres últimas iteraciones, entre dos `torch.cuda.synchronize`: la
primera (`SONDA_DESCARTADAS = 1`) compila `R.grad` para esa forma, porque va con `dynamic=False`,
y deja a cuDNN elegir algoritmos. Es lo que dice, en la tarjeta de destino, si llenar más memoria
acelera. Tres iteraciones ordenan micro-lotes; el costo de la corrida se mide en la corrida, con
la tarjeta sola (regla 3 de AGENTS.md). La sonda sube el `recompile_limit` de dynamo a
2·(tamaños) + 2: cada micro-lote son dos entradas de `R.grad` (el primer paso con x sin gradiente
y los demás con; medido en CPU el 2026-10-09), y pasado el límite seguiría en eager y mediría otra
memoria (61 % más a 64³ × 1).

La sonda no corre la evaluación. En una corrida de humo (48³ × 4, `--checkpointing`, sin
`--guard-amp`) el pico total, evaluación incluida, fue 3.59 GiB, y el modelo predice 3.53 para el
entrenamiento solo. Al terminar, el log y `estado.json` dan el pico de la corrida, evaluaciones
incluidas (`vram_pico_gb`), junto al predicho (`vram_predicha_gb`).

**La evaluación QSM no entra en el modelo ni en la sonda:** es sin gradiente sobre los 160³ de
COSMOS y pide 5.19 GiB de pico en `rapido` (6.92 en fp32; medido en la 5070 Ti el 2026-10-12), por
debajo del entrenamiento sin `--checkpointing` desde 48³ × 2 (7.69 medidos) y por encima de él
con `--checkpointing` a 48³ × 4 (3.53 predichos): ahí el pico de `estado.json` puede ser el suyo.
Si no entra, su OOM lo ataja `evalua_qsm` y el entrenamiento sigue (EVALUACIÓN QSM). **La
estabilidad tampoco entra:** guarda el grafo de `R.grad` sobre un cubo de 96³ y pide ~9 GB en
float32 (derivado de la RSS en CPU, sin medir aparte en GPU), más que la evaluación QSM; su OOM lo
ataja `mide_estabilidad` (ESTABILIDAD).

### CALENDARIO DEL LR

En `train.py` el lr vive en `opt.param_groups` y lo mueve `MultiStepLR` sólo en los hitos; el
respaldo lo copia con el resto del estado de Adam, y por eso `rebobina()` lo conserva a mano. Acá
el lr es una función pura de la iteración, `lr_en(it, hitos)`: `LR` a la mitad por cada hito
≤ it − 1, como `MultiStepLR`, que avanza después del paso (con un hito en 25 000, la 25 000 usa
4e-4 y la 25 001, 2e-4). `entrena` lo escribe en el optimizador al principio de cada iteración: ni
el respaldo ni un estado reanudado pueden traer un lr viejo, y una rama puede reanudar con otro
calendario. `test_lr_en_reproduce_multisteplr` lo compara con `MultiStepLR` iteración por
iteración. La columna 11 de la traza es el lr leído del optimizador, el que usó `opt.step()`.

`--hitos` acepta cualquier lista: sin la bandera son los múltiplos de `LR_HALVE_EVERY` (25 000)
hasta `--iterations`, como en el paper; sola, ninguno, que es lr constante. Con `--meseta`,
`lr_en` recibe además los hitos que puso la meseta, que viajan en el estado (MESETA).

### MICRO-LOTES

El micro-lote es cualquier entero hasta `--lote`, no un divisor: `particion` parte el lote en
ceil(lote / m) trozos que difieren a lo sumo en uno, los mayores primero (32 con 12 da
11 + 11 + 10). Lo que cuesta tiempo son las pasadas por iteración, no la memoria ocupada: un trozo
más grande que no quita una pasada sólo pide más memoria. Por eso `micro_lote_auto` toma el mayor
de `tamanos_utiles` que entra, que deja menos pasadas con el menor pico (11 y no 12 en 48 GiB), y
el micro-lote que guardan `config.json` y el estado es el mayor trozo: `--micro-lote 12` queda en
11. En la tarjeta de destino son dos pasadas de 16; en 48 GiB, tres en vez de las cuatro que daría
un divisor (8); en la 5070 Ti, once (diez de 3 y una de 2) en vez de dieciséis.

**La aritmética** (`gradiente_del_lote`): cada trozo entra a la media del lote con su peso,
`loss / veces` con `veces = lote / n`, y la columna 10 escala por el mismo `veces`. Con trozos
iguales ese cociente es la cantidad de trozos, entero y exacto, y la corrida es bit a bit la de un
divisor: medido el 2026-10-16 en `referencia` y CPU, con seis configuraciones de 6 iteraciones
(micro-lotes de 2 y de 1, con workers, ruido ciego, `--guard-amp` y `auto`), `traza.npy`,
`history.json` y estado idénticos a los de la versión que exigía un divisor. Con trozos desiguales
el gradiente es el del lote en aritmética exacta y cambia el orden de suma
(`test_un_lote_en_trozos_desiguales_es_el_mismo_lote`). El DataLoader entrega el lote entero y el
bucle lo corta con `torch.split`: los mismos parches en el mismo orden. El ruido se sigue
sorteando trozo por trozo, así que con `--sigma` fijo y P³ múltiplo de 16 es el mismo para
cualquier partición (REANUDACIÓN).

Cada forma de trozo compila `R.grad` aparte (`dynamic=False`, dos entradas por forma), y con
trozos desiguales hay dos formas: `arma_modelo` sube el `recompile_limit` de dynamo a 16 (el
default es 8), para que ninguna forma de la validación o del ADMM caiga en eager.

### SOBREAJUSTE

`--sobreajuste` es la prueba de que la tubería puede aprender: cada iteración ve el mismo lote,
los parches 0 a `--lote` − 1 sorteados de los primeros `--lote` volúmenes de la lista (los únicos
que carga), armado una vez en `arma_entrenamiento`, con el mismo ruido, porque `entrena` vuelve el
generador del ruido a su semilla en cada iteración. Si la tubería aprende, la pérdida de ese lote
baja mucho más que la de validación: la red lo memoriza. La validación agrega `psnr_lote`, el PSNR
de ese lote leído de la pérdida de la iteración (antes de su paso, dividida por C·P³), al lado del
PSNR de validación.

Va sin las dos cotas del disyuntor (`revisa_banderas` pone `--guard` y `--guard-abs` en infinito);
siguen el rechazo de un gradiente no finito, que no pasa nunca, y el rebobinado tras
`GUARD_SEGUIDOS` rechazos. Con el mismo lote, un rechazo deja la iteración siguiente idéntica (sin
paso de Adam no cambia nada) y se repite para siempre, y el rebobinado vuelve al mismo estallido:
en `runs/sobreajuste-3d` (2026-10-16, semilla 1, lote 4 de 48³) el estallido de la 8 (‖g‖ 360 706
contra una mediana de 2 314) volvió en la 37, la 66 y la 95, con los mismos ‖g‖ y pérdida en todos
los decimales impresos. Con datos frescos el disyuntor sí sale: después de rebobinar, los parches
siguen adelante. El resultado de la prueba: bitácora, «Lo que mostraron las sondas».

### MESETA

`--meseta N` mira el checkpoint en el ADMM y, si su NRMSE deja de bajar, parte el lr o corta. La
regla es `paso_meseta`; la aplica `aplica_meseta`, llamada desde `valida_qsm`. Cada evaluación QSM
sin fallo compara su NRMSE mínimo con el mejor visto: si lo mejora, la cuenta vuelve a cero; si
no, suma uno, y a las N seguidas el lr se divide por 2 desde la iteración siguiente (esa iteración
pasa a ser un hito de `lr_en`) y la cuenta vuelve a cero sin olvidar el mejor. Agotadas
`--meseta-mitades` (`MESETA_MITADES = 5`, el tope que pidió el autor), la meseta siguiente corta:
`corrida.corte` queda en esa iteración, se guarda el estado, `estado.json` dice
`"terminada": true` con `corte` (motivo, iteración y mitades) e `iteraciones_hechas`, y el bucle
sale por su único `break` (`test_la_meseta_parte_el_lr_y_corta`). Sin COSMOS, `--meseta` es un
error al arrancar: no habría NRMSE con qué decidir.

- **Mira `nrmse_min_pct`**, el mínimo de la trayectoria del ADMM, de toda evaluación sin fallo.
  Con la parada de dos subidas (EVALUACIÓN QSM) coincide con el final salvo en un ADMM no finito.
  Se eligió el mínimo cuando el ADMM corría 100 iteraciones sin parada: mientras no convergía, el
  final saltaba decenas de puntos entre dos evaluaciones seguidas y el mínimo se movía menos de
  uno (144.02 → 102.53 % contra 34.23 → 34.30 %, y 37.79 → 69.71 % contra 32.15 → 32.98 %, en
  `ventana-3d-12h`). El mínimo elige su iteración con el χ de COSMOS: sirve para seguir el
  entrenamiento, no da un mapa sin referencia. `best_qsm.pth` se elige con la misma cantidad. Una
  evaluación con el mínimo no finito no cuenta (`paso_meseta` devuelve `"nula"`: ni mejora ni
  suma a la paciencia, y el log lo dice); una que levanta excepción no llega a la regla. Si fallan
  todas, la meseta no actúa y el lr queda constante hasta el techo: lo avisan «qsm FALLÓ» en el
  log y `qsm/fallo` = 1 en W&B.
- **Sin umbral de mejora:** cualquier baja reinicia la cuenta, también una de centésimas. N = 1 es
  el «inmediatamente» del pedido. Exigir una mejora de 1σ sube las mitades falsas en vez de
  bajarlas (Monte Carlo del 2026-10-19: 2 500 × 1 pasa del 8 al 27 %); sólo sirve para ver antes
  una meseta de verdad.
- **La paciencia que importa es en iteraciones**, N por la cadencia de la evaluación QSM, no en
  evaluaciones: con la misma paciencia la cadencia casi no cambia nada. En el Monte Carlo, con la
  tendencia subiendo 1σ cada 1 000 iteraciones, en 5 000 iteraciones parten el lr por error
  100 × 10 (1 000 iteraciones) el 80 % de las veces, 2 500 × 1 el 8 %, 100 × 25 el 1 %, y
  500 × 10, 2 500 × 2 o 100 × 50 (5 000) el 0 %. En 2D, entre la 10 000 y la 24 000 a lr
  constante, «sin récord de PSNR» con 2 000 iteraciones de paciencia habría disparado en 4 de 8
  corridas con el PSNR todavía subiendo 0.024–0.091 dB; con 4 000, en 1; con 6 000, en ninguna.
  La configuración de la corrida larga: bitácora, «La meseta».
- **`--meseta-desde`** (pedido del autor): las evaluaciones de antes de esa iteración no cuentan
  ni dejan mejor, y la primera desde ahí fija el de las dos métricas; `best_qsm.pth` sí mira
  todas. Arrastrar el mejor de antes la dejaría estancada para siempre: en 3 de las 4 corridas 3D
  con 3 o más evaluaciones el mejor NRMSE mínimo es el de la primera, en la 250 (32.89, 33.22 y
  32.89 %, con el mínimo en la iteración 23–27 del ADMM), y la rama de 2 500 no lo venció en
  2 250 iteraciones.
- **`--meseta-psnr`** (pedido del autor): la mitad y el corte piden además que el PSNR que elige
  `best.pth` (en el cerebro con `--mascara-cerebro`) no haya hecho récord en las mismas N
  evaluaciones; cada métrica lleva su mejor y su cuenta, y las dos vuelven a cero al partir. Es la
  regla «Y». En las series 3D (cada 250, hasta la 2 500), la de sólo NRMSE con N = 2 habría
  partido el lr en la 750, 1 250, 1 750 y 2 250, con el PSNR en el cerebro subiendo de 31.85 a
  33.38 dB y el NRMSE recuperando 2.03 puntos a lr constante; la Y no disparó nunca. En el Monte
  Carlo, con ruidos independientes, baja las mitades falsas de 100 × 10 del 80 al 21 %; si los dos
  ruidos se correlacionan, protege menos (no medido). El PSNR del recorte entero no sirve: en la
  rama de 2 500 no hizo récord en la 750 ni en la 2 250 mientras el del cerebro subía. Un PSNR no
  finito anula la evaluación, como un NRMSE no finito.
- **En la última iteración no decide** (`valida_qsm` pide `it < --iterations`): ya no queda
  corrida que cambiar. Por eso el `mejor` de la meseta en `estado.json` puede quedar detrás del de
  `best_qsm`.
- **Su estado** (los hitos, la cuenta, el mejor y su iteración; con `--meseta-psnr`, también los
  del PSNR) viaja en `estado-<it>.pth`: reanudar sigue la misma regla, y una rama puede cambiar
  `--meseta`, `--meseta-mitades`, `--meseta-desde` y `--meseta-psnr` como cambia `--hitos`
  (`CALENDARIO`), heredando las mitades del tronco. W&B recibe `meseta/mitades`,
  `meseta/sin_mejora` y los mejores en cada evaluación que pasa por la regla.
- **Cortar va contra la regla de `train.py`**, que no corta nunca porque los análisis de
  supervivencia cuentan filas de la traza. Por eso viene apagada y el corte queda escrito: una
  corrida cortada es censurada por diseño en `corte.iteracion`, ni viva ni muerta.

### REANUDACIÓN

`estado-<it>.pth` (`guarda_estado`) guarda todo lo que el bucle lee: pesos, `opt.state_dict()`, la
ventana `escala`, el respaldo completo con `it_respaldo`, los contadores (`rechazos`, `seguidos`,
`max_seguidos`, `rebobinados`, `limpias`, `descartadas`, `it_rebobinado`), `history`, `best_psnr`,
`max_grad`, las horas, el generador del ruido, los de torch (CPU y CUDA), la iteración, la config
de la VNet y las banderas. Además, lo que el directorio tenía hasta `it`: la traza, `mejor` (el
payload de `best.pth`), `qsm` (las filas de `qsm.jsonl`), `mejor_qsm` (el payload de
`best_qsm.pth`, o None) y `meseta`. Así el estado se basta solo: no depende de que `traza.npy`,
`best.pth`, `best_qsm.pth` o `qsm.jsonl` de su directorio sigan enteros ni de que sean de su
linaje, y una rama puede arrancar mientras el tronco los reescribe. Lo leen `lee_estado` (las
banderas), `arma_modelo` (los pesos), `main` (Adam y el ruido), `corrida_desde_estado` (las
cuentas) y `restaura_generadores`. `corte` no viaja: al reanudar vuelve a None.

- **Un estado anterior a una bandera de receta** la tiene en su default: `lee_estado` compara con
  `parser.get_default` la que falta. Los de `runs/humo3d` no tienen `mascara_cerebro` y se
  reanudan sin ella; con la bandera prendida, se rechaza
  (`test_reanudar_un_estado_sin_banderas_nuevas`). Uno sin `qsm` arranca sin evaluaciones QSM
  previas; uno sin `mejor_qsm`, sin `best_qsm.pth`; uno sin `meseta`, con la meseta nueva.
- **Los parches no necesitan estado:** el parche `i` sale de `(seed, offset + i)`, así que se
  reanuda con `offset = it · lote`. El generador global de torch no lo usa nada del bucle; se
  restaura igual, después de `iter(loader)`, que saca de él una semilla para los workers.
- **Qué se puede cambiar:** seguir en el mismo `--out` sólo admite cambiar `LIBRES` (cadencias,
  workers, seguimiento, evaluación QSM, estabilidad, `micro_lote`, `vram_gb` y `sonda_vram`); una
  rama, además, `CALENDARIO` (`--hitos`, `--meseta`, `--meseta-mitades`, `--meseta-desde` y
  `--meseta-psnr`). Lo demás cambiaría la aritmética, y `lee_estado` lo rechaza en vez de
  ignorarlo.
- **Seguir en otra tarjeta** cambia `--micro-lote` (o `--vram-gb`, que lo elige con `auto`), y está
  permitido: el lote, los parches y el gradiente en aritmética exacta son los mismos, pero el orden
  de suma no, así que desde el corte la corrida no es bit a bit la que hubiera seguido, y el log lo
  dice con un AVISO que nombra lo que cambia. `config.json` y el estado guardan el micro-lote
  efectivo, no `auto`. El ruido sale de `torch.randn` con el generador en CPU, micro-lote por
  micro-lote: con `--sigma` fijo y P³ múltiplo de 16 da los mismos valores partido de cualquier
  manera, porque se generan de a 16 (verificado el 2026-10-09 con torch 2.13: 4 parches de 48³ de
  una vez, 2 + 2 y 1 + 1 + 1 + 1, idénticos bit a bit; a 15³ no). Con `--sigma-max` no: cada
  micro-lote sortea sus sigmas antes que su ruido, y otra partición le da a cada parche otro
  ruido. Con `--guard-amp`, la amplificación adjunta suma normas micro-lote por micro-lote: su
  valor depende de la partición, y con él lo que el disyuntor rechaza.
- **Verificado en `tests/test_train3d.py`:** en `referencia` y CPU, 12 iteraciones seguidas contra
  6 + 6 y 9 + 3 reanudadas dan `traza.npy`, `history.json`, log y estado final idénticos bit a bit,
  con ruido ciego, aumentos y rebobinados a los dos lados del corte
  (`test_reanudar_da_la_misma_corrida_bit_a_bit`). `rebobina()` restaura una copia del respaldo,
  así que el respaldo guardado en el estado es el del momento en que se tomó
  (`test_el_respaldo_del_estado_es_el_del_momento_en_que_se_tomo`). Seguir desde la 6 con
  `--micro-lote 2` un estado de micro-lote 1 hereda la traza bit a bit y queda a 7.8e-7 relativo,
  como mucho, de la corrida seguida en las 6 iteraciones siguientes, sin ser bit a bit; con otro
  `--lote` se rechaza (`test_reanudar_con_otro_micro_lote_avisa_y_sigue`).

### DIRECTORIO DE SALIDA

Al arrancar, `prepara_salida` deja `--out` como lo describe el estado (vacío, sin `--reanudar`):

- `estado.json` se borra: mientras falta, la corrida está en vuelo o la mataron, como en
  `train.py`, y los análisis de supervivencia cuentan con eso.
- `config.json` se escribe, y `history.json`, `traza.npy` y `qsm.jsonl` se reescriben hasta `it0`
  (`qsm.jsonl` no queda si el estado no trae evaluaciones QSM).
- Un `last.pth`, `best.pth` o `best_qsm.pth` que no sea el del estado (otra iteración u otro PSNR)
  se aparta como `<nombre>-descartada-<it>.pth`, sin borrarlo (`aparta`), y `best.pth` y
  `best_qsm.pth` se reponen desde el estado. Uno que no se puede leer no traba el arranque: se
  aparta como `<nombre>-descartada-ilegible.pth`. Si el nombre ya está tomado, por otro linaje
  descartado que llegó a la misma evaluación, se le agrega `-2`, `-3`…: nada apartado se pisa.

Así, seguir en el mismo `--out` desde un estado anterior al último no deja un `estado.json`
«terminada» de la corrida corta, ni un `best.pth` de iteraciones que no están en la traza ni en
`history.json`, y `mejor_psnr` de `estado.json` es siempre el criterio de `best.pth`: su PSNR o,
con `--mascara-cerebro`, su PSNR en el cerebro (y `best_qsm`, el de `best_qsm.pth`). Una rama
hereda `history`, `best_psnr`, `best.pth` y `best_qsm.pth` del tronco. Los `estado-<it>.pth` de un
linaje descartado se quedan: cada uno se basta solo, y se pisan cuando la corrida nueva llega a su
iteración.

### TRAZA

`float64`, una fila por iteración, `COLUMNAS = 14`. Las diez primeras son las de `train.py`, en su
orden, porque los `scripts/` las leen por posición; una columna nueva va al final. Sus nombres en
W&B, en `seguimiento.COLUMNAS_TRAZA`.

| Col. | Qué |
|---|---|
| 0 | pérdida |
| 1 | T |
| 2 | ∂L/∂T |
| 3 | λ |
| 4 | ∂L/∂λ |
| 5 | ‖g‖ sin recortar |
| 6 | mediana de la ventana del disyuntor |
| 7 | amplificación adjunta (NaN sin `--guard-amp`) |
| 8 | rechazado (0/1) |
| 9 | rebobinado (0/1) |
| 10 | ‖g‖ del primer micro-lote, como si fuera el lote |
| 11 | lr que usó el paso, leído del optimizador |
| 12 | iteración efectiva |
| 13 | σ medio del lote |

Las cinco primeras se escriben antes del paso de Adam: son los valores que produjeron esa
pérdida.

- **Columna 10:** la norma del gradiente tras el primer backward, por `veces` (`lote / n`). Con la
  5 estima offline la escala de ruido del gradiente, B_simple = tr(Σ)/|G|² (McCandlish et al.,
  2018), con los lotes `--micro-lote` y `--lote`: el efectivo, que imprime el log al arrancar; si
  cambia al reanudar, la columna cambia de tamaño desde el AVISO. Sólo lee `.grad` entre dos
  backward, que acumulan sobre el mismo tensor: no cambia ningún bit
  (`test_leer_el_gradiente_entre_backwards_no_cambia_un_bit`). Con `--micro-lote` = `--lote` es
  la 5 y no estima nada, y en la iteración 1 es bit a bit la ‖g‖ de una corrida con lote de
  `--micro-lote` (mismo parche, ruido y pesos; los factores 2 son exactos en binario).
- **Columna 12:** `it` menos lo descartado por los rebobinados, min(it − it_respaldo,
  it − rebobinado anterior) por evento (`aplica_disyuntor`). Es la abscisa del PSNR a iteración
  pareada.
- **Columna 13:** `--sigma`, o la media de los sorteados con `--sigma-max`.

### DISYUNTOR EN 3D

`GUARD_RATIO`, `GUARD_VENTANA`, `GUARD_SEGUIDOS`, `GUARD_RESPALDO` y `GUARD_LIMPIAS` se importan de
`train.py`, los de BSDS: el relativo es invariante de escala. La cota absoluta no: la pérdida es
una suma por parche, y en 3D el arranque sano la cruza (medido con una semilla en 80 iteraciones:
máximo 8 367 a 48³ y 15 985 a 64³). `cota_absoluta(P)` la lleva por el cociente de valores
sumados, `GUARD_ABS · P³ / VALORES_BSDS` con `VALORES_BSDS` = 93² · 3 = 25 947: 4.26e4 a 48³
(42 622) y 1.01e5 a 64³, 5.1× y 6.3× sobre esos máximos. Se aplica cuando `--guard-abs` no se pasa.
Es derivada, no calibrada: los máximos son de una semilla y 80 iteraciones, y que el hueco
[6 113, 1.97e5] de BSDS escale igual es un supuesto. `GUARD_SEGUIDOS = 25` viene de la racha sana
más larga en BSDS (18); en 3D la tiene que medir la corrida larga. Lo observado en las sondas 3D
(razón 5 contra 10, el arranque, la cota) y el `--guard 10` de la corrida larga: bitácora, «El
disyuntor en 3D».

### RUIDO CIEGO Y AUMENTOS

Apagados por defecto: con los defaults la corrida es la receta del paper en 3D. `--sigma-max` > 0
sortea un σ por parche en U(`--sigma-min`, `--sigma-max`) con `add_noise_ciego`, del mismo
generador del ruido (`SIGMA_MIN = 0`: U(0, 55) es la convención de DnCNN-B). `--aug-signo` y
`--aug-ganancia` los aplica `PatchDataset3D` después de elegir el parche y su orientación, así que
un brazo y su control ven los mismos parches. La evaluación es siempre a `--sigma` fijo.

### MÁSCARA DE CEREBRO

Apagada por defecto, y receta: no está en `LIBRES`, así que reanudar con otro valor se rechaza.
`--mascara-cerebro` lee `<data-root>/mascaras/train/<sujeto>.npy` para cada volumen de
`--train-list` (las escribe `datasets/mascaras_ldm100k3d.py combina`, alineadas vóxel a vóxel con
`images/train`), una por vez con un generador (2 000 juntas serían ~10 GB), y cada parche pasa a
ser un cubo entero dentro del cerebro, a `MARGEN_CEREBRO` = 1 vóxel o más de su borde
(`ddr/data3d.py`). Con la máscara el parche no ve cráneo, cuero cabelludo ni aire, y casi nada del
LCR externo (HD-BET deja 1–3 vóxeles sobre la superficie pial); sin ella, con el filtro de tejido,
sólo el 4.0 % de los parches de 48³ cae entero en el cerebro. El precio es la periferia: a 48³ los
cubos cubren el 66 % del cerebro y el 18.5 % de su franja cortical (≤ 6 mm del borde); a 32³, el
80 % y el 38 % (bitácora, «Datos: LDM_100k en 3D»; por qué deepbet ∩ HD-BET sin TTA, en el
bloque final de `datasets/mascaras_ldm100k3d.py`; por qué margen 1, en el de `ddr/data3d.py`).

Si falta alguna máscara, `arma_entrenamiento` sale con error antes de que `prepara_salida` toque
`--out`, y dice cuántas faltan y los tres comandos que las generan. Con la bandera también son
obligatorias las de los volúmenes de validación (`arma_validacion`), porque `best.pth` se elige
con el PSNR en el cerebro. Un volumen sin cubos queda fuera del sorteo con un aviso, y el log de
arranque dice cuántos tienen cubos y la mediana de esquinas por volumen. La evaluación no depende
de la bandera (los mismos recortes de 128³ y los mismos PSNR, en todo el recorte, en tejido y, si
están las máscaras de `mascaras/val`, en el cerebro), así que una corrida con la bandera y una sin
ella se miden sobre lo mismo.

### EVALUACIÓN Y ARCHIVOS DE SALIDA

Cada `--eval-every` iteraciones y en la última, `valida_denoising` llama a `evaluate_denoising3d`,
que mide el PSNR y el NRMSE medios sobre los recortes fijos de `EVAL_RECORTE`³ (128³) de la lista
de evaluación, con ruido de semilla fija por volumen (`SEMILLA_VAL` + i), en todo el recorte y sólo
en tejido (vóxeles limpios > `UMBRAL_TEJIDO` = 0.05), sin gradiente y en el modo de la corrida
(bf16 en `rapido`). `train3d` no tiene épocas (sortea parches): esta validación es la «época» del
seguimiento. Si están las máscaras de cerebro de todos los volúmenes de evaluación
(`<data-root>/mascaras/val/<sujeto>.npy`), mide además el PSNR en el cerebro, con la máscara
recortada con la misma ventana: es la región que ve el entrenamiento con `--mascara-cerebro`, que
no ve cráneo ni aire, mientras el recorte entero los tiene. Agregarla no cambia los PSNR de antes
(`test_la_mascara_de_cerebro_no_cambia_los_psnr_de_antes`). El NRMSE es el de
`ddr/qsm3d.nrmse_pct`, 100·‖x̂ − x‖/‖x‖ por volumen y promediado como el PSNR: en un volumen es
función del PSNR, y deja hablar en las unidades de QSM (bloque final de `ddr/seguimiento.py`,
MÉTRICAS). Después de guardar todo eso vienen, si toca, la estabilidad (ESTABILIDAD) y la
evaluación QSM (EVALUACIÓN QSM).

En `--out` quedan:

- **`config.json`:** las banderas de la última invocación, defaults incluidos, con el micro-lote
  efectivo en vez de `auto`, y `commit` (`commit_git`: el HEAD, con `+cambios` si lo versionado
  tiene cambios, o None sin git), también sin W&B.
- **`history.json`:** una entrada por validación: `iteration`, `efectiva`, `psnr`,
  `psnr_tejido`, `nrmse_pct`, `nrmse_pct_tejido`, `psnr_cerebro` (sólo con las máscaras),
  `psnr_lote` (sólo con `--sobreajuste`), `T`, `lambda` y, en las que toca la estabilidad,
  `estabilidad`, `lambda_max` y `estabilidad_cambio` (o `estabilidad_fallo`).
- **`qsm.jsonl`:** una línea por evaluación QSM: `iteration`, `efectiva` y lo de `evalua_qsm` sin
  el volumen `chi` (con el NRMSE y el update de cada iteración del ADMM, `nrmse_trayectoria_pct` y
  `update_trayectoria`), o `fallo`, más las tres de la estabilidad en COSMOS (o
  `estabilidad_fallo`). No existe si la corrida no hizo ninguna.
- **`traza.npy`:** `float64`, `COLUMNAS` = 14 por iteración (TRAZA). Se reescribe al arrancar, en
  cada validación y en cada estado.
- **`last.pth`:** la última validación; **`best.pth`**, la de mejor PSNR en el recorte entero o,
  con `--mascara-cerebro`, en el cerebro. Claves: `config` (la de `vnet_config3d`, con R de tipo
  `tdv3d`), `model`, `iteration`, `psnr`, `optimizer`.
- **`best_qsm.pth`:** el checkpoint para QSM (EVALUACIÓN QSM): `config`, `model`, `iteration`,
  `psnr`, `nrmse_min_pct`, `iter_min`, `nrmse_final_pct`, `iteraciones`, `motivo` y
  `estabilidad`, sin `optimizer`. No existe si ninguna evaluación QSM dio un NRMSE mínimo finito.
- **`estado-<it>.pth`:** cada `--guardar-cada`, en la última y en un corte, todo lo que hace falta
  para reanudar (REANUDACIÓN).
- **`estado.json`:** sólo al terminar (`cierra`), y se borra al arrancar: `terminada`,
  `iteraciones_pedidas`, `iteraciones_hechas`, `iteracion_efectiva`, `corte`, `meseta` (el estado
  de la regla, con `--meseta` o si heredó hitos de ella), `mejor_psnr`, `psnr_final`, `rechazos`,
  `max_seguidos`, `rebobinados`, `reanudada_de`, `horas`, `vram_pico_gb` (el medido, evaluaciones
  incluidas) y `vram_predicha_gb` (el de MEMORIA); las `qsm_*` de `resumen_qsm` (evaluaciones,
  fallos, el último NRMSE final con su iteración y su motivo, y el menor con la suya; None sin
  evaluaciones QSM), y `best_qsm`, las claves de `RESUMEN_BEST_QSM` de `best_qsm.pth` (o None). Con
  `--mascara-cerebro`, `mejor_psnr` es el PSNR en el cerebro (el criterio de `best.pth`) y
  `psnr_final`, el del recorte entero en la última validación: no se comparan entre sí (31.35
  contra 26.77 en `runs/prep3d-punta`).
- **`seguimiento.json`, `wandb/`:** con W&B (SEGUIMIENTO), el id de la corrida y, offline, la
  corrida.
- **`sonda-vram.json`:** sólo con `--sonda-vram`, que no escribe nada más y no entrena.

La traza y los `.pth` se escriben a un temporal y después `replace` (`guarda_traza`,
`guarda_pth`): un corte no deja ninguno a medio escribir con el nombre bueno. `qsm.jsonl` se agrega
línea por línea (`anota_jsonl`), y un error al escribirla se avisa sin cortar: el estado lleva las
filas.

### SEGUIMIENTO

W&B es una vista de `--out`, no otra fuente: todo lo que registra está también en la traza,
`history.json`, `qsm.jsonl` o `estado.json`. Lo hace `ddr/seguimiento.py`, donde está el porqué de
los modos, la identidad de la corrida y la protección de cada llamada; acá van dónde se llama y qué
se registra.

- **`abre`**, desde `abre_seguimiento`, después de `iter(loader)`: los workers ya existen y no se
  forkea con los hilos de wandb vivos. Con `padre` = `--reanudar` y `desde` = it0, `abre` sigue la
  misma corrida sólo si W&B no registró nada después de `desde`: extender una terminada, o un
  corte que cayó justo en un estado. Tras un corte común (registra cada `--log-every` y los
  estados van cada `--guardar-cada`) lo registrado pasa del estado: abre otra corrida, con el
  mismo nombre, y el id viejo queda en `descartadas` de `seguimiento.json`. Una rama es otra
  corrida, en el grupo del tronco (los tres casos, con test offline). El grupo lo elige `abre`
  (el del tronco, o su nombre) y no se pasa: así una rama de una rama queda en el grupo del tronco
  original. La config es la de `config.json`: `vars(args)` con el micro-lote efectivo, más
  `commit`. Con `--seguimiento online`, si wandb o la credencial faltan (`falta_para_online`) o W&B
  no abre, la corrida sale con error antes de entrenar.
- **Cada `--log-every`** (`registra_progreso`), en la fila de la iteración: `metricas_traza` (las
  columnas de la traza con los nombres de `COLUMNAS_TRAZA`, más los rechazos y rebobinados
  acumulados), `train/grad_norm_max` (el max‖g‖ del intervalo, que no pierde un estallido entre
  dos muestras), `train/s_por_it` (sin validaciones ni estados: el reloj se corre tras cada uno)
  y, con GPU, `train/vram_pico_gib` (el pico desde el arranque, evaluaciones incluidas). Cada
  `--log-every` y no por iteración: W&B muestra lo que muestra el log, y la traza tiene todas las
  filas. Por iteración también entraba (0.185 ms por llamada, medido; 53 MB por corrida de 40 000,
  derivado), pero no agregaba nada que la traza no tenga.
- **En cada validación:** `val/psnr`, `val/psnr_tejido`, `val/psnr_cerebro` (con las máscaras),
  `val/nrmse_pct`, `val/nrmse_pct_tejido`, `val/psnr_lote` (con `--sobreajuste`) y la lámina
  `val/denoising` (`lamina_denoising` del primer volumen de evaluación, siempre el mismo: limpia |
  ruidosa | salida | 5·|error|).
- **En cada evaluación QSM:** `qsm/nrmse_final_pct`, `qsm/nrmse_min_pct`, `qsm/iter_min`,
  `qsm/iteraciones`, `qsm/alpha`, `qsm/segundos`, `qsm/fallo` (0 o 1) y la lámina
  `qsm/reconstruccion` (`lamina_qsm`: GT | reconstrucción | error en ±0.1 ppm); con `--meseta`,
  `meseta/*`. `motivo` es texto: va al log, a `qsm.jsonl` y, el de la última, al resumen.
- **Con la cadencia de la evaluación QSM**, haya COSMOS o no: `val/estabilidad` y
  `val/lambda_max`; con COSMOS, también `qsm/estabilidad` y `qsm/lambda_max` (ESTABILIDAD). Una
  estabilidad que falla no registra nada en W&B: queda en el log y en `estabilidad_fallo`.
- **Al final**, después de `estado.json`: `resumen` con lo mismo que `estado.json`, y `cierra`.

Las láminas se arman dentro de `registra_lamina`: un error al armarla cuesta la imagen, no la
corrida ni las métricas QSM de esa evaluación (`test_una_lamina_qsm_rota_no_tira_sus_metricas`),
y sin seguimiento no se arma. Con `--seguimiento no`, wandb ni se importa (test). No toca la
aritmética: con W&B offline y la evaluación QSM, la traza es bit a bit la de la corrida sin
ninguna de las dos (`test_seguimiento_offline_de_una_corrida`, con wandb instalado), y `abre` y
`registra` no tocan los generadores (`tests/test_seguimiento.py`).

### EVALUACIÓN QSM

Lo que el autor pidió: medir el modelo como denoiser dentro de la reconstrucción QSM, que es su
destino. Es el ADMM plug-and-play de `../tdv/wTDV_QSM_torch_3d.py` en `ddr/qsm3d.admm_wtdv`, con
k = 1 (la red 3D sobre los 160³ de COSMOS de una vez) y su protocolo: μ = 0.008, μ₂ = 1.0 y
`target_std` = 0.06; la parada no (abajo). No viene de `../QSM-TDV`, que no tiene ADMM:
reconstruye con pasos de gradiente explícitos. Las máscaras, los umbrales de parada del autor y el
número que reproduce con su red 2D, en `ddr/qsm3d.py` (PROTOCOLO DEL AUTOR). La métrica es la
suya, el NRMSE contra `chi_cosmos` en su máscara, medido sobre M·χ.

- **La parada** (decisión del autor, 2026-10-19). `evalua_qsm` llama a `admm_wtdv` con
  `max_iter = --qsm-max-iter` (`QSM_MAX_ITER = 25`), `tol = -inf` (sin el corte por TOL y CORTE
  del script) y `subidas_para_cortar = QSM_SUBIDAS = 2`: el ADMM para cuando el NRMSE contra
  COSMOS sube dos iteraciones seguidas (motivo `empeora`), o si el update no es finito, y pare como
  pare devuelve el χ de menor NRMSE que vio. Así `nrmse_final_pct` y `nrmse_min_pct` son el mismo
  número salvo en un ADMM no finito, y los dos eligen su iteración con el χ de COSMOS: es una
  parada que mira la referencia, sirve para comparar los checkpoints de una corrida y no para un
  sujeto sin χ (`test_la_evaluacion_qsm_para_si_empeora_con_techo_de_25`).
- **Por qué sin TOL y CORTE:** se ajustaron para la red 2D del autor en el pseudo-3D, no para una
  red 3D, y con ellos ninguna de las 31 evaluaciones 3D cortó en 100 iteraciones.
- **Por qué 25:** con 25 el orden entre redes sale casi el del final. En las 31 trayectorias 2D y
  pseudo-3D de 25 o más iteraciones, la correlación de rangos del NRMSE en t con el final es +0.07
  en t = 1, +0.82 en t = 2, +0.87 en t = 3, +0.93 en t = 10 y +0.98 en t = 24, y el de t = 24
  queda una mediana de 0.12 pp sobre el mínimo (máximo 0.80, salvo una oscilante de 5.39). En 3D
  el mínimo cayó en t ≤ 24 en 29 de 31 evaluaciones (t ≤ 7 en 19 de 25 con máscaras).
- **Por qué dos subidas y no una:** sobre 26 trayectorias, parar en la primera subida da el mínimo
  de t ≤ 24 en 19, y en la segunda, en 25 de 26, con una iteración más; la que no sube 16 seguidas
  desde t = 3 (bitácora, «La evaluación QSM»).
- **La referencia, para reportar y no para decidir:** el pseudo-3D oficial da 26.08 % en t = 24
  sobre M·χ (25.94 % a 100), y el 2D oficial, 26.57 % a 25. El objetivo del autor para la red 3D
  es ~26 % a 25 iteraciones; la corrida se compara contra sí misma. `qsm.jsonl` guarda el NRMSE y
  el update de cada iteración (`nrmse_trayectoria_pct`, `update_trayectoria`): con ellos se mide
  después el orden temprano en 3D y se calibra una parada sin referencia.
- **Cuándo** (`valida`): en las validaciones, después de que `valida_denoising` guardó
  `history.json` y los `.pth` (un corte duro a mitad del ADMM no pierde la validación) y antes del
  estado, que lleva la fila. Con `--eval-qsm-cada 0`, en todas; con N > 0, en las de iteración
  múltiplo de N y en la última.
- **Los datos:** `--qsm-datos` (`QSM_DATOS` = `datasets/qsm_cosmos`) con `params.mat` y
  `chi_cosmos.mat`, copiados de `../tdv/data_mat` (`QSM_ORIGEN`); `carga_qsm` los carga una vez al
  arrancar, en numpy y en CPU (102 MB y 0.5 s, medido el 2026-10-12). Si falta alguno, un AVISO
  dice cuál y de dónde copiarlo, la evaluación queda apagada y el entrenamiento sigue: no es
  receta, y una corrida sin ella entrena igual. Con `--meseta`, en cambio, faltar COSMOS es un
  error al arrancar. `--qsm-datos ''` la apaga sin aviso (los tests que no la miran).
- **En el modo de la corrida:** `denoiser_3d` con el autocast de `--mode` (bf16 en `rapido`) y el
  `R.grad` compilado, que en 160³ compila una forma más (`dynamic=False`). Con las del
  entrenamiento y la validación son 4 entradas de dynamo con trozos iguales (dos más con trozos
  desiguales), bajo el `recompile_limit` de 16, y volver a entrenar no recompila (medido en CPU
  con formas chicas).
- **No puede matar el entrenamiento:** `evalua_qsm` ataja toda `Exception`, incluido
  `torch.cuda.OutOfMemoryError`, libera la caché de CUDA fuera del `except` (adentro la traza del
  error retiene los tensores) y devuelve `fallo`, que `valida_qsm` manda al log, a `qsm.jsonl` y a
  W&B como `qsm/fallo` = 1 (`test_un_fallo_de_la_evaluacion_qsm_no_mata_la_corrida`, con un ADMM
  que levanta un OOM). La lámina va fuera de `evalua_qsm`: un error al armarla no tira las
  métricas (SEGUIMIENTO).
- **No perturba el entrenamiento:** corre en modo eval y sin gradiente, deja el modelo en modo
  train, no usa ningún generador (ni el del ruido ni el de los parches) y no escribe nada que el
  bucle lea. En `referencia` y CPU, con un COSMOS de juguete de 24³, la traza, la historia, el log
  y los pesos finales salen bit a bit iguales con y sin ella
  (`test_la_evaluacion_qsm_no_toca_el_entrenamiento`).
- **Costo**, medido en la 5070 Ti en `rapido` (2026-10-12 y 2026-10-19): 3.41 s por iteración de
  ADMM (la primera evaluación de cada proceso compila: +10 a +43 s) y 5.19 GiB de pico; en fp32 sin
  compilar, 25.95 s y 6.92 GiB. Las 25 iteraciones son ~85 s, y la evaluación entera, con la
  validación (22.6 s) y las dos estabilidades (61.9 s), ~170 s. Contra 8.77 s por iteración de
  entrenamiento, todo cada 100 suma un 19.4 % (+47 h en 100 000), cada 500 un 3.9 % y cada 2 500
  un 0.8 %; la validación de sólo PSNR cada 100 suma un 2.6 %, y PSNR cada 100 con QSM cada 500,
  5.9 %. En la tarjeta de destino no está medido; como todo corre la misma red, el cociente
  debería ser parecido. Con la parada de dos subidas, una evaluación puede usar menos de 25.
- **`best_qsm.pth` es el checkpoint que se pasa a QSM**, porque el PSNR de denoising no sirve para
  elegir un prior PnP (AGENTS.md, «Lo que ya se sabe»). El criterio: la evaluación de menor
  `nrmse_min_pct`, lo mismo que mira la meseta, entre todas las evaluaciones (también las de antes
  de `--meseta-desde`). El final, las iteraciones, el motivo y la estabilidad viajan con él y no
  eligen. La estabilidad bajo `UMBRAL_ESTABILIDAD` no predice la convergencia: en `ventana-3d-12h`
  habría elegido la 200, con 1.54 y el ADMM en 144 % tras 100 iteraciones, y en 2D `ciego2d-35k`
  converge con 2.14. Exigir que el ADMM corte por TOL y CORTE tampoco sirve: esos umbrales
  dependen de cada red, y una red 3D nueva podía dejar el archivo sin escribir en toda la corrida.
  Ojo: el mínimo elige su iteración con el χ de COSMOS, así que puede ganar una red cuyo ADMM pasa
  cerca y después se aleja (en `ventana-3d-12h`, 34.23 % en t = 6 y 144 % en la 100); si una
  parada calibrada llega ahí se sabe recién con el grid de después del entrenamiento.
- **Cómo se escribe** (`valida_qsm`): tras una evaluación sin fallo, si su NRMSE mínimo es el menor
  hasta ahora, con `guarda_pth` y una línea en el log; un mínimo NaN (ninguna iteración finita) no
  lo escribe. Lleva lo de `best.pth` salvo el optimizer (es para usar; para seguir entrenando está
  el estado) más `nrmse_min_pct`, `iter_min`, `nrmse_final_pct`, `iteraciones`, `motivo` y
  `estabilidad` (None si no se midió o falló). Viaja en el estado como `mejor_qsm`, y al reanudar
  se aparta o se repone como `best.pth` (DIRECTORIO DE SALIDA). Sin COSMOS no se busca, y el
  arranque lo dice en una línea. Con `--guardar-cada` igual a `--eval-every` queda además un
  `estado-<it>.pth` por evaluación, para elegir otro a mano; con la cadencia de la corrida larga,
  4 de cada 5 evaluaciones QSM no dejan pesos (bitácora, «Abierto»).
- **Ojo con `estado.json`:** `qsm_nrmse_mejor_pct` es el menor NRMSE final de todas
  (`resumen_qsm`); lo de `best_qsm.pth` está en `best_qsm` (`RESUMEN_BEST_QSM`). Con la parada de
  dos subidas suelen coincidir. Comparar sólo vale con los mismos COSMOS y `--qsm-max-iter`, que
  son libres al reanudar: cambiarlos a mitad mezcla criterios en la misma elección, y nada lo
  verifica.
- **Tests** (`test_best_qsm_es_el_de_menor_nrmse_minimo` y vecinos): con los NRMSE forzados, se
  escribe en la iteración del menor mínimo (no en la del menor final, ni en una sin NRMSE finito, y
  sí en una que no cortó, con el final en 90 % y estabilidad 7); sin ningún mínimo finito no hay
  archivo; reanudar repone el del estado y aparta uno ajeno; y la traza es la de la corrida sin
  QSM.
- **Abierto:** `TARGET_STD` = 0.06 se calibró para la red 2D color del autor, no para una 3D
  entrenada en T1, y una parada sin referencia para la red 3D está sin calibrar. Las dos se
  calibran con un grid después del entrenamiento (bitácora, «Abierto»).

### ESTABILIDAD

El número del paso explícito del esquema, (T/S)·λ_max(∇²R(x)), que `ddr/qsm3d.estabilidad` mide en
un punto x: el paso amplifica si pasa de `UMBRAL_ESTABILIDAD` = 2 (el porqué y el método, en
`ddr/qsm3d.py`, ESTABILIDAD). Viene de la 2D: explicó por qué los checkpoints 2D propios rinden
peor que el oficial como prior del ADMM de QSM y no convergen. Medido a la amplitud de
entrenamiento (σ = 25) los cuatro dan 1.38–2.01 y el PSNR no los distingue; a la de uso (la
estimación inicial del ADMM llevada a std 0.06), con los dos extremos del espectro, el paso del
oficial expande ×1.27 (ρ de −0.28 a 1.03) y el de los propios ×1.77–23.9 (2026-10-12). El PSNR de
la validación es ciego a esto; este número no. Ojo: `estabilidad` devuelve el MÓDULO del
dominante, y con el dominante negativo (Rayleigh < 0) el paso expande aunque el módulo quede bajo
`UMBRAL_ESTABILIDAD`.

- **Dónde:** en dos puntos, armados una vez al arrancar (`arma_puntos_de_estabilidad`). `val`, el
  punto de entrenamiento (`punto_validacion`): el cubo central de `--estabilidad-lado`³ del primer
  recorte de evaluación, con el ruido que le pone la validación (σ = `--sigma`, semilla
  `SEMILLA_VAL`, la que `valida_denoising` pasa explícita a `evaluate_denoising3d`; se sortea
  sobre el recorte entero y después se corta), así que es lo que la validación ve en ese cubo y no
  usa el generador del ruido. `qsm`, sólo con COSMOS: `punto_qsm3d`, la entrada del primer
  denoiser del ADMM (α·M·χ₀) recortada a un cubo centrado en la máscara, en el device de la
  corrida, que es donde corre el ADMM. Un lado que no entra se recorta al volumen (en los tests,
  el 96 por defecto queda en los 16³ del recorte de evaluación). Si `punto_qsm3d` falla, un AVISO
  apaga sólo la estabilidad en QSM.
- **Cuándo:** con la cadencia de la evaluación QSM (`--eval-qsm-cada`), haya COSMOS o no: `val`
  después de guardar la validación (`valida_estabilidad`, que reescribe `history.json` con la
  entrada completada), y `qsm` antes del ADMM, en la misma fila de `qsm.jsonl` (`valida_qsm`).
- **Cómo:** en float32 IEEE (sin autocast ni TF32, aunque la corrida sea `rapido`) y con el
  `R.grad` de la clase, no el compilado de la instancia: no suma entradas de dynamo (medido en CPU
  con formas chicas). No toca pesos, `.grad` ni generadores, y una excepción, también un OOM, la
  ataja `mide_estabilidad` como `evalua_qsm`: va al log y a `estabilidad_fallo`, y el bucle sigue
  (test con una `estabilidad` que levanta). Con y sin la medición (`--estabilidad-lado 0`), en
  `referencia` y CPU, la traza, el log del entrenamiento, los pesos, Adam y los generadores salen
  bit a bit iguales (`test_la_estabilidad_no_toca_el_entrenamiento`).
- **Qué se guarda** (`claves_estabilidad`): `estabilidad` (el paso), `lambda_max` (‖Hv‖, el módulo
  del autovalor dominante, como en el script 2D) y `estabilidad_cambio`, el cambio relativo de la
  última iteración de la potencia; no el reloj, que cambiaría la fila al reanudar. El log
  (`linea_estabilidad`) da además el cociente de Rayleigh (el signo del dominante: ≈ −λ_max si es
  negativo), T, los segundos y `*INESTABLE*` desde `UMBRAL_ESTABILIDAD`. Un cambio grande dice que
  las 40 iteraciones (`ITERS_ESTABILIDAD`) no alcanzaron: el número queda debajo del verdadero.
- **El lado:** `--estabilidad-lado` = `LADO_ESTABILIDAD` = 96, el del script 2D. El número
  depende del lado (en el modelo de humo, 0.85 a 32³ y 1.27 a 64³ en COSMOS): se compara al mismo
  lado. `--estabilidad-lado 64` baja la memoria ~3.4×, y 0 la apaga.
- **Costo:** en CPU con 8 hilos y la CPU compartida (2026-10-12), 13.9 s por validación a 32³ y
  208 s a 64³, los dos puntos. En la 5070 Ti a 96³, ~33 s por punto (133–134 s por corrida en
  `ventana-3d-12h`, dos validaciones de dos puntos, 2026-10-13) y 61.9 s por validación con los
  dos puntos (2026-10-19). Memoria, derivada de la RSS: ~9 GB a 96³ (~2.7 a 64³); en GPU no se
  midió aparte, y el pico de las corridas de `ventana-3d-12h`, con ella incluida, fue 8.16 GiB. Si
  no entra, queda `estabilidad_fallo`.
- **Qué decide: nada.** Se registra (log, `history.json`, `qsm.jsonl`, W&B y el propio
  `best_qsm.pth`) y no corta, no cambia el entrenamiento ni elige el checkpoint (EVALUACIÓN QSM).

## ddr/qsm3d.py

### PROTOCOLO DEL AUTOR

El ADMM de `admm_wtdv` parte χ en `z2 = Dχ` (datos) y en k copias `z_i = χ` (prior), con duales
escalados:

```
χ   = F⁻¹[(μ₂·D·F(z2 − s2) + (μ/k)·Σ F(z_i − s_i)) / (μ₂·D² + μ)]      (D real)
z_i = M·D_i(α·M·(χ + s_i))/α,                                       s_i += χ − z_i
z2  = (W²·phase + μ₂·(Dχ + s2))/(W² + μ₂),                          s2  += Dχ − z2
```

y devuelve M·χ.

- **M es la máscara dura de α** (`datos["mascara"]`, el `refined_mask` del script del autor), no
  la del NRMSE (`mascara_eval`, χ_COSMOS ≠ 0). Son las máscaras del solver de `../tdv`
  (`wTDV_QSM_torch_3d.py`), las mismas en su 2D y en su 3D. Con la entrada en M, el prox no
  depende de lo de afuera. Con la salida en M, el prior vale 0 donde no hay datos (W² = 0), que es
  donde el TDV amplifica.
- **k = 1 y k = 3 con el mismo denoiser son el mismo ADMM salvo redondeo**, porque cada split del
  prior pesa μ/k y juntos pesan μ. Antes de las máscaras, en el sintético del test, se separaban
  2.5e-8 sobre un χ de 0.1. `test_k1_es_k3_con_el_mismo_denoiser` pide ≤ 1e-6·max|χ|. k = 3 es
  el pseudo-3D del autor (la red 2D en los tres ejes) y k = 1, la red 3D.
- **`alpha` se fija en t = 0 con χ₀**, que sale sólo del término de datos (los `z_i` valen 0): no
  depende de la red. En COSMOS da 4.8591, el mismo valor que los logs del autor.
- **TOL y CORTE.** El update relativo, sobre el volumen entero, baja, toca un mínimo y vuelve a
  subir. El ADMM se da por convergido al bajar de TOL y corta en la primera iteración cuyo update
  *anterior* pasa de CORTE; el χ final es el de esa iteración. `TOL = 1.15` y `CORTE = 1.30` los
  ajustó el autor con la red 2D oficial cuando sólo enmascaraba la salida. Con la entrada también
  en M, el update de esa red baja sólo a ~1.24 y no corta: las 100 iteraciones dan 25.94 %
  (`../tdv/README.md`, «Script pseudo-3D»). Ese README usa ahora `tolUpdate = 1.25` y `finUpdate =
  1.35`, con los que la misma red da 26.01 % en 29 iteraciones; `ddr/qsm3d.py` conserva los de la
  referencia de 25.94 %.
- **Los umbrales dependen de la red:** en el 2D, el grid del autor eligió `tolUpdate` de 0.3 a 1.5
  para cinco redes (`PARAMS` de `../tdv/wTDV_QSM_torch.py`). Para la red 3D no hay calibración
  todavía, y la corrida no depende de ellos: `evalua_qsm` de `train3d.py` llama a `admm_wtdv` con
  `tol = -inf`, un techo de `QSM_MAX_ITER = 25` y `subidas_para_cortar = QSM_SUBIDAS = 2`, y
  `best_qsm.pth` y `--meseta` miran el NRMSE mínimo de la trayectoria («## train3d.py»,
  EVALUACIÓN QSM y MESETA; bitácora, «La evaluación QSM»). `MAX_ITER = 100`, `TOL` y `CORTE` siguen
  siendo los defaults de `admm_wtdv`: el protocolo del autor, la referencia.

### REPRODUCCIÓN

- **Sin máscaras** (k = 3, CPU float32, sin compilar; 2026-10-12): 26.8898 % con corte en t = 26
  (27 x-updates), contra 26.89 % y el mismo corte del autor en GPU con bf16. Iteración a
  iteración, el NRMSE se separa de su log a lo sumo 0.0026 pp y el update, 0.0094.
- **Con las máscaras** (GPU bf16, sin compilar; 2026-10-18): 25.9393 % en las 100 iteraciones,
  sin corte, contra 25.94 % del autor. El update se separa de su log a lo sumo 0.0136.

Los dos con la red 2D color oficial, su `process_axis` y el mismo α. Sobre un sintético, contra el
bucle del script transcripto en el test, `admm_wtdv` es igual bit a bit: χ, updates, NRMSE de cada
iteración y α (`test_admm_k3_es_el_script`).

### COSTO DEL 3D

- **CPU** (2026-10-12): `denoiser_3d` con el VNet3D de la corrida de humo (TDV33 3D, 32 features,
  S = 10) sobre los 160³ de COSMOS de una vez, en float32 con 8 hilos de una CPU compartida:
  466 y 478 s por iteración, y 7.77 GB de pico de RSS contra 0.63 GB antes del ADMM.
- **GPU** (5070 Ti, `rapido`, 2026-10-19): 3.413 s por iteración de ADMM, 85 s las 25 del techo de
  `train3d.py` (bitácora, «La evaluación QSM»). En la Blackwell, sin medir.
- Cada iteración corre un denoiser. En `train3d.py` (`tol = -inf`) el ADMM recorre el techo de 25
  salvo que el NRMSE suba dos iteraciones seguidas; con el protocolo del autor, las máscaras y
  TOL/CORTE sin calibrar para la red 3D, una red que no corta recorrería las `MAX_ITER = 100`.

### ESTABILIDAD

**El número.** Cada paso del esquema desenrollado es un paso de gradiente explícito sobre T·R
seguido del prox del término de datos (`VNet2D.forward` en `ddr/model.py`, que `VNet3D` hereda;
la receta lleva `use_prox`, en `vnet_config` de `train.py`):

```
x ← (x − (T/S)·∇R(x) + (λ/S)·z)/(1 + λ/S)
```

Su jacobiano es `(I − (T/S)·∇²R)/(1 + λ/S)`: el paso amplifica en la dirección de un autovalor μ
de ∇²R si `|1 − (T/S)·μ| > 1 + λ/S`, o sea si `(T/S)·μ > 2 + λ/S` o `(T/S)·μ < −λ/S`. Con S = 10
pasos encadenados, la amplificación se compone. `estabilidad` devuelve `paso` = (T/S)·max|μ| en un
punto, y se compara con `UMBRAL_ESTABILIDAD = 2`: el λ/S de la receta mueve ese umbral menos de
0.7 %. Por debajo de 2 un paso también expande, por curvatura negativa (AGENTS.md, «Entrenamiento
3D»).

**Por qué importa** (2026-10-12; bitácora, «Lo que dejó el 2D»): los checkpoints 2D propios no
convergen como prior del ADMM. A σ = 25, donde se entrenan, los cuatro dan un índice de 1.38–2.01.
A la amplitud de uso de QSM, el paso del oficial expande ×1.27 (ρ de −0.28 a 1.03) y el de los
propios ×1.77–23.9, con dos de los tres dominantes negativos. El orden coincide con el de cuánto se
degradan, y el PSNR no los distingue.

**Cómo se mide:**

- **El producto Hessiano-vector es exacto:** la VJP por autograd de `R.grad`, que es el gradiente
  exacto de la energía escrito a mano (`tests/test_gradient3d.py`) y diferenciable (el
  entrenamiento retropropaga a través de él). Así `J_grad(x)ᵀ·v = ∇²R(x)·v`, sin ε que elegir. El
  script 2D usaba diferencias centradas con ε = 1e-5·mean|x| en float64; en float32 la cancelación
  dejaría ~2 de 7 dígitos (ulp/ε ≈ 6e-3) y en bf16 ninguno (la ulp, 4e-3, pasa a ε): derivado, no
  medido. El grafo de `R.grad` se arma una vez y se recorre `ITERS_ESTABILIDAD` veces
  (`retain_graph`): cada producto cuesta un backward, no un forward más un backward.
- **En el dtype de los pesos, sin autocast y sin TF32** (`_sin_tf32` lo apaga mientras dura y deja
  los flags como estaban): float32 IEEE en la corrida, float64 en los tests. Llama al método de la
  clase (`type(R).grad`), no al atributo: `train3d.py` pone en la instancia el `R.grad` compilado
  con `dynamic=False`, que sumaría una entrada de dynamo por cada forma nueva y correría en bf16
  bajo el autocast de `rapido`. Con el de la clase no suma ninguna, y el compilado da después lo
  mismo bit a bit (medido en CPU).
- **No toca el modelo:** `autograd.grad` sólo respecto de x no escribe ningún `.grad`, y el vector
  de arranque sale de un generador propio en CPU (la misma secuencia en CPU y en GPU), no del
  global. `test_estabilidad_no_toca_el_modelo_ni_los_generadores`: con los `.grad` de una
  iteración puestos, pesos, `.grad` y generador global quedan bit a bit, y dentro de un autocast
  bf16 el resultado es el mismo bit a bit.
- **`lambda_max` es ‖Hv‖ del último v (unitario)**, como en el script 2D: converge desde abajo al
  módulo del autovalor dominante, max|μ|, que nunca es menor que el mayor positivo y es él si el
  dominante es positivo. Si `paso` pasa de 2, el paso amplifica cualquiera sea el signo: con
  μ ≤ −2S/T, `|1 − (T/S)·μ| ≥ 3`.
- **`rayleigh`, vᵀHv, da el signo, no el número:** ≈ −`lambda_max` si el dominante es negativo
  (con pesos al azar pasa: semilla 0 del test, −5.35). Con dos dominantes de signo opuesto y módulo
  parecido no converge a ninguno: semilla 3 a amplitud 0.1, −2.6731 y 2.6705, y `rayleigh` da
  0.57 con 200 iteraciones, con ‖Hv‖ entre los dos módulos.
- **`cambio_relativo`** es |Δλ|/λ entre las dos últimas iteraciones. El error es
  ~cambio·r²/(1 − r²), con r = |μ₂|/|μ₁|: con r = 0.98, cuarenta iteraciones dejan λ 0.9 % abajo
  con un cambio de 2.2e-4 (red de juguete, 2026-10-12). Un cambio grande dice que el número es una
  cota inferior de max|μ|; uno chico no garantiza que no lo sea.
- **Las constantes:** `ITERS_ESTABILIDAD = 40` y `SEMILLA_ESTABILIDAD = 11` son las del script con
  que se midió el número en los checkpoints 2D, para que los números se comparen.
  `UMBRAL_ESTABILIDAD = 2` es la cuenta de arriba. `LADO_ESTABILIDAD = 96` es el `lado` de aquel
  script, ahora un cubo.

**El test contra la Hessiana** (`test_estabilidad_es_el_autovalor_dominante_de_la_hessiana`): la
Hessiana de la energía armada entera con autograd, en una red de 4 features sobre 6³ en float64.
Con 200 iteraciones la potencia da el módulo del autovalor dominante de `eigvalsh` a ~1e-15
relativo, y `rayleigh` su signo, con un dominante negativo (semilla 0) y con uno positivo (semilla
1, 3.61); en el empate de signos opuestos, `lambda_max` queda entre los dos módulos.

**El punto QSM.** `punto_qsm3d` es la entrada del primer denoiser del ADMM 3D (k = 1): α·M·χ₀, con
χ₀ la primera actualización de χ (z = s = 0, así que sólo el término de datos), α =
`TARGET_STD`/std(χ₀ en M) con la std de `torch.std` (n − 1), y cero fuera de M, como la ve el
denoiser. No se recalcula: corre `admm_wtdv` una iteración con un denoiser que guarda su entrada,
así que no puede divergir del ADMM (`test_punto_qsm3d_es_la_entrada_del_primer_denoiser`: bit a bit
la primera entrada de un ADMM de cuatro iteraciones, y a 1e-5 de la fórmula cerrada en float64).
El script 2D armaba otra cosa: tres cortes axiales adyacentes como RGB de 96², con la misma fórmula
(μ₂ = 1) y la std de `np.std` (n). Aquí es un cubo de `LADO_ESTABILIDAD`³ centrado en la caja de M,
con la ventana de los recortes de evaluación (`_recorte_centrado` de `ddr/data3d.py`): en COSMOS,
la ventana `[32:128, 34:130, 26:122]` de los 160³, con el 77.6 % de sus vóxeles en M (medido el
2026-10-12 y el 2026-10-20).

## train.py (la receta que hereda el 3D)

`train3d.py` importa de `train.py` lr, betas, σ, S, la arquitectura (`vnet_config`), los init, las
constantes del disyuntor, `rechaza_lote`, `respalda`, `rebobina` y `MODES`, como los mismos objetos
(`tests/test_train3d.py` lo exige con `is`): lo que sigue vale para los dos.

### BUCLE

`main()` de `train.py` arma modelo, datos, Adam y los hitos del lr, y repite `--iterations` veces
lo mismo que hace `entrena` en `train3d.py`:

1. Dos micro-lotes de `MICRO_BATCH = 16` parches de 93×93 con flips y rotaciones: los 32 del paper
   con la mitad de memoria y el mismo gradiente, porque ninguna capa mezcla muestras.
2. Ruido gaussiano nuevo sobre cada parche limpio `y`; la imagen ruidosa `z` es también x_0.
3. S = 10 pasos `x_s = prox(x_{s-1} − (T/S)·∇R(x_{s-1}))`: R es la energía TDV aprendida, y el
   prox, `(x + (λ/S)·z)/(1 + λ/S)`, tira hacia `z` con peso λ/S.
4. Pérdida ‖x_S − y‖², sumada por imagen y promediada sobre el lote: sólo cuenta x_S.
5. Backward. `clip_grad_norm_` devuelve ‖g‖, la norma global del gradiente antes de recortar (con
   `--clip inf`, el default, no recorta), y con ella el disyuntor decide si hay paso de Adam. El lr
   avanza igual.
6. `model.project_()`: cada filtro de K1, la primera convolución, con media cero y norma ≤ 1, que
   es la restricción del paper; T y λ recortados a [0, 1000], el rango del `config` oficial.
7. Una fila de traza; cada `--log-every`, una línea de log; cada `--eval-every`, evaluación.

El bucle no tiene `break` ni chequeo de NaN: una corrida que estalla sigue hasta la última
iteración, y sólo ahí se escribe `estado.json`. Si ese archivo falta, se cortó desde afuera. En qué
se aparta `entrena` (parches 3D, micro-lote libre, lr por `lr_en`, cota absoluta escalada,
reanudación y el corte de `--meseta`): «## train3d.py», BUCLE.

### CONTROLES APRENDIDOS

- **T es el tiempo de parada:** el largo total del descenso, repartido en S pasos de T/S. **λ**
  (`lmbda`) es el peso del término de datos. Se aprenden junto con los pesos y tienen pared en 0:
  con T = 0 el esquema devuelve `z` sin tocar; λ = 0 apaga el término de datos, y al checkpoint
  oficial le cuesta 0.033 dB en denoising (en QSM, sin medir).
- **`DELTA_ADAM` = lr/√(1−β₂) = 0.01265** es lo que Adam mueve un parámetro ante un pico de
  cualquier tamaño, en régimen (t ≫ 1/(1−β₂) = 1000). No hay piso en `T_init`: arrancar en 2·δ o
  más es higiene de procedencia, no corrección de un modo de fallo, y `test_regla_2delta` de
  `tests/test_recipe.py` lo pide a `T_INIT` y `LAMBDA_INIT` (su par 3D,
  `test_t_init_respeta_la_regla_2delta`, sólo a `T_INIT`).
  Ningún init es del paper.
- **`T_INIT = 8·δ_Adam ≈ 0.1012`:** con él, una corrida 2D de 10⁵ iteraciones en `referencia`
  llega a 31.547 dB en CBSD68, contra 31.53 publicado. Bajarlo cuesta rechazos, no PSNR: a 10⁵,
  2·δ da −0.011 dB y 1.99× rechazos; 1.1·δ, −0.022 dB y 4.7× (bitácora, «Lo que dejó el 2D»). El
  riesgo de que un pico lleve T a 0 está en la primera fase del lr, porque un pico mueve T según el
  lr vivo; en 3D vale igual, con las mismas constantes.
- **`LAMBDA_INIT = 0.1`.**

### DISYUNTOR DE LOTES

‖g‖ vale ~10² en una corrida 2D sana; de una iteración a la otra puede saltar 400×, y llega a
~10¹³. No es un bug (`tests/test_gradient.py` compara el gradiente con autograd): unos pocos lotes
de ‖g‖ moderado llevan los pesos a una cuenca mala, donde todo lote da un ‖g‖ enorme y de la que no
se sale solo.

- **Un rechazo saltea `opt.step()`:** el lote no entra en `m` ni en `v`, los promedios móviles de
  Adam del gradiente y de su cuadrado. Un clip no alcanza: Adam divide `m` por √v y la escala se
  cancela.
- **`rechaza_lote`** rechaza si ‖g‖ no es finito, si pasa la cota absoluta, si pasa `--guard` veces
  la mediana de las últimas `GUARD_VENTANA` normas aceptadas (con la ventana llena) o si `amp` pasa
  `--guard-amp`. Sin rechazos no toca la aritmética, y ninguna bandera deja pasar un ‖g‖ no finito.

Cada número, medido en BSDS:

- **`GUARD_RATIO = 5`:** los lotes que meten a una corrida en la cuenca mala van a 7.70–8.97× la
  mediana (tres medidos); 5 queda debajo y rechaza el 0.15 % de los sanos (318 de 205 829,
  `tests/test_disyuntor.py`).
- **`GUARD_VENTANA = 100`:** mediana y no media exponencial: al arrancar, ‖g‖ sube de ~26 a ~830
  en doce iteraciones y una EMA lo rechazaría. Un pico no sube su propio umbral.
- **`GUARD_ABS = 1e4`:** la única prueba armada en las 100 primeras iteraciones. Separa excursiones
  que se resuelven solas (≤ 6 113) de las que no (≥ 1.97·10⁵).
- **`GUARD_LIMPIAS = 3`:** aceptadas seguidas que pide el respaldo, porque en un estallido hay
  aceptadas sueltas (hueco máximo 1).
- **`GUARD_RESPALDO = 250`:** iteraciones desde el respaldo anterior, contadas desde él y no por
  múltiplos de `it`: con múltiplos, una corrida que se daña antes del primero se queda sin respaldo.
- **`GUARD_SEGUIDOS = 25`:** rechazos seguidos que rebobinan; un tramo sano llega a 18.

En 3D la cota absoluta es `GUARD_ABS` escalada por los valores sumados por parche (`cota_absoluta`:
42 622 a 48³), y la corrida larga usa `--guard 10` (bitácora, «El disyuntor en 3D»).

- **El rebobinado.** Rechazar llega un paso tarde: ‖g‖ se mide en los pesos de ahora. Por eso
  `respalda` copia en RAM pesos, `m`, `v` y ventana (~4.5 MB en el 2D; derivado, ~14 MB con los
  1.16 M parámetros del 3D), y a `GUARD_SEGUIDOS` rechazos seguidos `rebobina` vuelve ahí y sigue
  con lotes nuevos.
- **`it` y el calendario no retroceden:** `rebobina` restaura Adam pero conserva el lr vigente,
  porque el `state_dict` del optimizador trae el lr del respaldo y deshacía un halving cruzado
  (`test_un_rebobinado_que_cruza_un_hito_no_deshace_el_halving`).
- **Restaura una copia:** `opt.load_state_dict` adopta los tensores que recibe, así que sin ella los
  pasos siguientes mutan el respaldo y el segundo rebobinado al mismo deja a Adam como estaba
  (`test_el_segundo_rebobinado_al_mismo_respaldo_tambien_devuelve_adam`).
- **`--guard-amp`** prueba amp = ‖p_0‖/‖p_S‖, con p_s = ∂L/∂x_s (no la clave `amp` de `MODES`);
  sana, mediana 0.697 en cinco trazas 2D instrumentadas. Viene apagada: alarga el backward hasta
  x_0, y en los eventos medidos ‖g‖ dispara igual o antes.

### MODOS DE EJECUCIÓN

Misma receta; cambian precisión, velocidad y reproducibilidad. Cada entrada de `MODES` fija:

- **`amp`:** dtype de autocast: bf16 (7 bits de mantisa) en `rapido`, `None` en los otros dos.
- **`tf32`:** convoluciones y matmuls en TF32 (10 bits) en `intermedio`; si no, fp32 IEEE.
- **`deterministic`:** kernels deterministas, sin sumas atómicas cuyo orden varía entre corridas.
- **`benchmark`:** cuDNN cronometra y elige el algoritmo de convolución: dos corridas idénticas
  pueden elegir distinto y separarse desde la iteración 1–2.

Sólo `referencia`, fp32 IEEE de 23 bits, apaga `benchmark` y repite bit a bit; es también el único
modo donde `--checkpointing` (recomputar el forward en el backward) es exacto. cuDNN trae TF32
encendido de fábrica, así que `main` fija `fp32_precision` a mano en los dos backends. En 3D el
default es `rapido`: `referencia` cuesta ~21× por vóxel (bitácora, «Costo y memoria»).

### OPTIMIZADOR

Adam con un solo grupo y `LR = 4e-4` para todo, como en el paper; bajar el lr de T y λ por
separado creó un modo de fallo nuevo (AGENTS.md, regla 1). El lr se divide por 2 en cada hito.

- **En `train.py`** el primer hito es `--lr-halve-start` (0 lo pone en `--lr-halve-every`), los
  demás van cada `--lr-halve-every`, y `--lr-halvings` les pone tope, con `MultiStepLR` avanzando
  después del paso. Con 10⁵ iteraciones son 25k/50k/75k/100k, el StepLR del paper; con las 10 000
  del default (`ITERACIONES_TANDA`), ninguno.
- **En `train3d.py`** los hitos por defecto son los mismos (`range(LR_HALVE_EVERY, …)`, o
  `--hitos`), y `lr_en(it, hitos)` reproduce el lr de `MultiStepLR` iteración por iteración: con un
  hito en 25 000, la 25 000 usa 4e-4 y la 25 001, 2e-4.
- **Entre dos hitos conviene dejar varias veces la memoria de Adam**, 1/(1−β₂) = 1000 iteraciones.
- **Adelantar los hitos sirve en 2D** porque el final rinde poco (de 50 000 a 100 000, +0.030 dB
  en 18.8 h), pero los números no se trasladan al 3D: un parche de 48³ tiene 4.26× los valores de
  uno de 93²·3, y el modelo 3.0× los parámetros (bitácora, «El modelo y la receta 3D»).
