# Bitácora

Lo vigente del entrenamiento 3D (`train3d.py`): estado, decisiones con el número que las sostiene,
lo medido que sigue valiendo y lo abierto. Las entradas nuevas van al final, fechadas, y cuando una
cambia algo de las secciones de arriba, se corrige la sección.

Reescrita el 2026-10-20, al cerrar el 2D: la historia de las fases anteriores no se conserva, y lo
que de ella sigue valiendo está aquí. Cada punto lleva entre paréntesis la fecha en que se midió o
se decidió. El porqué de cada constante del código está en
[`porque-del-codigo.md`](porque-del-codigo.md).

---

## Estado

- **`train3d.py` está listo para la corrida larga**, que el autor lanza en la RTX PRO 6000
  Blackwell Max-Q de 96 GB, sin probar allí. Qué copiar, cómo comprobar y el comando: README,
  «Llevar el entrenamiento 3D a otra máquina». Lo fijado y por qué: «La corrida larga», abajo.
- **Lo más largo que corrió en 3D** son 2 500 iteraciones (`prep3d-g10-s2-2500`). Del 3D hay sólo
  sondas cortas en la 5070 Ti local: lo que dejaron está en «Lo que mostraron las sondas» y en
  «El disyuntor en 3D».
- **Para QSM se usa `best_qsm.pth`**, no `best.pth`. En `../QSM-TDV` no carga tal cual: lo
  convierte `a_qsm_tdv.py`, y falta la escala de χ («Hacia QSM-TDV»).
- **La fase 2D está cerrada**: `t8delta-100k-s1` llegó a 31.547 dB en CBSD68 (σ = 25) contra
  31.53 publicado. Lo que dejó para el 3D: «Lo que dejó el 2D», al final.

## Abierto

- **Antes de lanzar, en la Blackwell:** la sonda (micro-lote 16 contra 8), la prueba de 40
  iteraciones del README (paso 5) con las banderas finales, y el s/it y el costo de una evaluación.
- **Si la tasa de escalones de ‖g‖** con micro-lote 16 y la lista de 2 000 es la de las sondas
  (lista de 1 000), con qué frecuencia hay escalones sobre 10× en 10⁵ iteraciones, y si aceptar los
  de 5–10× le cuesta algo al prior.
- **Las pendientes del PSNR y del NRMSE después de la 10 000** (la extrapolación logarítmica del
  PSNR, optimista, da 3.8σ por 1 000 iteraciones en la 10 000 y 1.0 en la 40 000), cuánto se
  correlacionan sus ruidos, y si hay pozos del NRMSE más largos que la paciencia.
- **Si partir el lr mejora el NRMSE QSM:** ninguna serie QSM cruzó una mitad.
- **Si las trayectorias del ADMM de la red larga oscilan** como la de `sobreajuste-3d-control`: la
  parada de dos subidas le restaría ~1 pp a ese checkpoint.
- **Por qué la red hace crecer el error en el cono del dipolo**, y si sigue pasando a 10 000 o más
  iteraciones; y qué le hace al NRMSE que la red no vea el borde del cerebro.
- **Pesos de cada evaluación QSM:** con QSM cada 500 y estados cada 2 500, 4 de cada 5
  evaluaciones QSM no dejan pesos. Si `best_qsm.pth` quedara mal elegido, no hay otro para elegir a
  mano (auditoría, 2026-10-16; sin decidir).
- **Una falla en la validación de denoising corta la corrida:** `valida` llama a
  `valida_denoising` sin `try`, a diferencia de la estabilidad y el ADMM (auditoría, 2026-10-16;
  verificado el 2026-10-20).
- **El punto de operación del ADMM es el del 2D:** `TARGET_STD = 0.06` (α = 4.86 en COSMOS),
  μ = 0.008, sin calibrar para la red 3D entrenada en T1 (2026-10-12).
- **Después del entrenamiento:** una parada del ADMM sin referencia (TOL/CORTE con un grid, o
  otra), con las trayectorias de `qsm.jsonl`; la escala de χ en QSM-TDV; `λ` en QSM, sin medir.
- **Si `wandb` queda como dependencia obligatoria** (hoy lo es en `pyproject.toml`) o pasa a extra.
- **`ITERACIONES_TANDA = 40 000` y `GUARDAR_CADA = 10 000`** (`train3d.py:80-81`) son los defaults
  de un piloto de tronco y ramas que nunca se lanzó; la corrida larga pasa los suyos por bandera.

---

## La corrida larga

### Configuración fijada

- **Máquina:** RTX PRO 6000 Blackwell Max-Q de 96 GB, entera (2026-10-15). `VRAM_GB = 96`
  (`train3d.py:88`): `auto` da micro-lote 16 con 57.06 GiB predichos, dos pasadas por iteración;
  con 8 serían 28.8 GiB y con 32, 113.5, que no entran. Son extrapolaciones de la 5070 Ti.
- **Micro-lote 16 contra 8, sin medir:** en la 5070 Ti, doblar el micro-lote bajaba el costo por
  parche 2–6 %. La sonda lo mide; si 16 no baja claramente el `s_por_parche` de 8, va
  `--micro-lote 8`. Sin `--checkpointing`: cuesta +22–27 % de tiempo (2026-10-15, 2026-10-16).
- **Receta:** `T_init = 8·δ_Adam` (0.1012, el default); σ = 25 fijo; sin `--aug-ganancia` (no
  arregla la estabilidad en COSMOS: «Lo que dejó `ventana-3d-12h`»); lr 4e-4 constante hasta que
  decida la meseta (2026-10-15, 2026-10-18).
- **Datos:** 2 000 cerebros (`lista_train_2000.txt`). Por paso, 32 cubos de 48³ independientes,
  cada uno de un volumen al azar, en una posición y con una de las 48 orientaciones del cubo. En
  100 000 iteraciones son 1 600 cubos por cerebro (2026-10-15).
- **`--mascara-cerebro`**, en el comando: cubos enteros dentro del cerebro. Al arrancar lee las
  máscaras a 0.06 s por volumen (~2 min) y las esquinas ocupan ~0.4 GB de RAM; exige las máscaras
  de los 12 volúmenes de validación. El precio es de receta: a 48³ los cubos cubren el 66 % del
  cerebro y el 18.5 % de la franja cortical (≤ 6 mm del borde); la red no ve el borde y el ADMM sí
  se lo pasa (2026-10-15, 2026-10-18).
- **Comando:** README, paso 6: `--iterations 100000 --hitos --meseta 10 --meseta-desde 10000
  --meseta-psnr --guard 10 --mascara-cerebro --seguimiento online --eval-every 100
  --eval-qsm-cada 500 --guardar-cada 2500`, como unidad `tdv-3d-largo` con `.venv/bin/python` y
  log en `runs/3d-largo.log` (2026-10-19).
- **Quién decidió qué:** el ADMM de 25 con parada tras dos subidas, la meseta desde la 10 000 con
  NRMSE y PSNR, `--guard 10` y el ~26 % como referencia, el autor (2026-10-19). La cadencia (PSNR
  cada 100, QSM cada 500) y la paciencia (`--meseta 10` = 5 000 iteraciones) son una recomendación
  sobre su propuesta (QSM cada 100 con `--meseta 10` = 1 000): las fija él al lanzar. Para QSM cada
  100 con la misma paciencia: `--eval-every 100 --meseta 50` (2026-10-19).
- **Seguimiento:** W&B, proyecto `training-tdv3d`, público (2026-10-18).
- **Estados:** uno cada 2 500 iteraciones (~47 MB cada uno). `config.json` guarda el commit del
  código, también sin W&B (2026-10-18).

### La meseta

- **Qué hace.** `--hitos` sola deja el lr constante; `--iterations 100000` es sólo el techo. Con
  `--meseta N`, si en N evaluaciones QSM seguidas no hay un mejor nuevo, lr/2 desde la iteración
  siguiente, hasta `--meseta-mitades` (5); agotadas, la meseta siguiente corta la corrida, y
  `estado.json` lleva `corte` con motivo, iteración, mitades e iteraciones hechas (2026-10-16).
- **Mira `nrmse_min_pct`, no el final.** Una evaluación con el mínimo no finito no cuenta; una que
  levanta excepción no llega a la meseta. Si fallan todas, el lr queda en 4e-4 hasta el techo: lo
  avisan «qsm FALLÓ» en el log y `qsm/fallo` = 1 (2026-10-18).
- **`--meseta-psnr`:** cada métrica lleva su mejor y su cuenta; la mitad pide que las dos lleven N
  sin mejorar, y las dos vuelven a cero al partir. Usa `val/psnr_cerebro` (2026-10-19).
- **`--meseta-desde 10000`:** antes no cuenta nada, y la primera evaluación desde ahí fija los dos
  mejores. Tiene que reiniciar el mejor: en 3 de las 4 corridas 3D con 3 o más evaluaciones, el
  mejor NRMSE mínimo fue el de la 250 (32.89, 33.22 y 32.89 %), y la rama de 2 500 no lo venció en
  2 250 iteraciones. `best_qsm.pth` sigue mirando todas las evaluaciones (2026-10-19).
- **La última evaluación no pasa por la meseta:** una corrida terminada que se extiende no cuenta
  esa evaluación (2026-10-18).
- **Los hitos de la meseta viajan en el estado:** una rama desde un estado posterior a una mitad la
  hereda. Para rehacer una mitad prematura, rama (otro `--out`) desde el estado anterior, con otra
  `--meseta` (las banderas de `CALENDARIO`) (2026-10-19).
- **Peor caso, si nada mejora desde la 10 000:** con QSM cada 500 y `--meseta 10`, mitades en la
  15 000, 20 000, 25 000, 30 000 y 35 000, y corte en la 40 000; con QSM cada 100 y `--meseta 10`,
  mitades de la 11 000 a la 15 000 y corte en la 16 000 (2026-10-19).

**Por qué esa paciencia** (2026-10-19):

- **Cuenta en iteraciones**, N × cadencia QSM. Monte Carlo con la tendencia subiendo 1σ cada 1 000
  iteraciones más ruido: en 5 000 iteraciones parten el lr por error 100 × 10 el 80 % de las veces,
  2 500 × 1 el 8 %, 100 × 25 el 1 %, y 500 × 10, 2 500 × 2 o 100 × 50 el 0 %. Exigir una mejora de
  1σ empeora (2 500 × 1 pasa al 27 %).
- **En 2D**, entre la 10 000 y la 24 000 a lr constante, «sin récord de PSNR» con 2 000
  iteraciones de paciencia habría disparado en 4 de 8 corridas con el PSNR todavía subiendo
  0.024–0.091 dB; con 4 000, en 1; con 6 000, en ninguna.
- **Pedir las dos métricas** baja las mitades falsas de 100 × 10 del 80 al 21 % (con ruidos
  independientes). En las series 3D cada 250 hasta la 2 500, la de sólo NRMSE con N = 2 habría
  partido en la 750, 1 250, 1 750 y 2 250 mientras el PSNR en el cerebro subía de 31.85 a 33.38 dB;
  la de las dos no disparó nunca.

### La evaluación QSM

- **El ADMM:** techo de 25 iteraciones (`QSM_MAX_ITER`, `train3d.py:129`; el `MAX_ITER` de
  `ddr/qsm3d.py` sigue siendo el protocolo del autor), sin corte por TOL/CORTE (`tol` = −inf) y
  parando si el NRMSE sube dos iteraciones seguidas (`QSM_SUBIDAS = 2`, motivo `empeora`). Pare
  como pare, devuelve el χ de menor NRMSE visto (2026-10-19).
- **En `qsm.jsonl`**, `nrmse_final_pct` y `nrmse_min_pct` son el mismo número salvo un ADMM no
  finito, y quedan la trayectoria del NRMSE y del update de cada iteración, para calibrar después
  una parada (2026-10-19).
- **Por qué 25.** Sobre 31 trayectorias 2D y pseudo-3D de 25 o más iteraciones, la correlación de
  Spearman del NRMSE en t contra el final da +0.07 en t = 1, +0.82 en t = 2, +0.93 en t = 10 y
  +0.98 en t = 24; en t = 24 queda una mediana de 0.12 pp sobre el mínimo. Con un par de
  iteraciones no se sabe: `ciego2d-35k` va 2.º en t = 5 y termina último de 5. En 3D ninguna de 31
  evaluaciones cortó con TOL/CORTE en 100 iteraciones, y el mínimo cayó en t ≤ 24 en 29
  (2026-10-19).
- **Por qué dos subidas y no una.** Sobre 26 trayectorias, parar en la primera subida da el mínimo
  de t ≤ 24 en 19 (en 6 que suben un paso y siguen bajando queda 0.03–0.37 pp arriba); en la
  segunda, en 25 de 26, con una iteración más. La que no: `sobreajuste-3d-control` (it 1 000), que
  sube 16 seguidas desde t = 3; la regla la deja en 38.82 % y en t = 24 llegaba a 37.80 %
  (2026-10-19).
- **Es una parada que mira la referencia:** sirve para comparar los checkpoints de una corrida.
  Cómo parar en un sujeto sin χ se calibra después (2026-10-19).
- **Máscaras** (las del solver de `../tdv`, `wTDV_QSM_torch_3d.py`): z_i = M·D_i(α·M·(χ + s_i))/α, el χ final es M·χ, con M la
  máscara dura, y el NRMSE se mide sobre M·χ. `punto_qsm3d` = α·M·χ₀, la entrada del primer
  denoiser (2026-10-18).
- **El port reproduce al autor:** con la red 2D oficial en los tres ejes, bf16 y sin compilar, da
  25.9393 % en 100 iteraciones contra 25.94 % de su log (el NRMSE por iteración, a ≤ 0.0092 pp);
  sin máscaras, 26.8898 % contra 26.89 % (2026-10-12, 2026-10-18).
- **Costo** (5070 Ti, `rapido`): entrenamiento 8.772 s/it; una evaluación, 22.6 s de validación,
  61.9 s de estabilidad (dos puntos) y 3.413 s por iteración de ADMM (85 s las 25). QSM y
  estabilidad cada 100 suman 19.4 % (~47 h en 100 000); PSNR cada 100 con QSM cada 500, 5.9 %;
  todo cada 500, 3.9 %. En la Blackwell no está medido (2026-10-19).

### Qué checkpoint va a dónde

- **`best_qsm.pth`, a QSM:** la evaluación de menor `nrmse_min_pct` entre todas, incluidas las de
  antes de la 10 000. El final, las iteraciones, el motivo y la estabilidad se guardan y no eligen.
  Con ~200 evaluaciones es optimista: COSMOS lo elige y lo mide (2026-10-18, 2026-10-19).
- **No por la estabilidad:** estable no es convergente (en `v12-control`, estabilidad 1.54 y ADMM
  en 144 % a las 100; en 2D, `ciego2d-35k` converge con 2.14). Y exigir que el ADMM cortara por
  TOL/CORTE podía dejar `best_qsm.pth` sin escribir: esos umbrales dependen de cada red (el grid de
  `../tdv` eligió tolUpdate de 0.3 a 1.5 para cinco redes 2D) (2026-10-18).
- **`best.pth`:** con `--mascara-cerebro`, por `val/psnr_cerebro`. El PSNR del recorte entero salta
  24.7–28.5 dB entre validaciones, porque la red no ve fondo ni cráneo (2026-10-16, 2026-10-18).
- **Referencias de NRMSE**, para reportar y no para decidir: la corrida se compara contra sí misma
  (el autor, 2026-10-19). Con el solver de `../tdv` y sus máscaras: 2D oficial, 26.45 % (tolUpdate 0.3) y
  26.57 % a 25 iteraciones; pseudo-3D oficial, 25.94 % a 100 y 26.08 % a 25, la comparable con la
  validación 3D (2026-10-18, 2026-10-19).

---

## El disyuntor en 3D

- **Cómo funciona aquí.** Rechaza el lote si ‖g‖ no es finita, supera la cota absoluta o supera
  `--guard` veces la mediana de las últimas 100 normas aceptadas; 25 rechazos seguidos
  (`GUARD_SEGUIDOS`) rebobinan al último respaldo (pesos, Adam por copia y ventana; el lr no viaja).
  El respaldo se refresca tras 250 iteraciones sanas: hasta la ~253 el único es el de la it 3, y un
  rebobinado ahí vuelve al arranque. Un rechazo no da el paso: una racha evalúa lotes nuevos con los
  mismos pesos, así que el nivel de ‖g‖ es del estado, no de un lote (2026-10-08, 2026-10-18).
- **`--guard 10`**, no está en `LIBRES` (`train3d.py:158`): una corrida lanzada con 5 se empieza
  de cero (2026-10-19).
- **Con razón 5** (dos corridas de 1 000 y `prep3d-punta`): 15 rachas de 10 o más rechazos en
  1 394 filas con el relativo armado, una cada ~93; ‖g‖ sube a 5–12× la mediana y se queda, con la
  pérdida normal. Todas las de 25 rebobinaron, y las corridas conservaron el 32 y el 22 % de sus
  iteraciones (2026-10-18).
- **Con razón 10** (`prep3d-g10-s1/s2` hasta la 750 y la rama de 2 500): una racha en 1 103 filas
  armadas (a 10.72–12.06×), y se conservó el 69 y el 91 %. En la rama, ningún rechazo después del
  arranque en 2 400 iteraciones, con iteración efectiva del 97.3 %. De 20 lotes aceptados a 5–10×
  en 3 103 filas armadas, uno solo (7.55×) precedió una racha que rebobinó. 10 no cubre todos los
  escalones: hubo de 12.1× y 12.4× (2026-10-19).
- **La cota absoluta queda en 42 622** = 10⁴·48³/(93²·3): el `GUARD_ABS = 10⁴` del 2D, que en BSDS
  cae en el hueco entre excursiones que se resuelven solas (≤ 6 113) y las que no (≥ 1.97·10⁵),
  escalado por los valores sumados por parche (×4.26). Con la razón armada, la razón rechaza todo
  lo que pase de ~3·10³, y la cota no actuó en ~5 000 filas: decide sólo en el arranque. Las 6
  salidas de arranque observadas pasaron por un lote aceptado de 1.38·10⁴ a 3.96·10⁴; con 2·10⁴ se
  habrían rechazado 4, con 10⁴ las 6. Sin escalar, el arranque sano la cruza (2026-10-08,
  2026-10-19).
- **El arranque.** Las primeras ~75 iteraciones rechazan ~50 lotes por la cota y rebobinan a la
  it 3 dos o tres veces; `--guard` no lo toca, porque el relativo no está armado. Sin cotas, el pico
  de la it 8 de la semilla 1 (‖g‖ 3.3–7.9·10⁵) se aceptó y la pérdida volvió sola (lote 4, un solo
  evento). Un respaldo tomado en el arranque lo repite: en `prep3d-mil-s2` una racha tapó el
  refresco de la 253 y el respaldo siguiente fue un estado de iteración efectiva 6. Si
  `train/iteracion_efectiva` no avanza en unos cientos de iteraciones, se relanza con otra semilla.
  El arranque de la larga (semilla 0, lote 32, máscara) no está observado (2026-10-18, 2026-10-19).
- **`rebobina()` restaura una copia del respaldo de Adam y conserva el lr vigente**: un segundo
  rebobinado al mismo respaldo devuelve Adam, y un rebobinado no deshace un halving (los dos con
  test). Es inerte sin rebobinados. El único rebobinado repetido al mismo respaldo ejercido en vivo
  fue en `v12-control` (2026-10-08, 2026-10-13).
- Con `--aug-ganancia` la cota absoluta no se reescala con la ganancia (2026-10-13).

## Lo que mostraron las sondas

- **La tubería aprende** (`prep3d-sobreajuste` y `prep3d-control`: `--sobreajuste`, lote 4 de 48³,
  lr constante, semilla 1, 1 500 iteraciones). El lote fijo sube de 31.86 a 33.59 dB entre la 250 y
  la 1 500 y su pérdida baja de 1 072.7 (no hacer nada vale 1 063) a 48.4, mientras su validación en
  cerebro queda en 31.18–31.63. El control, con datos frescos, llega a 32.18 en la validación en la
  1 500. Desde la 500: +1.50 dB en el lote, +0.37 en su validación y +0.79 en la del control
  (2026-10-18).
- **El NRMSE QSM no sigue al PSNR de forma monótona** (`prep3d-g10-s2-2500`, razón 10, lr
  constante, lista de 1 000). El PSNR en el cerebro sube en cada evaluación, de 31.85 dB en la 500
  a 33.38 en la 2 500. El NRMSE mínimo empeora de 35.29 % (500) a 35.82 % (1 250) y mejora en cada
  evaluación hasta 33.79 % (2 500), con pasos cada vez menores; no baja del 33.22 % de la 250. El
  final a 100 iteraciones pasa de 123 a 38 %: el ADMM diverge menos con el entrenamiento
  (2026-10-19).
- **Ruido de la evaluación:** con los mismos pesos, el mínimo es el mismo y el final difiere 0.04
  pp; una réplica de un tramo de 250 iteraciones con otro redondeo mueve el mínimo 0.09, el PSNR
  0.027 dB y el final 4.7 puntos. En las primeras 1 000 el mínimo fluctúa 2–4 puntos (2026-10-18,
  2026-10-19).
- **Reanudar da lo mismo.** En `rapido`, la fila siguiente al estado es idéntica en sus 14
  columnas; después se separan por redondeo, con lr e iteración efectiva idénticos. En CPU y
  `referencia`, bit a bit (`tests/test_train3d.py`). Una racha de rechazos no se repitió al
  reanudar: depende del redondeo (2026-10-16, 2026-10-18).
- **Misma semilla en `rapido`:** dos corridas son idénticas bit a bit hasta una decisión distinta
  del disyuntor o hasta que el no determinismo numérico las separa (en una, desde la it 77)
  (2026-10-19).
- **El ruido es el de la receta** (28.3 M vóxeles de lotes reales con máscara): desviación
  0.0980348 contra 25/255 = 0.0980392, gaussiano, sin correlación entre vecinos ni entre
  iteraciones, sin recorte, y bit a bit igual con cualquier partición del lote. La intensidad media
  de los cubos de cerebro es 0.316: σ es el 31 % de la señal (2026-10-16).

## Cuando el ADMM 3D diverge

- Se partió el error con la FFT 3D en el cono del dipolo, |D(k)| < √(μ/μ₂) = 0.0894 (21.7 % de
  k-space, donde el x-update pesa más el prior que el dato), y el resto, sobre 8 checkpoints de las
  sondas (it 500–1 500), 100 iteraciones sin corte (2026-10-19).
- Seis divergen: en el cono el error sube a 77–106 pp, en el resto se queda en 28–35 pp. El pico del
  error final es una onda de 4.7–6.8 vóxeles a 53.6–55.0° de B0, con |D| ≤ 0.015: el ángulo
  mágico, las rayas de las láminas. Converge sólo uno (31.45 % a 100); otro oscila con periodo 2.
  Donde el dato no mide queda lo que hace la red, y ahí el error crece iteración a iteración; la
  parada de dos subidas corta antes, en 5–15 x-updates (2026-10-19).

## El número de estabilidad PnP

- **Qué es:** `(T/S)·ρ(∇²R)`, el índice del paso explícito; un paso no expande en la dirección
  dominante si cae en (0, 2). Con el prox el jacobiano es `(I − (T/S)·∇²R)/(1 + λ/S)` y el umbral
  exacto `2 + λ/S`: el 2 redondo (`ddr/qsm3d.py:84`) se queda corto por < 0.7 % (2026-10-12).
- **Mide max|μ|, y el umbral 2 cubre sólo la curvatura positiva.** En el `best.pth` de
  `v12-control`, 1.84 con μ_min ≤ −90.6: un paso que expande ×1.99. En COSMOS dio 2.38–5.95 en
  todas las sondas. Se registra como diagnóstico (`qsm/estabilidad`) y no elige (2026-10-19).
- **Cómo se mide** (`ddr/qsm3d.estabilidad`): iteración de potencia con el producto
  Hessiano-vector exacto, 40 iteraciones, semilla 11, float32 IEEE con TF32 apagado. `lambda_max`
  es ‖Hv‖, el módulo del dominante; el cociente de Rayleigh, con su signo, va sólo al log (con dos
  dominantes de signo opuesto y módulo parecido no es un autovalor). Contra la Hessiana entera, a
  ~1e-15 con 200 iteraciones; con 40, el error crece con |μ₂|/|μ₁| (9.0e-3 con 0.98) (2026-10-12).
- **Dos puntos:** `val`, el cubo central de `--estabilidad-lado`³ (96³) del primer recorte de
  evaluación con su ruido; y `qsm`, `punto_qsm3d` recortado a un cubo centrado en la máscara.
  Depende del lado (con el modelo de humo, 0.85 a 32³ y 1.27 a 64³): se compara al mismo lado
  (2026-10-12).
- **Costo:** 31–40 s por punto a 96³; memoria ~9 GB, más que la evaluación QSM;
  `--estabilidad-lado 0` la apaga. No toca la aritmética (test) (2026-10-12, 2026-10-13).

---

## El modelo y la receta 3D

- **Port mecánico del 2D** (`ddr/conv3d.py`, `ddr/tdv3d.py`, `VNet3D` en `ddr/model.py`): kernels
  3³, el binomial `[1,4,6,4,1]/16` aplicado al kernel antes del stride 2, bajada y subida
  independientes. En float64 las adjuntas cierran a ~10⁻¹⁵ y `grad` contra autograd a ~4·10⁻¹⁶;
  1 162 114 parámetros con `T` y `λ` (3.0× el 2D en gris) (2026-10-08, 2026-10-16).
- **`train3d.py` importa la receta de `train.py`**: `LR`, `BETAS`, `DELTA_ADAM`, `T_INIT`,
  `LAMBDA_INIT`, `SIGMA`, `STEPS` y el disyuntor, como los mismos objetos (`tests/test_train3d.py`
  lo exige con `is`) (2026-10-12).
- **El lr es una función pura de la iteración** (`lr_en(it, hitos)`), reescrita en cada
  iteración: ni un respaldo ni un estado reanudado traen un lr viejo (2026-10-12).
- **Lo que la dimensión obliga a cambiar:** las 8 orientaciones del paper pasan a las 48 simetrías
  del cubo (LDM_100k es isotrópica a 1 mm; con el dipolo dejarían de valer); `rapido` por defecto;
  parche de 48³, el menor costo por vóxel medido (2026-10-08).
- **Lote 32**, el número de parches del paper, no su contenido: 3.54 M vóxeles por paso, 4.3× los
  830 k de BSDS color. Cuál lote entrena mejor por hora no está medido (2026-10-09).
- **El lote es receta; el micro-lote, hardware:** ninguna capa acopla muestras, así que el gradiente
  es el mismo para cualquier partición y sólo cambia el orden de suma (2026-10-09). Con σ fijo el
  ruido tampoco depende de la partición; con `--sigma-max` sí (2026-10-09, 2026-10-16).
- **Parches:** sin máscara, al menos 35 % de tejido (> 0.05); el parche `i` depende sólo de
  `(semilla, i)` (2026-10-08).
- **Candidatas implementadas y apagadas:** ruido ciego (`--sigma-max`), signo (`--aug-signo`) y
  ganancia log-uniforme (`--aug-ganancia G`). No implementadas: EMA y una etapa con el término de
  datos del dipolo, que es por donde pasaría entrenar contra el streaking (2026-10-08).
- **El calendario 2D no se traslada en números:** el mecanismo sí, pero un parche de 48³ tiene
  4.26× los valores de uno de 93²·3, y el modelo, 3.0× los parámetros (2026-10-12).
- **Traza de 14 columnas:** las 10 de `train.py`, más ‖g‖ del primer micro-lote (10), lr vigente
  (11), iteración efectiva (12) y σ medio del lote (13) (bloque «TRAZA» de `train3d.py`)
  (2026-10-09).

## Costo y memoria

| modo (5070 Ti) | parche × micro-lote | s | µs/vóxel | VRAM |
|---|---|---|---|---|
| `rapido` | 48³ × 1 | 0.334 | 3.02 | 4.14 GB |
| `rapido` | 48³ × 4 | 0.999 | 2.26 | 14.72 GB |
| `rapido` | 64³ × 1 | 0.619 | 2.36 | 8.78 GB |
| `referencia` | 48³ × 1 | 5.299 | 47.9 | 7.60 GB |

- **`referencia` cuesta ~21× `rapido`** por vóxel e `intermedio` 2.8×: en 3D `referencia` sirve
  sólo para tests cortos (2026-10-08). `torch.compile` sobre `R.grad` ahorra 35 % de tiempo y 38 %
  de memoria a 64³ (2026-10-08).
- **Medido en la 5070 Ti:** 8.8–9.3 s/it con lote 32 de 48³ y micro-lote 2–3 (2026-10-13,
  2026-10-19).
- **Modelo de memoria** (`vram_predicha`, bloque MEMORIA): `pico = 0.61 + n · P³ · 3.19·10⁻⁵` GiB
  en `rapido`, a < 2.2 % en la grilla de la 5070 Ti, sin la evaluación. `MARGEN = 0.10` es un
  supuesto: con micro-lote 3 y validaciones, 11.47 GiB contra 11.19 predichos; a 24³ × 8, 8.16
  contra 4.14 (2026-10-09, 2026-10-16, 2026-10-18).
- **Micro-lote libre:** cualquier entero hasta `--lote`; el lote se parte en trozos casi iguales, y
  con trozos iguales la corrida es bit a bit la de un divisor. `auto` toma el que deja menos
  pasadas: 16 en 96 GiB, 11 en 48, 3 en la 5070 Ti. La sonda mide 1–8, 11, 16 y 32 (2026-10-16).
- **RAM** (lista de 2 000): 25.7 GiB de pico al compilar la primera evaluación, 23.4–24.9
  entrenando. Hacen falta más de 32 GB, mejor 48. Con la lista de 1 000, 14.5 GiB (2026-10-18).
- **`WORKERS = 2`:** 26 ms de CPU los 32 parches de una iteración, contra ~9 s de GPU (2026-10-09).

## Datos: LDM_100k en 3D

- **Lo que hay:** 100 000 T1 sintéticas de 160×224×160 a 1 mm, alineadas, **con cráneo**, todas
  `_T1w` sin contraste invertido, sin metadatos útiles; la variedad se mide en las imágenes
  (2026-10-08, 2026-10-09). Detalle, en `datasets/ldm100k3d/README.md`.
- **Selección «mezcla»** (mitad k-center sobre miniaturas en espacio PCA, mitad sorteo): con 2 000
  cubre las colas mejor que un sorteo de 8 000. Los 12 de evaluación son el más cercano a cada
  centroide de un k-means y no comparten sujetos con las listas de entrenamiento. Ningún volumen se
  excluye por calidad (z máximo 8.1, umbral 10) (2026-10-08, 2026-10-16).
- **Evaluación de denoising:** recortes fijos de 128³ de los 12, con 88.5–92.7 % del cerebro
  adentro (2026-10-12).
- **Sin máscara, los parches no son cerebro:** sólo el 4.0 % de los de 48³ cae entero en el
  cerebro, y el parche medio es 60 % cerebro; el resto, cráneo, cuero cabelludo y aire (2026-10-09).
- **Máscara = deepbet ∩ HD-BET (sin TTA)**, con margen de 1 vóxel; Dice entre las dos 0.957 de
  mediana. ~0.67 s por cerebro en la 5070 Ti (`datasets/mascaras_ldm100k3d.py`) (2026-10-09,
  2026-10-12).
- **Qué ve el entrenamiento con cubos 100 % cerebro** («franja» = ≤ 6 mm del borde, 27.8 % del
  cerebro): a 24³, 85.5 % del cerebro y 49.6 % de la franja; a 32³, 80.1 y 38.1; a 48³, 66.3 y
  18.5. Con 56³ o más se pierde ≥ 37 % del cerebro (2026-10-09).
- **Cubos sueltos, no cruces:** una cruz de 7 cubos no entra en el cerebro a 32³ o más, y sus cubos
  se parecen entre sí lo mismo que dos sueltos del mismo cerebro (ICC 0.397 contra 0.387)
  (2026-10-12).

## Reanudación y ramas

- **El estado completo viaja en `estado-<it>.pth`:** pesos, Adam, ventana, respaldo, contadores,
  generadores, posición en los datos, filas de QSM y la meseta. Una rama es otro `--out` reanudado
  desde un estado de su tronco, con otro calendario (2026-10-08).
- **Bit a bit en CPU y `referencia`** (`tests/test_train3d.py`): 12 iteraciones seguidas contra
  6 + 6 y 9 + 3, con ruido ciego, aumentos y rebobinados a los dos lados del corte (2026-10-08).
- **Libres al reanudar** (`LIBRES`): cadencias, seguimiento, evaluación QSM, estabilidad,
  micro-lote y `vram_gb`. Cambiar el micro-lote avisa y sigue; cambiar el lote o una bandera de
  receta se rechaza; las de `CALENDARIO` cambian sólo en una rama (2026-10-09, 2026-10-12).

## Implementación de la evaluación y del seguimiento

- **`ddr/qsm3d.py`** es el ADMM del autor (`../tdv/wTDV_QSM_torch_3d.py`) con el TDV como denoiser,
  sobre COSMOS (160³; 933 481 vóxeles en la máscara): μ = 0.008, μ₂ = 1.0, `target_std` = 0.06
  (α = 4.8591). `admm_wtdv` acepta k denoisers: k = 3 es el pseudo-3D del autor, k = 1 la red 3D
  sobre el volumen entero (2026-10-12).
- **Dónde va:** en las validaciones múltiplo de `--eval-qsm-cada` y en la última, después de
  guardar `history.json`, `last.pth` y `best.pth`. Un OOM o una excepción del ADMM quedan como
  `qsm/fallo` = 1 y la corrida sigue. No toca la aritmética ni consume el generador del ruido
  (test, también compilado) (2026-10-12).
- **COSMOS** va en `datasets/qsm_cosmos/` (`params.mat`, `chi_cosmos.mat`); sin ellos, aviso y
  entrenamiento sin QSM (2026-10-12).
- **W&B** (`ddr/seguimiento.py`): `train/*` cada `--log-every` (con el `max|g|` del intervalo),
  `val/*` y `qsm/*` en cada validación, láminas, y el resumen igual a `estado.json`. Toda llamada va
  protegida: un error apaga el seguimiento sin cortar el entrenamiento, salvo `--seguimiento
  online`, que falla al arrancar. Tras un corte, reanudar abre otra corrida con el mismo nombre; una
  rama va en el grupo del tronco. Cuesta 0.19 ms por fila (2026-10-12, 2026-10-16).

## Hacia QSM-TDV

- **`../QSM-TDV`** (de otra persona; rama `direct-grad-dc-pytorch`) no tiene ADMM:
  reconstruye con pasos explícitos χ ← M[χ − (T/S)∇R(χ) − λ·Aᴴ W²(Aχ − b)] (2026-10-19).
- **No carga directo:** otras claves (`analysis.`, `macro_blocks.`, `energy_head.`). Con los 56
  tensores mapeados carga con `strict=True`, pero ∇R difiere 1.2–3.6 % en las caras del volumen:
  la bajada/subida de escala no es el mismo operador junto al borde. Con nuestra bajada/subida
  dentro de su modelo, ∇R coincide a 1.5e-16 (2026-10-12).
- **Opción (a)**, elegida el 2026-10-18: el entrenamiento no cambia; en QSM-TDV va una bandera que
  use la bajada y subida de `ddr` (sin escribir: el repositorio es de otra persona), más el
  conversor de claves, `a_qsm_tdv.py` en la raíz; sus scripts de validación, en
  `scripts/compat_qsm_tdv/`, locales (2026-10-18, 2026-10-19).
- **Pendiente, la escala de χ:** nuestro ADMM multiplica χ por α ≈ 4.86 antes de la red y QSM-TDV
  no; se resolvería pasando la fase y χ multiplicadas por α (derivado, no corrido). Además su W² es
  14.47 veces menor que el nuestro, y corre en float16, sin medir con nuestros pesos (2026-10-12,
  2026-10-19).

## Lo que dejó `ventana-3d-12h` (2026-10-13)

Dos brazos de 400 iteraciones, semilla 1, lote 32 de 48³, lr constante, 1 000 cerebros: control
contra `--aug-ganancia 4`.

- **El 3D corre de punta a punta en GPU** con la validación completa (PSNR, ADMM, estabilidad,
  W&B), sin fallos.
- **`--aug-ganancia 4` queda apagada:** baja la estabilidad en el punto de validación 5.8–7× pero la
  sube en COSMOS (3.63 y 3.29 contra 1.54 y 2.79), sin costo de PSNR visible (+0.09 dB). Una
  semilla y 400 iteraciones: sin veredicto.
- **Estable no es convergente:** en la 200 del control, estabilidad 1.54 en COSMOS y ADMM divergido
  a 144 %. El número mira un solo punto, la entrada del primer denoiser.

## Verificado del código 3D

- **Auditoría** (2026-10-16): `R.grad` contra autograd en float64 a 3.0e-16, Hessiano-vector a
  7.0e-16 y simétrico; `project_` deja K1 con media ~3e-7 y norma ≤ 1; la acumulación da el
  gradiente del lote a ≤ 1e-15 también con trozos desiguales; las 48 orientaciones son el grupo
  completo del cubo; el dipolo, α y la parada del protocolo son los del autor.
- **`train3d.py` ordenado** (2026-10-20): `main` arma la corrida en pasos con nombre y llama a
  `entrena`, cuyas partes son `gradiente_del_lote`, `aplica_disyuntor`, `registra_progreso`,
  `valida` (`valida_denoising`, `valida_estabilidad`, `valida_qsm` → `aplica_meseta`) y
  `guarda_estado`. Contra la versión anterior, en 33 casos de CPU y `referencia`: salida,
  JSON, traza y `.pth` iguales bit a bit. `--log-every 0` divide por cero.

---

## Lo que dejó el 2D

BSDS400 en color, validación en CBSD68 a σ = 25. Receta común (`train.py`): lote 32 de 93², S = 10,
Adam lr 4e-4 con β = (0.9, 0.999), lr/2 cada 25 000, `λ_init` = 0.1, `δ_Adam` = lr/√(1−β₂) =
0.01265, disyuntor por defecto. El ADMM 2D de QSM es el de `../tdv/wTDV_QSM_torch.py` contra COSMOS
(tres cortes adyacentes como RGB).

| corrida | modo | `T_init` | it | σ | mejor PSNR (it) | rech / reb | h |
|---|---|---|---|---|---|---|---|
| `t8delta-100k-s1` (la réplica) | referencia | 8·δ | 100 000 | 25 | **31.547** (100 000) | 163 / 4 | 37.7 |
| `t8delta-100k-s0` | referencia | 8·δ | 66 000, cortada | 25 | 31.526 (64 000) | 18 / 0 | — |
| `t2delta-100k-s1` | referencia | 2·δ | 100 000 | 25 | 31.536 (94 000) | 325 / 5 | 37.4 |
| `rapido-100k-s1` | rapido | 8·δ | 100 000 | 25 | 31.546 (100 000) | 80 / 2 | 12.6 |
| `t1p1delta-rapido-100k` | rapido | 1.1·δ | 100 000 | 25 | 31.525 (100 000) | 378 / 8 | 12.7 |
| `ciego2d-35k` | referencia | 8·δ | 35 000 (lr/2 en 25 000 y 30 000) | U(0, 55) | 31.428 (34 000) | 202 / 7 | 13.1 |

Kobler et al. publican 31.53 dB. Los checkpoints de `t8delta-100k-s0`, `-s1` y `ciego2d-35k`
están en `checkpoints/Entrenados/`.

**Receta y reproducibilidad**

- La receta del paper con `T_init` = 8·δ llega a 31.547 dB a 10⁵. Sobre el camino sano el
  disyuntor es inerte: las dos semillas repiten bit a bit su trayectoria anterior hasta la 18 000
  (2026-10-07).
- Una sd entre semillas vale 0.164 dB: con una semilla por brazo, lo mínimo detectable es
  ≈ 0.45 dB. Todas las diferencias de endpoint de esta tabla quedan por debajo (2026-10-11).
- La exposición varía 3.2·10⁶× entre dos semillas iguales (`max‖g‖` de 1.295·10¹⁰ en s1, atravesado
  con 4 rebobinados, contra 4 006 en s0): una tasa de eventos no se estima con n = 2 (2026-10-07).

**`T_init`: 8·δ, y bajarla cuesta rechazos, no PSNR** (2026-10-11, 2026-10-12)

- A 10⁵: 2·δ, −0.011 dB y 1.99× rechazos; 1.1·δ, −0.022 dB y 4.7×. Con 2·δ y 1.1·δ, el 96 y el 91 %
  de los rechazos y todos los rebobinados caen antes del primer hito; con 8·δ, el 2 %.
- Por qué: un pico mueve `T` según el lr vivo, y la reserva `T/δ_vivo` se duplica en cada hito: el
  riesgo de que `T` llegue a 0 está en la primera fase del lr. En 3D vale igual (mismas constantes).
- `T` final conserva el orden de `T_init` (2.23, 2.76, 6.53 y 6.83·δ): no hay un `T*` que la
  atraiga, ni un piso «2-δ».

**`rapido` ≈ `referencia`** (2026-10-10)

- A 10⁵: 31.546 contra 31.547 dB, 50 de 50 puntos pareados dentro de 0.05 dB salvo uno, 2.98× más
  rápido (12.6 contra 37.7 h) y 5.07 contra 9.94 GB. Una semilla; `rapido` no repite bit a bit.

**Gauge: `T` y `λ` sueltas no describen el modelo** (2026-10-12)

- `(T, K_N) → (cT, K_N/c)` deja el modelo igual. Entre las cuatro de 10⁵, `w = λ/T` va de 0.32 a
  3.87 mientras `T·‖∇R‖` queda dentro de 1.13×, y los cuatro `∇R` tienen coseno 0.978–0.987.
- Lo que separa es el modo: el cociente dato/regularizador vale 0.15–0.18 en `referencia` y
  0.031–0.039 en `rapido`, con los denoisers a 0.08 dB entre sí. `λ` es casi plana en denoising;
  en QSM, sin medir. El solver ADMM no lee `T` ni `λ`: usa la red entera, `vn(z·α, z·α)/α`.

**Disyuntor y calendario**

- Los rebobinados recuperan la corrida; descartaron el 0.4 % de las iteraciones de la réplica. Un
  lote rechazado ya pagó su forward y backward: 378 rechazos costaron +0.3 % de reloj (2026-10-08,
  2026-10-11).
- El tramo a 4e-4 tiene el 88–91 % de su ganancia hecho a la 14 000. Por tramo, en s1: +0.450 dB
  (4e-4), +0.064 (2e-4), +0.018 (1e-4) y +0.005 (5e-5); de 50 000 a 100 000, +0.030 dB en 18.8 h.
  Cambiar los hitos no toca nada antes del primero: una rama de calendario es el tronco hasta su
  hito (2026-10-07, 2026-10-08, 2026-10-12).

**Como prior de QSM: el PSNR no elige** (2026-10-12, 2026-10-17, 2026-10-19)

- ADMM 2D sin máscara: el oficial converge en 33 iteraciones a 27.34 %; los tres propios con
  σ = 25 no convergen en 100 (8·δ, 28.36 %; 1.1·δ, 28.47; 2·δ, 28.50); `ciego2d-35k` converge en
  33, a 33.18 %. Con las máscaras y la grilla del autor: oficial 26.45 %, 1.1·δ 27.75, 8·δ 28.12,
  2·δ 28.19, ciego 29.11.
- Lipschitz del denoiser: a σ = 25, 0.243 en los cuatro de σ fijo; a `target_std` 0.06, de 0.085
  (oficial) a 244.6. Coinciden donde se mide el PSNR y se abren 2 900× donde los usa QSM.
- Factor por el que un paso desenrollado agranda un error a la amplitud de QSM: oficial ×1.27, 2·δ
  ×1.77, ciego ×3.13, 1.1·δ ×6.74 y 8·δ ×23.9. A σ = 25 van de ×1.03 a ×1.54: ninguno es no
  expansivo. En el oficial y el ciego expande la curvatura negativa.
- Ni el índice ni el factor predicen la convergencia: el ciego converge con ×3.13 y 2·δ no con
  ×1.77. `target_std` no es la perilla: 0.06 es el óptimo de todos.

**El entrenamiento ciego** (`ciego2d-35k`, σ ~ U(0, 55) por parche; 2026-10-17)

- El índice en el punto QSM bajó de 23.2 (8·δ) a 2.14, sin cruzar 2 y con el dominante negativo;
  PSNR 0.064 dB bajo la réplica a la misma iteración.
- Corrige casi lo mismo a toda amplitud (0.44 a 0.35 entre std 0.015 y 0.25), y en el punto del
  solver mueve el dato un tercio menos que los de σ fijo: converge más fácil y deja más artefactos.
- Sin un control con σ = 25 y el mismo calendario (se preparó, `control2d-35k`, y no corrió), la
  baja del índice no se separa del largo ni del calendario.

---

## Corridas que quedan en `runs/`

Local, en la máquina del autor. Las sondas que no figuran se borraron: sus nombres en esta bitácora
son sólo rótulos de dónde salió un número.

| corrida | qué es |
|---|---|
| `t8delta-100k-s1`, `t8delta-100k-s0` | la réplica 2D (31.547 dB) y su segunda semilla |
| `t2delta-100k-s1`, `t1p1delta-rapido-100k` | `T_init` = 2·δ y 1.1·δ a 10⁵ |
| `rapido-100k-s1` | `rapido` contra `referencia` a 10⁵ |
| `ciego2d-35k` | 2D ciega al ruido, σ ~ U(0, 55) |
| `prep3d-sobreajuste`, `prep3d-control` | prueba de sobreajuste 3D y su control con datos frescos |
| `prep3d-punta`, `prep3d-reanuda` | las banderas de la larga a 200 iteraciones, y su reanudación desde la 100 |
| `prep3d-g10-s2-2500` | la única 3D terminada con `--guard 10`, la más larga (2 500), lr constante |

---

## 2026-10-20 — Limpieza al cerrar el 2D

- **Qué:** a pedido del autor, la bitácora se reescribió con sólo lo vigente para el 3D; salieron del
  repositorio las mediciones crudas, las fichas por corrida, las métricas, las presentaciones, la
  comparación 3D contra 2D, `replica-v4.pth` y los cortes 2D de T1 (su
  `normaliza` pasó a `datasets/cache_ldm100k3d.py`). El porqué de las constantes pasó a
  `docs/porque-del-codigo.md`. El código ya no cita commits ni archivos borrados.
- **Reglas:** la 3 y la 5 se adaptaron a una sola GPU y a corridas 3D de días; se agregaron la 11
  (sólo la GPU 1), la 12 (sólo los hilos 16–31) y la 13 (pocos commits, ninguno citado).
- **Fuera del repositorio (locales):** de `scripts/` quedan 15; los otros 108 y las listas de
  selección sin usar de `datasets/ldm100k3d/` se movieron a `~/Documents/Training-TDV-archivo/`.
- **Verificado al reescribir:** `valida` llama a `valida_denoising` sin `try`.
