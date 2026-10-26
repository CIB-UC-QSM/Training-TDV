# Bitácora

Lo vigente del entrenamiento 3D (`train3d.py`): estado, decisiones con el número que las sostiene,
lo medido que sigue valiendo y lo abierto. Las entradas nuevas van al final, fechadas, y cuando una
cambia algo de las secciones de arriba, se corrige la sección.

Reescrita el 2026-10-20, al cerrar el 2D: la historia de las fases anteriores no se conserva, y lo
que de ella sigue valiendo está aquí. Cada punto lleva entre paréntesis la fecha en que se midió o
se decidió. El porqué de cada constante del código está en
[`porque-del-codigo.md`](porque-del-codigo.md).

**SNR = pico/σ, con el pico = 1** (el percentil 99.5 de cada T1) desde el 2026-10-23. Lo anterior
usa la media del cerebro, 0.316/σ: SNR_pico = SNR_media/0.316 (3.2 → 10.2, 25 → 79, 50 → 158,
100 → 316). Las figuras del 21 y 22-10 en `docs/Sanity Check/` usan la vieja: su «SNR 25»
tiene 3.16 veces menos ruido que un SNR 25 por el máximo («Auditoría del ruido…», 26-10).

---

## Estado

- **La red es 3D real:** `VNet3D`, con convoluciones 3D (`ddr/conv3d.py`, `ddr/tdv3d.py`),
  entrenada con `train3d.py` sobre T1 3D. La evaluación QSM es el ADMM de
  `../tdv/wTDV_QSM_torch_3d.py` con k = 1: la red 3D sobre el volumen entero. No es el pseudo-3D
  del autor (k = 3, la red 2D en los tres ejes, sólo para reproducir su 25.94 %) ni el código de
  `../QSM-TDV`, que reconstruye con pasos de gradiente, sin ADMM («Hacia QSM-TDV»).
- **`train3d.py` está listo para la corrida larga**, que el autor lanza en la RTX PRO 6000
  Blackwell Max-Q de 96 GB, sin probar allí. Qué copiar, cómo comprobar y el comando: README,
  «Llevar el entrenamiento 3D a otra máquina». Lo fijado y por qué: «La corrida larga», abajo.
- **Lo más largo en 3D:** `ciego-epocas-l32-10k`, 10 000 iteraciones en la 5070 Ti local
  (terminada el 24-10; W&B `kxein1yw`, consolidada): ruido ciego log-uniforme, σ ∈ [0.81, 55]/255
  por parche, épocas, lote 32 y `--mascara-cerebro`. Su rama a 20 000,
  `ciego-epocas-l32-20k` (lr constante, W&B `4027tbsv`), se cortó a mano en la 15 875 por meseta;
  **en vuelo**, su rama a 30 000 desde la 14 880, `ciego-epocas-l32-30k`, con lr/2 y meseta
  (desde el 26-10 a las 11:19; W&B `ata3j1cj`; «La rama a 30 000»). Lo que
  dejaron las sondas cortas: «Lo que mostraron las sondas» y «El disyuntor en 3D».
- **Para QSM se usa `best_qsm.pth`**, no `best.pth`. En `../QSM-TDV` no carga tal cual: lo
  convierte `a_qsm_tdv.py`, inexacto junto al borde, y falta la escala de χ («Hacia QSM-TDV»).
  El `best_qsm.pth` de `ciego-epocas-l32-10k` es el de la 1 488, elegido a target 0.25: el
  checkpoint de la 10 000 le gana a todos los targets de 0.02 a 0.06 (3.7–7.2 pp en COSMOS;
  «Grid de target…», 25-10), así que ese archivo no sirve. El de la rama a 20 000 ya se elige al
  mejor target.
- **Los dos sujetos QSM son simulados.** La fase de COSMOS es D∗χ_COSMOS, con el mismo kernel
  del ADMM, más ruido complejo de SNR 100 por el máximo: un crimen inverso, optimista frente a
  una fase adquirida. Sim2 es del challenge 2.0 y trae desajuste de modelo («Auditoría del
  ruido y de los datos QSM», 26-10).
- **La fase 2D está cerrada**: `t8delta-100k-s1` llegó a 31.547 dB en CBSD68 (σ = 25) contra
  31.53 publicado. Lo que dejó para el 3D: «Lo que dejó el 2D», al final.

## Abierto

- **Antes de lanzar, en la Blackwell:** la sonda (micro-lote 16 contra 8), la prueba de 40
  iteraciones del README (paso 5) con las banderas finales, y el s/it y el costo de una evaluación.
- **La tasa de escalones de ‖g‖ con micro-lote 16:** con qué frecuencia hay escalones sobre 10×
  en 10⁵ iteraciones, y si aceptar los de 5–10× le cuesta algo al prior. En `ciego-epocas-l32-10k`
  (ciega con épocas, lote 32, lista de 1 000; `--guard 10` desde la 4 960): una racha de 25 que
  rebobinó en 10 000 iteraciones (la 9 768), sin `TRABADA`.
- **Después de la 10 000:** en la 10 000 de `ciego-epocas-l32-10k`, PSNR en el cerebro 33.787 dB,
  el mejor de la corrida; el QSM al mejor target, casi plano desde la 8 432 (varía ≤ 0.3 pp;
  COSMOS a 0.02, 26.73 %; Sim2 a 0.06, 35.49 %). La rama con lr constante llegó a la 15 875
  (RESULTADO en «La rama a 20 000»); la rama a 30 000 sigue con lr/2 y meseta. Sin medir: cuánto se
  correlacionan los ruidos del PSNR y del NRMSE, y si hay pozos del NRMSE más largos que la
  paciencia.
- **Si partir el lr mejora el NRMSE QSM:** ninguna serie QSM cruzó una mitad.
- **Si las trayectorias del ADMM de la red larga oscilan** como la de `sobreajuste-3d-control`: la
  parada de dos subidas le restaría ~1 pp a ese checkpoint. Con la ciega, `best_qsm` a 0.15 oscila
  (en Sim2, el NRMSE en diente de sierra), y ninguna trayectoria llega a un punto fijo en 100
  iteraciones («Los saltos finales», 23-10).
- **Por qué la red hace crecer el error en el cono del dipolo.** Lo agranda la control a
  target_std 0.06; lo achican la misma red a 0.25–0.35, la ciega a 0.06 y `prep3d-g10-s2-2500` a
  0.06. No lo explican el piso sin datos ni las 20 pasadas aisladas (las dos dan < 1 en todo); la
  medición que separa contraste de amplitud del error está diseñada y sin correr («Etapa 2»,
  pendientes). Si sigue pasando a 10 000 o más iteraciones, sin medir; y qué le hace al NRMSE que
  la red no vea el borde del cerebro.
- **Ruido de la corrida larga:** el comando del README se alineó el 25-10 con las corridas
  vigentes, ciega log-uniforme σ ∈ [0.81, 55]/255 con `--epocas`. Si el autor prefiere σ = 25
  fijo, se cambia. Lo medido: a 1 500 iteraciones, en COSMOS, la ciega no diverge a 0.06 y queda
  1.7 pp bajo la control a 0.25, sin costo de PSNR a σ = 25 (+0.37 dB); una corrida por brazo, un
  sujeto, y su λ quedó en 0 («Etapa 2», C) (2026-10-21). En denoising, la ciega (`ciego-log-l32`,
  2 250 it) deja 2.4–3.6 pp menos NRMSE que la de σ fijo de SNR de pico 79 a 316, 1.65 pp más a
  10.2, y desde 126 empeora la entrada («Ruido ciego log-uniforme», resultado) (2026-10-22). Con la
  ciega, el QSM al mejor target mejora con el entrenamiento: en COSMOS a 0.02, de 35.1 % en la 496
  a 26.7 % en la 10 000 («curvas por target», 23-10).
- **Una falla en la validación de denoising corta la corrida:** `valida` llama a
  `valida_denoising` sin `try`, a diferencia de la estabilidad y el ADMM (auditoría, 2026-10-16;
  verificado el 2026-10-20).
- **Un sujeto con fase adquirida:** COSMOS es un crimen inverso y Sim2 una simulación, así que
  ningún NRMSE de la evaluación dice cómo le va a la red in vivo. Falta una fase adquirida con
  referencia (por ejemplo, la del challenge 2016, con COSMOS o χ33) («Auditoría del ruido…»,
  26-10).
- **α sin GT:** la evaluación elige el target con la GT (α oráculo). No sirven una target_std
  robusta («La distribución de χ₀»), α por el ruido estimado de χ₀ (MAD de wavelets) ni el
  principio de discrepancia («¿α sin GT?»). Propuesta, sin correr: aumento de amplitud en el
  entrenamiento («¿α sin GT?», 23-10).
- **`--sigma-max` 85/255** en la próxima corrida ciega, para que la primera iteración del ADMM
  quede en el ruido entrenado hasta target 0.22 en COSMOS y 0.19 en Sim2: propuesta, sin correr
  («Barrido de target y ADMM a 100 con la ciega», 23-10).
- **El pseudo-3D oficial a target_std 0.06, 0.25 y 0.35 en COSMOS**, en `../tdv`: si α alto
  también mejora la oficial, es del ADMM y del cono; si no, de esta red. Lo decide el autor
  («Etapa 2», pendientes).
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
- **Receta:** `T_init = 8·δ_Adam` (0.1012, el default); ruido ciego log-uniforme por parche,
  σ ∈ [0.81, 55]/255 (`--sigma-min 0.81 --sigma-max 55 --sigma-log`), el de las corridas vigentes
  (2026-10-25; si el autor prefiere σ = 25 fijo, se cambia: «Abierto»); sin `--aug-ganancia` (no
  arregla la estabilidad en COSMOS: «Lo que dejó `ventana-3d-12h`»); lr 4e-4 constante hasta que
  decida la meseta (2026-10-15, 2026-10-18).
- **Datos:** 1 000 cerebros (`lista_train_1000.txt`), con `--epocas`: cada época son 31
  iteraciones de 32 cubos de 48³, 992 cubos de volúmenes distintos en el orden de una permutación
  (sin reemplazo), cada uno en una posición y con una de las 48 orientaciones del cubo. En 100 000
  iteraciones son ~3 200 cubos por cerebro (2026-10-25).
- **`--mascara-cerebro`**, en el comando: cubos enteros dentro del cerebro. Al arrancar lee las
  máscaras a 0.06 s por volumen (~2 min) y las esquinas ocupan ~0.4 GB de RAM; exige las máscaras
  de los 12 volúmenes de validación. El precio es de receta: a 48³ los cubos cubren el 66 % del
  cerebro y el 18.5 % de la franja cortical (≤ 6 mm del borde); la red no ve el borde y el ADMM sí
  se lo pasa (2026-10-15, 2026-10-18).
- **Comando:** README, paso 6, uno solo: `--iterations 100000 --hitos --meseta 10 --meseta-desde
  10000 --meseta-psnr --guard 10 --mascara-cerebro --epocas --train-list
  datasets/ldm100k3d/lista_train_1000.txt --sigma-min 0.81 --sigma-max 55 --sigma-log
  --seguimiento online --eval-every 100 --eval-qsm-cada 500 --guardar-cada 500 --qsm-target-std
  0.15 0.02 0.05 --qsm-sim2 datasets/qsm_sim2 --val-snr 25 50 100`, como unidad `tdv-3d-largo`
  con `.venv/bin/python` y log en `runs/3d-largo.log`. Reanudar es el mismo comando más
  `--reanudar` (2026-10-19, 2026-10-25).
- **Quién decidió qué:** el ADMM de 25 con parada tras dos subidas, la meseta desde la 10 000 con
  NRMSE y PSNR, `--guard 10` y el ~26 % como referencia, el autor (2026-10-19). La cadencia (PSNR
  cada 100, QSM cada 500) y la paciencia (`--meseta 10` = 5 000 iteraciones) son una recomendación
  sobre su propuesta (QSM cada 100 con `--meseta 10` = 1 000): las fija él al lanzar. Para QSM cada
  100 con la misma paciencia: `--eval-every 100 --meseta 50` (2026-10-19).
- **Seguimiento:** W&B, proyecto `training-tdv3d`, público (2026-10-18).
- **Estados:** uno cada 500 iteraciones (~47 MB cada uno, ~9.4 GB en 100 000), para reevaluar
  después cada checkpoint a otros targets. `config.json` guarda el commit del código, también sin
  W&B (2026-10-18, 2026-10-25).

### La meseta

- **Qué hace.** `--hitos` sola deja el lr constante; `--iterations 100000` es sólo el techo. Con
  `--meseta N`, si en N evaluaciones QSM seguidas no hay un mejor nuevo, lr/2 desde la iteración
  siguiente, hasta `--meseta-mitades` (5); agotadas, la meseta siguiente corta la corrida, y
  `estado.json` lleva `corte` con motivo, iteración, mitades e iteraciones hechas (2026-10-16).
- **Mira `nrmse_min_pct`, no el final;** con varios `--qsm-target-std`, el del target de menor
  NRMSE mínimo de esa evaluación (`qsm/mejor_target`). Una evaluación con el mínimo no finito no
  cuenta; una que levanta excepción no llega a la meseta. Si fallan todas, el lr queda en 4e-4
  hasta el techo: lo avisan «qsm FALLÓ» en el log y `qsm/fallo` = 1 (2026-10-18, 2026-10-25).
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

- **Targets** (el autor, 2026-10-25): `--qsm-target-std 0.15 0.02 0.05`, cada uno en W&B como
  `qsm_t<target>/*` (0.15 → `qsm_t015`); `qsm/*` lleva el de menor NRMSE mínimo, con
  `qsm/mejor_target`, y Sim2 igual (`qsm_sim2/*`). μ = 0.008 y μ₂ = 1, fijos; α = target_std /
  std(χ₀ en la máscara), fijo desde t = 0. El mejor depende del sujeto: 0.02 en COSMOS y
  0.05–0.055 en Sim2 («Grid de target 0.02–0.06, μ y ADMM en dos rondas»). La regla de target_std
  (la std de χ₀) no se cambia: ninguna variante robusta separa a los sujetos («La distribución de
  χ₀»), ni un α por el ruido estimado de χ₀ («¿α sin GT?»).
- **El ADMM:** techo de 25 iteraciones (`QSM_MAX_ITER`, `train3d.py:134`; el `MAX_ITER` de
  `ddr/qsm3d.py` sigue siendo el protocolo del autor), sin corte por TOL/CORTE (`tol` = −inf) y
  parando si el NRMSE sube dos iteraciones seguidas (`QSM_SUBIDAS = 2`, motivo `empeora`). Pare
  como pare, devuelve el χ de menor NRMSE visto (2026-10-19).
- **En `qsm.jsonl`**, `nrmse_final_pct` y `nrmse_min_pct` son el mismo número salvo un ADMM no
  finito, y quedan la trayectoria del NRMSE y del update de cada iteración, para calibrar después
  una parada (2026-10-19). Con varios targets, una fila por evaluación, la del target de menor
  NRMSE mínimo, con cada uno en `por_target` (2026-10-25).
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
  todo cada 500, 3.9 %. En la Blackwell no está medido (2026-10-19). Con tres targets en COSMOS
  y Sim2, el ADMM suma 134–159 s en COSMOS y 340–363 s en Sim2 por evaluación
  (`ciego-epocas-l32-20k`, `segundos` de `qsm.jsonl` y `qsm_sim2.jsonl`, sus dos primeras): ~8
  min, ~11 % sobre el entrenamiento con QSM cada 500, por cuenta (2026-10-25).

### Qué checkpoint va a dónde

- **`best_qsm.pth`, a QSM:** la evaluación de menor `nrmse_min_pct` entre todas y entre sus
  targets (el mejor de cada evaluación, `qsm/mejor_target`), incluidas las de antes de la 10 000.
  El final, las iteraciones, el motivo, el target y la estabilidad se guardan y no eligen. Con
  ~200 evaluaciones es optimista: COSMOS lo elige y lo mide (2026-10-18, 2026-10-19, 2026-10-25).
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

- **Cómo funciona aquí.** Rechaza el lote si ‖g‖ no es finita o supera `--guard` (10) veces la
  mediana de las últimas 100 normas aceptadas; sin cota absoluta (`--guard-abs inf`). 25 rechazos
  seguidos (`GUARD_SEGUIDOS`) rebobinan al último respaldo (pesos, Adam por copia y ventana; el lr
  no viaja). El primer respaldo se toma tras 3 iteraciones limpias; después se refresca cada 50
  iteraciones sanas (`--guard-respaldo`) y sólo con el relativo armado: antes, el único es el de
  la it 3. Dos rebobinados seguidos al mismo respaldo son una corrida `TRABADA`, que se avisa
  (log y alerta de W&B) y no se corta. Un rechazo no da el paso: una racha evalúa lotes nuevos con
  los mismos pesos, así que el nivel de ‖g‖ es del estado, no de un lote (2026-10-08, 2026-10-18,
  2026-10-23).
- **`--guard 10` es el default en 3D, y `--guard`, `--guard-abs` y `--guard-respaldo` están en
  `LIBRES`** (`train3d.py:169-173`): una corrida se puede reanudar con otro disyuntor, que es
  inerte mientras no rechaza (2026-10-23).
- **Con razón 5** (dos corridas de 1 000 y `prep3d-punta`): 15 rachas de 10 o más rechazos en
  1 394 filas con el relativo armado, una cada ~93; ‖g‖ sube a 5–12× la mediana y se queda, con la
  pérdida normal. Todas las de 25 rebobinaron, y las corridas conservaron el 32 y el 22 % de sus
  iteraciones (2026-10-18).
- **Con razón 10** (`prep3d-g10-s1/s2` hasta la 750 y la rama de 2 500): una racha en 1 103 filas
  armadas (a 10.72–12.06×), y se conservó el 69 y el 91 %. En la rama, ningún rechazo después del
  arranque en 2 400 iteraciones, con iteración efectiva del 97.3 %. De 20 lotes aceptados a 5–10×
  en 3 103 filas armadas, uno solo (7.55×) precedió una racha que rebobinó. 10 no cubre todos los
  escalones: hubo de 12.1× y 12.4× (2026-10-19).
- **Sin cota absoluta en 3D** (2026-10-23): sólo decidía antes de que se armara el relativo, y
  ahí trababa la corrida (el primer arranque de `ciego-log-l32` y el primer intento de
  `ciego-epocas-l32-10k`). Con el relativo armado no decidió nada en ~5 000 filas («Qué pasó con
  los estallidos»).
- **El arranque.** Hasta que se arma el relativo (100 aceptadas) sólo se rechaza lo no finito. Los
  picos de las primeras 100 iteraciones de las corridas ciegas, de 1.06·10⁴ a 4.8·10⁷, se
  aplicaron y se resolvieron solos (en la semilla 3 de `ciego-epocas-l32-10k`, 1.11·10⁵ hacia la
  75); también el de la it 8 de la semilla 1 (‖g‖ 3.3–7.9·10⁵, lote 4). Si
  `train/iteracion_efectiva` no avanza en unos cientos de iteraciones, se relanza con otra semilla
  (2026-10-19, 2026-10-23).
- **`rebobina()` restaura una copia del respaldo de Adam y conserva el lr vigente**: un segundo
  rebobinado al mismo respaldo devuelve Adam, y un rebobinado no deshace un halving (los dos con
  test). Es inerte sin rebobinados. El único rebobinado repetido al mismo respaldo ejercido en vivo
  fue en `v12-control` (2026-10-08, 2026-10-13).

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
  de los cubos de cerebro es 0.316: σ es el 31 % de la señal (SNR 3.2 de la media; 10.2 de pico)
  (2026-10-16).

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
  exacto `2 + λ/S`: el 2 redondo (`ddr/qsm3d.py:87`) se queda corto por < 0.7 % (2026-10-12).
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
- **En uso en las corridas vigentes:** ruido ciego log-uniforme (`--sigma-min 0.81 --sigma-max 55
  --sigma-log`) y `--epocas`. **Implementadas y apagadas:** `--fraccion-limpias`, signo
  (`--aug-signo`) y ganancia log-uniforme (`--aug-ganancia G`). No implementadas: EMA y una etapa
  con el término de datos del dipolo, que es por donde pasaría entrenar contra el streaking
  (2026-10-08, 2026-10-22).
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
- **Libres al reanudar** (`LIBRES`): el techo (`--iterations`), cadencias, seguimiento, evaluación
  QSM (también `--qsm-target-std`, `--qsm-sim2` y `--val-snr`), estabilidad, disyuntor (`--guard`,
  `--guard-abs`, `--guard-respaldo`), micro-lote y `vram_gb`. Cambiar el micro-lote avisa y sigue;
  cambiar el lote o una bandera de receta se rechaza; las de `CALENDARIO` cambian sólo en una rama
  (2026-10-09, 2026-10-12, 2026-10-23).

## Implementación de la evaluación y del seguimiento

- **`ddr/qsm3d.py`** es el ADMM del autor (`../tdv/wTDV_QSM_torch_3d.py`) con el TDV como denoiser,
  sobre COSMOS (160³; 933 481 vóxeles en la máscara). μ = 0.008 y μ₂ = 1, fijos (2026-10-25).
  α = target_std / std(χ₀ en la máscara), fijo desde t = 0; el default del protocolo es 0.06
  (α = 4.8591 en COSMOS), y la evaluación vigente corre varios targets («La evaluación QSM»).
  `admm_wtdv` acepta k denoisers: k = 3 es el pseudo-3D del autor, k = 1 la red 3D sobre el
  volumen entero (2026-10-12).
- **El NRMSE de QSM** es `100·‖χ − χ_COSMOS‖/‖χ_COSMOS‖` en la máscara de evaluación (χ_COSMOS ≠ 0,
  933 481 vóxeles; `nrmse_pct` en `ddr/qsm3d.py`): un RMSE dividido por el RMS de χ_COSMOS en esa
  máscara, que es 0.02800 ppm. Un 26 % es un RMSE de 0.00728 ppm (medido con `carga_cosmos`,
  2026-10-20).
- **Dónde va:** en las validaciones múltiplo de `--eval-qsm-cada` y en la última, después de
  guardar `history.json`, `last.pth` y `best.pth`. Un OOM o una excepción del ADMM quedan como
  `qsm/fallo` = 1 y la corrida sigue. No toca la aritmética ni consume el generador del ruido
  (test, también compilado) (2026-10-12).
- **COSMOS** va en `datasets/qsm_cosmos/` (`params.mat`, `chi_cosmos.mat`); sin ellos, aviso y
  entrenamiento sin QSM (2026-10-12).
- **W&B** (`ddr/seguimiento.py`): `train/*` cada `--log-every` (con el `max|g|` del intervalo),
  `val/*` y `qsm/*` en cada validación, láminas, y el resumen igual a `estado.json`. Toda llamada va
  protegida: un error apaga el seguimiento sin cortar el entrenamiento, salvo `--seguimiento
  online`, que falla al arrancar. Reanudar en el mismo `--out` reabre la misma corrida (id en
  `seguimiento.json`), salvo que haya registrado pasos posteriores al estado: entonces abre otra.
  Una rama abre otra en el grupo del tronco, con `padre_id`. Además: `qsm_t<target>/*`,
  `qsm/mejor_target`, `qsm_sim2/*`, `val_snr<SNR>/*` y alertas. Cuesta 0.19 ms por fila
  (2026-10-12, 2026-10-16, 2026-10-23, 2026-10-25).

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
- **Pendiente, la escala de χ:** nuestro ADMM multiplica χ por α = target_std / std(χ₀) antes de
  la red (en COSMOS, ~1.6 a 0.02, ~4 a 0.05 y ~12 a 0.15) y QSM-TDV no; se resolvería pasando la
  fase y χ multiplicadas por α (derivado, no corrido). Además su W² es 14.47 veces menor que el
  nuestro, y corre en float16, sin medir con nuestros pesos (2026-10-12, 2026-10-19).

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
| `prep3d-g10-s2-2500` | la 3D de σ fijo más larga (2 500), con `--guard 10` y lr constante |
| `etapa2-ciega` | la rama ciega de la etapa 2, σ ~ U(0, 55)/255, 1 500 iteraciones |
| `ciego-log-l32` | ciega log-uniforme, σ ∈ [0.81, 55]/255, sin épocas ni disyuntor, 2 250 iteraciones |
| `ciego-epocas-l32-10k` | ciega log-uniforme con épocas y lote 32, terminada en 10 000 (W&B `kxein1yw`) |
| `ciego-epocas-l32-20k` | su rama a 20 000, lr constante, cortada a mano en la 15 875 (W&B `4027tbsv`) |
| `ciego-epocas-l32-30k` | rama de la anterior desde la 14 880, lr/2 y `--meseta 2`, en vuelo (W&B `ata3j1cj`) |
| `prueba-disyuntor-s2` | la prueba del disyuntor 3D arreglado, 500 iteraciones |
| `humo-panel` | la prueba de humo del panel de W&B, 4 iteraciones |
| `denoising-snr`, `etapa2-pnp`, `snr-datos` | salidas de la etapa 1, de la etapa 2 y de «El SNR propio de los datos» |

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

## 2026-10-20 — Etapa 1: denoising de la red control por nivel de ruido

**Veredicto: de los siete niveles medidos, la red control con su T entrenado sólo mejora la
entrada al del entrenamiento, con o sin reescalado; bajando T en inferencia, la mejora en los
cuatro del brazo T (abajo).** A SNR 3.2 baja el RMSE de 0.0980 a 0.0246 (+12.0 dB). De SNR 25 a
100 lo deja en 0.0312–0.0317 haga lo que haga la entrada (0.0126 a 0.0032): 2.5 a 10 veces el de
entrada, ~30 dB de PSNR (SNR efectivo ~10), peor que a SNR 3.2.
Con el reescalado del paper la salida mejora (33.5, 35.7 y 43.5 dB) pero sigue peor que la entrada
(−4.5, −8.3 y −6.5 dB): «Con el reescalado del paper», abajo.

### Configuración

- **Redes:** `runs/prep3d-control/last.pth` (it 1 500, datos frescos) para el barrido;
  `runs/prep3d-sobreajuste/last.pth` (it 1 500) sólo para la figura del lote fijo.
- **Datos:** los 12 volúmenes de `lista_eval.txt` con `VolumeDataset3D`, recorte de 128 y las
  máscaras de cerebro de `mascaras/val`, como `arma_validacion`; y el lote fijo de `--sobreajuste`
  regenerado (los 4 primeros de `lista_train_2000.txt`, `PatchDataset3D` con seed 100 y máscaras,
  ruido del generador 101 en dos micro-lotes de 2).
- **Niveles:** σ = μ/SNR con μ = 0.316 (la media de los cubos de cerebro), salvo el nivel «3.2»,
  que es el σ = 25/255 del entrenamiento (con μ = 0.316, un SNR de 3.22). Semillas de
  `evaluate_denoising3d` (1234 + i por volumen), iguales en todos los niveles: el patrón de ruido
  es el mismo, escalado. SNR 40, 60 y 80 se sumaron después (2026-10-21, `--niveles-nuevos`, GPU 1,
  12.3 min), con el mismo código; antes repitió SNR 50, con una diferencia relativa máxima del
  RMSE de salida de 1.5·10⁻⁹ contra el guardado. La media medida del cerebro en
  los 12 recortes es 0.312.
- **Métricas del barrido:** en la máscara del cerebro, en float64: PSNR, RMSE y NRMSE de entrada
  y salida, y SNR efectivo de salida μ/RMSE. NRMSE = 100·‖x̂ − y‖/‖y‖ (`nrmse_pct`, el de la
  validación y de QSM); las figuras lo dan primero (el autor, 2026-10-21). En los resultados ya
  guardados sale del RMSE, 100·RMSE/rms(y) en los mismos vóxeles, sin volver a correr la red. Los chequeos se calculan como se registraron:
  `evaluate_denoising3d` para la validación, y el MSE del lote entero para el lote fijo.
- **Dispositivo y precisión:** GPU 1 (RTX 5070 Ti), `taskset -c 16-31`. El barrido va en fp32 sin
  TF32, la aritmética exacta del modelo; los chequeos, en bf16, como se registraron. En bf16 la
  media de los 12 queda 0.007 (SNR 3.2) a 0.026 dB por debajo de fp32 (por volumen, 0.006–0.029),
  sin crecer al bajar σ: la pérdida es de la aritmética interna, no del redondeo de la entrada.
  13.0 min la corrida, 4 s las figuras.
- **Script:** `scripts/denoising_por_snr_3d.py` (local); las salidas, en float16, en
  `runs/denoising-snr/` (`--figuras` las rehace sin correr la red).

### Chequeos de reproducción

| chequeo | medido (bf16) | registrado en `history.json` | tolerancia |
|---|---|---|---|
| red control, validación a σ = 25/255, PSNR en el cerebro | 32.183 dB | 32.183 | ±0.02 |
| red de overfit, lote fijo con su ruido | 33.594 dB (33.595 en fp32) | 33.592 | ±0.1 |

El del lote es aproximado por construcción: el registrado sale de la pérdida previa al último paso
de Adam de la it 1 500, y `last.pth` es posterior a ese paso.

### Medias de los 12 volúmenes de validación (fp32, en el cerebro)

| SNR | RMSE entrada (= σ) | RMSE salida [mín–máx] | salida / entrada | σ (0–255) | SNR ef. salida | PSNR entrada | PSNR salida | ganancia (dB) |
|---|---|---|---|---|---|---|---|---|
| 3.2 | 0.09804 | 0.02458 [0.02371–0.02552] | 0.25 | 25.00 | 12.9 | 20.172 | 32.190 | +12.018 |
| 25 | 0.01264 | 0.03124 [0.03085–0.03166] | 2.47 | 3.22 | 10.1 | 37.965 | 30.106 | −7.859 |
| 40 | 0.00790 | 0.03150 [0.03112–0.03195] | 3.99 | 2.01 | 10.0 | 42.047 | 30.035 | −12.012 |
| 50 | 0.00632 | 0.03156 [0.03118–0.03203] | 4.99 | 1.61 | 10.0 | 43.985 | 30.016 | −13.969 |
| 60 | 0.00527 | 0.03160 [0.03122–0.03207] | 6.00 | 1.34 | 10.0 | 45.569 | 30.006 | −15.563 |
| 80 | 0.00395 | 0.03164 [0.03126–0.03211] | 8.01 | 1.01 | 10.0 | 48.068 | 29.996 | −18.072 |
| 100 | 0.00316 | 0.03166 [0.03128–0.03213] | 10.02 | 0.81 | 10.0 | 50.006 | 29.991 | −20.015 |

- **El RMSE de salida no depende de la entrada desde SNR 25:** 0.0312–0.0317, con la entrada de
  0.0126 a 0.0032. Lo que queda es el error de la red sobre la anatomía (abajo), no el ruido.

- **El lote fijo con la red control** (4 cubos, PSNR entrada → salida): 20.17 → 33.16 a SNR 3.2,
  37.96 → 32.01 a 25, 42.05 → 31.95 a 40, 43.98 → 31.93 a 50, 45.57 → 31.92 a 60, 48.07 → 31.91 a
  80 y 50.01 → 31.91 a 100. No es fuera de muestra: sus 4
  volúmenes son también los 4 primeros de `lista_train_1000.txt`, con la que se entrenó la red
  control, y son cubos enteros de cerebro; eso puede explicar parte de los ~1 dB sobre la
  validación.

- A SNR 3.2, lo que la red quita («ruidosa − salida») sigue a la gaussiana N(0, σ) del ruido
  sumado (correlación 0.97 con él, en los 12); a SNR 25, 50 y 100 baja a 0.35, 0.18 y 0.09: lo
  que quita y lo que deja mal son anatomía, con la estructura de la sustancia blanca y los surcos.
- El sesgo medio de la salida en el cerebro es −0.0001 a SNR 3.2 y −0.0010 a SNR 25–100: no oscurece
  en conjunto. El histograma de «salida − limpia» es asimétrico, con la moda en negativo.
- En la figura del lote fijo se deshizo la simetría aleatoria con que entró cada cubo (una de las
  48), para que el corte sea axial.

### Tres semillas (2026-10-21)

Los siete niveles con otras dos semillas de ruido por volumen: la semilla s usa 1234 + 1000·(s − 1)
+ i en el volumen i (la 1 es la de arriba). `scripts/denoising_por_snr_3d.py --semilla 2|3`, GPU 1,
fp32 sin el contraste bf16, 17.3 y 17.5 min; `runs/denoising-snr/seed2|seed3/`. El resumen:
`--resumen-semillas`, `docs/Sanity Check/resultados-control/21-10-resumen_semillas.png` y
`resumen_semillas.json`.

| SNR | RMSE entrada | RMSE salida, media de las 3 | rango entre semillas | salida / entrada |
|---|---|---|---|---|
| 3.2 | 0.09803 | 0.02459 | 0.024582–0.024607 | 0.25 |
| 25 | 0.01264 | 0.03124 | 0.031241–0.031242 | 2.47 |
| 40 | 0.00790 | 0.03150 | 0.031497–0.031498 | 3.99 |
| 50 | 0.00632 | 0.03156 | 0.031564–0.031565 | 4.99 |
| 60 | 0.00527 | 0.03160 | 0.031601–0.031602 | 6.00 |
| 80 | 0.00395 | 0.03164 | 0.031639–0.031639 | 8.01 |
| 100 | 0.00316 | 0.03166 | 0.031656–0.031657 | 10.02 |

- **La semilla no mueve la media de los 12:** a SNR 3.2 el rango es 0.1 % del RMSE de salida, y de
  SNR 25 a 100, ≤ 0.003 %. El ruido es otro (la salida de un mismo volumen cambia hasta 0.21 en
  un vóxel entre semillas), pero con ~10⁶ vóxeles por volumen y 12 volúmenes la media no lo ve. La
  dispersión que importa para comparar redes es la de entrenamiento, no la del ruido de
  evaluación.
- **En NRMSE** (lo que muestran las figuras), entrada → salida, media de las 3: 29.88 → 7.50 % a
  SNR 3.2, 3.85 → 9.52 % a SNR 25 y 0.96 → 9.65 % a SNR 100; los siete niveles, en
  `resumen_semillas.json` y en la figura.

### Figuras

Una carpeta por red, y en la de la control la de la semilla de ruido 1 (las otras dos dan lo
mismo: «Tres semillas», abajo), `docs/Sanity Check/resultados-control/seed1/`:
- `21-10-denoising_snr_3.2.png`, `…_25.png`, `…_40.png`, `…_50.png`,
  `…_60.png`, `…_80.png` y `…_100.png`: el volumen `sub-028881` (el de PSNR de salida mediano a
  SNR 3.2 con la semilla 1) en sus tres cortes centrales, con la tabla (el
  NRMSE primero, y el RMSE al lado) y el histograma de cada nivel;
- `21-10-grilla_snr_axial.png`: los siete niveles lado a lado, corte axial
  central del mismo volumen, con la limpia una vez de referencia y el NRMSE y el PSNR medios de los
  12 en cada columna;
- `21-10-grilla_snr_reescalado_axial.png` (lo mismo con el reescalado del paper)
  y `21-10-ganancia_vs_T.png` (la ganancia contra el T de inferencia, brazo T);
- en `resultados-control/`, `21-10-resumen_semillas.png`: el NRMSE por nivel con las tres semillas;
- `docs/Sanity Check/resultados-ciego-log-l32/`: lo mismo para la ciega log-uniforme en la 2 250 (fecha
  22-10; «Ruido ciego log-uniforme», resultado);
- `22-10-convergencia.png` en `resultados-control/`, `resultados-ciego-log-l32/` y
  `resultados-g10-s2-2250/`: pérdida, validación a σ = 25, λ y T contra la iteración;
- `docs/Sanity Check/resultados-sobreajuste/21-10-prueba_overfit_sigma25.png`: la red de overfit sobre
  su lote fijo.

**Convenciones de visualización**, las de `imshow_3d` de `../tdv` y de las láminas QSM de COSMOS:
gris, aspecto 1:1 y los tres cortes centrales; sagital con la cara a la izquierda, coronal y axial
con lo superior y la frente arriba, en convención radiológica (la derecha del paciente a la
izquierda: LDM_100k está en LAS según el affine del NIfTI); fuera del cerebro, en 0 (negro en el T1,
gris medio en las diferencias). El T1 va en [0, p99.9 de la limpia en el cerebro], la misma ventana
para limpia, ruidosa y salida (0.51 en `sub-028881`). Las diferencias, con una escala común de
±0.1 en las figuras de nivel y en la grilla: ~3 veces el mayor RMSE de salida (0.032), así que el
error de salida no satura en ningún nivel, y «ruidosa − salida» a SNR 3.2 (σ = 0.098) pasa de la
escala en el 29 % del cerebro. La del lote fijo, de un solo nivel, va a ±3σ. Las figuras se rehacen
con `--figuras` (o `--figuras niveles grilla overfit`, por partes) sin correr la red.

### Con el reescalado del paper (2026-10-21)

Kobler et al. (2020, §5.2) no aplican una red de σ = 25 tal cual a otro ruido: reescalan la entrada
por 25/σ y la salida por σ/25 (en 2D, `evaluate.py --sigma-ref`). En estas unidades, s =
(25/255)/σ: la red ve s·z y devuelve D(s·z)/s; las métricas, como siempre, en el cerebro contra la
limpia. Red control, los mismos 12 volúmenes y semillas, fp32, GPU 1; 7.2 min con
`scripts/denoising_por_snr_3d.py --reescalado`. A SNR 3.2, s = 1: es el barrido de arriba.

| SNR | s | PSNR entrada | PSNR salida [mín–máx] | ganancia | RMSE salida | SNR ef. salida | salida sin reescalar |
|---|---|---|---|---|---|---|---|
| 25 | 7.76 | 37.965 | 33.469 [33.16–33.79] | −4.496 | 0.02122 | 14.9 | 30.106 |
| 50 | 15.51 | 43.985 | 35.666 [35.29–36.24] | −8.319 | 0.01648 | 19.2 | 30.016 |
| 100 | 31.03 | 50.006 | 43.537 [42.91–44.31] | −6.469 | 0.00666 | 47.5 | 29.991 |

- El reescalado saca la salida del piso de ~30 dB (+3.4, +5.7 y +13.5 dB sobre la de sin
  reescalar), pero en ningún nivel la deja mejor que la entrada.
- La ganancia no es monótona con el ruido: −4.5 a SNR 25, −8.3 a 50 y −6.5 a 100.
- **La correlación de lo que la red deja mal (salida − limpia) con el ruido sumado**, en el
  cerebro de los 12 (`--correlacion`, sin GPU): +0.27 a s = 7.76, +0.15 a 15.5 y +0.30 a 31; sin
  reescalar, +0.03, +0.01 y +0.01 (ahí el error es anatomía), y a SNR 3.2, +0.27. En las unidades
  que ve la red, el RMSE de salida es 0.25, 1.68, 2.61 y 2.11 veces σ₀ = 25/255 (SNR 3.2, 25, 50 y
  100). La no monotonía admite dos lecturas; que a s = 31 la correlación vuelva a subir respecto de
  s = 15.5 apoya la de que, a contraste muy alto, la red deja sin tocar el ruido de los bordes
  fuertes. Hipótesis, no medida: la Student-t es redescendente, y ante contrastes muy altos actúa
  menos.
- **Ojo con la lectura:** s = 7.76–31.0 queda muy lejos del rango del paper (s de 0.5 a 1.67), y la
  red no es escala-equivariante. Con s·z la red ve su ruido de siempre, pero con las diferencias
  entre tejidos multiplicadas por s, hasta 31 (la red no ve el nivel medio: K₁ tiene media cero y
  una constante no entra): si falla a s = 31, puede ser por ese contraste fuera de rango y no
  porque el reescalado no sirva.
- Figura: `docs/Sanity Check/resultados-control/seed1/21-10-grilla_snr_reescalado_axial.png`.

### Barrido de s: la ventana de contraste (2026-10-21)

**PRE-REGISTRO**, escrito antes de lanzar. El reescalado del paper barriendo s, con la red viendo
siempre su ruido σ₀ = 25/255: z = y + n/s con el mismo n por volumen (ruido pareado), la red ve
s·z = s·y + n y devuelve D(s·z)/s; ganancia = PSNR de salida − de entrada, en el cerebro. s ∈
{0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 31} (SNR = 3.22·s; s < 1 es más ruido que el del
entrenamiento). Red control, los 6 volúmenes del brazo T, fp32, GPU 1 (~12 min).
- **Pregunta:** en qué ventana de s la red mejora la entrada (ganancia > 0), contra el rango del
  paper, de 0.5 a 1.67.
- **Lo que ya se sabe:** +12.0 dB a s = 1 y −4.5, −8.3 y −6.5 a s = 7.76, 15.5 y 31 (arriba).
- **Comparación con el ADMM:** el s que equivale a cada target_std es target_std / std(y en el
  cerebro), con la std de los 6 volúmenes limpios: la red ve en los dos casos una imagen de esa
  desviación. Es una equivalencia de amplitud, no de contenido (χ no es T1).
- Descriptivo. Figura `21-10-ganancia_vs_contraste.png`: s en escala logarítmica, ganancia en dB,
  una línea en 0, el rango del paper sombreado y los target_std marcados.

**RESULTADO del barrido de s: la red control mejora la entrada hasta s ≈ 3.5, el doble del rango
del paper.** `scripts/denoising_por_snr_3d.py --barrido-s`, GPU 1, fp32, 15.5 min;
`runs/denoising-snr/barrido_s.json`.

| s | 0.5 | 0.75 | 1 | 1.5 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 | 31 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ganancia (dB) | +14.9 | +13.2 | +12.0 | +9.7 | +6.9 | +1.6 | −1.1 | −3.4 | −4.7 | −7.1 | −8.4 | −7.9 | −6.4 |

- **La ventana:** ganancia positiva de s = 0.5 a 3 (en los 6 volúmenes) y negativa desde s = 4
  (también en los 6); el cero cae entre 3 y 4. Cubre todo el rango del paper (0.5–1.67), donde la
  red gana 9.7–14.9 dB.
- **Repite lo de arriba** a s = 1, 7.76, 15.5 y 31 (12.0, −4.5, −8.3 y −6.5 dB con 12 volúmenes),
  con el mínimo en s = 16 y la subida hasta 31.
- **El s que equivale a cada target_std, por amplitud** (std de y en el cerebro 0.104): 0.02 → 0.19,
  0.06 → 0.58, 0.25 → 2.41, 0.35 → 3.37. Si la equivalencia valiera, 0.06 estaría en el centro de
  la ventana y 0.25–0.35 en su borde, y en el ADMM es al revés: 0.06 es el peor. La amplitud sola
  no ordena el ADMM. Lo que la red ve ahí no es imagen más ruido gaussiano: es χ con el error que
  el dato no corrige, sobre todo el del cono.

### Brazo T: T en inferencia (2026-10-21)

**PRE-REGISTRO del brazo T**, escrito antes de lanzar (regla 5; las predicciones son del Agente
teórico, corregidas con el coordinador). T es la duración del flujo (τ = T/S se recalcula en
`forward`). Sin reescalar, con la red control, se fija `model.T` en:

- a SNR 3.2: T ∈ {0.05, 0.075, 0.125, 0.15};
- a SNR 25, 50 y 100: T ∈ {0.0016, 0.0031, 0.0063, 0.0125, 0.025, 0.05}, de 0.0125 hacia abajo
  dividiendo por 2, para ver si el óptimo baja con el ruido.

El T entrenado no se recalcula: son los valores por volumen del barrido de arriba. T = 0
es la identidad (x₀ = z y el prox de z hacia z es z, `ddr/model.py:104`), con ganancia 0: se
comprueba en un volumen. Volúmenes: 6 de los 12 de validación, entre ellos `sub-028881` (los
índices 0, 3, 5, 7, 9 y 11 de `lista_eval.txt`). Cada volumen usa la misma realización de ruido en
todos los T (semilla 1234 + i, como `evaluate_denoising3d`): la diferencia entre dos T es sólo
efecto de T, la comparación es pareada, y por eso alcanzan 6.

- **(a)** A SNR 3.2, el T de mayor ganancia cae en {0.075, el entrenado, 0.125}, a un paso del
  entrenado como mucho. Si cae en 0.05 o en 0.15, falla.
- **(b)** T* a SNR 100 ≤ T* a 50 ≤ T* a 25 < T entrenado.
- **(c)** En cada nivel hay algún T > 0 con ganancia estrictamente positiva.

Si (b) falla, se descarta la hipótesis de que el flujo se pasa de largo (que corre un tiempo T
ajustado para llevar una entrada a σ = 25 hasta la limpia, y que si parte casi limpia alisa). Si en
algún nivel el mejor T es 0, el T no basta como perilla a ese nivel. Para la lectura: con
T mayor que el entrenado crecen el paso τ = T/S y el índice de estabilidad (T/S)·ρ(∇²R) en la misma proporción;
si a 0.15 la salida se degrada mucho o diverge, es un dato de estabilidad, no un fallo del barrido.

**Una corrección al pre-registro:** el T entrenado de la red control es 0.0963 (`control.T` del
checkpoint), no el 0.1003 con que se escribió. El barrido leyó el valor del checkpoint, y las
predicciones se leen con él; el conjunto de (a) es {0.075, 0.0963, 0.125}.

**RESULTADO del brazo T.** `scripts/denoising_por_snr_3d.py --barrido-t`, GPU 1, fp32, 26.3 min.
Volúmenes: `sub-004207`, `sub-028881`, `sub-047377`, `sub-054205`, `sub-064462` y `sub-096471`.
T = 0, comprobado en `sub-028881` a SNR 25: |salida − entrada| máx 6.0·10⁻⁷ (un ULP de float32) y
ganancia −3·10⁻⁷ dB. Ganancia media de los 6 (dB, en el cerebro); en negrita, el óptimo de cada
nivel; la columna 0.0963, de la etapa 1:

| SNR | 0.0016 | 0.0031 | 0.0063 | 0.0125 | 0.025 | 0.05 | 0.075 | 0.0963 | 0.125 | 0.15 |
|---|---|---|---|---|---|---|---|---|---|---|
| 3.2 | | | | | | +10.07 | +10.97 | **+11.98** | +11.01 | +10.33 |
| 25 | +0.39 | +0.70 | +1.20 | **+1.58** | +0.67 | −3.05 | | −7.87 | | |
| 50 | +0.36 | **+0.53** | +0.44 | −0.88 | −4.27 | −9.14 | | −13.98 | | |
| 100 | **+0.14** | −0.25 | −1.93 | −5.39 | −10.03 | −15.19 | | −20.03 | | |

- **(a) se cumple:** a SNR 3.2 el óptimo es el T entrenado. El pico es angosto, pero no es un
  artefacto de la grilla: en `sub-028881`, con el mismo código, la ganancia es +11.82, +12.02,
  +12.06, +12.03, +11.91 y +11.71 dB a T = 0.085, 0.09, 0.0963, 0.10, 0.105 y 0.11, y el valor en
  0.0963 es el de la etapa 1 para ese volumen.
- **(b) se cumple:** T* = 0.0016 (SNR 100) ≤ 0.0031 (50) ≤ 0.0125 (25) < 0.0963. El de SNR 100 cae
  en el borde de la grilla: el óptimo puede ser menor.
- **(c) se cumple:** en cada nivel hay un T > 0 con ganancia positiva: +11.98, +1.58, +0.53 y +0.14
  dB. Son mejoras chicas a SNR 25–100.
- No se descarta que el flujo se pase de largo: el óptimo baja con el ruido, y con el T entrenado
  la ganancia cae ~6 dB por cada duplicación del SNR. A SNR 3.2, T = 0.15 da +10.33 dB: la salida no
  diverge (dato de estabilidad).
- Figura: `docs/Sanity Check/resultados-control/seed1/21-10-ganancia_vs_T.png`.

**Lo que dice y lo que no.** Con menos ruido la salida empeora (RMSE 0.0317 a SNR 100 contra
0.0246 a SNR 3.2); un filtro lineal fijo no haría eso, porque su error no sube cuando baja σ. Que
quede peor que la entrada, en cambio, lo hace cualquier suavizado demasiado fuerte. Hipótesis: el
flujo corre un tiempo T ajustado para llevar una entrada a σ = 25 hasta la limpia, y si
parte casi limpia se pasa de largo y alisa (Kobler et al., 2020, Fig. 4). Con T = 0 la salida es la
entrada (`ddr/model.py:104`), así que algún T evita el daño (como mínimo, T = 0); que el T
entrenado sea demasiado largo para un ruido menor lo apoya el brazo T: el óptimo baja con el ruido
y con él la ganancia vuelve a ser positiva en los cuatro niveles. A σ =
25, el error que queda se concentra en bordes y detalle: si es límite del modelo o del ruido, está
abierto (la red de overfit seguía subiendo, +0.57 dB en sus últimas 250 iteraciones). Sobre el
ciego: la evidencia de que una red de σ = 25 cubre otros niveles con el reescalado es 2D y con s de
0.5 a 1.67 (`checkpoints/Entrenados/README.md`; Kobler et al., Tablas 2–3); con s de 7.8 a 31 el
reescalado no alcanzó. Para SNR 25–100 entrenar ciego vuelve a tener motivo, y sigue sin medir;
el 33.18 % contra 27.34 % del 2D compara `ciego2d-35k` contra el oficial, no contra un control. Es una red de 1 500 iteraciones.

## 2026-10-21 — El SNR propio de los datos

**Pregunta (del autor):** qué SNR tienen de verdad los T1 de LDM_100k, para entrenar con uno
realista. `scripts/snr_datos.py`, CPU; `runs/snr-datos/snr_datos.json`. Se mide en el
cerebro (la máscara erosionada 2 vóxeles), en las unidades de la caché (cada volumen dividido por
su percentil 99.5, en [0, 1], float16), con dos estimadores robustos de σ:
- **haar:** la MAD de los coeficientes HHH de una wavelet de Haar 3D de escala fina, / 0.6745;
- **vecinos:** la MAD del residuo x − (media de los 6 vecinos), / 0.6745 / √(7/6).

SNR = media del cerebro / σ. En χ, RMS(χ) / σ, sólo como orientación: χ tiene media ~0 y no hay una
intensidad de fondo con qué compararlo.

| datos | volúmenes | media del cerebro | σ haar | SNR haar | σ vecinos | SNR vecinos |
|---|---|---|---|---|---|---|
| LDM_100k, validación | 12 | 0.321 [0.299–0.348] | 0.0041 [0.0038–0.0046] | 75 [67–91] | 0.0083 [0.0075–0.0094] | 38 [33–46] |
| LDM_100k, entrenamiento | 50 (uno de cada 20 de `lista_train_1000.txt`) | 0.326 [0.263–0.397] | 0.0040 [0.0033–0.0045] | 81 [63–104] | 0.0080 [0.0068–0.0094] | 42 [32–51] |
| χ COSMOS (`mascara_eval`) | 1 | RMS 0.0288 ppm | 0.0037 | 7.8 | 0.0053 | 5.4 |
| χ Sim2 (`mask_final`) | 1 | RMS 0.0289 ppm | 0.0022 | 13.1 | 0.0033 | 8.7 |

Medianas, con mínimo y máximo entre corchetes.

- **Los dos estimadores no coinciden, y por eso ninguno es «el σ del ruido».** Sobre ruido blanco
  conocido coinciden (σ 0.01 → 0.0100 y 0.0100; `chequeo` del script). En los T1, haar da la mitad
  que vecinos (razón 0.51), como un ruido suavizado: uno gaussiano de 0.7 vóxeles, de σ 0.01, da
  0.0027 y 0.0046 (razón 0.58). Lo que varía en la escala de un vóxel no es ruido blanco: es ruido
  espacialmente correlacionado, anatomía fina, o las dos cosas. Con ruido correlacionado los dos
  estimadores subestiman σ; con anatomía, la sobreestiman.
- **Lo que sí dice:** como ruido blanco equivalente en la escala de un vóxel, los T1 están en SNR
  ~40–80, contra 3.2 del entrenamiento (σ = 25/255 = 0.098, 12–24 veces la variación propia). El
  piso de float16 (0.7–1.4·10⁻⁴ en [0.25, 1)) queda muy debajo. LDM_100k es sintético (lo decodifica
  un modelo de difusión latente), y no es seguro que su variación fina se parezca a la de un T1
  adquirido.
- **Para entrenar a SNR 50** (σ = 0.316/50 = 0.0063): ese σ cae entre las dos estimaciones de la
  variación propia (0.0040–0.0083). La «limpia» del entrenamiento trae ya una variación del mismo
  tamaño que el ruido que se suma. Lectura, no medida: la red aprendería a quitar el ruido sumado y
  a conservar esa variación, sea anatomía o ruido.
- **χ:** RMS/σ de 5–8 en COSMOS y de 9–13 en Sim2. COSMOS, reconstruido de datos adquiridos, tiene
  más variación fina relativa que Sim2, simulado. (La fase de COSMOS, en cambio, es simulada:
  «Auditoría del ruido…», 26-10.)

## 2026-10-21 — Ruido ciego log-uniforme, lote 32: `ciego-log-l32`

**Las banderas** (esta rama): `--sigma-log` sortea σ log-uniforme en [`--sigma-min`, `--sigma-max`]
(pide `--sigma-min` > 0) y `--fraccion-limpias p` deja una fracción p de los parches sin ruido. Las
dos van en `add_noise_ciego` (`ddr/data3d.py`), y apagadas la corrida es bit a bit la de antes
(`tests/test_data3d.py`, y los tests de traza de `tests/test_train3d.py`). `--fraccion-limpias` no
se usa aquí.

**PRE-REGISTRO**, escrito antes de lanzar (regla 5).

- **Corrida:** `runs/ciego-log-l32`, la configuración de `prep3d-g10-s2-2500` (su `config.json` y
  su `plan.sh`: semilla 2, lote 32 de 48³, micro-lote 3, `--mascara-cerebro`,
  `--hitos` sin hitos y sin meseta, es decir lr 4e-4 constante, `lista_train_1000.txt`,
  `--log-every 25`, `--eval-every 250`, `--guardar-cada 250`), cambiando el ruido y el disyuntor:
  `--sigma-min 0.81 --sigma-max 55 --sigma-log`, σ log-uniforme entre SNR 100 y SNR ~1.5, y
  `--guard inf --guard-abs inf`, sin disyuntor, como `prep3d-control` y `etapa2-ciega`. Sin
  ADMM (`--qsm-datos ''`, `--estabilidad-lado 0`) y 2 250 iteraciones (~5.5–5.8 h a 8.8–9.3 s/it en
  la 5070 Ti). Código congelado de esta rama en `runs/ciego-log-l32/codigo`.
- **Por qué log-uniforme:** con U(0, 55), sólo ~3 % de los parches caen sobre SNR 50 (σ < 1.61);
  con log-uniforme en [0.81, 55], el 16 %, y cada octava de σ recibe lo mismo. Se busca mejor RMSE
  con poco ruido.
- **Diferencias con la comparada, además del ruido y del disyuntor:** `prep3d-g10-s2-2500` hizo sus primeras 500
  iteraciones en otra corrida (`prep3d-g10-s2`, borrada) con la misma configuración, y tuvo 60
  rechazos y 2 rebobinados allí; esta empieza de cero. Las dos se comparan a la misma iteración.
- **Comparación:** contra `prep3d-g10-s2-2500` en la 2 250 (sus `estado-002250.pth`), con la etapa
  1 de la semilla 1 a los 7 niveles (`scripts/denoising_por_snr_3d.py`, sin reescalar), en RMSE de
  salida en el cerebro, media de los 12 volúmenes.
- **Predicción:** mejor RMSE de salida a SNR 25–100 que la de σ fijo, y posiblemente algo peor a
  SNR 3.2, porque ve menos parches con mucho ruido.
- **Alcance:** es una corrida por brazo: ordena, no da el número final.
- **Primer arranque, con `--guard 10`** (lanzado a las 17:42 y detenido a los ~12 min): en 75
  iteraciones el disyuntor rechazó 60 lotes y rebobinó 2 veces, y a la 75 la red llevaba sólo 17
  pasos efectivos. Los rechazados tenían |g| de 3·10⁵ a 5·10⁶, contra una mediana de 710 fijada en
  los primeros 7 lotes; los rechazó la cota absoluta, porque el relativo se arma con 100
  aceptadas. No eran los lotes más ruidosos: en el relanzamiento sin disyuntor, el |g|
  de las iteraciones 26 a 75 no pasó de 2.6·10³, aunque cada lote trae en promedio 4.6 parches con
  σ > 30. La lectura más simple es una inestabilidad pasajera del arranque: el relanzamiento tuvo
  un pico de 1.5·10⁶ antes de la iteración 25, y `prep3d-control`, uno de 3.3·10⁵ antes de la 50.
  Aplicando esos pasos, que Adam acota, la red sale; rechazándolos, se queda, y el rebobinado la
  devuelve a un punto desde el que vuelve a entrar. Se relanzó sin disyuntor. El log de ese
  arranque está en `runs/ciego-log-l32/intento-guard10/`.

**RESULTADO (2026-10-22): se cumple la predicción.** Etapa 1 con la semilla 1, sobre el
`estado-002250.pth` de las dos corridas (`scripts/denoising_por_snr_3d.py --red … --nombre
ciego-log-l32` y `--nombre g10-s2-2250`; GPU 1, 17.2 min cada una; `runs/denoising-snr/<nombre>/`).
NRMSE de salida en el cerebro, media de los 12; la entrada es la misma para las tres redes:

| SNR | entrada | ciega log-uniforme (2 250) | σ fijo, `prep3d-g10-s2-2500` (2 250) | control (1 500) |
|---|---|---|---|---|
| 3.2 | 29.88 % | 8.25 % | 6.60 % | 7.49 % |
| 25 | 3.85 | 3.13 | 5.57 | 9.52 |
| 40 | 2.41 | 2.55 | 5.58 | 9.60 |
| 50 | 1.93 | 2.33 | 5.58 | 9.62 |
| 60 | 1.61 | 2.19 | 5.59 | 9.63 |
| 80 | 1.20 | 2.03 | 5.59 | 9.64 |
| 100 | 0.96 | 1.96 | 5.59 | 9.65 |

- **Mejor que la de σ fijo de SNR 25 a 100** (2.4–3.6 pp menos) **y peor a SNR 3.2** (1.65 pp más),
  como se predijo. En RMSE, el endpoint escrito, el orden es el mismo (a SNR 3.2, 0.0271 contra
  0.0216; a SNR 25, 0.0103 contra 0.0183; a SNR 100, 0.0064 contra 0.0184).
- **La salida de la ciega baja con el ruido** (de 8.25 a 1.96 %); la de σ fijo queda plana (5.6 %
  en `g10-s2`, 9.6 % en la control).
- **Mejora la entrada sólo hasta SNR 25** (+1.80 dB). Desde SNR 40 la empeora (−0.49 dB), y a SNR
  100 le duplica el error (1.96 contra 0.96 %, −6.15 dB): le queda un piso de ~2 %.
- Las semillas 2 y 3 dan lo mismo (≤ 0.01 pp). Figuras, como las de la control:
  `docs/Sanity Check/resultados-ciego-log-l32/seed1/` (fecha 22-10) y
  `22-10-resumen_semillas.png`.
- **Alcance:** una corrida por brazo y 2 250 iteraciones: ordena, no da el número final.
- **Convergencia: ninguna de las tres redes convergió** (`scripts/convergencia_3d.py`, de
  `history.json` y `traza.npy`; `22-10-convergencia.png` en `resultados-control/`,
  `resultados-ciego-log-l32/` y `resultados-g10-s2-2250/`, 2026-10-22). La validación a σ = 25 de
  la ciega está plana desde la 500 (31.1–31.4 dB, +0.10 en las últimas 250), pero su pérdida sigue
  bajando (la mediana de las últimas 250, −6.9 % contra las 250 anteriores) y su λ baja sin pausa
  (0.100 a 0.053). La de σ fijo sube despacio (+0.08 dB, 33.38 en la 2 500; pérdida −1.1 %; λ de
  0.109 a 0.053 desde la 750). La control sube +0.22 dB (32.18 en la 1 500; pérdida −3.1 %), con λ
  casi quieta (0.098) y T en 0.0963 desde un pico de la pérdida cerca de la 190.

## 2026-10-21 — Etapa 2: la red dentro del ADMM, el α de la evaluación y la rama ciega

**Lo que dice.**
- **Dentro del ADMM la red no quita el error del cono del dipolo** (A, B y su control): quita el de
  fuera, y en el cono el ADMM le devuelve su propia salida. Con target_std 0.06 la red control
  además lo agranda desde t = 8, y el NRMSE diverge (86.65 % en la 100).
- **El α importa, y 0.06 es el peor de los medidos.** Con la control y con `prep3d-g10-s2-2500`, en
  COSMOS y en Sim2, 0.06 fue el peor de {0.02, 0.06, 0.25, 0.35}: 2.6–7.7 pp de NRMSE sobre 0.25,
  con HFEN también peor. Con 0.25–0.35 la control converge y achica el cono. (Con la ciega, en
  Sim2 queda entre los mejores: ver «Barrido de target y ADMM a 100 con la ciega» y «Grid de
  target…».)
- **La regla pre-registrada da 0.25 en las dos redes**, y se le propone al autor como α fijo de la
  evaluación de la corrida larga, con estados cada 500 (README, paso 6). `--qsm-target-std` está
  implementada, con default 0.06. (Reemplazado el 25-10: ver «La evaluación QSM».)
- **La ciega no diverge a 0.06**, achica el cono, mejora el NRMSE 3.2 pp y no cuesta PSNR a σ = 25.
  A 0.25 también es la mejor de las dos (29.63 contra 31.36 %).

**Lo que no dice.**
- **Cuál es el mejor α:** depende del sujeto (0.35 en COSMOS, 0.02 en Sim2) y, entre 0.02, 0.25 y
  0.35, de la métrica. Son dos sujetos, uno simulado, y dos redes cortas (1 500 y 2 500
  iteraciones). (Con la ciega se invierte: 0.02 en COSMOS y 0.05–0.06 en Sim2; ver «Barrido de
  target y ADMM a 100 con la ciega» y «Grid de target…».)
- **Por qué la red agranda el cono:** no lo explican ni el piso sin datos ni las 20 pasadas
  aisladas; la medición que separa las hipótesis quedó diseñada y sin correr.
- **Si la ciega sirve para la corrida larga:** una corrida por brazo, un sujeto, λ trabada en 0.
- **Nada sobre 100 000 iteraciones.**

**Las partes:** A y B (abajo), D y su borde, la validación de α (Sim2, otra red, el piso y HFEN), el
control de B, C y la estimación de costo de tres α. El barrido de s está en la Etapa 1, «Barrido de
s».

**Veredicto de A y B: dentro del ADMM, la red control quita el error fuera del cono del dipolo y deja pasar
el del cono, y desde la iteración 8 lo agranda; en el cono el ADMM le devuelve a la red su propia
salida, y el NRMSE diverge.**
De la iteración 3 en adelante la red baja la energía del error fuera del cono a un 3–37 % de la que
entra, pero en el cono la deja en un 78–135 %. Una onda al ángulo mágico pasa la red casi entera
(91 % a 30 vóxeles de período) y, a la frecuencia de las rayas (5.9 vóxeles), en un 38–68 % según
su amplitud. Las rayas no salen de la red sola: la red no las quita (y desde la iteración 8 las
agranda un poco), y el ADMM, que no las mide, no las corrige.

### Configuración

- **A:** `admm_wtdv` de la evaluación sobre COSMOS (k = 1, μ = 0.008, μ₂ = 1, target_std = 0.06,
  α = 4.859), con la red control (`runs/prep3d-control/last.pth`, it 1 500) como en `evalua_qsm`:
  autocast bf16 y `R.grad` compilado. Una corrida de 100 iteraciones sin parada; la de dos subidas,
  aplicada después: la evaluación habría parado en la 7, con el mínimo, 39.01 %, en la 5. Un
  envoltorio del denoiser guarda lo que entra, v_t = α·M·(χ_t + s_t), y lo que sale, M·D(v_t); la
  referencia es α·M·χ_COSMOS, y todo se mide en la máscara de evaluación (χ_COSMOS ≠ 0). t es el
  número de x-update (1–100). El cono es |D| < √(μ/μ₂) = 0.0894, el 21.7 % de k-space; la energía se
  parte por Parseval.
- **Reproducción:** el NRMSE por iteración coincide con
  `runs/prep3d-control/admm-qsm/last-it1500/admm100.json` (misma red, mismo ADMM) a 0.0042 pp
  como máximo.
- **B:** a M·(ref + onda) se le pasa la red, y sobrevive ⟨D(ref + onda) − D(ref), onda⟩/⟨onda, onda⟩.
  Regla fijada en el código antes de mirar: la frecuencia del pico del error de entrada en el cono
  en la iteración del mínimo (la 5) y la amplitud del componente en el cono de ese error (RMS
  0.040). A 0.5× y 2×, descriptivo; y, también descriptivo, la frecuencia del pico del error final
  de χ (las rayas).
- **Script:** `scripts/pnp_adentro_del_admm.py` (`corre`, `puente`, `figuras`; las figuras en
  `scripts/pnp_figuras.py`), locales. Salidas en `runs/etapa2-pnp/`: los volúmenes de t ∈ {1, 2, 3,
  5, 10, 15, 20, 25, 50, 100} en float16, `adentro.json`, `puente*.json`.
- **GPU 1:** A 6.3 min, B ~3 min.

### A: por iteración

| t | NRMSE χ | RMSE entrada (×σ₀) | SNR ef. entrada | red (dB) | corr(quitó, error) | cono: % del error, entra → sale | energía en el cono, sale/entra | fuera del cono, sale/entra |
|---|---|---|---|---|---|---|---|---|
| 1 | 65.8 % | 0.090 (0.91) | 1.52 | −1.3 | −0.30 | 23 → 17 | 1.01 | 1.457 |
| 2 | 43.8 % | 0.072 (0.73) | 1.90 | −0.0 | +0.55 | 33 → 28 | 0.86 | 1.074 |
| 3 | 40.0 % | 0.096 (0.98) | 1.42 | +3.5 | +0.81 | 19 → 34 | 0.80 | 0.365 |
| 5 | 39.0 % | 0.138 (1.41) | 0.98 | +7.3 | +0.92 | 10 → 42 | 0.80 | 0.121 |
| 7 | 41.7 % | 0.171 (1.74) | 0.80 | +8.8 | +0.94 | 7 → 52 | 0.96 | 0.067 |
| 10 | 56.5 % | 0.205 (2.09) | 0.66 | +8.2 | +0.93 | 8 → 74 | 1.35 | 0.044 |
| 25 | 83.7 % | 0.258 (2.63) | 0.53 | +7.1 | +0.90 | 16 → 86 | 1.08 | 0.032 |
| 100 | 86.7 % | 0.264 (2.70) | 0.51 | +7.0 | +0.90 | 16 → 86 | 1.06 | 0.033 |

RMSE de entrada en unidades de la red y entre paréntesis en múltiplos de σ₀ = 25/255; SNR efectivo
= desvío de α·M·χ_COSMOS (0.136) / RMSE de entrada; red = 20·log10(RMSE entra / RMSE sale).

- **Cuánto ruido ve la red.** El SNR efectivo de lo que entra va de 1.9 (t = 2) a 0.5, siempre por
  debajo del 3.2 de su entrenamiento; en unidades absolutas, el error de entrada es 0.7–1.4 σ₀ en
  las primeras cinco iteraciones y crece a 2.7 σ₀. Ojo: el SNR de aquí es desvío/RMSE y el 3.2 del entrenamiento es media/σ; la comparación
  directa es la de unidades absolutas.
- **Qué hace la red.** Desde t = 3 baja el error total 3.5–8.8 dB, y lo que quita se correlaciona
  0.81–0.94 con el error de entrada: quita error, no señal. Pero lo quita fuera del cono (la
  energía de afuera sale al 3–37 %) y no en el cono (78–135 %): el error que sale es 34–86 % cono,
  contra 7–33 % el que entra.
- **El lazo.** Donde D = 0, el x-update da χ = z − s (con k = 1), así que lo que entra a la red,
  χ + s, es exactamente su salida anterior z, y la dual es sólo el último cambio (bloque «PROTOCOLO
  DEL AUTOR» de `ddr/qsm3d.py`): en el cono, el ADMM le devuelve a la red su propio dibujo y los
  datos no corrigen nada. Es exacto donde D = 0 y sin máscaras; en el resto del cono, una
  aproximación. La energía del error de entrada en el cono crece de 0.0018 (t = 1–5) a 0.0113
  (t = 100), y el NRMSE de χ en el cono de 27.8 a 80.5 %, mientras fuera del cono queda en 27–32 %.
- **Dos problemas, separados.** (i) La red no quita lo del cono (B: sobrevive el 91 % de una onda
  lenta): eso explica el mínimo de 39 %. (ii) La red lo agranda desde t = 8 (×1.06–1.35 por
  pasada, que en 10 pasadas es ×1.8–×20): eso explica la subida al 87 %. Para (ii): PnP no es
  convergente en general, y los resultados de convergencia piden un denoiser no expansivo o un
  término de datos fuertemente convexo (Schönlieb y Shumaylov, 2025, la viñeta «Convergence» de
  PnP, `docs/papers/data-driven-inverse-problems-2025.pdf`); el de QSM no lo es, justamente por el
  cono.

### B: una onda al ángulo mágico

| onda | período, ángulo con B0, \|D\| | 0.5× | 1× (la regla) | 2× |
|---|---|---|---|---|
| pico del error de entrada en el cono, t = 5 | 29.7 vóx, 56.1°, 0.027 | 0.906 | **0.908** | 0.915 |
| rayas: pico del error final de χ (descriptivo) | 5.9 vóx, 53.9°, 0.010 | 0.377 | 0.472 | 0.681 |

Fracción de la onda que sobrevive a la red (proyección; la razón de normas da lo mismo ±0.05). La
onda lenta pasa casi entera a toda amplitud. La de las rayas pasa a medias, y más cuanto más fuerte:
lo que la red recibe de vuelta más fuerte, lo quita cada vez menos.

### Figuras

- `docs/Sanity Check/21-10-pnp_entrada_salida.png`: referencia, lo que entra, lo que sale y sus
  diferencias, en ppm y ±0.1 (convención QSM de `../tdv`), corte sagital central (contiene B0), a
  t = 1, 5, 10 y 25: las rayas oblicuas crecen en «sale − referencia».
- `docs/Sanity Check/21-10-pnp_por_iteracion.png`: NRMSE, SNR de entrada, ganancia de la red, fracción
  del error en el cono y energía del error sale/entra en y fuera del cono, por iteración.

**Lo que dice y lo que no.** La red hace lo que aprendió en T1: quita error con estructura de ruido
y de alta frecuencia, que en QSM queda casi todo fuera del cono. El error del cono, que el dato no
mide, tiene la orientación de las rayas, y la red lo deja pasar; las rayas crecen porque, en el
cono, el ADMM le devuelve a la red su propia salida y nada las quita. Es una red de 1 500 iteraciones con su T entrenado y target_std 0.06; ni otra
amplitud (α) ni otro T dentro del ADMM están medidos (D).

### D: las perillas del acople (2026-10-21)

**PRE-REGISTRO**, escrito antes de lanzar (regla 5).

- **Pregunta:** ¿alguna perilla del acople baja el NRMSE mínimo del ADMM con la red control?
- **Protocolo:** el ADMM de la evaluación sobre COSMOS, 25 iteraciones, red control (it 1 500,
  bf16 y `R.grad` compilado, como `evalua_qsm`). Una variable por vez (regla 4); el resto, en la
  base: target_std 0.06, T 0.0963 (el entrenado), μ 0.008. Se corre sin la parada y se aplica
  después la de dos subidas: el NRMSE mínimo es el de lo recorrido hasta donde pararía.
  - target_std ∈ {0.02, 0.03, 0.045, 0.06, 0.09, 0.12, 0.25, 0.35}: con 0.06, A da un error de
    entrada de 0.7–1.4 σ₀ en t ≤ 5; la grilla lleva ese cociente por debajo y por encima.
  - T en inferencia (`modelo.T` antes del ADMM) ∈ {0.048, 0.072, 0.125, 0.15, 0.19}. Si a 0.19
    diverge, es un dato de estabilidad.
  - μ ∈ {0.004, 0.016, 0.032}. Cambia también el umbral del cono, √(μ/μ₂): 0.063, 0.126 y 0.179
    (0.089 en la base).
- **Por configuración:** el NRMSE mínimo y su iteración, la iteración en que pararía, el NRMSE en
  t = 25 sin la regla, el α resultante y la fracción del error de salida de la red que cae en el
  cono (con el umbral de su μ) en la iteración del mínimo.
- **Umbral de mejora: 2.07 pp** de NRMSE mínimo, el mayor salto entre evaluaciones consecutivas de
  `runs/prep3d-g10-s2-2500/qsm.jsonl` (de la 250 a la 500, durante el aprendizaje temprano; desde
  la 500, el mayor es 0.89 pp). Es una escala heurística: esa corrida usa otro lote y otra razón
  del disyuntor.
- **Advertencia:** COSMOS es un solo sujeto, y el mejor de 16 valores sobreestima la mejora. Es
  exploración, no calibración.
- **Lo que se espera, por A y B:** que ninguna perilla arregle el cono, porque su error pasa por la
  red casi sin cambio; a lo más moverían el mínimo y la iteración en que el error empieza a crecer.

**RESULTADO de D.** `scripts/pnp_adentro_del_admm.py perillas`, GPU 1, 23.0 min; la figura,
`docs/Sanity Check/21-10-perillas_admm.png`. La base reproduce A (39.01 % en t = 5).

| configuración | NRMSE mín | t mín | pararía en | NRMSE en 25 | α | umbral del cono | error de salida en el cono (t mín) |
|---|---|---|---|---|---|---|---|
| target_std 0.02 | **35.40 %** | 9 | 11 | 39.07 % | 1.62 | 0.089 | 34 % |
| target_std 0.03 | 35.90 % | 6 | 8 | 83.18 % | 2.43 | 0.089 | 35 % |
| target_std 0.045 | 38.02 % | 5 | 7 | 100.45 % | 3.64 | 0.089 | 39 % |
| **base** (0.06, T 0.0963, μ 0.008) | 39.01 % | 5 | 7 | 83.72 % | 4.86 | 0.089 | 42 % |
| target_std 0.09 | 39.65 % | 4 | 6 | 57.87 % | 7.29 | 0.089 | 41 % |
| target_std 0.12 | 39.39 % | 5 | 7 | 43.98 % | 9.72 | 0.089 | 48 % |
| target_std 0.25 | **31.36 %** | 25 | 25 | 31.36 % | 20.25 | 0.089 | 45 % |
| target_std 0.35 | **30.62 %** | 25 | 25 | 30.62 % | 28.34 | 0.089 | 42 % |
| T 0.048 | 37.19 % | 5 | 7 | 84.60 % | 4.86 | 0.089 | 49 % |
| T 0.072 | 38.23 % | 5 | 7 | 86.72 % | 4.86 | 0.089 | 45 % |
| T 0.125 | 39.84 % | 4 | 6 | 77.19 % | 4.86 | 0.089 | 35 % |
| T 0.15 | 40.59 % | 5 | 7 | 70.70 % | 4.86 | 0.089 | 37 % |
| T 0.19 | 42.64 % | 6 | 8 | 60.44 % | 4.86 | 0.089 | 37 % |
| μ 0.004 | 39.10 % | 5 | 7 | 94.44 % | 4.43 | 0.063 | 35 % |
| μ 0.016 | 41.88 % | 5 | 7 | 69.28 % | 5.46 | 0.126 | 49 % |
| μ 0.032 | 47.27 % | 5 | 7 | 53.86 % | 6.41 | 0.179 | 59 % |

- **La pregunta: sí, target_std.** Pasan el umbral de 2.07 pp tres valores: 0.02 (−3.6 pp), 0.25
  (−7.7 pp) y 0.35 (−8.4 pp). La curva no es monótona: baja a los dos lados de la base, que está
  cerca de su peor punto (0.045–0.12 dan 38.0–39.7 %).
- **A 0.25 y 0.35 el mínimo cae en el borde:** el NRMSE todavía baja en t = 25, la parada no actúa y
  la evaluación reportaría el valor de la 25. Qué pasa después de la 25 no está medido: puede
  seguir bajando o divergir más tarde.
- **T y μ no pasan el umbral.** Bajar T mejora un poco (37.19 % a 0.048, −1.8 pp); subir μ empeora
  (47.27 % a 0.032).
- **Lo que se esperaba se cumple a medias.** Ninguna perilla arregla el cono: el error de salida en
  el cono, en la iteración del mínimo, queda en 34–59 % en todas, contra 42 % en la base. Pero
  target_std no sólo movió el mínimo: lo bajó más que el umbral, a 0.25–0.35 frenando la subida
  dentro de las 25 iteraciones.
- **Advertencia, como se escribió:** COSMOS es un solo sujeto y el mejor de 16 valores sobreestima
  la mejora: es exploración, no calibración.

### D, el borde: 100 iteraciones con target_std 0.25 y 0.35 (2026-10-21)

**PRE-REGISTRO**, escrito antes de lanzar. Con target_std 0.25 y 0.35 el mínimo de D cayó en
t = 25, el borde. Se corre el ADMM instrumentado como A, 100 iteraciones sin parada, con cada uno,
y se lee:

- si el NRMSE sigue bajando después de la 25 o si diverge más tarde;
- en qué iteración aplicaría la parada de dos subidas (sin el techo de 25);
- el factor de energía del error del cono (sale/entra de la red) en t ≥ 8, contra el de la base
  (1.06–1.35): si con un α alto la red deja de agrandar el cono, o sólo lo agranda más lento.

Advertencia, como en D: un solo sujeto y el mejor de 16 valores. Es exploración.

**RESULTADO del borde: con target_std alto el ADMM converge, y la red achica el error del cono en
vez de agrandarlo.** `scripts/pnp_adentro_del_admm.py corre --target-std 0.25|0.35`, GPU 1, 6.0 min
cada una; la figura, `docs/Sanity Check/21-10-pnp_por_iteracion_alfa.png`.

| target_std | α | NRMSE mín (t) | pararía en (sin techo) | NRMSE en 100 | error de entrada / σ₀, t = 1–5 → 100 | energía del cono sale/entra, t ≥ 8 | fuera del cono, t ≥ 8 |
|---|---|---|---|---|---|---|---|
| 0.06 (base) | 4.86 | 39.01 % (5) | 7 | 86.65 % | 0.7–1.4 → 2.7 | 1.06–1.37 | 0.03–0.06 |
| 0.25 | 20.25 | 31.21 % (67) | 69 | 31.22 % | 3.0–4.6 → 5.2 | 0.49–0.53 | 0.08–0.09 |
| 0.35 | 28.34 | 30.38 % (72) | 74 | 30.39 % | 4.2–5.9 → 6.2 | 0.45–0.50 | 0.11–0.13 |

- **El NRMSE sigue bajando después de la 25 y no diverge:** llega a una meseta (31.2 % y 30.4 %)
  hacia la 60–70, y la parada de dos subidas, sin techo, actuaría recién en la 69 y la 74, por
  subidas de centésimas sobre la meseta. Con el techo de 25 de la evaluación, el valor reportado
  (31.36 % y 30.62 %) queda a 0.15–0.24 pp de la meseta.
- **La red deja de agrandar el cono:** con α alto, la energía del error del cono sale de la red al
  45–53 % de la que entra en cada pasada, contra 106–137 % en la base. No es que lo agrande más
  lento: lo achica. Fuera del cono quita un poco menos que en la base (sale al 8–13 % contra 3–6 %).
- **Lo que cambia es cuánto error ve la red:** con α alto, el error de entrada es de 3–6 σ₀ en sus
  unidades, contra 0.7–1.4 σ₀ en la base. Hipótesis, no medida: con un error del orden de su ruido
  de entrenamiento, la red trata el dibujo del cono como estructura y lo refuerza; con un error
  mucho mayor que ese ruido, lo trata como ruido y lo achica.
- **Advertencia:** un solo sujeto y el mejor de 16 valores de D. Es exploración, no calibración.

### Validación de α: Sim2, otra red, el piso de estabilidad y HFEN (2026-10-21)

**PRE-REGISTRO**, escrito antes de lanzar. Antes de proponer cambiar el α de la evaluación, se
elige α en Sim2 (`../tdv/data_mat/`, sólo lectura) y se reporta en COSMOS: red control;
target_std ∈ {0.02, 0.06, 0.12, 0.25, 0.35, 0.5}; el resto en la base (μ 0.008, μ₂ 1, T
entrenado). Seis corridas de 100 iteraciones instrumentadas como A (las de 25 son su prefijo, con la
parada de dos subidas), que guardan los χ finales. El target_std de menor NRMSE en Sim2 se reporta
en COSMOS (a 0.06, 0.25 y 0.35 ya está medido a 100 iteraciones; otro valor, se corre). Costo
estimado: ~58 min de GPU (Sim2 tiene 6.9 M de vóxeles, 1.7 veces los de COSMOS).

- **Cómo se arma Sim2** (`scripts/sim2_datos.py`; ningún código de `../tdv` lo carga):
  - la fase de `Sim2.mat` ([176, 208, 208]) recortada a la grilla de `Sim2ChiGT.nii` ([164, 205,
    205]) con el desplazamiento que maximiza la correlación con el modelo directo D∗χ_GT: (0, 0, 0),
    0.883, contra 0.74–0.83 de los vecinos y 0.40 del recorte centrado;
  - el kernel, el dipolo en esa grilla con B0 en el eje 2 (0.883; los otros ejes dan −0.53 y
    −0.35) y 0 en k = 0, como el de COSMOS;
  - la pendiente de la fase sobre D∗χ_GT es 0.965: la fase ya está en las unidades del ADMM;
  - la máscara de α y la de evaluación, `mask_final` (1 366 083 vóxeles; χ_GT no tiene ceros);
  - W2 = (w·M normalizado a media 1)², con `w` de `sim2_w.mat`, como `prepara_datos` hace con la
    magnitud. RMS de χ_GT en la máscara: 0.0282 ppm.
- **Predicción (del freno):** 0.25–0.35 convergen sin divergir en 100 iteraciones y quedan por
  debajo de 0.06, también en Sim2.
- **Regla:** si se cumple, se le propone al autor cambiar el α de la evaluación a target_std 0.25;
  si no, no se toca y se anota.
- **Advertencia:** el α bueno puede depender de la red y correrse durante el entrenamiento; Sim2 es
  simulado.
- **Se reporta:** NRMSE mínimo y su iteración, NRMSE en t = 25 y en t = 100, el factor del cono en
  t ≥ 8 y el error de entrada en unidades de σ₀, para las seis.

**RESULTADO de Sim2: la predicción se cumple a medias, y el α de la evaluación no se toca.**
`scripts/pnp_adentro_del_admm.py corre --sujeto sim2 --target-std …`, GPU 1, 10.2 min cada una; la
tabla sale de `scripts/pnp_resumen.py` (que con COSMOS reproduce la del borde). «Evaluación» es lo
que reportaría la evaluación (techo de 25, parada de dos subidas); las energías, lo que sale de la
red sobre lo que entra, por pasada.

| target_std | α | evaluación (t) | mín en 100 (t) | pararía en (sin techo) | NRMSE en 25 → 100 | error de entrada / σ₀, t = 1–5 → 100 | energía del cono sale/entra, t ≥ 8 | fuera del cono, t ≥ 8 |
|---|---|---|---|---|---|---|---|---|
| 0.02 | 1.74 | **40.75 % (13)** | 40.75 % (13) | 15 | 42.06 → 60.18 % | 0.3–0.6 → 1.3 | 0.64–0.97 | 0.02–0.06 |
| 0.06 | 5.21 | 46.11 % (5) | 46.11 % (5) | 7 | 78.50 → 83.53 % | 0.9–1.5 → 2.8 | 0.95–1.14 | 0.05–0.09 |
| 0.12 | 10.43 | 45.18 % (10) | 45.18 % (10) | 12 | 45.68 → 45.54 % | 1.8–2.9 → 3.7 | 0.73 | 0.08–0.11 |
| 0.25 | 21.72 | 43.54 % (25) | 42.97 % (100) | no para | 43.54 → 42.97 % | 3.7–5.2 → 5.8 | 0.60–0.64 | 0.16–0.18 |
| 0.35 | 30.41 | 46.09 % (10) | 46.09 % (10) | 12 | 46.95 → 47.93 % | 5.2–6.9 → 7.4 | 0.61–0.63 | 0.24–0.26 |
| 0.5 | 43.44 | 49.29 % (4) | 49.29 % (4) | 6 | 54.21 → 58.14 % | 7.4–8.9 → 9.8 | 0.66–0.70 | 0.42–0.48 |

- **La predicción, por partes.** 0.25 se cumple: baja hasta la 100 sin que la parada actúe y queda
  2.6–3.1 pp bajo 0.06. 0.35 no: para en la 12 y empata con 0.06 (46.09 contra 46.11 %). Por la
  regla, no se propone cambiar el α de la evaluación.
- **Validación cruzada, juntando los dos sujetos** (COSMOS a 100 iteraciones: base y borde de
  arriba, 0.02 de la corrida de abajo):
  - el mejor de Sim2, 0.02, da en COSMOS 35.39 % en t = 9, −3.6 pp frente a la base (D había dado
    35.40 %; `corre --target-std 0.02`, GPU 1, 6.0 min), y a 100 iteraciones sube a 59.94 %: la
    parada de dos subidas actúa en la 11;
  - el mejor de COSMOS, 0.35, da en Sim2 46.09 %, lo mismo que la base;
  - el segundo de COSMOS, 0.25, da en Sim2 −2.6 pp frente a la base en la evaluación (−3.1 pp a
    100 iteraciones);
  - ningún target_std es el mejor en los dos.
- **Lo único que coincide en los dos sujetos:** 0.06, la base, queda cerca del peor punto, y es el
  único α cuyo factor de energía del cono llega a pasar de 1 (0.95–1.14 en Sim2, 1.06–1.37 en
  COSMOS; los demás, 0.60–0.97 en Sim2 y 0.45–0.97 en COSMOS).
- **Con α alto, lo que crece es el error fuera del cono:** sale al 24–48 % de lo que entra a
  0.35–0.5 en Sim2, contra 11–13 % en COSMOS a 0.35. Lectura, no medida aparte: en Sim2, subir α
  cambia error del cono por error fuera de él, y el balance se da vuelta antes que en COSMOS.
- **Un factor del cono bajo 1 no es convergencia.** 0.02 lo cumple en los dos sujetos (0.56–0.97
  en COSMOS, 0.64–0.97 en Sim2), y aun así su NRMSE sube después del mínimo hasta ~60 % en la 100
  (59.94 y 60.18 %), con el error de entrada creciendo de 0.2–0.6 σ₀ a 1.2–1.3 σ₀. Lo que salva a
  0.02 en la evaluación es la parada de dos subidas, no la estabilidad.
- **Advertencia:** dos sujetos, uno simulado, una red. El α bueno depende del sujeto.

**PRE-REGISTRO: ¿el α bueno es el mismo en otra red?** `runs/prep3d-g10-s2-2500/last.pth` (2 500
iteraciones, la red 3D más entrenada) en COSMOS y en Sim2, a target_std 0.02, 0.06, 0.25 y 0.35:
100 iteraciones instrumentadas como A, guardando los χ finales (~65 min). El 0.02 y Sim2 se
agregaron antes de lanzar, después del RESULTADO de Sim2.

- **Lo que difiere de la control:** semilla, iteraciones (2 500 contra 1 500), `--guard`, lote y
  calendario. Si el mejor α o la ventana estable cambian, se escribe «en esta otra red está en otro
  lugar», no «se mueve con el entrenamiento» (regla 4). Es una comparación descriptiva, no rama
  contra tronco.
- **Regla de decisión (del teórico), por red y con los dos sujetos:**
  - α es estable si la energía del error del cono, sale/entra, queda bajo 1 en todo t ≥ 8, en los
    dos sujetos;
  - entre los estables, se elige el de menor distancia en el peor caso al mejor α de cada sujeto:
    máx sobre sujetos de (NRMSE de la evaluación en α − el menor NRMSE de la evaluación en ese
    sujeto).
- **Con la red control da 0.25** (COSMOS: 0.02 35.40 %, 0.06 39.01 %, 0.25 31.36 %, 0.35 30.62 %;
  Sim2: 40.75, 46.11, 43.54 y 46.09 %). Su peor distancia es 2.8 pp (0.7 en COSMOS, 2.8 en Sim2),
  contra 4.8 pp de 0.02 (4.8 y 0), 5.3 pp de 0.35 (0 y 5.3) y 8.4 pp de 0.06, que además no es
  estable. La estabilidad de 0.02 en COSMOS la dice la corrida de 100 iteraciones del RESULTADO
  de Sim2.
- **Para qué:** un solo α para la evaluación QSM de la corrida larga, fijo toda la corrida.
- **Condición, escrita antes de ver el resultado:** la regla se aplica a las dos redes, la control
  y `prep3d-g10-s2-2500`. Si en esta 0.25 deja de ser estable (energía del cono ≥ 1 en algún t ≥ 8,
  en algún sujeto) o la regla da otro valor, no se aplica tal cual y vuelve al autor. Si da 0.25 en
  las dos, se le propone al autor.

**RESULTADO: en `prep3d-g10-s2-2500` la regla también da 0.25, así que se le propone al autor.**
`corre --checkpoint runs/prep3d-g10-s2-2500/last.pth --sujeto cosmos|sim2 --target-std …`, GPU 1,
6.0 min (COSMOS) y 10.2 min (Sim2) cada una; la tabla, de `scripts/pnp_resumen.py`.

| sujeto | target_std | evaluación (t) | mín en 100 (t) | pararía en (sin techo) | NRMSE en 25 → 100 | error de entrada / σ₀, t = 1–5 → 100 | energía del cono sale/entra, t ≥ 8 | fuera del cono, t ≥ 8 |
|---|---|---|---|---|---|---|---|---|
| COSMOS | 0.02 | 30.88 % (14) | 30.88 % (14) | no para | 31.23 → 31.64 % | 0.2–0.5 → 1.1 | 0.43–0.50 | 0.01–0.04 |
| COSMOS | 0.06 | 33.79 % (8) | 33.79 % (8) | 10 | 36.74 → 40.29 % | 0.7–1.3 → 3.5 | 0.49–0.70 | 0.02–0.10 |
| COSMOS | 0.25 | 30.28 % (25) | 30.23 % (36) | 46 | 30.28 → 30.24 % | 3.0–6.0 → 10.4 | 0.46–0.53 | 0.02–0.04 |
| COSMOS | 0.35 | **28.47 % (25)** | 28.42 % (70) | 38 | 28.47 → 28.42 % | 4.2–8.1 → 11.2 | 0.43–0.49 | 0.02–0.04 |
| Sim2 | 0.02 | **35.93 % (23)** | 35.34 % (64) | no para | 36.08 → 35.35 % | 0.3–0.5 → 1.2 | 0.48–0.54 | 0.02–0.06 |
| Sim2 | 0.06 | 40.18 % (25) | 38.95 % (99) | no para | 40.18 → 40.76 % | 0.9–1.4 → 3.2 | 0.56–0.73 | 0.03–0.11 |
| Sim2 | 0.25 | 36.99 % (25) | 35.99 % (100) | no para | 36.99 → 35.99 % | 3.7–6.5 → 11.0 | 0.53–0.60 | 0.03–0.06 |
| Sim2 | 0.35 | 38.79 % (25) | 37.99 % (100) | no para | 38.79 → 37.99 % | 5.1–9.1 → 11.9 | 0.52–0.60 | 0.06–0.08 |

- **La regla.** Los cuatro α son estables en los dos sujetos (energía del cono 0.43–0.73 por
  pasada desde t = 8). Peor distancia al mejor de cada sujeto (COSMOS 0.35, 28.47 %; Sim2 0.02,
  35.93 %): **0.25, 1.8 pp** (1.8 en COSMOS, 1.1 en Sim2); 0.02, 2.4 pp (2.4 y 0); 0.35, 2.9 pp
  (0 y 2.9); 0.06, 5.3 pp (5.3 y 4.3). Con la control había dado 0.25 con 2.8 pp. Da 0.25 en las
  dos redes: por la condición escrita, se le propone al autor un α fijo de target_std 0.25 para
  la evaluación QSM de la corrida larga. (Reemplazado el 25-10: ver «La evaluación QSM».)
- **Qué tan firme es la elección:** en esta red, 0.25 le gana a 0.02 por 0.6 pp de peor distancia
  (1.8 contra 2.4), menos que la escala de 2.07 pp de D; con la control, por 2.0 pp. Lo que no
  depende de ese margen: en las dos redes y los dos sujetos, 0.06 (el α actual) es el peor de los
  cuatro, 2.6–7.7 pp sobre 0.25 (3.2–3.5 en esta red).
- **El mejor α de cada sujeto es el mismo en las dos redes:** 0.35 en COSMOS y 0.02 en Sim2.
- **En esta red todo converge mejor:** 0.06 deja de agrandar el cono (0.49–0.73 contra 1.06–1.37
  de la control en COSMOS) y 0.02 ya no se va después del mínimo (31.64 y 35.35 % en la 100,
  contra ~60 % de la control). Con las diferencias de arriba (semilla, iteraciones, `--guard`,
  lote, calendario), eso es «en esta otra red está en otro lugar», no «mejora con el
  entrenamiento».
- **El criterio de estabilidad no discrimina en esta red:** los cuatro lo cumplen, y la elección la
  hace sólo la distancia.
- **La métrica de la regla.** La regla pre-registrada es la del NRMSE, y da 0.25. Con HFEN
  elegiría 0.02 en g10 y 0.35 en la control («RESULTADO de las métricas», abajo). Esa duda la cubre
  evaluar después los checkpoints de la corrida larga a 0.02, 0.06 y 0.25. (Reemplazado el 25-10:
  ver «La evaluación QSM».)

**PRE-REGISTRO: un piso de estabilidad que no usa la χ verdadera.** Desde la primera
reconstrucción del ADMM (sólo el término de datos), z ← M·D(α·z)/α 20 veces, con el α de cada
target_std de la grilla de Sim2, en COSMOS y en Sim2 (~18 min). Se mide el factor de crecimiento
de la diferencia entre pasadas, ‖z_t − z_{t−1}‖², en el cono (media geométrica de las últimas 10).
Se compara qué α dan un factor < 1 con cuáles convergieron de verdad en el ADMM: si coinciden, es un
criterio para elegir α sin la referencia.

**RESULTADO del piso: no discrimina, y no sirve para elegir α.** `scripts/pnp_adentro_del_admm.py
piso --sujeto cosmos|sim2`, red control, GPU 1, 7.0 min (COSMOS) y 12.0 min (Sim2);
`runs/etapa2-pnp/piso_{cosmos,sim2}.json`. Factor de la diferencia entre pasadas en el cono, media
geométrica de las últimas 10:

| target_std | 0.02 | 0.06 | 0.12 | 0.25 | 0.35 | 0.5 |
|---|---|---|---|---|---|---|
| COSMOS | 0.862 | 0.771 | 0.868 | 0.858 | 0.855 | 0.812 |
| Sim2 | 0.874 | 0.850 | 0.794 | 0.791 | 0.810 | 0.860 |

- **Los doce dan < 1** (0.77–0.87), así que el criterio no separa nada. En el ADMM, en cambio, 0.06
  agranda el cono (1.06–1.37 en COSMOS) y se va a 86.65 % en la 100, y aquí da el factor más bajo
  de COSMOS (0.771).
- **Por qué, como lectura:** sin el término de datos, la red sola se asienta para cualquier α. Lo
  que hace crecer el error del cono en el ADMM es el lazo con el dato, donde D = 0 deja pasar lo
  que la red devuelve; el piso no lo tiene.

**PRE-REGISTRO: métricas además del NRMSE.** Sobre los χ finales de Sim2, de `prep3d-g10-s2-2500`
y de la base y el borde de D en COSMOS (sin corridas nuevas): HFEN (el error filtrado con un
laplaciano de gaussiana, σ = 1.5 vóxeles, relativo al de la referencia) y la energía de M·χ en
altas frecuencias (|k| > 0.25 ciclos/vóxel, la mitad superior del espectro) contra la de la
referencia. Predicción (del teórico): con α alto, χ sale más liso que la
referencia (menos energía en altas frecuencias).

**RESULTADO de las métricas: el NRMSE de 0.25 no se gana a costa de detalle, y χ no sale más liso
con α alto.** `scripts/metricas_chi.py`, CPU, sobre el χ que devuelve la evaluación (el del mínimo
dentro de su ventana); `runs/etapa2-pnp/metricas_chi.json` tiene también los de t = 100. HFEN en %
(menor es mejor); AF, energía de M·χ con |k| > 0.25 ciclos/vóxel sobre la de la referencia (< 1 es
más liso que ella). Los χ del borde de D (control, COSMOS, 0.25 y 0.35) se rehicieron para esto
(`corre --target-std 0.25|0.35 --etiqueta _ts025_chi|_ts035_chi`, GPU 1, 6.0 min cada una):
repiten la evaluación al centésimo (31.36 y 30.62 %); el mínimo en 100 cae en t = 68 y 74, contra
67 y 72, sobre la meseta.

| red | sujeto | target_std | NRMSE | HFEN | AF |
|---|---|---|---|---|---|
| control | COSMOS | 0.02 | 35.39 | 30.61 | 0.36 |
| control | COSMOS | 0.06 | 39.01 | 39.97 | 0.51 |
| control | COSMOS | 0.25 | 31.36 | 26.66 | 0.74 |
| control | COSMOS | 0.35 | 30.62 | 23.69 | 0.87 |
| control | Sim2 | 0.02 | 40.75 | 44.39 | 0.40 |
| control | Sim2 | 0.06 | 46.11 | 49.96 | 0.80 |
| control | Sim2 | 0.12 | 45.18 | 50.28 | 0.86 |
| control | Sim2 | 0.25 | 43.54 | 42.96 | 1.31 |
| control | Sim2 | 0.35 | 46.09 | 44.37 | 1.48 |
| control | Sim2 | 0.5 | 49.29 | 47.06 | 1.50 |
| g10-s2-2500 | COSMOS | 0.02 | 30.88 | 25.59 | 0.46 |
| g10-s2-2500 | COSMOS | 0.06 | 33.79 | 30.44 | 0.57 |
| g10-s2-2500 | COSMOS | 0.25 | 30.28 | 26.31 | 0.46 |
| g10-s2-2500 | COSMOS | 0.35 | 28.47 | 24.38 | 0.58 |
| g10-s2-2500 | Sim2 | 0.02 | 35.93 | 36.28 | 0.57 |
| g10-s2-2500 | Sim2 | 0.06 | 40.18 | 40.47 | 0.79 |
| g10-s2-2500 | Sim2 | 0.25 | 36.99 | 37.95 | 0.65 |
| g10-s2-2500 | Sim2 | 0.35 | 38.79 | 38.41 | 0.97 |

- **0.25 contra 0.06, el cambio que se propondría:** HFEN baja en los cuatro casos (−13.3 en
  control/COSMOS, −7.0 en control/Sim2, −4.1 en g10/COSMOS, −2.5 en g10/Sim2), con el NRMSE. El
  detalle no paga la mejora.
- **La predicción (α alto, más liso que la referencia) no se cumple como patrón.** Casi todos los χ
  tienen menos energía alta que la referencia (0.36–0.97), y el más liso en cada caso es 0.02, el α
  más bajo. 0.25 sale más liso que 0.06 en g10 (0.46 contra 0.57, 0.65 contra 0.79) y más áspero en
  la control (0.74 contra 0.51 en COSMOS; 1.31 contra 0.80 en Sim2, por encima de la referencia,
  como 0.35–0.5 ahí).
- **Fuera de lo pre-registrado, descriptivo:** con HFEN en vez de NRMSE, la regla minimax elegiría
  0.02 en g10 (peor distancia 1.2, contra 1.9 de 0.25 y 2.1 de 0.35) y 0.35 en la control (1.4,
  contra 3.0 de 0.25 y 6.9 de 0.02). Entre 0.02, 0.25 y 0.35 la métrica decide; contra 0.06, no.

**Estimación, sin GPU: registrar tres α (0.02, 0.06 y 0.25) en la evaluación de la corrida larga.**
Supuestos, no mediciones:
- **Costo por iteración del ADMM en COSMOS, en la 5070 Ti:** 3.41 s (`runs/prep3d-g10-s2-2500/qsm.jsonl`,
  100 iteraciones en 341 s).
- **Iteraciones con el techo de 25 y la parada de dos subidas,** las medidas arriba:
  - control: 11, 7 y 25 (43);
  - g10: 25, 10 y 25 (60);
  - peor caso, 75, contra 7–10 de la evaluación actual a 0.06.
- **Cadencia del comando de la corrida larga:** QSM cada 500, 200 evaluaciones en 100 000
  iteraciones. La estabilidad se mide una vez por evaluación y no cambia.
- **Extrapolación a la Blackwell:** ÷2.0–2.5, por el ancho de banda (1.79 contra 0.896 TB/s) y la
  FP32 (109.7 contra 43.9 TFLOPS) de las fichas técnicas. Supone que la red escala como el ancho
  de banda o como el cómputo, y que el ADMM y el entrenamiento, dominados los dos por ∇R, escalan
  igual. Así, el porcentaje no depende del cociente.

| | 5070 Ti | Blackwell (estimado) |
|---|---|---|
| una evaluación con tres α | 2.4–3.4 min (peor, 4.3) | 1.0–1.7 min (peor, 1.7–2.1) |
| lo que se suma por evaluación | +2.0–2.8 min (peor, +3.9) | +0.8–1.4 min (peor, +1.5–1.9) |
| lo que se suma en 100 000 it | +6.8–9.5 h (peor, +12.9) | +2.7–4.8 h (peor, +5.2–6.4) |
| sobre el entrenamiento (8.77 s/it en la 5070 Ti) | +2.8–3.9 % (peor, +5.3 %) | igual |

La otra forma de registrarlos es sin tocar la corrida: estados cada 500 (~9.4 GB) y la evaluación de
los tres α después, con el mismo costo de GPU fuera de la corrida.

**Pendiente, diseñada y sin correr: qué hace que la red agrande o achique el cono.** Dos hipótesis
(«RESULTADO del borde»): la amplitud del error en unidades de σ₀ (con un error del orden de su
ruido, la red lo trata como estructura) o el contraste de la anatomía que ve (α escala las dos a la
vez). Para separarlas, una pasada de la red control sobre v = a·α·M·χ_ref + b·e, con e el error de
entrada del ADMM de A en t = 8 (base, target_std 0.06), a ∈ {1, 4} y b ∈ {1, 4}: cuatro pasadas
(~1 min de GPU), y el factor de energía del error del cono, sale/entra, en cada una. Si cambia con
b y no con a, es la amplitud del error; si cambia con a y no con b, el contraste. Queda fuera de
esta etapa (decisión del coordinador).

**Pendiente para `../tdv`, no se corre aquí:** el ADMM pseudo-3D (k = 3) con la red 2D oficial a
target_std 0.06, 0.25 y 0.35 en COSMOS. Si α alto también mejora la oficial, es una propiedad del
ADMM y del cono; si no, es de esta red. Que se corra lo decide el autor.

### Control de B: la onda fuera del cono (2026-10-21)

**PRE-REGISTRO**, escrito antes de lanzar. Las dos ondas de B (períodos 29.7 y 5.9 vóxeles), con la
misma amplitud (RMS 0.040 en la máscara de evaluación) y la misma regla de proyección, con el vector
de onda en el plano de los ejes 0 y 2 (B0 es el eje 2) a 0°, 54.7° (el cono) y 90° de B0.

- **Predicción (del freno):** en cada período, las tres direcciones dan la misma fracción de
  supervivencia, ±0.05. La red se entrenó con ruido isótropo y no tiene información sobre B0: una
  diferencia sólo podría venir de kernels aprendidos anisótropos, y si sale, es un hallazgo.
- **Si sobreviven igual que en el cono,** la red no distingue el cono: sólo mira el tamaño y la
  frecuencia de la onda.

**RESULTADO del control de B: se cumple.** `scripts/pnp_adentro_del_admm.py control-b`, GPU 1,
~2 min. Fracción de la onda que sobrevive a la red (proyección; RMS 0.040):

| período | 0° de B0 (\|D\| 0.667) | 54.7°, en el cono (\|D\| 0.009–0.023) | 90° (\|D\| 0.333) |
|---|---|---|---|
| 29.7 vóx | 0.914 | 0.923 | 0.925 |
| 5.9 vóx | 0.521 | 0.548 | 0.547 |

- En cada período, las tres direcciones quedan dentro de 0.011 y 0.027: la red no distingue el
  cono. Lo que decide cuánto deja pasar es el período de la onda (y, en B, su amplitud), no su
  dirección respecto de B0. Que en el ADMM el error que crece sea el del cono es del acople, no de
  la red: fuera del cono los datos lo corrigen, y en el cono no.
- La onda de las rayas de B (5.9 vóx, con componentes en los tres ejes de la grilla) dio 0.47, algo
  menos que estas, en el plano de los ejes 0 y 2 (0.52–0.55). Con la grilla podría haber alguna
  anisotropía de los kernels; no está medida.

### C: la rama ciega (2026-10-21)

**PRE-REGISTRO**, escrito antes de lanzar (regla 5).

- **Corrida:** las banderas exactas de `runs/prep3d-control/plan.sh` (el mismo código congelado,
  `runs/prep3d-control/codigo/train3d.py`), más `--sigma-max 55`, a 1 500 iteraciones en la GPU 1,
  en `runs/etapa2-ciega/`. Difiere de la control sólo en `--sigma-max 55` (semilla, datos,
  iteraciones y calendario iguales): verificado al arrancar, los `config.json` difieren sólo en
  `out` y en `sigma_max` (0 contra 55).
- **Motivo (de A):** desde t = 8 la red control agranda el error del cono (×1.06–1.35), y eso es la
  divergencia. En 2D, entrenar ciego bajó el número de estabilidad de 23.2 a 2.14 y el ADMM
  convergió. Entrenar con ruido gaussiano no le enseña a la red a llenar el cono, pero puede
  cambiar si lo agranda.
- **Principal:** el factor de energía del error del cono (sale/entra de la red) en t ≥ 8, en el
  ADMM instrumentado como A (100 iteraciones, COSMOS, target_std 0.06). La ciega lo da ≤ 1; la
  control da 1.06–1.35. No se cambia después de ver los resultados.
- **Secundario 1:** que el NRMSE mínimo (25 iteraciones con la parada de dos subidas) no empeore
  más que el umbral de 2.07 pp (el de D).
- **Secundario 2, mecanismo: 20 pasadas de la red sobre su salida.** Desde el χ del ADMM de A en el
  mínimo (t = 5), escalado como en el ADMM, v = α·M·χ, se aplica v ← M·D(v) 20 veces, para las dos
  redes, con el mismo χ de partida y el mismo umbral |D| < 0.089. Endpoint: el factor medio de
  energía del cono por pasada, (E₂₀/E₀)^(1/20). Es lo que pasa en el cono dentro del ADMM, aislado:
  donde D = 0, χ + s es la salida anterior de la red. Predicción, con dirección, del freno: la
  ciega agranda menos que la control; falla si su factor es mayor o igual que el de la control.
  La teoría no da el signo: el teórico advirtió que podría ir al revés.
- **Secundario 3, agregado antes de que termine el entrenamiento:** la ciega y la control
  comparadas también a target_std 0.25, a 100 iteraciones. La pregunta: ¿entrenar ciego quita la
  necesidad de afinar α, es decir, la ciega a 0.06 se comporta como la control a 0.25?
- **Descriptivos:** la etapa 1 completa sobre la ciega (los cuatro niveles, sin reescalar) y su
  PSNR de validación a σ = 25 contra la control (32.183 dB en el cerebro), como costo.
- **Alcance, escrito antes:** con la parada tras dos subidas, la divergencia no toca el NRMSE
  mínimo. Si la ciega sólo arregla la divergencia, el endpoint de la evaluación de la corrida larga
  no lo va a ver.

**El entrenamiento de C.** `runs/etapa2-ciega/plan.sh` (el de la control con `--out` y
`--sigma-max 55`), unidad systemd `tdv-etapa2-ciega`, GPU 1: 1 500 iteraciones en 0.51 h,
terminada, sin rechazos ni rebobinados (las dos van sin disyuntor, `--guard inf`). PSNR de
validación a σ = 25 en el cerebro, por validación (250 a 1 500):

| red | 250 | 500 | 750 | 1 000 | 1 250 | 1 500 | T | λ |
|---|---|---|---|---|---|---|---|---|
| control | 28.93 | 31.39 | 31.67 | 31.82 | 31.96 | 32.18 | 0.0963 | 0.0978 |
| ciega | 31.30 | 31.43 | 31.41 | 32.07 | 31.87 | 32.56 | 0.1011 | 0.0000 |

- **Costo (descriptivo):** a la 1 500 la ciega no cuesta PSNR a σ = 25: queda 0.37 dB sobre la
  control. Es una corrida por brazo.
- **λ de la ciega está trabada en su pared, 0:** la proyección la recorta a [0, 1 000] después de
  cada paso (`ddr/model.py:83-84`). Bajó de forma sostenida (0.082 en la 100, 0.067 en la 250,
  0.037 en la 500), tocó 0 en la 950 y quedó exactamente en 0 las 545 iteraciones siguientes: es un
  parámetro en su pared, no uno libre. Con λ = 0 el paso proximal ya no tira hacia la entrada, y la
  red es un flujo puro desde x₀ = z. En 2D, `ciego2d-35k` la había bajado a 0.025.

**RESULTADO de C: el principal se cumple, el mecanismo aislado no.** GPU 1, 25 min en total;
`scripts/pnp_adentro_del_admm.py corre --checkpoint runs/etapa2-ciega/last.pth --etiqueta
_ciega[_ts025]`, `itera [--checkpoint … --etiqueta _ciega]`, `scripts/denoising_por_snr_3d.py
--red … --etiqueta _ciega` y `--grad-r`; la figura, `docs/Sanity Check/21-10-pnp_ciega_vs_control.png`.

| red | target_std | evaluación (t) | pararía en (sin techo) | NRMSE en 100 | energía del cono sale/entra, t ≥ 8 | fuera del cono, t ≥ 8 |
|---|---|---|---|---|---|---|
| control | 0.06 | 39.01 % (5) | 7 | 86.65 % | 1.06–1.37 | 0.03–0.06 |
| ciega | 0.06 | 35.78 % (8) | 10 | 39.51 % | **0.50–0.64** | 0.01–0.06 |
| control | 0.25 | 31.36 % (25) | 70 | 31.22 % | 0.49–0.53 | 0.08–0.09 |
| ciega | 0.25 | **29.63 % (24)** | no para | 29.64 % | 0.44–0.51 | 0.02–0.04 |

- **Principal, se cumple:** a target_std 0.06 la ciega achica el error del cono (0.50–0.64 por
  pasada desde t = 8), donde la control lo agranda (1.06–1.37). Su NRMSE sube despacio después del
  mínimo (39.51 % en la 100, contra 86.65 % de la control).
- **Secundario 1, se cumple:** el NRMSE de la evaluación no empeora, mejora 3.2 pp (35.78 contra
  39.01 %), más que el umbral de 2.07.
- **Secundario 2, falla la predicción del freno:** en las 20 pasadas aisladas, v ← M·D(v) desde el
  χ de A en t = 5, el factor medio de energía del cono por pasada es 0.883 en la control y 0.981 en
  la ciega. La ciega achica menos, no más. Además ninguna agranda: el 1.06–1.37 de la control
  dentro del ADMM no aparece sin el lazo con el dato, como en el piso. Este mecanismo aislado no
  explica la diferencia de las dos redes en el ADMM.
- **Secundario 3:** la ciega a 0.06 no se comporta como la control a 0.25 (35.78 contra 31.36 %).
  Entrenar ciego quita la divergencia a 0.06, pero no la necesidad de afinar α: la ciega también
  mejora con 0.25 (−6.2 pp, a 29.63 %), el mejor NRMSE de COSMOS de la etapa con una red de 1 500
  iteraciones.
- **Descriptivos, la etapa 1 sobre la ciega** (`runs/denoising-snr/resultados_ciega.json`, los 12
  volúmenes, sin reescalar): validación a σ = 25 de 32.556 dB en el cerebro (la del entrenamiento).

  | SNR | PSNR entrada | control: salida (ganancia) | ciega: salida (ganancia) |
  |---|---|---|---|
  | 3.2 | 20.17 | 32.19 (+12.0) | 32.56 (+12.4) |
  | 25 | 37.96 | 30.11 (−7.9) | 39.50 (+1.5) |
  | 50 | 43.99 | 30.02 (−14.0) | 40.30 (−3.7) |
  | 100 | 50.01 | 29.99 (−20.0) | 40.41 (−9.6) |

  La ciega no tiene el piso de ~30 dB de la control: el suyo está en ~40 dB, y mejora la entrada
  hasta SNR 25. A SNR 50–100 sigue dejándola peor.
- **‖∇R‖ en el cerebro, limpia contra ruidosa a σ = 25** (3 volúmenes, `grad_r.json`): la control
  da 595–625 contra 3 040–3 115 (cociente 0.20); la ciega, 225–292 contra 7 384–7 717 (0.030–0.038).
  La ciega casi se frena sobre las imágenes limpias, 5–6 veces más que la control.
- **Lo que dice:** con σ fijo a 0.06, la divergencia es de esta red y la ciega no la tiene. Con
  0.25, las dos convergen y la ciega queda 1.7 pp abajo, en un solo sujeto y una corrida por
  brazo. **Lo que no dice:** por qué. El freno sobre lo limpio (‖∇R‖) es compatible, y las
  pasadas aisladas no lo explican. Tampoco si la ventaja se mantiene en Sim2 ni con 100 000
  iteraciones.

## 2026-10-21 — La distribución de χ₀ no explica el α distinto de COSMOS y Sim2

α se fija con χ₀, el primer x-update: χ₀ = F⁻¹[μ₂·D·F(Wy/(W2 + μ₂)) / (μ₂·D² + μ)], con μ = 0.008 y
μ₂ = 1, en la máscara M (`scripts/distribucion_chi0.py`, en CPU). La pregunta: ¿alguna estadística
de χ₀ separa a los dos sujetos tanto como su mejor α? Con la red control y la parada de la evaluación, el mejor target es 0.35 en
COSMOS y 0.02 en Sim2, o sea α ≈ 28 y 1.7: 16×. Ese 16× no es firme: esa parada mira el χ real
(in vivo no existe), a t = 100 el mejor de la control en Sim2 es 0.25, y los dos mejores están en
el borde de su grilla. En ppm:

| | std | MAD·1.48 | p99 de \|χ\| | p99.9 | máximo | curtosis |
|---|---|---|---|---|---|---|
| χ₀ de COSMOS | 0.0124 | 0.0089 | 0.046 | 0.087 | 0.195 | 13.6 |
| χ₀ de Sim2 | 0.0115 | 0.0116 | 0.031 | 0.057 | 0.207 | 5.2 |
| χ real de COSMOS | 0.0280 | 0.0195 | 0.106 | 0.208 | 0.391 | 18 |
| χ real de Sim2 | 0.0282 | 0.0238 | 0.078 | 0.163 | 2.034 | 140 |

- Ninguna estadística de χ₀ difiere más de 1.5× entre los sujetos, y las colas más pesadas son las
  de COSMOS, no las de Sim2. Una normalización robusta (MAD, percentiles) daría casi el mismo α a
  los dos: no explica el 16×.
- Donde sí difieren es en el χ real: Sim2 tiene unas pocas fuentes fuertes (hasta 2.0 ppm, contra
  0.39; curtosis 140, contra 18) que χ₀ no muestra (llega a 0.21). Si esas fuentes son la causa,
  ninguna regla calculada sobre χ₀ en t = 0 puede anticiparlas. Es hipótesis (E1 la descartó: abajo), y no la primera: el
  Agente teórico pone antes la calidad del dato (en Sim2 la fase correlaciona 0.883 con D∗χ
  real). Recortar las fuentes y rehacer el campo cambia dos cosas a la vez. Su control es Sim2 con
  el campo rehecho como D∗(M·χ real) más ruido y sin recortar (E1); antes, en CPU, va el desajuste
  del campo en cada sujeto (E0).
- **Para el diseño:** con desvíos de χ₀ que difieren 7 %, target_std 0.25 ya es en la práctica un
  α fijo (20.2 en COSMOS, 21.7 en Sim2), a ≤ 2.8 pp del mejor en las ocho lecturas (dos redes, dos
  sujetos, evaluación y t = 100). Ninguna versión de target_std, robusta o no, ve los artefactos:
  mide contenido, no error. (Con la ciega no vale: en la 7 936, 0.25 queda 8.3 pp sobre el mejor en
  COSMOS y 9.1 pp en Sim2; ver «curvas por target», 23-10.)

## 2026-10-21 — E0: el desajuste del campo, y el pre-registro de E1

Las dos pruebas las propuso el Agente teórico para decidir si α debe seguir al dato. E0 corre en CPU,
sin la red (`scripts/e0_desajuste.py`; los números, en `runs/etapa2-pnp/e0_desajuste.json`).
σ_d = rms de W·(φ − D∗(M·χ real)) en M; α_MAP = σ₀√μ/σ_d, de λ_ef = μ(σ₀/α)² = σ_d².

| | σ_d | correlación de φ con D∗(M·χ real) | desajuste a ≤ 5 vóxeles de \|χ\| > 0.3 ppm | α_MAP (target) |
|---|---|---|---|---|
| COSMOS | 0.00199 | 0.80 | 0.9 %, en el 0.9 % de los vóxeles | 4.4 (0.054) |
| Sim2 | 0.00251 | 0.88 | 4.7 %, en el 0.1 % de los vóxeles | 3.5 (0.040) |

- **La calidad del dato no separa a los sujetos:** el desajuste de Sim2 es 1.26× el de COSMOS, y
  COSMOS correlaciona peor (0.80 contra 0.88; en χ ≠ 0, sin el borde, 0.967: «Auditoría del
  ruido…», 26-10). La predicción de la hipótesis (COSMOS con
  correlación ≥ 0.99) falla porque COSMOS trae ruido, no porque su modelo no cuadre (ver abajo).
- **La forma del desajuste** (agregado tras la revisión del Agente teórico):
  - COSMOS: es el de una simulación directa con ruido. Pendiente 0.998; blanco frente al cono (el
    21.7 % de su energía ponderada cae en el cono, igual que la fracción del k-space); concentrado
    donde W es chico (0.0064 ppm sin ponderar contra 0.0020 ponderado), como el ruido de una señal
    compleja.
  - Sim2: pendiente 0.965, y sólo el 13.9 % de su energía en el cono, contra 21.6 % si fuera
    blanco: cerca de un tercio tiene forma de campo. El campo de E1 es, aproximadamente, Sim2 sin
    esa parte.
  - En los 41 vóxeles de Sim2 con |χ| > 0.3 ppm, W² vale 0.084 (0.58 en los 133 de COSMOS): ahí el
    dato casi no pesa, y por eso χ₀ llega sólo al 10 % del máximo real.
- **α según el error del dato (la opción c) cae en la joroba:** α_MAP da target 0.054 en COSMOS y
  0.040 en Sim2, la zona de los peores resultados.
- **El entorno de las fuentes no explica que Sim2 prefiera α bajo:** recalculado sin ese entorno, el
  NRMSE de los χ guardados mantiene el mejor target en las cuatro lecturas, y la ventaja del lado
  bajo crece (control, evaluación: de 2.79 a 4.34 pp; g10: de 1.06 a 1.55 y de 0.64 a 1.25 pp).

**PRE-REGISTRO de E1**, escrito antes de lanzar (regla 5):
- **Qué:** Sim2 con φ = D∗(M·χ real) + σ_d(COSMOS)·N(0, 1)/W en M (semilla 2026), sin recortar las
  fuentes, con los mismos W, M y kernel. Red control; target_std 0.02, 0.06, 0.25, 0.35 y 0.5; 100
  iteraciones (`scripts/pnp_adentro_del_admm.py corre --sujeto sim2-rehecho`). El desvío de su χ₀ es
  0.01142 (en Sim2, 0.01151): target y α absoluto se mueven juntos.
- **Cuándo:** en cola en la GPU 1, tras `ciego-log-l32` (unidad `tdv-e1-sim2`, ~50 min; log en
  `runs/etapa2-pnp/e1.log`).
- **Lectura:** el mejor α absoluto, en la evaluación (`min_eval`) y a t = 100 (`final`).
- **Qué decide:** si lo que importa del dato de Sim2 es la estructura de su desajuste, no su tamaño.
  Si el mejor α sube al lado de COSMOS (≥ 0.25) y el NRMSE baja varios pp, es esa estructura. Si
  queda en ≤ 0.02, como Sim2, es el contenido del sujeto (sus fuentes o su anatomía), y recién ahí
  se prueba Sim2 con las fuentes recortadas.

**RESULTADO de E1 (2026-10-22): es la estructura del desajuste.** GPU 1, ~50 min
(`runs/etapa2-pnp/adentro_sim2r_ts*.json` y `e1.log`). NRMSE de la red control en Sim2 y en E1, en
la parada de la evaluación (entre paréntesis, su t) y en t = 100:

| target_std | Sim2, evaluación | Sim2, t = 100 | E1, evaluación | E1, t = 100 |
|---|---|---|---|---|
| 0.02 | **40.75 %** (13) | 60.18 % | 34.28 % (25) | 33.78 % |
| 0.06 | 46.11 (5) | 83.53 | 38.31 (7) | 74.84 |
| 0.25 | 43.54 (25) | **42.97** | 30.99 (25) | 29.44 |
| 0.35 | 46.09 (10) | 47.93 | **30.50** (25) | **28.73** |
| 0.5 | 49.29 (4) | 58.14 | 33.17 (25) | 31.26 |

- **El mejor target sube de 0.02 a 0.35, el de COSMOS, y el NRMSE baja ~10 pp** (de 40.75 a
  30.50 % en la evaluación, de 42.97 a 28.73 % en t = 100). Por la regla pre-registrada, lo que
  hace que Sim2 prefiera α bajo es la estructura de su desajuste (la parte con forma de campo), no
  el contenido del sujeto: E1 conserva sus fuentes fuertes y su anatomía. La prueba de las fuentes
  recortadas ya no hace falta.
- El tamaño del desajuste no lo explica: E1 lo cambia 1.26× (σ_d de 0.00251 a 0.00199), y el mejor
  α, 17×.
- **0.06 sigue siendo el peor:** la evaluación para en t = 9 (mínimo en t = 7), y en t = 100 llega a
  74.8 %.
- **target_std 0.25 queda a 0.49 pp del mejor en la evaluación y a 0.71 pp en t = 100.**
- **Lo que no dice:** si un dato in vivo trae desajuste con forma de campo (un fondo mal quitado,
  por ejemplo), α bajo podría volver a convenir. Es una red y una simulación.

**Pesar la regularización en cada paso del ADMM** (pregunta del autor, 2026-10-21; respuesta del
Agente teórico, derivada y sin medir). Se hace en la literatura (DPIR, Zhang et al. 2021; Chan et
al. 2017; Wei et al. 2020), pero aquí α es la única perilla y cambia también el contraste que ve la
red. Ajustarlo al error de cada paso lo lleva a la zona de α_MAP (target 0.040–0.054, E0), junto a
0.06, el peor. La versión con fundamento es la red ciega, que se adapta al ruido sin reescalar. No
se implementa. (Un calendario de target fuerte → suave se probó el 25-10 y no sirve: ver «ADMM en
dos rondas».)

## 2026-10-22 — El panel de W&B y su prueba corta: `panel-l8`

Pedido del autor: que W&B informe lo necesario para decidir si cortar o seguir la corrida larga,
con la estructura del panel de Comet del laboratorio (métricas externas en COSMOS y Sim2, pérdidas
en varias validaciones, láminas por evaluación). Lo nuevo es todo evaluación, sin tocar la
aritmética del entrenamiento:
- **QSM, en COSMOS (`qsm/*`) y en Sim2 (`qsm_sim2/*`, `--qsm-sim2 datasets/qsm_sim2`):** además
  del NRMSE, el HFEN y el XSIM de FANSI (`compute_hfen.m`, `compute_xsim.m`), el residuo con el
  dato 100·‖W(φ − D∗χ)‖/‖Wφ‖ (con el de χ_GT como referencia), el factor del cono, y las mismas
  métricas de χ₀, la primera reconstrucción sin la red (`chi0_*`), como línea base. Sim2 no elige
  checkpoints ni toca la meseta. Sim2 se arma una vez con `datasets/prepara_sim2.py`.
- **Denoising a otros SNR (`--val-snr 25 50 100`, `val_snr25/*`, `val_snr50/*`, `val_snr100/*`):** PSNR y NRMSE en
  el cerebro de la entrada y de la salida, además de la validación a σ = 25 (SNR 3.2), que gana
  `val/nrmse_pct_cerebro`.
- **Láminas:** QSM con GT | χ₀ | reconstrucción | error, y denoising a cada SNR.
- **Épocas (`--epocas`, pedido del autor):** una época es una pasada por los volúmenes de
  entrenamiento, un parche de cada uno en el orden de una permutación por época (sin reemplazo),
  con un DataLoader que se agota al terminarla (`lotes_por_epoca` en `train3d.py`; el volumen de
  cada parche, en `PatchDataset3D._volumen_de` de `ddr/data3d.py`). Iteraciones por época =
  volúmenes // lote: con los 1 000 de `lista_train_1000.txt`, 125 con lote 8 y 31 con lote 32.
  W&B registra `train/epoca` (con fracción) como eje x alternativo. Sin la bandera, cada parche
  sale de un volumen al azar con reemplazo, bit a bit como antes; con ella cambia la receta, y
  una corrida con épocas no se compara bit a bit con una sin ellas.

**PRE-REGISTRO de `panel-l8`**, escrito antes de lanzar (regla 5).
- **Corrida:** la configuración de `ciego-log-l32` (ciega log-uniforme, σ en [0.81, 55]/255,
  semilla 2, lr constante, sin disyuntor, `--mascara-cerebro`, `lista_train_1000.txt`) con lote 8,
  500 iteraciones, validación y QSM cada 50 (COSMOS y Sim2, target_std 0.25), `--val-snr 25 50
  100`, sin estabilidad, W&B online (`cienfueg0z-tdv/training-tdv3d`). Un primer arranque con
  `--val-snr 25 100` se paró en la iteración 50 para sumar SNR 50 (pedido del autor); quedó en
  `runs/panel-l8/intento-sin-snr50/` (borrada el 25-10).
- **Pregunta:** ¿el panel deja ver, en 10 evaluaciones, si la red aprende, y cuánto cuesta cada
  evaluación (para fijar la cadencia de la corrida larga)?
- **Predicción:** la pérdida y el NRMSE de validación bajan sobre todo en las primeras ~200
  iteraciones; el NRMSE QSM queda bajo el de χ₀ (65.8 % en COSMOS y 70.3 % en Sim2 en la prueba
  de humo); cada evaluación cuesta 3–4 min en la 5070 Ti.
- **Alcance:** prueba del panel, no del modelo: lote 8 y 500 iteraciones no dicen nada del
  resultado de la corrida larga.

**RESULTADO de `panel-l8` (2026-10-22): el panel muestra que aprende, y que la ganancia a mucho
ruido se paga a poco ruido.** 500 iteraciones en 57 min en la 5070 Ti, con las 10 evaluaciones; W&B
`cienfueg0z-tdv/training-tdv3d`, run `nl54gxl9` (borrada el 25-10, con su carpeta). NRMSE en %,
entrada → salida, con SNR de la media (de pico: 10.2, 79, 158 y 316):

| it | SNR 3.2 | SNR 25 | SNR 50 | SNR 100 | COSMOS (mín) | HFEN | XSIM | Sim2 (mín) |
|---|---|---|---|---|---|---|---|---|
| 50 | 29.9 → 26.3 | 3.85 → 3.53 | 1.93 → 1.94 | 0.96 → 1.26 | 47.6 | 35.9 | 0.639 | 54.8 |
| 250 | → 25.2 | → 3.51 | → 1.99 | → 1.38 | 39.7 | 24.0 | 0.710 | 52.9 |
| 500 | → 13.3 | → 4.24 | → 3.52 | → 3.31 | 29.9 | 23.0 | 0.807 | 44.9 |

- **χ₀, la línea base sin red:** COSMOS 65.8 % y XSIM 0.462; Sim2 70.3 % y 0.428. Desde la primera
  evaluación la red queda muy por debajo.
- **Se cumple lo predicho para el aprendizaje:** el NRMSE a SNR 3.2 y el de COSMOS bajan todo el
  tramo, y no se aplanan en 500 iteraciones (la mayor caída es al final, no en las primeras ~200,
  como se predijo).
- **No predicho: a poco ruido empeora mientras aprende.** Al principio la red casi no toca la
  entrada (a SNR 25 la mejora, 3.53 contra 3.85); a medida que aprende a quitar ruido fuerte,
  empeora la entrada a SNR 25, 50 y 100 (a la 500, 4.24, 3.52 y 3.31 %). Es el compromiso que
  `val_snr*` hace visible y que la validación a σ = 25 sola no muestra.
- **Residuo con el dato:** las reconstrucciones dejan 8–13 %, menos que χ_GT (22.0 %): explican la
  fase mejor que la verdad, o sea que también ajustan su ruido y su desajuste (lectura, no
  medición). Sube hacia el de la GT a medida que la red mejora. El factor del cono, 1.9–2.3: el
  error se concentra en el cono, como en la etapa 2.
- **Costo de cada evaluación:** el ADMM de COSMOS, 14–86 s (86 cuando corre las 25 iteraciones),
  y el de Sim2, 17–69 s.

## 2026-10-22 — La ciega con épocas, lote 32 y 5 000 iteraciones: `ciego-epocas-l32`

**PRE-REGISTRO**, escrito antes de lanzar (regla 5).
- **Corrida:** `runs/ciego-epocas-l32`, la configuración de `ciego-log-l32` (σ log-uniforme en
  [0.81, 55]/255, semilla 2, lote 32 con micro-lote auto, lr 4e-4 constante, sin disyuntor,
  `--mascara-cerebro`, `lista_train_1000.txt`) más `--epocas` (31 iteraciones por época), 5 000
  iteraciones (161 épocas), validación y QSM cada 496 (16 épocas) y en la última, COSMOS y Sim2
  con target_std 0.25, `--val-snr 25 50 100`, estados cada 496, sin estabilidad, W&B online.
  Código congelado en `runs/ciego-epocas-l32/codigo` (borrada el 25-10; la 10k tiene su copia en
  `runs/ciego-epocas-l32-10k/codigo`).
- **Pregunta:** ¿la ciega sigue mejorando después de la 2 250? En `ciego-log-l32` la validación a
  σ = 25 quedó plana desde la 500 mientras la pérdida y λ seguían bajando (su resultado, arriba).
  Se lee en el panel: NRMSE a SNR 3.2, 25, 50 y 100, y NRMSE, HFEN y XSIM en COSMOS y Sim2, por
  evaluación.
- **Predicción:** el NRMSE QSM y el de SNR 3.2 siguen bajando, cada vez más lento; a SNR 50–100 la
  salida sigue peor que la entrada (en `ciego-log-l32` a la 2 250, ~2 % contra 0.96–1.93 %).
- **No es una comparación con `ciego-log-l32`:** las épocas cambian el sorteo de los volúmenes y
  la cadencia es otra; una corrida por brazo ordena, no da el número final.
- **Costo:** ~9 s por iteración en la 5070 Ti (12.5 h) más 11 evaluaciones de 3–4 min.

## 2026-10-22 — Dónde está el error del ADMM y el piso de ~2 %

- **El cono es el 21.7 % del k-space** (|D| < `UMBRAL_CONO`, sin la componente continua; 21.7 % en
  COSMOS, 160³, y 21.6 % en Sim2). Entonces, en la tabla A de la Etapa 2, el error que entra a la
  red en t = 1 (23 % en el cono) se reparte casi como uno blanco entre cono y fuera (sólo en ese
  reparto: espectralmente no es blanco, «¿α sin GT?», 23-10): no nace en el cono. Se concentra
  ahí después, porque la red quita el de afuera y no el del cono (34–86 % del error que sale); en
  las reconstrucciones de `panel-l8`, el factor del cono 1.9–2.3 es 41–50 % de la energía del error.
  Corregido en `docs/Sanity Check/por-que-no-reescalar/` (txt y figura 3).
- **El piso de ~2 % a SNR 50–100, lectura, no medida:** la variación fina propia de los T1
  (σ 0.0041–0.0083, «El SNR propio de los datos») está en la referencia, y sobre rms(y) = 0.337 en
  el cerebro equivale a 1.2–2.5 % de NRMSE. Una red que la quita como ruido queda en ese rango
  aunque acierte la anatomía; y a SNR 100 el ruido sumado (σ 0.0032) es menor que esa variación.
  Se probaría correlacionando el error de salida con el residuo fino de la referencia.

## 2026-10-22 — `ciego-epocas-l32` estalló; se relanza a 10 000: `ciego-epocas-l32-10k`

**RESULTADO de `ciego-epocas-l32`: corrida perdida, cortada a mano en la 3 200** (23:10;
`estado.json` escrito a mano). Estalló hacia la iteración 200, sin disyuntor (`--guard inf
--guard-abs inf`): la `loss` pasó de 72 (it 175) a 4 179 (200) y a 9.5·10⁶ (225), y el |g| medio
de la ventana, de 583 a 5·10⁶ (máximo 1.8·10¹¹ en la 225). El primer |g| no finito llegó en la 461;
los dos rebobinados (485 y 529) volvieron a la 253, un respaldo de después del estallido. PSNR en
el cerebro: −32.3 dB en la 496 y −8.2 dB de la 992 a la 2 976, sin volver. El ADMM devolvió χ0
(65.82 % en COSMOS, 70.25 % en Sim2) en las seis evaluaciones. Por la regla 6, la muerte no se le
atribuye a las épocas: una corrida sola dice qué pasó, no la tasa de la configuración.

**PRE-REGISTRO de `ciego-epocas-l32-10k`** (`runs/ciego-epocas-l32-10k/plan.sh`, mismo código
congelado). Lo pidió el autor: la misma corrida a 10 000 iteraciones. No había de dónde
reanudarla: el primer checkpoint, el de la 496, ya está roto. Es la misma configuración, con dos
cambios:
- **10 000 iteraciones;**
- **la cota absoluta del disyuntor encendida** (`--guard inf`, sin `--guard-abs inf`: 42 622 a
  48³). El relativo sigue apagado, como en `ciego-log-l32`, cuyo primer arranque trabó la cota y
  no el relativo (abajo, «Qué pasó con los estallidos»). Mientras no rechaza, la corrida es bit a bit la de sin disyuntor: `ciego-log-l32`, sana,
  no pasó de |g| ~10³ en 2 250 iteraciones, 40 veces debajo de la cota, y este estallido la cruzó
  por 100.

**El riesgo:** un pico de arranque sobre la cota (el relanzamiento de `ciego-log-l32` tuvo uno de
1.5·10⁶) podría trabarla como al primer arranque de `ciego-log-l32`. Si en la 75 lleva menos de 50 pasos efectivos (`ef`
en el log), se para y lo decide el autor.

**La pregunta:** si la ciega sigue mejorando hasta la 10 000. Se lee en cada evaluación (cada 496):
- el NRMSE de `val_snr25/50/100`, y el de SNR 3.2;
- el NRMSE mínimo del ADMM en COSMOS y en Sim2, contra sí misma.

`ciego-log-l32` en la 2 250 es una referencia, no una comparación.

**Costo:** ~9.2 s por iteración con las evaluaciones, ~25–27 h.

**RESULTADO del primer intento (semilla 2 con la cota): trabada, cortada a mano en la 275**
(`runs/ciego-epocas-l32-10k/intento-cota-s2/`, borrada el 25-10).
- Hasta la 175, la `loss` fue idéntica a la de `ciego-epocas-l32`, a tres decimales: misma semilla,
  mismas épocas.
- En la 180 la cota rechazó el estallido (|g| 1·10⁵–1.6·10⁶ con `loss` 116–2 372), pero el
  respaldo era el de la 3, porque se refresca recién a las 250 sanas.
- Hubo tres ciclos de rebobinar a la 3 y volver a entrar en ~5 pasos: 83 rechazos, 3
  rebobinados y 7 pasos efectivos en la 275.
- Lectura: la cota sólo protege una vez refrescado el primer respaldo. Un estallido antes de la 250
  la deja dando vueltas, como `--guard 10` en el arranque de `ciego-log-l32`.

**Relanzamiento: semilla 3, sin disyuntor** (`--guard inf --guard-abs inf`, la configuración que el
autor eligió para `ciego-epocas-l32`). La semilla 2 con épocas repite el estallido de la 180, y
la cota no lo saca. La pregunta y la lectura no cambian. Si estalla, se pierde como
`ciego-epocas-l32`: se ve en W&B como el PSNR en el cerebro hundido y el ADMM devolviendo χ0.

**Extender una corrida sana sí se puede:** `--iterations` está en `LIBRES` (`train3d.py:169`), así
que se reanuda con `--reanudar` y otro techo.

**Arranque de la semilla 3 (00:38 del 23-10):** sana en la 300, con `loss` 42–66, sin rechazos y
24 h restantes según el log. Hacia la 75 tuvo un pico de |g| 1.11·10⁵, sobre la cota absoluta: sin
disyuntor lo absorbió en ~50 iteraciones. Es otra señal de que la cota habría trabado este
arranque.

## 2026-10-23 — Qué pasó con los estallidos y cómo se arregla el disyuntor 3D

**Qué pasó**, en las tres corridas ciegas de lote 32:
- **`ciego-epocas-l32` (semilla 2, sin disyuntor).** El estallido fue de golpe. |g| sobre la
  mediana de las últimas 100: 2× en la 176, 18× en la 178, 1 389× en la 179 y 10⁴–10⁵× después. Hubo
  precursores aislados a 25× (167) y 19× (171). Se aplicaron todos los pasos. Con `--guard inf`,
  «sano» quiere decir «finito», así que el respaldo se refrescó en la 253 con la red ya rota, y los
  rebobinados por gradiente no finito (485 y 529) volvieron ahí.
- **Primer intento de `ciego-epocas-l32-10k` (semilla 2, con la cota).** La cota rechazó el
  estallido en la 180, pero el respaldo era el de la 3, porque se refresca a las 250. Rebobinó
  tres veces a la 3, volvió a entrar las tres, y quedó trabada.
- **Primer arranque de `ciego-log-l32` (`--guard 10`).** Lo trabó la cota, no el relativo, que
  no estaba armado. La decisión de entonces, apagar el disyuntor entero, apagó también el relativo,
  que no tenía la culpa.

**El relativo a 10× sirve con ruido ciego** (`scripts/razones_disyuntor_3d.py`; la razón es |g| sobre la mediana de la traza, columnas 5 y 6, con el relativo armado):

| corrida | filas armadas | p99.9 | máximo | lotes > 10× | racha máxima > 10× |
|---|---|---|---|---|---|
| `ciego-log-l32` | 2 150 | 6.9 | 16.9 | 2 | 1 |
| `ciego-epocas-l32-10k`, semilla 3, hasta la 4 464 | 4 364 | 4.4 | 8.9 | 0 | 0 |
| `panel-l8` (lote 8) | 400 | 4.7 | 5.1 | 0 | 0 |
| `ciego-epocas-l32`, hasta la 178 | 79 | 24.4 | 24.9 | 3 | 1 |

El estallido de la 179 (1 389×) queda tres órdenes por encima, y los rechazos sanos son sueltos,
lejos de los 25 seguidos que rebobinan.

**La cota absoluta sólo actúa en el arranque, y en 3D ahí traba.** Los picos de las primeras 100
iteraciones fueron 1.46·10⁶, 1.11·10⁵, 1.06·10⁴ y 4.8·10⁷ (`panel-l8`), y los cuatro se resolvieron
solos sin cota. La cota (42 622) habría rechazado tres. Con el relativo armado, ya no decidió nada
en ~5 000 filas («El disyuntor en 3D»).

**El arreglo, a implementar en `train3d.py`** (el 2D no se toca):
1. **Por defecto en 3D:** `--guard 10` y sin cota absoluta (`--guard-abs inf`). Lo único que
   cubre el arranque es el rechazo de lo no finito.
2. **El respaldo sólo se refresca con el relativo armado y finito**, y cada 50 iteraciones sanas
   en vez de 250: un rebobinado vuelve a un estado certificado reciente, no a la 3. Sin relativo,
   el respaldo queda en el del arranque, en vez de en uno roto.
3. **Aviso de corrida trabada:** con dos rebobinados seguidos al mismo respaldo, el log dice
   `TRABADA` y W&B manda una alerta. No se corta el bucle (AGENTS.md).
4. **Alertas de W&B** en cada rebobinado; cuando el PSNR en el cerebro cae 3 dB o más bajo el
   mejor; y cuando el ADMM devuelve χ0 (mínimo en la iteración 0). Así el autor se entera sin mirar.
5. **`guard`, `guard_abs` y `guard_respaldo` pasan a `LIBRES`:** el disyuntor es inerte mientras
   no rechaza, así que una corrida sin él se puede reanudar protegida.

**La prueba, PRE-REGISTRO:** la semilla 2 con épocas reproduce el estallido. Hasta la 175, la
`loss` coincidió a tres decimales en dos lanzamientos. Se corren 500 iteraciones con el arreglo, y
pasa si:
- el estallido se rechaza;
- rebobina a lo sumo a un respaldo de la 150 o después;
- llega a la 500 con más del 80 % de iteraciones efectivas, sin `TRABADA`.

Si se traba igual, el respaldo reciente no alcanza, y se vuelve a pensar antes de tocar nada
más (regla 1).

**α y el «máximo».** Normalizar χ0 como los T1 (percentil 99.5 de |χ0| = 1) equivale a target_std
0.205 en COSMOS y 0.289 en Sim2, cerca del 0.25 en uso. Con el máximo literal (el máximo de |χ0|
= 1), a 0.063 y 0.056: el 0.06 del 2D, el peor medido en 3D (`scripts/alfa_por_maximo.py`).

## 2026-10-23 — El disyuntor 3D arreglado y el SNR unificado

**Lo implementado** (con tests). El autor lo pidió («arregla todos
nuestros mecanismos de trabajo con explosiones»), y es el plan de «Qué pasó con los estallidos»:
- **Defaults 3D:** `--guard 10`, `--guard-abs inf` (`cota_absoluta` y `VALORES_BSDS` salen) y
  `--guard-respaldo 50`.
- **El respaldo** se refresca sólo con el relativo armado (`aplica_disyuntor`). El primero se sigue
  tomando a las 3 limpias.
- **`TRABADA`:** dos rebobinados seguidos al mismo respaldo. Se anota, se avisa y el bucle sigue.
- **Alertas** (`alerta` en `train3d.py`, `seguimiento.alerta`): van al log como `ALERTA:` y a W&B
  con `run.alert`, en cinco casos:
  - cada rebobinado;
  - `TRABADA`;
  - el PSNR de validación peor que el de la entrada ruidosa;
  - el PSNR de validación 3 dB o más bajo el mejor;
  - el ADMM con su mínimo en la iteración 0, o sea que devuelve χ0.

  Si la alerta falla, no apaga el registro de métricas.
- **`guard`, `guard_abs` y `guard_respaldo` pasan a `LIBRES`.**
- **Para que lleguen por correo:** en W&B, *User settings → Alerts*, «Scriptable run alerts».

**SNR unificado: SNR = pico/σ, con el pico = 1** (el percentil 99.5 al que se normaliza cada T1).
σ = 1/SNR, el PSNR de entrada es 20·log10 SNR, y σ = 25/255 es SNR 10.2.
- `--val-snr` pasa a esta convención: σ = 1/SNR en vez de 0.316/SNR.
- Antes, SNR 25, 50 y 100 eran σ = 3.2, 1.6 y 0.8 /255 (PSNR de entrada 38–50 dB): casi no se
  veía ruido. Ahora son σ = 10.2, 5.1 y 2.55 /255 (PSNR de entrada 28.0, 34.0 y 40.0 dB).
- La etapa 1, sus figuras y lo escrito antes de hoy usan media/σ: SNR_pico = SNR_media / 0.316.
  Rehacer la etapa 1 en la convención nueva pide la GPU.
- Por qué el pico y no la media:
  - el σ del entrenamiento ya está en esa escala (25/255, como Kobler);
  - el PSNR también;
  - un mismo número de SNR tiene que querer decir lo mismo en el entrenamiento, en la validación
    y en el ADMM.

  La media/σ es la convención de MRI, pero acá mezclaba dos escalas.
- **En el ADMM**, la misma convención es SNR = 1/RMSE de lo que entra a la red. En la tabla A de la
  Etapa 2 (target 0.06) da 11.1 en t = 1, 13.9 en t = 2, 7.2 en t = 5, 4.9 en t = 10 y 3.8 en
  t = 100. El entrenamiento es 10.2: al empezar, el ADMM está en el nivel de ruido del
  entrenamiento.

**RESULTADO de la prueba con la semilla 2** (`runs/prueba-disyuntor-s2/`, 500 iteraciones con el
código de la rama, sin QSM ni W&B):
- **Contención.** El primer lanzamiento corrió con la suite de tests ocupando la CPU, y se
  apartó del camino de referencia desde la 25 (pérdida 76.5 contra 79.5). Es la regla 3:
  `cudnn.benchmark` elige otros algoritmos. No estalló en 225 iteraciones; quedó en
  `intento-con-contencion/`.
- **Sin contención** repitió el camino a tres decimales hasta la 150.
- **El precursor.** El relativo rechazó el de la 168 (25×), y desde ahí el camino cambió: el
  estallido de la 179 no llegó.
- **Otro estallido.** Llegó en la 497. Entre la 488 y la 496 se aceptaron pasos a 2.4–6.8×, bajo
  la razón 10. Con los pesos de la 496, cada lote da 237–1 089× y una pérdida de 382–805, contra
  ~45, y el relativo rechazó los cuatro que quedaban (497–500). El último respaldo era el de la
  450, sano.
- **La corrida terminó en la 500**, con 4 rechazos seguidos, antes de los 25 que rebobinan. La
  validación de la 500 mide los pesos de la 496: 25.56 dB en el cerebro.
- **Veredicto:** cumple lo pre-registrado. Rechaza los dos estallidos, no hay `TRABADA` y la
  iteración efectiva es del 100 %. Pero el rebobinado al respaldo de la 450 no se llegó a ver en
  vivo, sólo en el test.
- **Lectura, no medida:** con lr 4e-4 constante, la semilla 2 con épocas entra en zonas
  inestables seguido (dos en 500 iteraciones). La razón 10 no impide entrar, porque el paso que
  entra es normal; rechaza una vez adentro.

**Reanudada `ciego-epocas-l32-10k`** desde la 4 960 (23-10, 15:25), con `codigo-disyuntor/` y
`plan-desde-4960.sh`: `--guard 10`, sin cota, respaldo cada 50 y `--val-snr` en SNR de pico.
- W&B abrió otra corrida, `b3vm1g37`, porque la anterior (`1tfszcg4`) ya había registrado hasta
  la 4 975, después del punto de reanudación. En la nueva, `val_snr25/50/100` miden σ = 10.2, 5.1
  y 2.55 /255; en la anterior, 3.2, 1.6 y 0.8. (Las dos se borraron el 25-10: la 10k entera es
  `kxein1yw`; ver «W&B, limpio».)

## 2026-10-23 — El ruido máximo del entrenamiento contra la primera iteración del ADMM

**El ruido del entrenamiento.** Cada parche recibe un σ log-uniforme en [0.81, 55]/255, en las
unidades de la caché, donde el T1 tiene su percentil 99.5 en 1.
- El sorteo: `add_noise_ciego`, `ddr/data3d.py`. El generador es `noise_gen`, con semilla
  `--seed`·100 + 1, y viaja en el estado.
- El σ cuenta sólo el ruido sumado. La variación propia del T1 (σ 0.004–0.008, «El SNR propio
  de los datos») está también en la referencia limpia. Con σ = 55/255 = 0.216 no pesa:
  √(0.216² + 0.008²) = 0.2161. Pesa en el extremo de poco ruido.
- En SNR de pico, el rango es 4.6–315.

**La primera iteración del ADMM:** su entrada es α·χ0. El error escala con α: RMSE = k·target_std,
con k = RMSE(χ0 − χ)/std(χ0) = 1.53 en COSMOS (tabla A: 0.090 con target 0.06) y ~1.77 en Sim2
(de NRMSE(χ0)·RMS(χ), aproximado).
- Con target 0.25: σ efectivo 0.38–0.44, o sea SNR de pico 2.3–2.6. Es más ruido que el máximo
  del entrenamiento.
- Para quedar dentro (σ ≤ 0.216, SNR ≥ 4.6) hace falta target ≤ 0.14 en COSMOS y ≤ 0.12 en
  Sim2.
- La razón entre el pico propio de χ0 (percentil 99.5 de |χ0|) y su error no depende de α: 3.2 en
  COSMOS y 2.0 en Sim2. Bajar α mete el error en el rango entrenado, pero deja a χ más tenue que
  un T1: su percentil 99.5 queda en ~0.7, no en 1.
- Con la red control, en COSMOS no se midió entre 0.06 y 0.25; en Sim2, 0.12 dio 45.2 % contra
  43.6 % a 0.25. Con la ciega no se barrió α.
- Las dos salidas: barrer target 0.10–0.35 con la ciega, o subir `--sigma-max` a ~85/255 (SNR
  3) en la próxima corrida.

**Lectura de `ciego-epocas-l32-10k` en la 4 960** (`qsm.jsonl` y `qsm_sim2.jsonl`; ADMM de la
evaluación, techo de 25):
- **COSMOS:** el NRMSE mínimo fue 35.87 (496), 30.88 (1 488, el `best_qsm.pth`) y 33.09 % (4 960),
  y sube desde la 1 488. El HFEN baja todo el tiempo, de 35.7 a 22.1 %, y el XSIM cae de 0.799 a
  0.770. Su mínimo está en la iteración 24 de 25: no convergió.
- **Sim2:** el mínimo fue 41.99 % (1 984) y 44.57 % (4 960), con la mejor iteración del ADMM
  bajando de 24 a 5.
- **El denoising:** el PSNR en el cerebro sube a la par, de 33.03 a 33.21 dB entre la 3 968 y la
  4 960, y λ baja de 0.1 a 0.005.
- **Lectura:** mejor denoiser no es mejor prior (como en 2D, «el PSNR no elige»). Las altas
  frecuencias mejoran y crece el error de baja frecuencia (el cono, según el factor del panel,
  ×2.2). (Refutado el 23-10: el empeoramiento era del target fijo en 0.25; ver «Barrido de target
  y ADMM a 100 con la ciega».)

## 2026-10-23 — Barrido de target y ADMM a 100 con la ciega

**El pico propio contra las unidades de la red.** Son dos números para dos preguntas distintas.
- **Pico propio de χ0 sobre el RMSE de su error.** No depende de α: es la calidad de la
  inversión del dipolo.
  - Con el percentil 99.5, la misma convención que en los T1: 3.2 en COSMOS y 2.0 en Sim2.
  - Con el máximo literal: 10.3 y 10.2. Pero el máximo es un solo vóxel, 3× el percentil 99.5 en
    COSMOS y 5× en Sim2.
  - Corolario: target 0.06 equivale a normalizar χ0 por su máximo literal. Deja el error en
    σ ≈ 0.097, el σ = 25/255 del entrenamiento 2D. Lectura, no medida: es coherente con que la red
    oficial anduviera mejor con 0.06.
- **En unidades de la red, con el pico de los T1 = 1:** σ = k·target. Es lo que la red reconoce,
  porque no es escala-equivariante (15–28 %): aprendió con un pico de 1 y σ ≤ 0.216. Para una red
  equivariante bastaría el primer número; para la nuestra cuentan los dos.

**Propuesta, no corrida:** subir `--sigma-max` a 85/255 (σ 0.333, SNR 3.0) en la próxima corrida
ciega.
- Cubre la primera iteración con target ≤ 0.22 en COSMOS y ≤ 0.19 en Sim2.
- Con 0.25 hace falta σ 0.38–0.44, o sea ~112/255.

**PRE-REGISTRO tardío:** escrito con el barrido ya lanzado, antes de leerlo.
- `scripts/barrido_target_ciega.py`, sobre `best_qsm.pth` (it 1 488) y el último estado (4 960).
- target_std 0.15, 0.20, 0.25 y 0.30; ADMM de 100 iteraciones sin corte; COSMOS y Sim2.
- Corre en la GPU 0, con permiso del autor por única vez (23-10), porque estaba sin uso.
- Salidas: `runs/ciego-epocas-l32-10k/barrido-target/resultados.json` y
  `runs/ciego-epocas-l32-10k/barrido-target.log`.
- **Preguntas:**
  1. ¿El target bueno de la ciega cae en 0.15–0.30, y es el mismo para los dos checkpoints?
  2. A 100 iteraciones, ¿el ADMM converge o se va, y cuál checkpoint lo hace mejor?
- **Se lee:**
  - el NRMSE mínimo y su t;
  - el NRMSE en t = 25, 50 y 99;
  - la primera t con update < TOL (1.15).
- **Predicción:** el último estado baja de su 33.09 % con más iteraciones, porque en la
  evaluación su mínimo estaba en la 24 de 25. `best_qsm` se va antes.

**RESULTADO del barrido** (`runs/ciego-epocas-l32-10k/barrido-target/resultados.json`; GPU 0,
16:00–18:10). NRMSE mínimo de 100 iteraciones, con su t entre paréntesis:

| target | COSMOS, `best_qsm` (1 488) | COSMOS, último (4 960) | Sim2, `best_qsm` | Sim2, último |
|---|---|---|---|---|
| 0.15 | 32.82 (34) | **32.15** (11) | **37.72** (99) | 38.98 (99) |
| 0.20 | 31.22 (34) | 32.45 (34) | 39.30 (98) | 41.72 (99) |
| 0.25 | **30.81** (35) | 32.99 (69) | 41.51 (99) | 44.57 (5) |
| 0.30 | 31.11 (65) | 33.88 (69) | 44.34 (99) | 46.06 (3) |

- **Convergencia:** en las 16 corridas el update baja del TOL del autor (1.15 %) entre t = 15 y
  24 y no vuelve a pasarlo, pero ninguna llega a un punto fijo en 100 iteraciones (abajo, «Los
  saltos finales»).
  - COSMOS queda plano: en t = 99 está a ≤ 0.44 pp del mínimo, y el techo de 25 de la
    evaluación cuesta ≤ 0.3 pp.
  - Sim2 con `best_qsm` sigue bajando lento hasta t = 99 (~1 pp entre la 25 y la 99).
  - Sim2 con el último, a 0.25–0.30, se va: el mínimo cae en t = 3–5 y después sube a 45.0 y
    48.4 %.
- **`best_qsm` es mejor prior que el último** en 7 de las 8 combinaciones. La excepción es COSMOS
  con 0.15. Sólo en 0.15–0.30: con el mejor target de cada uno, el último gana en los dos sujetos
  (RESULTADO del barrido 0.02 / 0.06 / 0.12, abajo).
- **El target bueno baja con el entrenamiento.** En COSMOS es 0.25 con `best_qsm` y 0.15 con el
  último. En Sim2 es 0.15, el borde, con los dos.
  - Con 0.15 el último le gana a `best_qsm` en COSMOS (32.15 contra 32.82 %). Parte del QSM que
    empeora durante el entrenamiento es del target fijo en 0.25 (lectura, no medida).
  - Es coherente con la cuenta de arriba: con target ≤ 0.14–0.12, el error de la primera
    iteración queda dentro del ruido entrenado.
- **Predicción: falló en las dos partes.** El último sólo baja a 32.99 % con 0.25 (contra 33.09 %
  en la evaluación). `best_qsm` no se va: en COSMOS queda plano y en Sim2 sigue mejorando.
- **Lo que sigue:**
  - barrer 0.05–0.15, porque Sim2 y el último piden menos;
  - la propuesta de `--sigma-max` (arriba).

**PRE-REGISTRO del target óptimo al final del entrenamiento** (2026-10-23, pedido por el autor;
antes de medir).
- **El criterio:** α se fija en t = 0 (`ddr/qsm3d.py`), y la red sabe quitar ruido hasta
  σ_max/α. El target «justo» es el que iguala eso al error de χ₀: target* = σ_max/k. Con la
  oficial (σ₀ = 25/255) da 0.098/1.53 = 0.064, el 0.06 del 2D, que además pone el máximo literal
  de χ₀ en ~1. Con la ciega (σ_max = 55/255) da 0.14 en COSMOS y 0.12 en Sim2 (~0.019–0.020 ppm,
  el error de χ₀). Bajo eso no se gana ajuste de ruido: la imagen queda más tenue que cualquier T1
  y la red puede tomar por ruido contrastes de χ cada vez mayores (a 0.02, hasta ~0.13 ppm).
- **Predicción:** el óptimo sigue bajando mientras la red se especializa y se detiene un poco bajo
  ese borde: ~0.11 en COSMOS y ~0.09 en Sim2; falla si alguno cae fuera de 0.07–0.13.
- **A favor:** la parábola del último en COSMOS tiene el mínimo en 0.11 (32.1 %).
- **En contra:** la parábola de `best_qsm` en Sim2 lo pone en 0.05 (36.5 %); con la control de
  σ fijo, el target «justo» (0.06) fue el peor, y 0.02 dio el mejor mínimo de Sim2 pero se fue a
  ~60 % en la 100.
- **Medición propuesta, sin correr:** checkpoint final y `best_qsm`, COSMOS y Sim2, target ∈ {0.04,
  0.06, 0.08, 0.10, 0.12, 0.15}, 100 iteraciones: 24 corridas, ~3 h de GPU 1 cuando termine la
  10k. (Hecha el 25-10 con la grilla 0.02–0.06: ver «Grid de target…».)

**Los saltos finales del ADMM** (pregunta del autor, 2026-10-23, después del pre-registro; de
`resultados.json`, sin correr nada). El update es el cambio relativo de χ por iteración, en %, el
de la parada del autor.
- **Ninguna converge a un punto fijo en 100:** en t = 99 el update sigue en 0.10–0.50 % en COSMOS y
  0.11–0.81 % en Sim2. La suma de updates entre t = 50 y 99 (una cota de cuánto se mueve χ) es
  6–11 % en 13 de las 16, 20 % en `best_qsm` con Sim2 a 0.20, y 24 y 41 % en las dos que no
  decaen.
- **Cómo decae:** como t^−p, con p ≈ 1 a 0.30 y 0.55–0.85 a 0.15–0.25 en COSMOS: cuanto menor el
  target, más lento.
- **Dos no decaen y oscilan:** `best_qsm` a 0.15, en COSMOS (0.47 → 0.50 % entre t = 50 y 99) y en
  Sim2 (0.85 → 0.81 %, con el NRMSE en diente de sierra: 18 giros en 18 pasos).
- **El NRMSE casi no se mueve en COSMOS** (≤ 0.03 pp entre t = 50 y 99): χ se mueve en direcciones
  que no cambian el error. En Sim2 sigue derivando: baja con `best_qsm` y sube con el último a
  0.25–0.30.
- **Un mínimo temprano no es convergencia:** el último en COSMOS a 0.15 tiene el mínimo en t = 11
  (32.15 %), y en t = 99 está en 32.59 % y se mueve 0.18 % por iteración.
- **Contra el pre-registro de arriba:** los targets bajos son los que peor convergen.

**PRE-REGISTRO del barrido 0.02 / 0.06 / 0.12** (2026-10-23, antes de lanzar; regla 5). Los mismos
checkpoints del barrido de arriba (`best_qsm` de la 1 488 y `estado-004960`), ADMM de 100 sin
corte, COSMOS y Sim2: `scripts/barrido_target_ciega.py 0.02 0.06 0.12`, en la GPU 0 (permiso del
autor, 23-10 a las 22:00; ~1.6 h). El autor pidió 0.02 y 0.12; 0.06 se agrega para elegir el
tercer target. Para W&B el autor decidió 0.25 y 0.06, sin 0.12. (Reemplazado el 25-10 por 0.15,
0.02 y 0.05: ver «La rama a 20 000».)
- **Predicción:**
  - en COSMOS, 0.12 es el mejor de los tres con los dos checkpoints; con `best_qsm` los tres quedan
    sobre el 30.81 % de 0.25 (la parábola da ~34 % a 0.12);
  - en Sim2, 0.06 y 0.12 quedan a menos de 1 pp entre sí, y bajo el 37.72 % de 0.15 con
    `best_qsm` (la parábola da 37.1 y 36.5 %);
  - 0.02 tiene el mínimo antes de t = 20 y después sube, en al menos 3 de las 4 combinaciones;
  - el update final crece al bajar el target: a 0.12 queda sobre el de 0.15 en las 4.
- **Falla** cada punto por separado.

**¿α sin GT?** (pregunta del autor, 2026-10-23; `scripts/sigma_chi0.py`, CPU, y el barrido de
arriba; sin GPU).
- **El error de χ₀ no es ruido fino:** la MAD de la subbanda Haar HHH (exacta para ruido blanco) ve
  el 30 % de su RMS en COSMOS (0.0056 de 0.0184 ppm) y el 20 % en Sim2 (0.0039 de 0.0198). Es
  sobre todo error suave. El «casi blanco» de «Dónde está el error del ADMM» vale sólo para el
  reparto entre cono y fuera.
- **Un α por ruido estimado de χ₀ es peor que la regla actual:** RMSE/σ̂(χ₀) es 3.81 en COSMOS y
  5.54 en Sim2 (45 % de diferencia), contra k = 1.49 y 1.72 de la std (15 %).
- **El principio de discrepancia tampoco:** todas las reconstrucciones del barrido dejan menos
  residuo de dato (12.5–21.7 %) que el χ verdadero (22.0 % en COSMOS, 29.2 % en Sim2, con el
  desajuste del campo de E1). Igualarlos pide targets de ~0.03–0.13, y con `best_qsm` en COSMOS el
  óptimo es 0.25.
- **Contra el criterio del pre-registro del target óptimo:** el borde 0.14/0.12 trata todo el RMSE
  como ruido blanco. La parte fina, en unidades de la red, es 0.39·target en COSMOS y 0.31·target
  en Sim2: a 0.25 queda en 0.098 y 0.078, dentro del rango, y saldría de él recién a 0.55–0.69. La
  predicción queda como se escribió.
- **Propuesta, sin correr:** sacar la dependencia en vez de calibrarla. Aumento de amplitud en el
  entrenamiento (cada parche limpio por una ganancia log-uniforme en [1/16, 1], con el ruido en
  unidades absolutas), para que la red vea imágenes tenues con todo el rango de ruido. Se mide con
  una corrida corta y el barrido 0.02–0.30: sirve si el NRMSE queda a ±1 pp en un rango de targets
  de 4×. Mientras tanto, la evaluación elige el mejor de sus targets (α oráculo, con GT).

**RESULTADO del barrido 0.02 / 0.06 / 0.12** (GPU 0, 21:57–23:30; mismo `resultados.json`). La
tabla entera, NRMSE mínimo (t) · update en t = 99 (%), con el mejor de cada columna en negrita:

| target | COSMOS, `best_qsm` (1 488) | COSMOS, último (4 960) | Sim2, `best_qsm` | Sim2, último |
|---|---|---|---|---|
| 0.02 | **30.71** (12) · 0.84 | **27.77** (14) · 1.32 | 39.67 (99) · 0.91 | 38.56 (99) · 0.23 |
| 0.06 | 38.07 (4) · 0.96 | 31.95 (10) · 0.28 | 39.86 (99) · 0.24 | **37.08** (99) · 0.23 |
| 0.12 | 34.64 (63) · 0.99 | 32.28 (13) · 0.20 | **37.37** (99) · 0.91 | 37.68 (99) · 0.21 |
| 0.15 | 32.82 (34) · 0.50 | 32.15 (11) · 0.18 | 37.72 (99) · 0.81 | 38.98 (99) · 0.19 |
| 0.20 | 31.22 (34) · 0.15 | 32.45 (34) · 0.14 | 39.30 (98) · 0.38 | 41.72 (99) · 0.15 |
| 0.25 | 30.81 (35) · 0.12 | 32.99 (69) · 0.11 | 41.51 (99) · 0.13 | 44.57 (5) · 0.13 |
| 0.30 | 31.11 (65) · 0.10 | 33.88 (69) · 0.10 | 44.34 (99) · 0.11 | 46.06 (3) · 0.11 |

- **En COSMOS el NRMSE no es monótono en el target:** 0.06 es el peor con los dos checkpoints y
  0.02 el mejor, con un segundo valle ancho en 0.15–0.25. Es el patrón de la control del 21-10
  (0.06, el peor de cuatro).
- **El último a 0.02 da 27.77 % en COSMOS:** 5.2 pp bajo su 0.25, y cerca del pseudo-3D oficial
  (25.94–26.08 %). No converge: el mínimo cae en t = 14, en t = 99 está en 29.25 % y el update
  volvió sobre el TOL (1.32 %). Aun así, en t = 99 queda bajo todo lo demás.
- **En Sim2 los targets bajos siguen bajando en t = 99,** y de 0.02 a 0.15 caen en ~2.5 pp.
- **Con el mejor target de cada checkpoint, el último gana en los dos sujetos** (27.77 contra
  30.71 % en COSMOS; 37.08 contra 37.37 % en Sim2). El empeoramiento del QSM entre la 1 488 y la
  4 960 era del target fijo en 0.25, no de la red.
- **Predicciones: se cumple 1 de 4.** Fallan «0.12 el mejor de los tres en COSMOS» (lo es 0.02),
  «0.06 y 0.12 a menos de 1 pp en Sim2» (2.5 pp con `best_qsm`) y «0.02 sube después de t = 20 en
  3 de 4» (sólo en COSMOS). Se cumple «el update a 0.12 queda sobre el de 0.15» (en las 4). El
  pre-registro del target óptimo final (0.07–0.13) ya queda fuera con el último en COSMOS.
- **Control de la evaluación de W&B** (`scripts/qsm_targets_wandb.py evalua 0.25 --estados 1488`):
  reproduce la del entrenamiento (30.88 % en COSMOS y 42.13 % en Sim2, t = 24 de 25).
- **Consecuencia:** a la evaluación de la rama a 20 000 se le suma 0.02 (0.25, 0.06 y 0.02).
  (Reemplazado el 25-10 por 0.15, 0.02 y 0.05: ver «La rama a 20 000».)

**RESULTADO de las curvas por target** (`scripts/qsm_targets_wandb.py`, GPU 0, 23:40–01:20; la
evaluación del entrenamiento, 25 iteraciones, sobre los 16 estados de la 496 a la 7 936; W&B
`ciego-epocas-l32-10k-targets`, borrada el 25-10; `kxein1yw` tiene el QSM de los 21 estados a
0.15, 0.02 y 0.05). NRMSE mínimo en la 496 → 1 488 → 4 960 → 7 936:

| | 0.25 (la del entrenamiento) | 0.06 | 0.02 |
|---|---|---|---|
| COSMOS | 35.87 → 30.88 → 33.09 → 35.36 | 39.34 → 38.07 → 31.95 → 30.82 | 35.12 → 30.71 → 27.77 → **27.08** |
| Sim2 | 47.27 → 42.13 → 44.57 → 45.16 | 45.52 → 40.79 → 37.99 → **36.02** | 46.14 → 40.64 → 40.02 → 39.81 |

- **A 0.02 y 0.06 el QSM mejora con el entrenamiento;** a 0.25 empeora desde la 1 488. La curva de
  W&B de la corrida, a 0.25, mostraba la contraria de lo que aprende la red.
- **El mejor target depende del sujeto:** 0.02 en COSMOS (27.08 %), 0.06 en Sim2 (36.02 %), y el
  segundo de cada uno queda 3.7–3.8 pp atrás.
- **Hasta la 10 000** (los 5 estados que faltaban, 24-10, GPU 0): en la 10 000, COSMOS a 0.02 da
  26.73 % (el mínimo, 26.51 %, en la 8 928 y la 9 920) y Sim2 a 0.06 da 35.49 %. Desde la
  8 432 las dos curvas quedan casi planas: varían 0.3 pp o menos.

**RESULTADO de `ciego-epocas-l32-10k`: terminó la 10 000 sin cortes** (24-10, 04:18; 25.89 h en
total; W&B `b3vm1g37` desde la 4 960, borrada el 25-10: la consolidada es `kxein1yw`).
- **Denoising en el cerebro, en la 10 000:** el mejor PSNR de la corrida, 33.787 dB. NRMSE de
  entrada → salida: SNR de pico 25, 12.19 → 4.18 %; 50, 6.10 → 3.27 %; 100, 3.05 → 2.17 %.
- **El disyuntor 3D actuó una vez:** 25 rechazos seguidos en la 9 768 y un rebobinado a la 9 711,
  sin TRABADA. 25 lotes rechazados de 10 000; iteración efectiva 9 943. VRAM pico 11.51 GiB.
- **QSM a 0.25 (la evaluación de la corrida):** 34.95 % en COSMOS y 45.06 % en Sim2; `best_qsm`
  sigue siendo el de la 1 488 (30.88 %). A 0.02 y 0.06 la red mejora (curvas por target, arriba).

## 2026-10-24 — La rama a 20 000: `ciego-epocas-l32-20k`

**PRE-REGISTRO**, escrito antes de lanzar (regla 5) y rehecho el 25-10 antes del segundo
lanzamiento, con los targets nuevos. Rama de `ciego-epocas-l32-10k` desde `estado-010000.pth`, con
las mismas banderas, lr constante (sin lr/2: el autor no ve aplanarse la curva) y
`--qsm-target-std 0.15 0.02 0.05`, μ = 0.008. El ruido queda igual (`--sigma-max 55`): a 0.02 y
0.05 el error de la primera iteración que ve la red está dentro del rango entrenado (a 0.05, 0.075
en COSMOS y 0.086 en Sim2, contra 0.216); a 0.15 queda en el borde (0.22–0.26), con su parte fina
(0.05–0.06) dentro. Además, una rama no puede cambiar la receta.
W&B abre una corrida nueva desde la 10 001, en el grupo `ciego-epocas-l32`. ~28 h en la GPU 1.
- **Pregunta:** si 10 000 iteraciones más, sin lr/2, siguen mejorando el denoising y el QSM al
  mejor target.
- **Predicción:**
  - el PSNR en el cerebro sube 0.2–0.6 dB sobre 33.79;
  - el QSM al mejor target (evaluación de 25 iteraciones) sigue bajando: COSMOS (0.02) de 26.7 a
    25.5–26.7 %, Sim2 (0.05) de 36.0 a 34–35.5 %;
  - 0.15 queda peor que los otros dos en los dos sujetos;
  - el mejor target no cambia: 0.02 en COSMOS y 0.05 en Sim2 en las últimas 5 evaluaciones;
  - a lo sumo 3 rebobinados y ninguna TRABADA.
- **Se reporta:** el PSNR en el cerebro, el NRMSE de los tres niveles de SNR, el QSM por target y
  sujeto en cada evaluación, los rebobinados y las alertas.

**El primer intento murió a los 6 minutos, sin culpa del entrenamiento** (24-10, 05:12; visto a
las 16:17). A las 05:12:14 se cerró la última sesión de login del usuario (la que abrió el 19-10 la
sesión de Claude Code que lanzaba las corridas), y con `Linger=no` systemd paró todo el gestor de
usuario (`exit.target`), con la unidad de la corrida adentro. El log no tiene error: llega a la
10 025. No hay estado nuevo (el último es el de la 10 000) y se pierden ~6 minutos. La corrida de
W&B `zkuxpf7m` queda con un punto (borrada el 25-10, con la carpeta del intento). Arreglo:
`loginctl enable-linger`, para que el gestor de usuario no dependa de una sesión abierta; las de
10 000 sobrevivieron porque esa sesión siguió abierta.

**Segundo lanzamiento** (25-10, 18:07): en `runs/ciego-epocas-l32-20k`, desde
`estado-010000.pth`, con el pre-registro rehecho (arriba) y las banderas de su `plan.sh`. W&B
`4027tbsv`, en el grupo `ciego-epocas-l32`, enlazada a `kxein1yw` (`padre_id`). El linger quedó
aplicado: `Linger=yes`.

**RESULTADO: cortada a mano en la 15 875** (26-10, decisión del autor; `estado.json` escrito a
mano, último estado `estado-015872.pth`). 12 evaluaciones, de la 10 416 a la 15 872, ~16.4 h en la
GPU 1. Se lee contra el pre-registro, que era a la 20 000, como parcial:
- **PSNR en el cerebro:** récord 34.019 dB en la 14 880, +0.23 sobre 33.79 (predicho +0.2–0.6 a la
  20 000).
- **QSM al mejor target:** COSMOS (0.02), mejor 25.95 % en la 12 896 (predicho 25.5–26.7), pero de
  una evaluación a otra va de 25.95 a 27.76 %. Sim2 (0.05), mejor 35.75 % en la 14 384 (predicho
  34–35.5: no llegó), entre 35.75 y 37.64 %.
- **Se cumplió:** 0.15 quedó peor que los otros dos en los dos sujetos en las 12 evaluaciones, y el
  mejor target fue siempre 0.02 en COSMOS y 0.05 en Sim2.
- **Disyuntor:** 2 rebobinados, en la 12 429 (a la 12 371, |g| 671) y en la 14 223 (a la 14 182,
  |g| 1.77·10³), cada uno tras 25 rechazos seguidos, y ninguna `TRABADA` (predicho ≤ 3). Los
  rechazos pasaron de 25 a 77. La validación de la 12 400 cayó dentro de la excursión que rebobinó
  en la 12 429: PSNR en el cerebro 32.95 dB y COSMOS 27.76 %, las peores de la rama.
- **Pérdida de entrenamiento**, promedio cada 1 000 iteraciones desde la 10 000: 20.87, 20.58,
  20.39, 20.17, 21.70 (con el rebobinado de la 14 223) y 19.87 (15 000–15 872). T bajó de 0.089
  (10 025) a 0.062 (15 875).
- **Lectura:** sigue mejorando, despacio (+0.23 dB en 4 880 iteraciones), y la oscilación entre
  evaluaciones (~1 dB de PSNR, 1.8 pp en COSMOS) es mayor que el avance. Se corta para bajar el lr
  («La rama a 30 000», 26-10). Ningún checkpoint posterior a la 14 880 la mejora en NRMSE con el
  protocolo de la evaluación: la 15 376 da COSMOS 26.29 % y Sim2 36.55 %, la 15 872 27.56 % y
  36.37 %, contra 26.11 y 36.08 %.
- **La meseta, a posteriori:** `paso_meseta` sobre las 12 evaluaciones, con `--meseta-psnr` y la
  rama empezando de cero, habría partido el lr en la 15 872 con paciencia 2, y ninguna vez con 3 o
  4.

## 2026-10-25 — Grid de target 0.02–0.06, μ y ADMM en dos rondas, con la red de la 10 000

**RESULTADO del grid de target** (`scripts/barrido_target_ciega.py --ckpts final`, checkpoint de
la 10 000, 100 iteraciones sin corte; GPU 0 y 1 a la vez con permiso del autor, 13:20–14:50;
`barrido-target/grid-gpu{0,1}.json`). NRMSE mínimo (t) · NRMSE en t = 99 · update en t = 99 (%) y
la primera t con update < TOL (— si nunca):

| target | COSMOS | Sim2 |
|---|---|---|
| 0.02 | **26.73** (16) · 27.67 · 1.46 · — | 38.55 (99) · 38.55 · 1.66 · — |
| 0.025 | 27.27 (16) · 28.55 · 0.46 · 18 | 37.95 (98) · 37.95 · 2.08 · — |
| 0.03 | 27.80 (13) · 29.57 · 0.26 · 19 | 37.23 (99) · 37.23 · 1.85 · — |
| 0.035 | 28.48 (11) · 30.59 · 0.42 · 19 | 36.26 (99) · 36.26 · 2.87 · — |
| 0.04 | 29.13 (12) · 31.24 · 0.52 · 23 | 35.32 (99) · 35.32 · 1.62 · — |
| 0.045 | 29.60 (10) · 31.89 · 0.86 · 21 | 34.66 (98) · 34.67 · 2.28 · — |
| 0.05 | 30.14 (9) · 32.38 · 1.81 · 25 | 34.29 (98) · 34.34 · 2.68 · — |
| 0.055 | 30.54 (10) · 32.78 · 2.52 · 34 | **34.11** (98) · 34.22 · 3.43 · — |
| 0.06 | 30.83 (10) · 33.01 · 3.68 · 37 | 34.20 (99) · 34.20 · 3.46 · — |

- **Los sujetos tiran para lados opuestos:** en COSMOS el NRMSE sube con el target (0.02 el
  mejor), en Sim2 baja hasta 0.055.
- **Ninguno converge de verdad en Sim2:** el update en t = 99 queda en 1.6–3.5 % y nunca baja del
  TOL. En COSMOS, 0.025–0.045 bajan del TOL en t = 18–23; 0.02 no.
- **0.055 domina a 0.06 en los cuatro números** (NRMSE y update, en los dos sujetos); 0.05 los
  domina salvo el NRMSE de Sim2 (+0.09 pp).
- **Con `best_qsm` (la 1 488)**, a 0.03, 0.04 y 0.05 (`grid-gpu1b.json`), junto con 0.02 y 0.06 del
  23-10: COSMOS 30.71, 31.51, 33.52, 36.07 y 38.07 %; Sim2 39.67, 40.32, 40.36, 40.34 y 39.86 %. El
  checkpoint de la 10 000 es mejor a todos los targets, 3.7–7.2 pp en COSMOS y 1.1–6.1 pp en Sim2. Con
  `best_qsm`, Sim2 sí baja del TOL (t = 22) y COSMOS también (t = 18–26).

**RESULTADO de μ** (`barrido_target_ciega.py --ckpts final --mu …`, GPU 0 a target 0.02 y GPU 1 a
0.05; `mu-gpu{0,1}.json`). NRMSE mínimo · update en t = 99 (%), con μ₂ = 1:

| μ | COSMOS 0.02 | Sim2 0.02 | COSMOS 0.05 | Sim2 0.05 |
|---|---|---|---|---|
| 0.006 | 26.65 · 2.23 | 38.08 · 2.33 | 29.33 · 6.46 | 34.52 · 6.13 |
| 0.007 | 26.73 · 1.40 | 38.28 · 2.11 | 29.74 · 3.78 | 34.29 · 4.24 |
| 0.008 | 26.73 · 1.46 | 38.55 · 1.66 | 30.14 · 1.81 | 34.29 · 2.68 |
| 0.009 | 26.72 · 1.04 | 38.87 · 1.59 | 30.45 · 0.82 | 34.37 · 1.86 |
| 0.01 | 26.77 · 0.77 | 39.19 · 1.17 | 30.83 · 0.47 | 34.52 · 1.37 |

- **Subir μ siempre mejora la convergencia** (el update en t = 99 baja en los cuatro casos; COSMOS
  baja del TOL desde 0.009) y, como se predijo, no arregla el cono: el NRMSE mínimo se mueve
  ≤ 1.5 pp.
- **El costo en NRMSE cae en un sujeto u otro según el target:** a 0.02, plano en COSMOS y +1.1 pp
  en Sim2 de 0.006 a 0.01; a 0.05, +1.5 pp en COSMOS y plano en Sim2.
- **Más allá de 0.01, a 0.02** (el autor pidió seguir si 0.01 ganaba; ganaba en convergencia):
  μ = 0.0125 y 0.015 dan COSMOS 27.00 y 27.49 % (estable: 27.09 y 27.50 en t = 99, bajo el
  TOL desde t = 17–18) y Sim2 39.51 y 39.92 %, con el update de Sim2 peor que a 0.01 (2.71 y
  2.85 %).
- **Se queda μ = 0.008:** ningún μ gana en los cuatro casos; subirlo compra convergencia con
  NRMSE, y pasado 0.01 empeora Sim2 en las dos cosas. La convergencia se ataca con el ADMM en
  dos rondas (abajo).

**RESULTADO del ADMM en dos rondas** (`scripts/admm_dos_rondas.py`, checkpoint de la 10 000,
μ = 0.008, 100 iteraciones; GPU 0 y 1; `rondas-gpu{0,1}.json`). Empieza con el primer target y
pasa al siguiente cuando el update baja del TOL o se estanca (≥ 0.95 del de dos iteraciones antes);
cambió en t = 18–21, salvo 0.05 → 0.15 en Sim2 (t = 7). NRMSE mínimo (t) · en t = 99 · update en
t = 99 (%):

| calendario | COSMOS | Sim2 |
|---|---|---|
| 0.02 sola (grid) | 26.73 (16) · 27.67 · 1.46 | 38.55 (99) · 38.55 · 1.66 |
| 0.05 sola (grid) | 30.14 (9) · 32.38 · 1.81 | 34.29 (98) · 34.34 · 2.68 |
| 0.02 → 0.05 | 26.73 (16) · 32.34 · 1.96 | 34.31 (99) · 34.31 · 2.45 |
| 0.02 → 0.15 | 26.73 (16) · 34.65 · 8.20 | 39.94 (21) · 41.73 · 6.44 |
| 0.02 → 0.25 | 26.73 (16) · 35.69 · 8.22 | 40.32 (20) · 48.73 · 5.86 |
| 0.05 → 0.15 | 30.14 (9) · 34.75 · 7.97 | 40.17 (9) · 41.29 · 6.50 |
| 0.02 → 0.05 → 0.15 | 26.73 (16) · 34.72 · 8.17 | 38.67 (21) · 41.83 · 6.35 |

- **Pasar de un denoiser fuerte a uno suave no ayuda:** después del cambio el ADMM se va a la
  solución del target suave, peor, y el update sube a 6–8 %. Ninguno converge mejor que su target
  solo.
- **0.02 → 0.05 en Sim2 queda igual que 0.05 sola** (34.31 contra 34.29 %); en COSMOS pierde lo de
  0.02 (32.34 en t = 99).
- **Veredicto: no.** Se descartan los calendarios fuerte → suave con este cambio; queda en «Lo que no
  sirve».

**W&B, limpio** (25-10, con el OK del autor). Queda una corrida por tramo:
- **`ciego-epocas-l32-10k`**, una sola (`scripts/consolida_wandb_10k.py`), armada con los archivos
  locales:
  - `train/*` de las 10 000 iteraciones, de `traza.npy`;
  - la validación de `history.json`, con los niveles de SNR sólo desde la 5 456 (antes, la
    convención vieja);
  - el QSM de los 21 estados a 0.15, 0.02 y 0.05 con las claves de `train3d.py`.
  - Su id, `kxein1yw`, quedó en `seguimiento.json`, así que la rama a 20 000 se enlaza a ella.
- **Se borraron** las corridas de W&B de panel-l8, ciego-epocas-l32, las tres de la 10k partida,
  la de targets y el intento de 6 minutos de la 20k.
- **También las carpetas locales** `runs/panel-l8`, `runs/ciego-epocas-l32`,
  `runs/ciego-epocas-l32-10k/intento-cota-s2` y la del primer intento de la 20k (la rama se
  relanzó en `runs/ciego-epocas-l32-20k` el 25-10 a las 18:07; W&B `4027tbsv`). Los números
  quedan en esta bitácora.

**Auditoría y «Sanity Check»** (25-10, pedido del autor). Cuatro agentes leyeron README, AGENTS.md,
`docs/`, la bitácora, el estilo del código y las figuras, y otros cuatro aplicaron los arreglos
(comentarios y docs; ninguna lógica). Lo que queda:
- **Las figuras:** `docs/figuras/` pasó a `docs/Sanity Check/`, con 68 archivos y 33 MB, fuera de
  git. Todas son del 21–22-10, de redes cortas ya superadas, y usan la convención de SNR vieja.
  Se borraron `seed2/` y `seed3/`, que repetían `seed1/`, y `22-10-3-nrmse_qsm_vs_alfa.png`, que
  `25-10-grid-target.png` rehízo con la red vigente: quedan 35 archivos y 14 MB.
- **`docs/Sanity Check/resumen/` sí va en git** (~0.4 MB): `RESUMEN.md`, con lo vigente, lo
  descartado y el inventario de las figuras viejas, y tres figuras nuevas
  (`scripts/resumen_sanity_check.py`): QSM por target durante el entrenamiento, grid de target y μ.
- **`train.py` y `train3d.py`:** el bloque final de los dos, que repetía «BUCLE» de
  `docs/porque-del-codigo.md`, queda en un puntero; más un comentario que citaba una corrida
  borrada y 12 ajustes de comentarios y docstrings. Sin docstrings, el AST es idéntico al de
  antes. Se aplicaron con la rama a 20 000 en vuelo y con el OK del autor: la corrida lee su copia
  en `runs/ciego-epocas-l32-20k/codigo/`, así que no la toca.

## 2026-10-26 — La rama a 30 000: `ciego-epocas-l32-30k`

**Antes de elegir de dónde sale, un grid de target y μ** (`scripts/grid_ckpts_26_10.py`, GPU 1,
desde las 10:38; `runs/ciego-epocas-l32-20k/26-10-grid-target-mu.json`). Pedido del autor: si con
target y μ ajustados ningún checkpoint posterior a la 14 880 la supera en el NRMSE de COSMOS, la
rama sale de la 14 880. Etapa 1: la 14 880, la 15 376 y la 15 872 a target 0.01, 0.015, 0.02, 0.025
y 0.03, con μ = 0.008. Etapa 2: al mejor target de cada una, μ = 0.006, 0.007, 0.009 y 0.01. Es el
ADMM de la evaluación (para tras 2 subidas seguidas del NRMSE y devuelve el mínimo), con techo de
50 en vez de 25. Sim2 no entra: el entrenamiento ya lo midió a 0.02 y 0.05 en los tres.

**RESULTADO del grid** (10:38–11:18). NRMSE mínimo en COSMOS (%) y su t; con `max_iter`, el ADMM
llegó al techo sin dos subidas seguidas:

| target (μ = 0.008) | 14 880 | 15 376 | 15 872 |
|---|---|---|---|
| 0.01 | 27.55 (14) | 26.50 (18, max_iter) | 27.49 (14, max_iter) |
| 0.015 | 26.17 (46, max_iter) | 26.90 (14) | 27.50 (14) |
| 0.02 | **26.11** (18) | **26.29** (24) | 27.04 (21) |
| 0.025 | 27.03 (12) | 26.79 (15) | **27.03** (13) |
| 0.03 | 28.33 (9) | 27.59 (15) | 27.98 (9) |

| μ, al mejor target | 14 880 (0.02) | 15 376 (0.02) | 15 872 (0.025) |
|---|---|---|---|
| 0.006 | 26.60 | 26.26 | 27.52 |
| 0.007 | 26.26 | **26.15** | 27.23 |
| 0.008 | 26.11 | 26.29 | **27.03** |
| 0.009 | **26.09** | 26.66 | 27.04 |
| 0.01 | 26.15 | 26.93 | 27.76 |

- **Gana la 14 880:** 26.09 % ajustada, contra 26.15 de la 15 376 y 27.03 de la 15 872. La
  diferencia con la 15 376 (0.06 pp) es menor que la oscilación entre evaluaciones, así que lo que
  decide es la regla del autor: ninguna posterior la supera. La rama sale de la 14 880.
- **0.02 sigue siendo el mejor target** (en la 15 872, 0.025 empata: 27.03 contra 27.04).
- **μ no ordena nada:** el mejor μ es 0.009, 0.007 y 0.008 según el checkpoint, a ≤ 0.14 pp de
  μ = 0.008. Queda 0.008, como en el grid de la 10 000.

**PRE-REGISTRO**, escrito antes de lanzar (regla 5). Rama de `ciego-epocas-l32-20k` desde
`estado-014880.pth`, con las mismas banderas salvo el calendario: `--iterations 30000`,
`--hitos 14880` (lr 2e-4 desde la 14 881) y `--meseta 2 --meseta-psnr --meseta-mitades 4` (lr/2
cuando pasan 2 evaluaciones seguidas sin récord del NRMSE de COSMOS ni del PSNR en el cerebro, hasta
4 veces; agotadas, la meseta siguiente corta). La meseta parte de cero: la primera evaluación de la
rama fija las dos referencias. μ = 0.008, targets 0.15, 0.02 y 0.05. ~42 h si llega a la 30 000.
- **Pregunta:** si bajar el lr saca a la corrida de la meseta, en denoising y en QSM al mejor
  target, y si achica la oscilación entre evaluaciones.
- **Predicción:**
  - el PSNR en el cerebro supera 34.02 dB en alguna de las 4 primeras evaluaciones (hasta la
    16 864) y termina en 34.1–34.5 dB;
  - la oscilación baja: el NRMSE de COSMOS a 0.02 varía menos de 1 pp en 4 evaluaciones seguidas
    (en la rama a 20 000, 1.8 pp);
  - el mejor QSM baja de lo visto: COSMOS (0.02) ≤ 25.9 %, Sim2 (0.05) ≤ 35.7 %;
  - el mejor target no cambia: 0.02 en COSMOS y 0.05 en Sim2;
  - la meseta parte el lr al menos 2 veces antes de la 30 000;
  - a lo sumo 1 rebobinado, y ninguna `TRABADA`.
- **Se reporta:** el PSNR en el cerebro, el NRMSE a SNR 25, 50 y 100, el QSM por target y sujeto,
  las mitades de la meseta con su iteración, los rebobinados, las alertas y, si corta, dónde.

**Lanzada** (26-10, 11:19): `runs/ciego-epocas-l32-30k`, unidad `tdv-ciego-epocas-l32-30k`, con
las banderas de su `plan.sh` y la copia del código de la rama a 20 000. W&B `ata3j1cj`, en el grupo
`ciego-epocas-l32`, enlazada a `4027tbsv` (`padre_id`). Hereda `best_qsm.pth` de la 12 896
(25.95 %).

## 2026-10-26 — Auditoría del ruido y de los datos QSM (pedido del supervisor)

**Pedido** (del supervisor, por segunda vez, vía el autor): que el SNR del ruido inyectado está mal,
porque las imágenes no muestran el ruido que dicen; que el ruido se pondere por el máximo absoluto;
que el NRMSE de COSMOS no sea un falso positivo, y que Sim2 no sea simulado.
`scripts/auditoria_ruido_26_10.py` (CPU, con la GPU 1 en la rama a 30 000); números en
`runs/auditoria-ruido/`, figuras en `docs/Sanity Check/26-10-auditoria-ruido/`.

**El ruido del denoising está bien; las figuras viejas lo rotulaban con otra convención.**
- **El código:** `--val-snr` suma N(0, σ²) con σ = 1/SNR (`valida_niveles` →
  `evaluate_denoising3d` → `add_noise`), sin recortar, con una realización fija por volumen
  (semilla 1234 + i). El entrenamiento usa la misma escala: σ ∈ [0.81, 55]/255 es SNR de 4.6 a 315.
- **El máximo absoluto vale 1:** cada T1 se divide por el percentil 99.5 de sus vóxeles positivos y
  se recorta a [0, 1]. En los 12 volúmenes de validación el máximo es exactamente 1, en el volumen
  entero y en el recorte de 128³ (0.25–0.38 % de los vóxeles del recorte vale 1). Así, SNR =
  máximo/σ = 1/σ: la convención que pide el supervisor es la vigente desde el 23-10.
- **Medido:** la sd de ruidosa − limpia es 0.09805, 0.04000, 0.02000 y 0.01000 a SNR 10.2, 25, 50
  y 100, en los 12 a ±0.3 %, con media ≤ 1.5·10⁻⁴ y forma gaussiana
  (`26-10-1-ruido_inyectado_t1.png`).
- **Lo que estaba mal son las figuras del 21 y 22-10** (`resultados-*/seed1/`, las grillas y los
  resúmenes de semillas): usan σ = 0.316/SNR, con la media del cerebro. Su «SNR 25» es σ = 0.0126,
  que por el máximo es SNR 79, con 3.16 veces menos ruido que el rótulo. Además se mostraban con
  ventana [0, 0.51] y fuera del cerebro en 0. Las figuras nuevas usan σ = máximo/SNR, en gris
  [0, 1].
- **Cuánto ruido es:** en el cerebro la intensidad media es 0.314 y el p99.9, 0.52. A SNR 25 el
  ruido es el 13 % de la media del cerebro; a SNR 100, el 3 %, del orden de la variación fina
  propia de LDM_100k (σ 0.004–0.008, «El SNR propio de los datos»). El máximo dentro del cerebro
  tiene mediana 0.97 (0.68–1.0), por unos pocos vóxeles; ponderando por él, el SNR casi no cambia.

**La red vigente a esos SNR** (it 14 880, la de `best.pth` de la rama a 20 000;
`26-10-2-denoising_red_vigente.png`). En `sub-028881` (CPU, fp32), en el cerebro:

| SNR | PSNR entrada → salida | NRMSE entrada → salida | 12 volúmenes, NRMSE (validación, bf16) |
|---|---|---|---|
| 10.2 | 20.18 → 34.12 dB | 29.08 → 5.84 % | PSNR 34.02 dB |
| 25 | 27.97 → 37.74 dB | 11.87 → 3.85 % | 12.19 → 4.01 % |
| 50 | 33.99 → 40.74 dB | 5.93 → 2.73 % | 6.10 → 2.84 % |
| 100 | 40.01 → 43.92 dB | 2.97 → 1.89 % | 3.05 → 1.98 % |

Mejora la entrada en los cuatro niveles (en los 12: +9.7, +6.6 y +3.7 dB a SNR 25, 50 y 100).

**COSMOS: la fase es simulada desde χ_COSMOS, con el mismo kernel que invierte el ADMM.**
`params.mat` (del repositorio CIB-UC-QSM/TDV, escrito el 2020-11-10) trae `phase_use`, `magn_use`
y `kernel`; su fase no es la adquirida del challenge 2016 (`26-10-3-campos_cosmos_sim2.png`):
- **φ − D∗χ_COSMOS es ruido blanco gaussiano.** Con el kernel de `params.mat` (el dipolo continuo
  con B0 en el eje 2, a 4·10⁻³), la pendiente es 0.9996 y la correlación 0.967 en χ ≠ 0. La
  autocorrelación del residuo a un vóxel es ≤ 0.007 (la del campo, 0.78–0.89). Tras suavizar 1.5
  vóxeles le queda la misma energía que a un ruido blanco con su heterocedasticidad (0.0065 y
  0.0065), la curtosis de r·m es 3.01 y la sd de r va como 1/m, salvo en el decil de menor
  magnitud (el borde).
- **Sale del ángulo de una señal compleja.** φ está acotada en ±0.16233 ppm = ±π/(γB0TE),
  con γB0TE = 19.35 rad/ppm (B0·TE = 0.0723 T·s; por ejemplo, 2.89 T y 25 ms). Fuera de la cabeza
  la fase es uniforme en ese intervalo (rms 0.09372, el de la uniforme), y la magnitud es Rayleigh
  de σ = 0.01000 (por la mediana y por la media). Un σ redondo, el mismo en el fondo y en el
  residuo de la fase, es el de un ruido sumado en la simulación.
- **El ruido es de SNR 100 por el máximo:** σ = 0.0100 con la magnitud de máximo 1 (1.012 con el
  ruido), que es SNR 26 por la media de la magnitud en el cerebro. El fondo y el residuo de la fase
  dan el mismo σ (0.009996 y 0.009999).
- **Lectura:** el 25.95 % no es un error de código, pero es optimista. Es un crimen inverso: el
  dato sale del mismo χ, con el mismo kernel y en la misma grilla con que se invierte, y no trae
  desajuste de modelo, sólo ruido blanco. No dice cómo le iría a la red con una fase adquirida.
  Comparar con el 25.94 % del autor es justo, porque es el mismo dato; compararlo con resultados
  in vivo, no.
- **Además se elige con la GT** (ya anotado en «La evaluación QSM» y en «Abierto», α sin GT). La
  parada pesa poco al mejor target: en la 14 880 a 0.02, el NRMSE queda en 26.11–26.15 % de la
  iteración 18 a la 22, y a 50 iteraciones (target 0.02, μ 0.01) en 26.15–26.17 %. El update no
  baja del TOL (1.15 %) en ninguna, así que la parada del autor correría las 100. El target pesa
  más: al mejor de Sim2 (0.05), COSMOS da 29.88 % (+3.8 pp), y al mejor de COSMOS (0.02), Sim2 da
  39.44 % (+3.4 pp).
- **Referencias en el mismo dato**, con el λ elegido también con la GT: Tikhonov de gradiente en
  forma cerrada, 46.0 % en COSMOS y 50.9 % en Sim2; TKD, 62.5 y 63.6 %; el χ₀ del ADMM, 65.8 y
  70.3 %.
- **Corrige a E0** (21-10): su correlación de COSMOS, 0.80, se midió en la máscara refinada, que
  suma ~3 800 vóxeles de borde con χ = 0 y magnitud casi nula, donde la fase es casi uniforme. En
  χ ≠ 0 es 0.967. Su lectura («una simulación directa con ruido») queda.

**Sim2 es simulado, pero no con este modelo.** Es el sujeto simulado de `../tdv/data_mat`, y por
su nombre y su grilla (164 × 205 × 205, 1 mm) corresponde al fantoma del QSM Reconstruction
Challenge 2.0, que entregó imágenes GRE multieco sintetizadas con un simulador de RM a partir de
un fantoma de cabeza in silico (Marques et al., *Magn. Reson. Med.* 2021). La fase (`Sim2.mat`,
escrita el 2025-05-06) no la arma este repositorio:
- **φ − 0.965·D∗χ_GT tiene estructura:** la pendiente es 0.965 y la correlación 0.883. La
  autocorrelación a un vóxel es 0.18–0.26, y tras suavizar le queda el 4.2 % de la energía, contra
  0.57 % del ruido blanco. La curtosis es 118 y la sd del residuo, 0.0126 ppm en el decil de menor
  peso (el borde). Por Haar, sólo el 14 % del residuo es blanco a la escala de un vóxel (en COSMOS,
  el 83 %).
- **Lectura:** no es un crimen inverso con el kernel del ADMM. Trae un desajuste de modelo, de
  borde y cerca de las fuentes (E0), y por eso es la prueba más exigente de las dos: ahí la red da
  35.75–36.08 %. Ninguno de los dos sujetos es una fase adquirida.

**Qué cambia:** nada en la corrida en vuelo. `--val-snr` ya pondera por el máximo, y COSMOS sigue
eligiendo `best_qsm.pth`, sabiendo que es una simulación sin desajuste. Para decir algo in vivo
falta un sujeto con fase adquirida y una referencia («Abierto», arriba).
