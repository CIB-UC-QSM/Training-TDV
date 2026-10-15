# Contexto para agentes

Training-TDV reentrena Total Deep Variation (Kobler et al., 2020) desde cero para denoising 2D,
como paso previo a usarlo de prior en QSM 3D. **La fase 1 —reproducir el entrenamiento del paper—
está cerrada:** `t8delta-100k-s1` llega a 31.547 dB en CBSD68 (σ = 25) contra 31.53 publicado. Lo
que sigue es otro dominio: cortes de T1 (`datasets/ldm100k`) y después QSM.

Acá va sólo lo que no está en otro lado. El estado y la historia viven en la bitácora; el uso y la
teoría, en el README.

## Por dónde empezar

- **Qué pasa y qué sigue:** `## Cómo retomar` de la bitácora
  (`sed -n '/^## Cómo retomar/,/^## Inventario/p' docs/bitacora.md`) y sus últimas entradas
  (`grep -n '^## 20' docs/bitacora.md | tail -5`), que suelen ir por delante. La tabla `## Estado`
  es historial congelado: se consulta con `grep`, no se lee.
- **Qué corre ahora:** `systemctl --user list-units 'tdv-*'` y `nvidia-smi`.
- **Si una idea ya se probó:** `grep -in "<término>" docs/bitacora.md` y la tabla `## Descartado`.
- **Uso, banderas y conceptos:** `README.md`. **Por qué cada constante vale lo que vale:** las notas
  al final de `train.py`. **Qué hay en `docs/`:** `docs/README.md`.
- **Qué significa cada número:** `docs/metricas.md` (la sección 0 explica sin fórmulas semilla,
  brazo y hazard). Sus rangos son de BSDS en color: en gris la `loss` suma 93·93 valores y no hacer
  nada vale 83.1, no 249.4.

## Mecánica de la sesión

- **Consultar, no leer entero:** la bitácora pesa ~290 kB, el README ~50 kB y
  `docs/presentacion/pipeline-explicado.txt` ~60 kB. Las entradas anteriores al 2026-10-08 citan
  rutas viejas; la traducción está en la entrada «Cierre de la fase 1 y reordenamiento».
- **Python de más de ~20 líneas va a un archivo**, no pegado en un comando. Una medición nueva deja
  su script en `scripts/` y su salida cruda en `docs/mediciones/<pregunta>.txt`, con script y fecha.
- **Una tanda la lanza el autor, o el agente si el autor lo pide en ese turno**, mirando antes las
  GPUs (los `plan.sh` lanzan igual): en segundo plano (`systemd-run --user` con una unidad
  `tdv-<nombre>`, o `tmux`), nunca encadenada a otra ni esperada con `until`/`sleep`.
- **`pgrep`/`pkill`, anclados:** `^[^ ]*python[0-9.]*( +-[^ ]+)* +[^ ]*train(3d)?\.py`, que cuenta
  también las corridas de `train3d.py` (`pgrep -f train.py` se encuentra a sí mismo). Cuenta procesos, no corridas —cada una suma 16 workers— y un
  `pkill` las mata a todas; una sola se para con `systemctl --user stop <unidad>`.
- **Con el autor:** el número antes que la interpretación, cada término definido la primera vez,
  suposición y medición dichas como tales, y un puntero (`archivo:línea`) antes que una paráfrasis.

## Reglas

Cada una salió de un error caro. Otros archivos citan las reglas por número y las secciones por
título: no renumerar ni renombrar sin un `grep -rn` sobre `.claude/`, `scripts/` y `docs/`.

1. **No parchear un síntoma.** Lo que el paper no tiene no entra para tapar un fallo: el lr propio de
   los escalares y el piso `t_min` crearon modos de fallo nuevos, el clip no rescató las dos veces
   que se probó, y el warmup lo descarta esta regla sin probarlo. La única pieza ajena al paper, el
   disyuntor de lotes, entró con mecanismo medido y siendo inerte mientras no rechaza (sin
   rechazos, la corrida es bit a bit la de la receta); su efecto sobre la tasa no está medido.
2. **No tocar `train.py` ni el `plan.sh` de una tanda en vuelo.** `plan.sh` llama a `train.py` una
   vez por semilla —una bandera quitada en vivo mató diez corridas— y `bash` lo lee de a pedazos.
   Si no puede esperar, se para la tanda, y eso lo decide el autor. Un worktree sirve para preparar
   el cambio, no para integrarlo.
3. **No medir con las GPUs ocupadas:** bajo contención, `cudnn.benchmark` elige otros algoritmos; y
   un tiempo sólo se compara en la misma tarjeta y **a la misma iteración**: el ETA que imprime
   `train.py` es el promedio acumulado desde `t0`, así que a it 500 es casi todo arranque de
   `torch.compile` y no dice nada contra uno de it 16 000. Con una tanda en vuelo, los tests van
   en CPU y sin los cuatro PSNR del paper, que tardan minutos cada uno:
   `CUDA_VISIBLE_DEVICES="" uv run pytest -k "not published_psnr"`.
4. **Una variable por vez.** `intermedio` y `referencia` difieren en tres cosas (TF32, determinismo,
   `benchmark`); `rapido` e `intermedio`, sólo en la precisión.
5. **Potencia antes de lanzar, sobre el endpoint que se va a leer.** PSNR: la sd del mejor PSNR
   entre semillas iguales es 0.16 dB, así que con una semilla por brazo lo mínimo detectable es
   ~0.33 dB. Exposición al estallido: log-rank exacto sobre los **tiempos** de evento, nunca
   mortalidad binaria, que subestima (mismos grupos: Fisher p = 0.45, log-rank p = 0.022). Con 12
   semillas por brazo, censura a 4 000 it y sin disyuntor, el poder es 0.11 contra 1.35× de hazard,
   0.70 contra 3× y 0.94 contra 5×: 1.35× pide ~300 semillas por brazo y no se lanza.
   `scripts/poder_logrank.py` cubre sólo ese caso y no da la potencia condicional a un control ya
   corrido. Sin potencia, se lanza sólo si la pregunta principal es otra, escrita **antes**.
6. **No re-correr una semilla para entender una muerte.** Un ULP de float32, 284 iteraciones antes,
   borra un estallido (`docs/mediciones/replay-ulp.txt`): una muerte no es atribuible a un estado,
   un lote ni un checkpoint, y lo que es de una configuración es la **tasa**. De una corrida sola sí
   se dice qué pasó y si el disyuntor la contuvo (`REBOBINADO` en el log, columnas 8–9 de la traza).
7. **Una hipótesis descartada vale tanto como una confirmada**, si queda escrito por qué, con su
   fila en `## Descartado`.
8. **La bitácora se escribe en cuanto corre el test**: configuración, número y veredicto, aunque el
   veredicto sea «nada».
9. **El repositorio es la única fuente.** Nada queda sólo en la memoria de un agente, en `runs/` o
   en una conversación: va al README, a este archivo, a `docs/bitacora.md` o a `docs/metricas.md`.
10. **Una afirmación se verifica antes de publicarse y se retira en todos los sitios** (skill
    `afirmar`): `grep -rin` del término y del número, en el mismo commit.

## El código

- **`ddr/` es un port de VLOGroup/tdv**: los nombres de atributo son las claves del `state_dict` y
  los oficiales cargan con `strict=True`, así que renombrar rompe todos los checkpoints. Antes de
  tocar el modelo, leer el original. `../tdv` es el port 2D del que sale `ddr/` (no es de los
  autores), con el solver ADMM-QSM; `../QSM-TDV`, la implementación 3D.
- **Los autores publicaron sólo inferencia** (modelo, `denoise.py`, dos checkpoints). «Lo que
  hicieron ellos» se verifica contra ese repo o el `config` de los `.pth`, que vale para
  arquitectura y modos pero no para los init: su `T = 0.001` mató tres corridas.
- **Los tests son el contrato** (`uv run pytest`): si un cambio los rompe, el cambio está mal. Un
  cambio de `train.py` que no debe tocar la aritmética se valida además con 120 iteraciones en
  `referencia`, antes y después: `traza.npy` idéntica bit a bit.
- **`train.py` no corta nunca el bucle** (ni `break`, ni aborto por `NaN`, ni early stopping): los
  análisis de supervivencia cuentan filas de la traza. Una corrida terminó si su `estado.json` dice
  `"terminada": true`; las cortadas a mano llevan uno escrito a mano, con `false` y el motivo.
- **Clases, sólo si heredan de `nn.Module` o `Dataset`, o si las tiene el port** (skill `huella`).
- **Sin test que los cubra:** la pérdida mira sólo `x_S`; el ruido se sortea nuevo en cada
  iteración; y no se parte del checkpoint oficial si el resultado se compara con él.
- **El respaldo del rebobinado se restaura como COPIA, y el lr no vuelve con él.** Arreglado el
  2026-10-09 en `main` y, por separado y con el mismo código, el 10-08 en la rama `3D-Training`
  (`train3d.py` importa `rebobina()` de `train.py`); sin rebobinados nada cambia. Las dos mitades
  siguen siendo trampas vivas para quien sume estado:
  `opt.load_state_dict` **adopta** los tensores que recibe —sin `deepcopy`, los pasos siguientes
  mutan el respaldo y el 2º rebobinado al mismo ya no devuelve Adam, que es el lazo de
  `t1-guard-2delta-5k` (`docs/mediciones/alias-respaldo-adam.txt`)—, y del `state_dict` del
  optimizador cuelga el lr, que es del calendario y no retrocede. Al sumar cualquier estado (EMA,
  otro scheduler) se decide si viaja con el respaldo —la ventana sí, el lr no— y que nada se
  restaure por referencia. Los dos casos tienen test en `tests/test_disyuntor.py`. **La mitad del
  alias tiene una sola prueba en vivo, en 3D:** en `v12-control` (`ventana-3d-12h`, 2026-10-13,
  `rapido`, una semilla) los dos rebobinados, en la 37 y la 72, fueron al mismo respaldo de la it 3,
  y de la 73 a la 400 no hubo ni un rechazo (con el defecto, en `t1-guard-2delta-5k`, los repetidos
  rechazaban el 96 % de las 25 siguientes; `docs/mediciones/ventana-3d-12h.txt`). En 2D sigue sin
  ejercerse: los 19 rebobinados de las cuatro corridas de 10⁵ van a respaldos distintos. **La mitad
  del lr no tiene prueba en vivo:** en 2D ningún rebobinado cruzó un hito, y `train3d.py` reescribe
  el lr en cada iteración, así que ahí no se puede ejercer.

## Lo que ya se sabe

Medido en BSDS400 color a σ = 25: otro dominio pide volver a medirlo.

- **El gradiente es exacto** (contra autograd en float64, a ~10⁻¹⁶): los estallidos de 10⁸–10¹³
  son el gradiente verdadero y nacen en el **backward**, con la `loss` todavía normal.
- **`cudnn.benchmark` rompe la reproducibilidad**: `referencia` lo apaga y repite bit a bit; en
  `rapido` e `intermedio` se mide la tasa sobre semillas, nunca una corrida contra otra.
- **Evento, muerte y corrida perdida son tres cosas.** Evento: primera iteración con `‖g‖ > 10⁴`.
  Muerte: `NaN`, o `T` en 0, donde la salida es la entrada (20.17 dB). Perdida: PSNR hundido sin
  morir. El hazard de eventos crece con la corrida (Weibull `k` = 2.07,
  `docs/mediciones/hazard-fase1.txt`): no se extrapola, y se compara sobre la misma ventana.
  Endpoint: sin disyuntor, el mejor PSNR; con disyuntor, el PSNR a iteración pareada —la efectiva,
  si hay muchos rebobinados (`docs/mediciones/rebobinados-t1.txt`)—.
- **`‖g‖` se lee contra la mediana móvil de la corrida, nunca en absoluto entre configuraciones**:
  cambia con `T_init` (`|∂L/∂T| ∝ 1/T`) y con los canales —la `loss` suma 93·93·C—, y corregido por
  ×3 no queda brecha medible entre T1 y BSDS (`docs/mediciones/nivel-g-t1-vs-bsds.txt`). **El guard
  relativo es invariante de escala**: reescalar `‖g‖` entera deja el set de rechazos idéntico, así
  que el nivel de la mediana no explica ninguna diferencia de tasa
  (`docs/mediciones/umbral-guard-escala.txt`). La cota `GUARD_ABS` no se recalibra por dominio:
  queda 6.0× (BSDS) y 5.0× (T1, corregido por canales) sobre el mayor `‖g‖` aceptado, y no ha
  atrapado un lote por su cuenta en ninguno (`docs/mediciones/cotas-disyuntor-dominio.txt`).
  `max‖g‖` incluye los rechazados: se miran los aceptados.
- **`T` tiene una pared absorbente en 0 y dos caminos hasta ella**, que el disyuntor no ve: un
  **pico**, si arranca cerca de `δ_Adam = lr/√(1−β₂)` —lo que un pico desplaza bajo Adam, sea del
  tamaño que sea; derivado y medido con 3–6 % de error sobre cinco órdenes de magnitud— y una
  **deriva** en corridas largas. **`δ_Adam` no es una constante: va con el lr vivo, y el calendario
  lo parte a la mitad en cada hito** (`train.py:51` lo calcula una vez desde el `LR` inicial, que
  es la trampa), así que la reserva `T/δ_vivo` se duplica sola y el riesgo se concentra en la
  primera fase de lr.
- **NO hay un piso en `T_init`; hay un costo continuo, y está medido a 10⁵.** La «regla 2-δ» se
  **retiró el 2026-10-12** (fila en `## Descartado`): el factor 2 era una elección, el mismo
  razonamiento ya estaba refutado para `λ`, y la evidencia va al revés —`tinit/B-larga` arrancó
  **en** 2·δ y murió con `T = 0`, y `t1p1delta-rapido-100k` arrancó en **1.1·δ** y terminó las
  100 000—. Lo medido (`docs/mediciones/modo-y-tinit-100k-s1.txt`): 8·δ → 31.547 dB con 163
  rechazos; 2·δ → −0.011 dB con **1.99×**; 1.1·δ → −0.022 dB con **4.7×** y el 91 % de ellos en la
  primera fase. El default es 8·δ **porque es el de la réplica**, no porque haya umbral. Nada
  verifica lo que se pase por `--t-init`, y `λ → 0` es inocuo.
- **`T` sola no significa nada: es una coordenada de gauge.** `(T, K_N) → (cT, K_N/c)` deja el
  modelo **idéntico** (1e-15 en float64, `docs/mediciones/degeneracion-T-KN.txt`) porque `K_N` entra
  lineal en `R`, así que `T·∇R` es lo invariante y `T` sola no se compara entre corridas. De ahí
  que **`T*` no sea un atractor** —cerrado el 10-11: `T_init/δ` → `T` final`/δ` da 1.1 → 2.23,
  2 → 2.76, 8 → 6.53, 8 → 6.83, el orden intacto— y que los 49 de 50 puntos pareados por debajo
  del control sean un sesgo chico y consistente pero no un test de signos limpio (los puntos de
  eval están autocorrelacionados).
- **En denoising, `λ` es casi plana**: apagarla cuesta 0.033 dB en el oficial; daña `λ` grande
  (0.5 cuesta 0.6–1.8 dB, `docs/mediciones/lambda-sensibilidad.txt`). Con el dipolo de QSM no se
  traslada sin medir.
- **`w = λ/T` es de gauge y NO lo usa el solver de QSM. Las dos cosas las afirmé mal y se
  retiraron el 2026-10-12** (dos filas en `## Descartado`). Bajo `(T, K_N) → (cT, K_N/c)` el modelo
  no cambia y `w → w/c`, así que `w` no es propiedad de la función aprendida. Y
  `../tdv/wTDV_QSM_torch.py:188` usa la red **entera** como denoiser PnP —`vn(z·α, z·α)/α`, con
  `α = target_std/qsm_std`— sin leer `T` ni `λ`. Lo invariante son las dos **fuerzas** del esquema
  `x ← x − (1/S)[T·∇R(x) + λ·(x − z)]`: `T·‖∇R‖` y `λ·‖x − z‖`, y su cociente `ρ`. Medido en las
  cuatro corridas a 10⁵ (`docs/mediciones/gauge-4-corridas.txt`): `w` abre **12.2×** pero `T·‖∇R‖`
  queda dentro de **1.13×** y los cuatro `∇R` tienen coseno **0.978–0.987** entre sí — o sea casi
  todo el spread de `w` era gauge. Lo que **sí** es real y separa por **modo** (no por `T_init`) es
  la fuerza del dato: `ρ ≈ 0.15–0.18` en `referencia` contra **0.031–0.039** en `rapido`, 4.8×. Aun
  así los cuatro denoisers coinciden a **42–46 dB** entre sí.
- **Para un checkpoint destinado a PnP, la PSNR de denoising NO es criterio de selección.** El
  esquema es descenso **explícito** sobre `T·R`, estable sólo si `(T/S)·λ_max(∇²R) < 2`. El
  entrenamiento fija ese número justo bajo 2 **a la amplitud a la que entrena** —los cuatro
  checkpoints de 10⁵ caen en 1.38–2.01 a σ = 25— y **no controla nada 4× más abajo**, que es donde
  lo corre el solver de QSM (`target_std = 0.06`): ahí el oficial baja a **0.99** y los tres
  nuestros suben a **2.79 / 5.76 / 23.2**, con los cuatro coincidiendo en **0.027 dB** de PSNR.
  Dentro del ADMM eso es 27.34 % de RMSE convergiendo en 33 iteraciones contra 28.36–28.50 % sin
  converger en 100 (2026-10-12). **Medilo antes de elegir**, que es barato y sólo necesita el
  checkpoint: `scripts/diag_qsm_estabilidad.py`. Y ojo: la Lipschitz del denoiser con una dirección
  aleatoria es **cota inferior** y engaña (`1.1δ` da 0.108 ahí y 5.76 por el Hessiano).
- **Antes de investigar el solver de QSM, mirá `git log --all` de `../tdv`.** El 2026-10-12 re-derivé
  entero un mecanismo que ya estaba medido el 10-09 en la rama `worktree-wtdv-channels-flag`
  (commit `331491c`), porque miré el working tree de `master` y ahí no estaba: el trabajo de ese
  repo vive en ramas de worktree, no en `master`.
- **El solver de QSM apila tres cortes adyacentes como R, G, B** (`wTDV_QSM_torch.py:176`): usa el
  acoplamiento de color aprendido en BSDS como prior *through-plane*. Por eso un checkpoint de **1
  canal** —`checkpoint_qsm_rapido.pth`, `tdv3-3-25-f32-gray.pth`— **no corre ahí**: el batch es
  `(B, 3, H, W)`.
- **La red no es escala-equivariante** (la Student-t no es 1-homogénea; 15–28 %,
  `docs/mediciones/escala-equivariancia.txt`): el reescalado entre σ es una aproximación.
- **La precisión no es un efecto grande** (TF32 contra bf16: 1.31×, p = 0.61,
  `docs/mediciones/precision.txt`), y a 10⁵ iteraciones **`rapido` reproduce `referencia`**: misma
  semilla, 49 de 50 puntos pareados dentro de 0.05 dB, 31.546 contra 31.547 al final, 2.98× más
  rápido (`docs/mediciones/rapido-vs-referencia-100k.txt`). Una semilla, y `rapido` no se puede
  replicar contra sí misma. **Lo que sí se separa, y es invariante de gauge, es la fuerza del
  término de dato**: `λ·‖x − z‖` queda **4.8× más chica** en `rapido`, o sea `ρ = dato/reg` de
  0.031–0.039 contra 0.15–0.18 de `referencia`, y eso **separa por modo, no por `T_init`** (las dos
  corridas de cada modo caen juntas). El regularizador en cambio es el mismo en los cuatro:
  `T·‖∇R‖` dentro de 1.13× y coseno 0.978–0.987 (`docs/mediciones/gauge-4-corridas.txt`). Con un
  término de dato que pesa 3–18 % del regularizador, 4.8× sobre él mueve el denoiser **0.08 dB**.
- **Lo que recupera una corrida es el rebobinado**, no el rechazo: los estallidos entran por lotes
  aceptados (`docs/mediciones/tinit-10k.txt`) y todo detector sobre `‖g‖` llega un paso tarde.
  Visto en una o dos semillas, no medido como tasa. Su costo es el % de iteraciones que descarta
  (0.4 % en la réplica, 72 % en T1 a 2·δ).
- **Las dos GPUs no difieren en velocidad de forma medida**, aunque la 1 negoció PCIe gen 4 ×4 y la
  0 gen 5 ×16: en `referencia` son 0.9 % y la 1 es la rápida —la 0 maneja el display—, y dos
  corridas rinden 1.99× (`docs/mediciones/dos-gpu.txt`). En `rapido` no hay medición limpia: el
  «~1.3× más lenta la 1» que decía acá era arranque de `torch.compile`, no régimen
  (`docs/mediciones/velocidad-dos-tarjetas.txt`). DDP está clausurado.

## Lo que no sirve

Cerrado en `## Descartado` de la bitácora; se busca por el término.

- **Para estabilizar:** el clip, a cualquier umbral —los lotes que meten la corrida en la cuenca
  mala tienen `‖g‖` ≤ 510, debajo de todo umbral que no recorte lo sano, y adentro el `‖g‖`
  sostenido vuelve el clip una escala que Adam cancela; se re-abre sólo actuando sobre el signo—;
  lr propio para los escalares; piso `t_min`; atar `λ = T`. `T_init` (de 2·δ a 10·δ), `λ_init` y
  la precisión juegan entre 0.9× y 2.5× de hazard.
- **Para subir el PSNR:** mover `T_init` o `λ`; más iteraciones (de la 50 000 a la 100 000:
  +0.030 dB por 18.9 h de GPU, sin overfitting).
- **Para el disyuntor:** otro detector sobre `‖g‖` (`--guard-amp` incluido); rechazar sin
  rebobinar; la razón 31.62; la densidad de rechazos como disparador; bajar la cota a 5 000;
  abortar tras N rechazos; reconstruir la ventana con los rechazados o ensanchar la razón ante un
  rebobinado repetido —el lazo era el alias del respaldo, no la ventana—.
- **Como criterio de muerte o de corte:** `τ·L > 2` (el oficial opera a 4.8), `growth`, `amp₀`,
  `‖Δθ‖` (dos semillas a 131.8 % de `‖θ‖` dan el mismo PSNR), extrapolar `T` o el hazard. `σ_max`
  del adjunto separa a posteriori, pero no avisa.
- **Como causa de estallidos o muertes:** el port, la inicialización, `torch.compile`, el
  checkpointing, DDP, un desborde de `v` en Adam, la estructura residual, la Student-t
  redescendente.
- **Normalizar `‖K_N‖`:** casi no-op.

## Mejoras candidatas

Una por vez, contra la réplica a **iteración pareada** (`docs/resultados/t8delta-100k-s1.md`:
31.340 a la 10 000, no 31.547).

- **La de más valor para QSM: entrenar ciego al ruido**, con σ sorteado por parche en vez del
  `--sigma` único (`U(0, 55)` es la convención de DnCNN-B). Elimina el reescalado de `evaluate.py
  --sigma-ref` y el `alpha` de calibración del solver ADMM-QSM (`../tdv/wTDV_QSM_torch.py`), que
  existen porque la red conoce un solo σ.
- **Baratas, con condiciones:** EMA (su estado viaja con el respaldo), `channels_last`, AdamW con
  `T` y `λ` fuera del weight decay (si no, los empuja a su pared; derivado, no medido).
- **Cambian la receta:** cosine u otro calendario (heredan el defecto del lr); DIV2K/Flickr2K.
- **Qué NO tocar** (el porqué, en el README): BatchNorm/LayerNorm, ReLU (sin segunda derivada),
  atención con softmax, bilineal en vez del binomial, más features o bloques, otro potencial ψ.
  GELU y SiLU son suaves: probarlas es un experimento, no algo prohibido.
- **Búsqueda de hiperparámetros**, fuera de la receta: Optuna o Ray Tune con ASHA, presupuestando
  varias semillas por configuración.
- **Otro dominio:** el PSNR no se compara entre datasets (T1 tiene ~38 % de fondo casi plano); la
  referencia común es la entrada ruidosa, 20.17 dB. `datasets/ldm100k` es T1, no QSM; qué cambia en
  QSM, en el README.

## Lo que no está en un clon

`scripts/`, `.claude/` (las skills `tanda`, `afirmar`, `huella`, `papers`), `CLAUDE.md`, `runs/` y
las imágenes de `datasets/` existen sólo en la máquina del autor.

- Un worktree se crea desde `HEAD` (`git worktree add -b <rama> <ruta> HEAD`; `origin/main` puede
  ir atrás) y se le enlazan `runs` y `datasets/bsds500/images`; sin imágenes,
  `test_reproduction.py` se saltea. Para `train3d.py`, también `datasets/ldm100k3d/images`,
  `datasets/ldm100k3d/mascaras` y `datasets/qsm_cosmos` (sin ella, la evaluación QSM se apaga con
  un aviso).
- `scripts/guardia_tanda.py`, el hook de la regla 2: con cualquier `train.py` o `train3d.py` vivo
  en la máquina, bloquea escribir **cualquier** `train.py`, `train3d.py` o `runs/*/plan.sh`, también
  de un worktree o de `../QSM-TDV`. Los tests de `train3d.py` lanzan corridas cortas en
  subprocesos, así que mientras corre la suite el hook también bloquea. En Bash juzga el texto del comando: bloquea también una lectura que nombre
  `train.py` y redirija, y deja pasar un `python3 - <<EOF`.
- `scripts/poder_logrank.py` (regla 5), `hazard_fase1.py`, `replay_lambda_adam.py` (re-simula `λ`
  bajo otro `init` sin GPU: contesta si cruza un umbral, no cuánto tarda), los `veredicto_*.py` y
  `resumen.py`, que genera `docs/resultados/`.
