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
- **`pgrep`/`pkill`, anclados:** `^[^ ]*python[0-9.]*( +-[^ ]+)* +[^ ]*train\.py` (`pgrep -f
  train.py` se encuentra a sí mismo). Cuenta procesos, no corridas —cada una suma 16 workers— y un
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
   un tiempo sólo se compara en la misma tarjeta. Con una tanda en vuelo, los tests van en CPU y
   sin los cuatro PSNR del paper, que tardan minutos cada uno:
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
- **Defecto conocido, sin corregir:** `rebobina()` restaura `opt.state_dict()`, y con él el lr: un
  rebobinado que cruza un hito deshace ese halving hasta el final de la corrida. Pesa con
  rebobinados frecuentes e hitos juntos (~46 % en `t1-rapido-30k`), y se ve en la columna `lr=` del
  log después de un `REBOBINADO`. Al arreglarlo, y al sumar cualquier estado (EMA, otro scheduler),
  se decide si viaja con el respaldo —la ventana sí, el lr no— y sin rebobinados nada cambia.

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
  cambia con `T_init` (`|∂L/∂T| ∝ 1/T` es el 70–86 % de `‖g‖` a 2·δ y el 27–36 % a 8·δ) y con el
  dominio (mediana ~10 en T1 contra ~54 en BSDS, `docs/mediciones/disyuntor-t1.txt`). La cota 10⁴
  y el hueco [6 113, 1.97·10⁵] del disyuntor son de BSDS. Con disyuntor, `max‖g‖` incluye los
  lotes rechazados (una corrida viva llega a 1.3·10¹⁰): se miran los aceptados.
- **`T` tiene una pared absorbente en 0 y dos caminos hasta ella**, que el disyuntor no ve: un
  **pico**, si arranca cerca de `δ_Adam = lr/√(1−β₂) = 0.01265` (lo que un pico desplaza bajo Adam,
  sea del tamaño que sea), y una **deriva** en corridas largas (desde 2·δ, `tinit/B-larga` bajó a
  0.59·δ y murió en la it 32 750). Por eso el default es 8·δ y 2·δ es un piso, no un valor de
  trabajo; nada verifica lo que se pase por `--t-init`. `λ → 0` es inocuo.
- **`T_init` cambia la exposición al estallido, no el endpoint**: de 2·δ a 10·δ el PSNR pareado se
  mueve ≤ 0.036 dB. Bajo Adam `T` no busca ningún valor, y `T* = 0.0247` es la coordenada del
  oficial, no un destino (`(T, K_N) → (cT, K_N/c)` da el mismo modelo).
- **En denoising, `λ` es casi plana**: apagarla cuesta 0.033 dB en el oficial; daña `λ` grande
  (0.5 cuesta 0.6–1.8 dB, `docs/mediciones/lambda-sensibilidad.txt`). Con el dipolo de QSM no se
  traslada sin medir: `w = λ/T` ya sale 1.6–2.0× más alto en T1.
- **La red no es escala-equivariante** (la Student-t no es 1-homogénea; 15–28 %,
  `docs/mediciones/escala-equivariancia.txt`): el reescalado entre σ es una aproximación.
- **La precisión no es un efecto grande** (TF32 contra bf16: 1.31×, p = 0.61,
  `docs/mediciones/precision.txt`), pero `rapido` contra `referencia` no está medido: un PSNR de
  uno no se compara con el del otro.
- **Lo que recupera una corrida es el rebobinado**, no el rechazo: los estallidos entran por lotes
  aceptados (`docs/mediciones/tinit-10k.txt`) y todo detector sobre `‖g‖` llega un paso tarde.
  Visto en una o dos semillas, no medido como tasa. Su costo es el % de iteraciones que descarta
  (0.4 % en la réplica, 72 % en T1 a 2·δ).
- **Las GPUs no son iguales**: la 1 (PCIe gen 4 ×4) tarda ~1.3× más que la 0 en `rapido`, así que
  lo que se compara va a la 0. En `referencia`, dos corridas rinden 1.99×
  (`docs/mediciones/dos-gpu.txt`). DDP está clausurado.

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
  abortar tras N rechazos.
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
  `test_reproduction.py` se saltea.
- `scripts/guardia_tanda.py`, el hook de la regla 2: con cualquier `train.py` vivo en la máquina,
  bloquea escribir **cualquier** `train.py` o `runs/*/plan.sh`, también de un worktree o de
  `../QSM-TDV`. En Bash juzga el texto del comando: bloquea también una lectura que nombre
  `train.py` y redirija, y deja pasar un `python3 - <<EOF`.
- `scripts/poder_logrank.py` (regla 5), `hazard_fase1.py`, `replay_lambda_adam.py` (re-simula `λ`
  bajo otro `init` sin GPU: contesta si cruza un umbral, no cuánto tarda), los `veredicto_*.py` y
  `resumen.py`, que genera `docs/resultados/`.
