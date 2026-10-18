# Contexto para agentes

Training-TDV reentrena Total Deep Variation (Kobler et al., 2022) desde cero, para usarlo después
como prior en QSM. **Fase 1, 2D, cerrada:** `t8delta-100k-s1` (`train.py`) llega a 31.547 dB en
CBSD68 a σ = 25, contra 31.53 publicado. **Fase 2, 3D, en curso:** `train3d.py` entrena sobre
volúmenes T1 con cráneo (`datasets/ldm100k3d`, no QSM) y valida con PSNR, una reconstrucción QSM de
COSMOS y el número de estabilidad PnP. La corrida larga la lanza el autor en otra máquina.

Aquí va sólo lo que no está en otro lado: uso y teoría, en el README; estado e historia, en la
bitácora; el porqué de cada constante, en los bloques finales de los `.py` (`grep -n '^# --- '`).
Un `.txt` citado sin ruta está en `docs/mediciones/`.

## Por dónde empezar

- **Qué pasa y qué sigue:** `## Cómo retomar` de la bitácora
  (`sed -n '/^## Cómo retomar/,/^## Inventario/p' docs/bitacora.md`) y sus últimas entradas
  (`grep -n '^## 20' docs/bitacora.md | tail -5`), que suelen ir por delante. La tabla `## Estado`
  es historial congelado: se consulta con `grep`, no se lee.
- **Qué corre ahora:** `systemctl --user list-units 'tdv-*'` y `nvidia-smi`; la corrida larga 3D,
  en W&B (proyecto `training-tdv3d`) o en su `--out`.
- **Si una idea ya se probó:** `grep -in "<término>" docs/bitacora.md` y la tabla `## Descartado`.
- **README:** «Uso», «Banderas», «Qué deja una ejecución y cómo leerla» (el log, qué checkpoint),
  «Diferencias entre 2D y 3D» y «Cómo funciona». **El 3D frente al 2D, ítem por ítem:**
  `docs/decisiones-3d-vs-2d.md`, foto de `897fb8b` (ante una duda manda el código). **Qué hay en
  `docs/`:** `docs/README.md`.
- **Qué significa cada número:** `docs/metricas.md` (la sección 0 explica sin fórmulas semilla,
  brazo y hazard). Sus rangos son de BSDS en color: en gris la `loss` suma 93·93 valores y no hacer
  nada vale 83.1, no 249.4.

## Mecánica de la sesión

- **Consultar, no leer entero:** la bitácora pesa ~550 kB; `docs/decisiones-3d-vs-2d.md`, ~115 kB.
  Las entradas anteriores al 2026-10-08 citan rutas viejas (traducción en «Cierre de la fase 1 y
  reordenamiento»). El 2026-10-15 la historia publicada se aplastó en `ff4aa3c`: los hashes que
  cita la bitácora antes de esa fecha están en las etiquetas locales `historia-*`. El README de
  `ff4aa3c` guarda el banco de modos 2D (s/it, VRAM) y cuántas iteraciones hacen falta.
- **Python de más de ~20 líneas va a un archivo**, no pegado en un comando. Una medición nueva deja
  su script en `scripts/` y su salida cruda en `docs/mediciones/<pregunta>.txt`, con script y fecha.
- **Una tanda la lanza el autor, o el agente si el autor lo pide en ese turno**, mirando antes las
  GPUs (los `plan.sh` lanzan igual): en segundo plano (`systemd-run --user` con una unidad
  `tdv-<nombre>`, o `tmux`), nunca encadenada a otra ni esperada con `until`/`sleep`.
- **`pgrep`/`pkill`, anclados:** `^[^ ]*python[0-9.]*( +-[^ ]+)* +[^ ]*train(3d)?\.py`
  (`pgrep -f train.py` se encuentra a sí mismo). Cuenta procesos, no corridas (`train.py` suma 16
  workers; `train3d.py`, `--workers`), y un `pkill` las mata a todas; una sola se para con
  `systemctl --user stop <unidad>`.
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
2. **No tocar `train.py` (ni `train3d.py`) ni el `plan.sh` de una tanda en vuelo.** `plan.sh` llama
   a `train.py` una vez por semilla —una bandera quitada en vivo mató diez corridas— y `bash` lo lee
   de a pedazos. Si no puede esperar, se para la tanda, y eso lo decide el autor. Un worktree sirve
   para preparar el cambio, no para integrarlo.
3. **No medir con las GPUs ocupadas:** bajo contención, `cudnn.benchmark` elige otros algoritmos; y
   un tiempo sólo se compara en la misma tarjeta y **a la misma iteración** (el ETA es el promedio
   desde `t0`: a la it 500 es casi todo arranque de `torch.compile`). Con una tanda en vuelo, los
   tests van en CPU y sin los cuatro PSNR del paper, que tardan minutos cada uno:
   `CUDA_VISIBLE_DEVICES="" uv run pytest -k "not published_psnr"`.
4. **Una variable por vez.** `intermedio` y `referencia` difieren en tres cosas (TF32, determinismo,
   `benchmark`); `rapido` e `intermedio`, sólo en la precisión.
5. **Potencia antes de lanzar, sobre el endpoint que se va a leer.** PSNR: la sd del mejor PSNR
   entre semillas iguales es 0.164 dB (6 semillas de `runs/tinit01`; bitácora, PRE-REGISTRO del
   2026-10-04, punto 4), así que con una semilla por brazo lo mínimo detectable al 95 % es
   1.96·√2·0.164 ≈ **0.45 dB** con σ conocida, ≈ 0.60 con σ estimada en esas 6 (t₅). El «~0.33 dB»
   usado hasta el 2026-10-16 era 2σ sin el √2: retirado. Exposición al estallido: log-rank exacto
   sobre los **tiempos** de evento, nunca mortalidad binaria, que subestima (mismos grupos: Fisher
   p = 0.45, log-rank p = 0.022). Con 12 semillas por brazo, censura a 4 000 it y sin disyuntor, el
   poder es 0.11 contra 1.35× de hazard, 0.70 contra 3× y 0.94 contra 5×: 1.35× pide ~300 semillas
   por brazo y no se lanza. `scripts/poder_logrank.py` cubre sólo ese caso, sin la potencia
   condicional a un control ya corrido. Sin potencia, se lanza sólo si la pregunta principal es
   otra, escrita **antes**.
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
  autores), con los solvers ADMM-QSM (antes de investigarlos, `git log --all`: su trabajo vive en
  ramas de worktree, no en `master`). `../QSM-TDV`, la implementación 3D, no carga directo un
  checkpoint de `train3d.py` (`compat-qsm-tdv.txt`).
- **Los autores publicaron sólo inferencia** (modelo, `denoise.py`, dos checkpoints): «lo que
  hicieron» se verifica contra ese repo o el `config` de los `.pth`, que vale para arquitectura y
  modos pero no para los init (su `T = 0.001` mató tres corridas).
- **`train3d.py` importa la receta de `train.py`** (lr, betas, σ, S, arquitectura, inits,
  disyuntor, modos y `rebobina()`): tocar `train.py` mueve los dos.
- **Los tests son el contrato** (`uv run pytest`): si un cambio los rompe, el cambio está mal. Un
  cambio de `train.py` que no debe tocar la aritmética se valida además con 120 iteraciones en
  `referencia`, antes y después: `traza.npy` idéntica bit a bit.
- **Ningún entrenador corta el bucle** (ni `break`, ni aborto por `NaN`, ni early stopping): los
  análisis de supervivencia cuentan filas de la traza. Una corrida terminó si su `estado.json` dice
  `"terminada": true`; las cortadas a mano llevan uno escrito a mano, con `false` y el motivo. La
  única excepción es `--meseta` de `train3d.py`, pedida el 2026-10-16 y apagada por defecto:
  agotadas las mitades del lr, corta y lo deja en `estado.json` (`corte`, con `iteraciones_hechas`
  menor que las pedidas), así que se cuenta como censurada por diseño en `corte.iteracion`.
- **Clases, sólo si heredan de `nn.Module` o `Dataset`, o si las tiene el port** (skill `huella`).
- **Sin test que los cubra:** la pérdida mira sólo `x_S`; el ruido se sortea nuevo en cada
  iteración; y no se parte del checkpoint oficial si el resultado se compara con él.
- **Estado que se sume al rebobinado (EMA, otro scheduler): se restaura por copia** y se decide si
  viaja con el respaldo (la ventana sí, el lr no): `opt.load_state_dict` adopta lo que recibe
  (`alias-respaldo-adam.txt`, `tests/test_disyuntor.py`). En vivo, la copia la ejerció sólo
  `v12-control` (3D); la mitad del lr, nunca.

## Entrenamiento 3D

- **`rapido` por defecto:** `referencia` cuesta 16–21× por vóxel (`costo-tdv3d.txt`, estimado). En
  la 5070 Ti, ~9 s/it con lote 32 de 48³ (`runs/v12-sonda*`) y ~7.3 min por validación si el ADMM
  no corta (enmienda de `ventana-3d-12h`). De `rapido` se lee la tasa sobre semillas; una corrida
  contra otra, sólo rama contra tronco (mismos parches y ruido), sobre la dispersión del tronco.
- **La evaluación QSM y la estabilidad no tocan la aritmética** (tests: traza bit a bit igual).
  Dependen de COSMOS, `--qsm-max-iter` y `--estabilidad-lado`, libres al reanudar: cambiarlos a
  mitad de corrida mezcla criterios, y nada lo verifica.
- **`(T/S)·ρ(∇²R)` es un índice, no una garantía.** `*INESTABLE*` usa el umbral 2
  (`ddr/qsm3d.py:78`), no `2 + λ/S` (< 0.7 %). Por debajo de 2 un paso también expande, por
  curvatura negativa: en el oficial, en `x_0`, 1.81 con `‖J_0‖` = 1.84
  (`adjunta-sigma-max.txt:51`). El artículo mide un cociente secante en direcciones de ruido
  (Fig. 10, `α₁ < 1`); aquí, la peor dirección: no se contradicen.
- **A QSM se pasa `best_qsm.pth`, y estable no es convergente:** en `v12-control`, estabilidad 1.54
  y ADMM divergido a 144 % (`ventana-3d-12h.txt`); se mira `nrmse_final_pct`.
- **Con `--mascara-cerebro`, el PSNR del recorte entero no sirve:** la red no ve fondo ni cráneo, y
  salta 24.7–28.5 dB entre validaciones (prueba de sobreajuste, 2026-10-16). Se mira
  `val/psnr_cerebro`; `best.pth` se sigue eligiendo con el del recorte entero.
- **El micro-lote es cualquier entero** (2026-10-16): el lote se parte en trozos casi iguales y
  `auto` deja menos pasadas (16 en 96 GiB, 11 en 48, 3 en la 5070 Ti). Con trozos iguales la
  corrida es bit a bit la de antes; la sonda mide los tamaños que cambian las pasadas.
- **`--sobreajuste` va sin disyuntor:** con datos fijos, un lote rechazado deja igual la iteración
  siguiente y el rebobinado vuelve al mismo estallido (`runs/sobreajuste-3d`). La prueba pasó:
  lote fijo a 33.29 dB contra 31.38 de su validación en cerebro, y 32.17 con datos frescos en la
  1 000 (bitácora, 2026-10-16).
- **Fallar al arrancar:** `--seguimiento online` sale con error si falta wandb o la credencial, o
  si W&B no abre; sin GPU visible y sin `--sin-compilar`, `train3d.py` también, en vez de
  entrenar en CPU.
- **Memoria, abierto:** `MARGEN = 0.10` (`train3d.py:84`) es un supuesto, y la sonda no corre la
  validación. Un lector midió (2026-10-15) a 24³ × 8 un pico de 8.16 GiB contra 4.14 predichos, y
  la sonda ~15 % sobre el modelo (a 48³, ~1 %: `runs/v12-sonda*`); bitácora, «README y AGENTS.md
  reescritos» (2026-10-16).
- **El estado de una corrida** está en su `--out`: `estado.json` (falta mientras corre),
  `history.json`, `qsm.jsonl`, `traza.npy` (columnas: bloque «TRAZA» de `train3d.py`) y el log; W&B
  es una vista. `scripts/resumen_ventana_3d.py` resume varias.

- **La corrida larga va en otra máquina** (RTX PRO 6000 Blackwell Max-Q de 96 GB, sin probar
  allí). El procedimiento está en el README, «Llevar el entrenamiento 3D a otra máquina». Lo fijado y
  lo abierto, en `## Cómo retomar` y en la entrada «Decisiones para la corrida larga» (2026-10-15).
  Como servicio se lanza con `.venv/bin/python`, no con `uv run`: el `PATH` de una unidad systemd
  no incluye `~/.local/bin`. Al vigilarla: `train/rechazos`, `train/rebobinados`, `train/T` lejos
  de 0, `qsm/nrmse_final_pct`, `qsm/iteraciones` < 100 y `qsm/estabilidad` < 2.

## Lo que ya se sabe

En BSDS400 color a σ = 25 salvo aviso; en 3D, supuesto (el codo del lr no se traslada: entrada «¿El
recorte del calendario se traslada al 3D?»).

- **El gradiente es exacto** (`tests/test_gradient.py`): los estallidos de 10⁸–10¹³ son reales y
  nacen en el backward, con la `loss` normal.
- **`cudnn.benchmark` rompe la reproducibilidad;** `referencia` lo apaga y repite bit a bit.
- **Evento, muerte y corrida perdida son tres cosas** (`docs/metricas.md`); el hazard crece con la
  corrida (`hazard-fase1.txt`). Endpoint: sin disyuntor, el mejor PSNR; con él, a iteración
  pareada (`rebobinados-t1.txt`).
- **`‖g‖` se lee contra la mediana móvil**, no en absoluto (`nivel-g-t1-vs-bsds.txt`); el guard
  relativo es invariante de escala (`umbral-guard-escala.txt`), y la cota absoluta, 5–6× sobre el
  mayor aceptado (`cotas-disyuntor-dominio.txt`), en 3D se escala con `P³`, sin calibrar.
- **`δ_Adam` va con el lr vivo** (`train.py:51` usa el inicial): el riesgo de que un pico lleve `T`
  a 0 está en la primera fase (`adam-respuesta-al-pico.txt`). `tinit/B-larga`, sin disyuntor, no
  derivó: cayó de 2.15·δ a 0.72·δ en un estallido (`runs/tinit/todo.log`). Con él no murió
  ninguna de las cuatro de 10⁵ (semilla 1); que contenga esos picos no está medido.
- **No hay piso en `T_init`, hay un costo** («regla 2-δ» retirada): a 10⁵, 2·δ, −0.011 dB y 1.99× de
  rechazos (`modo-y-tinit-100k-s1.txt`); 1.1·δ, −0.022 dB y 4.7× (`t1p1delta-100k.txt`). 8·δ es el
  de la réplica; nada verifica `--t-init`.
- **`T` sola es de gauge** (`degeneracion-T-KN.txt`): ni `T*` es destino ni `w = λ/T` es del
  modelo; lo invariante es `T·‖∇R‖` (igual en las cuatro de 10⁵) y `λ·‖x − z‖` (4.8× menor en
  `rapido`, a 0.08 dB; `gauge-4-corridas.txt`).
- **`rapido` reproduce `referencia` a 10⁵** (una semilla, `rapido-vs-referencia-100k.txt`); TF32
  contra bf16, 1.31× de hazard, p = 0.61 (`precision.txt`).
- **`λ` es casi plana en denoising** (apagarla, −0.033 dB, `lambda-sensibilidad.txt`); en QSM, sin
  medir.
- **Para un prior PnP el PSNR no elige:** a la amplitud de QSM el paso del oficial expande ×1.27
  (ρ de −0.28 a 1.03) y el de los tres propios de σ = 25, ×1.77–23.9, con 0.022 dB de PSNR entre
  ellos, y de esos cuatro sólo el oficial converge (`espectro-extremos-qsm.txt`; el 0.99 de la §2
  de `qsm-prior-4corridas.txt` era la potencia sin converger, y un solo extremo). Ninguno tiene un
  paso no expansivo: el oficial da ×1.47 a σ = 25. Una Lipschitz con dirección aleatoria es cota
  inferior. El dominante del oficial es curvatura positiva y el de `t8delta-100k-s1`, negativa;
  `ciego2d-35k` expande ×3.13, sólo por curvatura negativa, y converge igual, a 33.18 %
  (`qsm-prior-ciego2d.txt`), mientras el de 2·δ no converge con ×1.77: el índice no predice la
  convergencia en ningún sentido.
- **El solver 2D de QSM** (`../tdv/wTDV_QSM_torch.py`) usa la red entera, `vn(z·α, z·α)/α`, sin
  `T` ni `λ`; `--channels 3` apila tres cortes como RGB, y `1` va corte a corte.
  `checkpoint_qsm_rapido.pth` es `runs/t1-rapido-30k/best.pth`.
- **La red no es escala-equivariante** (15–28 %, `escala-equivariancia.txt`).
- **Recupera el rebobinado, no el rechazo** (`tinit-10k.txt`, una o dos semillas); cuesta lo que
  descarta: 0.4 % en la réplica, 72 % en T1 a 2·δ.
- **Las dos GPUs no difieren de forma medida** (`dos-gpu.txt`); DDP, clausurado.

## Lo que no sirve

Cerrado en la bitácora, casi todo con fila en `## Descartado`: se busca el término, o lo que va
entre «».

- **Clip, a cualquier umbral:** los lotes que llevan a la cuenca mala tienen `‖g‖` ≤ 510, y adentro
  Adam cancela la escala; se re-abre sólo sobre el signo («clipping»).
- **lr propio de los escalares, piso `t_min` y warmup** (regla 1; «`scalar-lr-scale`», «`t_min`»,
  «warmup»). **Atar `λ = T`:** Adam no es invariante («atar `λ = T`»).
- **`T_init` (2·δ–10·δ), `λ_init` o la precisión para estabilizar:** 0.9×–2.5× de hazard, y
  `T_init` no mueve el endpoint («`T_init` mueva el endpoint»).
- **Más iteraciones tras la 50 000:** +0.030 dB en 18.8 h (`t8delta-100k.txt:59`; cuántas hacen
  falta, `calendario-lr.txt`).
- **Para el disyuntor:** otro detector sobre `‖g‖` («guard-amp»), la cota absoluta como prevención
  («cota ABSOLUTA»), rechazar sin rebobinar, la razón 31.62, la densidad de rechazos, la cota en
  5 000, abortar tras N rechazos y tocar la ventana ante un rebobinado repetido («lazo»: era el
  alias).
- **Como criterio de muerte o de corte:** `τ·L > 2` (el oficial opera a 4.8), `growth`, `amp₀`,
  `‖Δθ‖`, extrapolar `T` o el hazard; `σ_max` del adjunto separa a posteriori, pero no avisa.
- **Como causa:** el port, la inicialización, `torch.compile`, el checkpointing, DDP, TF32, un
  desborde de `v` en Adam, la estructura residual, la Student-t redescendente.
- **Para elegir un prior PnP:** `w = λ/T`, `T·‖K_N‖`, las eigenfunciones de `R`, subir
  `target_std` y el Rayleigh con signo como `λ_max`.
- **La «regla 2-δ»**, **`T*` como «atractor»** y **«normalizar `‖K_N‖`»** (casi no-op).

## Mejoras candidatas

Una por vez, contra la réplica a **iteración pareada** (`docs/resultados/t8delta-100k-s1.md`:
31.340 a la 10 000, no 31.547).

- **La de más valor para QSM: entrenar ciego al ruido**, con σ sorteado por parche (`U(0, 55)`,
  DnCNN-B): quitaría el reescalado de `evaluate.py --sigma-ref` y el `alpha` del solver ADMM-QSM.
  En 3D ya es bandera (`--sigma-max 55`), sin medir. En 2D, `ciego2d-35k` (una semilla, 35 000
  iteraciones) bajó el número en QSM de 23.2 a 2.14, sin cruzar 2 y con el dominante todavía
  negativo; en el ADMM converge, pero a 33.18 % contra 27.34 % del oficial. Falta el control con
  σ = 25 y su calendario (RESULTADO, 2026-10-17).
- **Baratas, con condiciones:** EMA (su estado viaja con el respaldo), `channels_last`, AdamW con
  `T` y `λ` fuera del weight decay (si no, los empuja a su pared; derivado, no medido).
- **Cambian la receta:** cosine u otro calendario; DIV2K/Flickr2K; búsqueda con Optuna o Ray Tune
  (ASHA), con varias semillas.
- **Qué NO tocar.** ReLU, BatchNorm y autoatención: README, «El esquema». **LayerNorm:** su
  jacobiano crece como `1/√ε` donde la varianza se anula (zonas planas tras un `K₁` de media cero).
  Y lo que eligió el artículo (comparable con el oficial): 32 features, tres bloques y
  tres escalas; potencial final identidad (empató con los otros dos en la Tabla 1: 0.006–0.013 dB,
  un valor por potencial); antialias binomial, no bilineal (`tests/test_conv.py`); sin sesgo, y
  `K₁` de media cero y norma ≤ 1.
- **Hacia QSM** (dipolo, ppm, eje del campo, métricas, y lo que falta medir): entradas «Fase 2 en
  3D: qué cambia respecto del paper» y «El checkpoint 3D no carga exacto en QSM-TDV…».

## Lo que no está en un clon

`scripts/`, `.claude/` (skills `tanda`, `afirmar`, `huella`, `papers`), `CLAUDE.md`, `runs/`, las
imágenes de `datasets/` y las etiquetas `historia-*` existen sólo en la máquina del autor.

- Un worktree se crea desde `HEAD` (`git worktree add -b <rama> <ruta> HEAD`; `origin/main` puede
  ir atrás) y se le enlazan `runs` y `datasets/bsds500/images` (sin ellas, `test_reproduction.py`
  se saltea); para `train3d.py`, también `datasets/ldm100k3d/images`, `datasets/ldm100k3d/mascaras`
  y `datasets/qsm_cosmos` (sin ella, la evaluación QSM se apaga).
- `scripts/guardia_tanda.py`, el hook de la regla 2: con cualquier `train.py` o `train3d.py` vivo
  (también la suite, que lanza corridas cortas), bloquea escribir **cualquier** `train.py`,
  `train3d.py` o `runs/*/plan.sh`, también de un worktree o de `../QSM-TDV`. En Bash juzga el
  texto: bloquea una lectura que nombre `train.py` y redirija; deja pasar un `python3 - <<EOF`.
- `scripts/` también: `poder_logrank.py` (regla 5), `replay_lambda_adam.py` (si `λ` cruza un
  umbral bajo otro `init`, sin GPU), los `veredicto_*.py` y `resumen.py`.
