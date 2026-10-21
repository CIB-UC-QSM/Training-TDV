# Contexto para agentes

Training-TDV entrena Total Deep Variation (Kobler et al., 2022) en 3D (`train3d.py`) para usarlo
como prior de un ADMM plug-and-play en QSM. Entrena sobre volúmenes T1 con cráneo
(`datasets/ldm100k3d`, no QSM) y valida con PSNR, una reconstrucción QSM de COSMOS y el número de
estabilidad PnP. La corrida larga la lanza el autor en otra máquina. **La fase 2D está cerrada**:
`train.py` (`t8delta-100k-s1`, 31.547 dB en CBSD68 a σ = 25, contra 31.53 publicado) queda porque
`train3d.py` importa su receta, y lo que el 2D dejó vigente está en la bitácora, «Lo que dejó el 2D».

Aquí va sólo lo que no está en otro lado: uso y teoría, en el README; estado, decisiones y qué
sigue, en `docs/bitacora.md`; el porqué de cada constante, en `docs/porque-del-codigo.md` y en los
bloques finales de los `.py` (`grep -n '^# --- '`). La historia de las fases anteriores no se
conserva: lo que de ella sigue valiendo está resumido en esos documentos.

## Por dónde empezar

- **Qué pasa y qué sigue:** `docs/bitacora.md`, que es corta y se lee entera: «Estado», «La
  corrida larga», «Abierto» y las entradas fechadas del final, que van por delante.
- **Qué corre ahora:** `systemctl --user list-units 'tdv-*'` y `nvidia-smi`; la corrida larga, en
  W&B (proyecto `training-tdv3d`) o en su `--out`.
- **Si una idea ya se probó:** «Lo que no sirve», abajo.
- **README:** «Entrenar en 3D», «Qué deja una ejecución y cómo leerla» (el log, qué checkpoint),
  «Reanudar», «Llevar el entrenamiento 3D a otra máquina», «Diferencias entre 2D y 3D» y «Cómo
  funciona».

## Mecánica de la sesión

- **Python de más de ~20 líneas va a un archivo**, no pegado en un comando. Una medición nueva deja
  su script en `scripts/` y su número, con el script y la fecha, en una entrada de la bitácora.
- **Una corrida la lanza el autor, o el agente si el autor lo pide en ese turno**, mirando antes
  la GPU 1 (reglas 11 y 12): en segundo plano, nunca encadenada a otra ni esperada con
  `until`/`sleep`. Con systemd:
  `systemd-run --user --collect --unit tdv-<nombre> -p AllowedCPUs=16-31 -E CUDA_VISIBLE_DEVICES=1
  -p WorkingDirectory=$PWD bash -c ".venv/bin/python -u train3d.py ... > runs/<nombre>.log 2>&1"`.
- **`pgrep`/`pkill`, anclados:** `^[^ ]*python[0-9.]*( +-[^ ]+)* +[^ ]*train(3d)?\.py`
  (`pgrep -f train3d.py` se encuentra a sí mismo). Cuenta procesos, no corridas (`train3d.py` suma
  `--workers`), y un `pkill` las mata a todas; una sola se para con `systemctl --user stop <unidad>`.
- **Con el autor:** el número antes que la interpretación, cada término definido la primera vez,
  suposición y medición dichas como tales, y un puntero (`archivo:línea`) antes que una paráfrasis.

## Reglas

Las 1–10 salieron de errores caros; las 11–13 son condiciones del autor. Otros archivos citan las
reglas por número: no renumerar sin un `grep -rn` sobre `.claude/`, `scripts/` y `docs/`.

1. **No parchear un síntoma.** Lo que el paper no tiene no entra para tapar un fallo (el lr propio
   de los escalares y el piso `t_min` crearon modos de fallo nuevos; el clip no rescató). La única
   pieza ajena al paper, el disyuntor de lotes, es inerte mientras no rechaza: sin rechazos, la
   corrida es bit a bit la de la receta.
2. **No tocar `train.py`, `train3d.py` ni el script de lanzamiento de una corrida en vuelo.**
   `bash` lee un script de a pedazos, y una bandera quitada en vivo mató diez corridas; `train3d.py`
   importa la receta de `train.py`. Si no puede esperar, se para la corrida, y eso lo decide el
   autor. Un worktree sirve para preparar el cambio, no para integrarlo.
3. **No medir con la GPU 1 ocupada:** bajo contención, `cudnn.benchmark` elige otros algoritmos; y
   un tiempo sólo se compara en la misma tarjeta y **a la misma iteración** (el ETA es el promedio
   desde `t0`, con el arranque de `torch.compile` adentro). Con una corrida en vuelo, los tests van
   en CPU y sin los PSNR publicados:
   `CUDA_VISIBLE_DEVICES="" taskset -c 16-31 uv run pytest -k "not published_psnr"`.
4. **Una variable por vez.** `referencia` e `intermedio` difieren en tres cosas (TF32,
   determinismo, `benchmark`); `rapido` e `intermedio`, sólo en la precisión.
5. **La comparación se diseña antes de lanzar, sobre el endpoint que se va a leer.** En 3D no hay
   dispersión medida entre semillas, y cada corrida cuesta días: una corrida contra otra se lee sólo
   rama contra tronco (mismos parches y ruido, `scripts/pares_trazas_3d.py`), y una diferencia entre
   dos corridas sueltas no es una conclusión. Como escala, en 2D la sd del mejor PSNR entre semillas
   iguales era 0.164 dB (lo mínimo detectable con una semilla por brazo, ≈ 0.45 dB). Una tasa de
   estallidos se compara por log-rank sobre los **tiempos** de evento, nunca por mortalidad binaria
   (`scripts/poder_logrank.py`). La pregunta principal y lo que la contestaría se escriben en la
   bitácora **antes** de lanzar.
6. **No re-correr una semilla para entender una muerte.** Un ULP de float32, 284 iteraciones antes,
   borra un estallido: una muerte no es atribuible a un estado, un lote ni un checkpoint; lo que es
   de una configuración es la **tasa**. De una corrida sola sí se dice qué pasó y si el disyuntor la
   contuvo (`REBOBINADO` en el log, columnas 8–9 de la traza).
7. **Una hipótesis descartada vale tanto como una confirmada**, si queda escrito por qué: una línea
   en «Lo que no sirve» y su entrada en la bitácora.
8. **La bitácora se escribe en cuanto corre el test**: configuración, número y veredicto, aunque el
   veredicto sea «nada».
9. **El repositorio es la única fuente.** Nada queda sólo en la memoria de un agente, en `runs/` o
   en una conversación: va al README, a este archivo o a `docs/bitacora.md`.
10. **Una afirmación se verifica antes de publicarse y se retira en todos los sitios** (skill
    `afirmar`): `grep -rin` del término y del número, en el mismo cambio.
11. **Sólo la GPU 1 (la segunda).** La 0 la usa otra persona: no se lanza, mide ni mata nada ahí.
    Todo proceso con CUDA lleva `CUDA_VISIBLE_DEVICES=1` (adentro, torch la ve como `cuda:0`), y
    antes se mira `nvidia-smi`. Vale para esta máquina; la corrida larga va en otra.
12. **Sólo la mitad de la CPU, la segunda: los hilos 16–31** (los 16 núcleos E del i9-14900KF; los
    0–15 son los P). Todo lo pesado va con `taskset -c 16-31` (o `-p AllowedCPUs=16-31` en
    systemd), y lo que abre procesos se limita a 16 (`--workers`, `--procesos`).
13. **Pocos commits, y ningún commit citado.** El autor sube cerca de un commit por semana y aplasta
    los intermedios. Un agente no commitea salvo que se lo pidan; si trabaja en un worktree, entrega
    un solo commit. Nunca se cita un hash, una etiqueta ni un estado pasado del repositorio en el
    código, los tests o la documentación: se escribe el hecho vigente, y lo reemplazado se borra.

## El código

- **`ddr/` es un port de VLOGroup/tdv**: los nombres de atributo son las claves del `state_dict` y
  los oficiales cargan con `strict=True`, así que renombrar rompe todos los checkpoints. Antes de
  tocar el modelo, leer el original. `../tdv` es el port 2D del que sale `ddr/`, con los solvers
  ADMM-QSM (su trabajo vive en ramas de worktree: `git log --all`). `../QSM-TDV`, la implementación
  3D de otra persona, no carga directo un checkpoint de `train3d.py`: lo convierte `a_qsm_tdv.py`,
  inexacto junto al borde (bitácora, «Hacia QSM-TDV»).
- **Los autores publicaron sólo inferencia** (modelo, `denoise.py`, dos checkpoints): «lo que
  hicieron» se verifica contra ese repo o el `config` de los `.pth`, que vale para arquitectura y
  modos pero no para los init (su `T = 0.001` mató tres corridas).
- **`train3d.py` importa la receta de `train.py`** (lr, betas, σ, S, arquitectura, inits,
  disyuntor, modos y `rebobina()`): tocar `train.py` mueve los dos.
- **Los tests son el contrato** (`uv run pytest`): si un cambio los rompe, el cambio está mal. Un
  cambio que no debe tocar la aritmética se valida además con la traza: en 3D, los tests de
  `tests/test_train3d.py` que comparan `traza.npy` bit a bit; en 2D, 120 iteraciones en
  `referencia`, antes y después.
- **Ningún entrenador corta el bucle** (ni `break`, ni aborto por `NaN`, ni early stopping): los
  análisis cuentan filas de la traza. Una corrida terminó si su `estado.json` dice
  `"terminada": true`; las cortadas a mano llevan uno escrito a mano, con `false` y el motivo. La
  única excepción es `--meseta`: agotadas las mitades del lr, corta y lo deja en `estado.json`
  (`corte`, con `iteraciones_hechas` menor que las pedidas), censurada por diseño.
- **Clases, sólo si heredan de `nn.Module` o `Dataset`, o si las tiene el port** (skill `huella`).
- **Sin test que los cubra:** la pérdida mira sólo `x_S`; el ruido se sortea nuevo en cada
  iteración; y no se parte del checkpoint oficial si el resultado se compara con él.
- **Estado que se sume al rebobinado (EMA, otro scheduler): se restaura por copia** y se decide si
  viaja con el respaldo (la ventana sí, el lr no): `opt.load_state_dict` adopta lo que recibe
  (`tests/test_disyuntor.py`).

## Entrenamiento 3D

- **`rapido` por defecto:** `referencia` cuesta 16–21× por vóxel (estimado). En la 5070 Ti,
  ~9 s/it con lote 32 de 48³. De `rapido` se lee la tasa sobre semillas; una corrida contra otra,
  sólo rama contra tronco.
- **La evaluación QSM y la estabilidad no tocan la aritmética** (tests: traza bit a bit igual).
  Dependen de COSMOS, `--qsm-max-iter` y `--estabilidad-lado`, libres al reanudar: cambiarlos a
  mitad de corrida mezcla criterios, y nada lo verifica.
- **`(T/S)·ρ(∇²R)` es un índice, no una garantía.** `*INESTABLE*` usa el umbral 2
  (`ddr/qsm3d.py:84`). Por debajo de 2 un paso también expande, por curvatura negativa; y en 2D el
  índice no predijo la convergencia del ADMM en ningún sentido (bitácora, «Lo que dejó el 2D»).
  Estable no es convergente.
- **A QSM se pasa `best_qsm.pth`, elegido por el NRMSE mínimo del ADMM** en COSMOS. El ADMM de la
  evaluación tiene un techo de 25 iteraciones, sin el corte por TOL y CORTE del autor (no
  calibrados para la red 3D), y para si el NRMSE sube dos iteraciones seguidas, devolviendo el
  mejor χ visto: el final y el mínimo son el mismo número. El ~26 % es una referencia para
  reportar; la corrida se compara contra sí misma.
- **Con `--mascara-cerebro`, el PSNR del recorte entero no sirve:** la red no ve fondo ni cráneo, y
  salta 24.7–28.5 dB entre validaciones. `best.pth` se elige por `val/psnr_cerebro`, y la corrida
  exige al arrancar las máscaras de los volúmenes de validación.
- **El micro-lote es cualquier entero:** el lote se parte en trozos casi iguales y `auto` deja
  menos pasadas (16 en 96 GiB, 11 en 48, 3 en la 5070 Ti). Con trozos iguales la corrida es bit a
  bit la de antes.
- **`--sobreajuste` va sin las cotas del disyuntor** (siguen el rechazo de un gradiente no finito y
  el rebobinado): con datos fijos, un lote rechazado se repetiría para siempre.
- **Fallar al arrancar:** `--seguimiento online` sale con error si falta wandb o la credencial, o
  si W&B no abre; sin GPU visible y sin `--sin-compilar`, `train3d.py` también.
- **Memoria, abierto:** `MARGEN = 0.10` (en `train3d.py`) es un supuesto, y la sonda no corre la
  validación (a 24³ × 8 se midió un pico de 8.16 GiB contra 4.14 predichos; a 48³ y micro-lote 2,
  ~1 % de error).
- **El estado de una corrida** está en su `--out`: `estado.json` (falta mientras corre),
  `config.json` (con el commit del código), `history.json`, `qsm.jsonl` y `traza.npy` (columnas:
  bloque «TRAZA» de `train3d.py`); el log, donde lo mande el comando (el de la corrida larga,
  `runs/3d-largo.log`, fuera de `--out`). W&B es una vista. `scripts/resumen_ventana_3d.py`
  resume varias.
- **La corrida larga va en otra máquina** (RTX PRO 6000 Blackwell Max-Q de 96 GB, sin probar
  allí): procedimiento y comando en el README, «Llevar el entrenamiento 3D a otra máquina»; lo
  fijado y lo abierto, en la bitácora. Como servicio se lanza con `.venv/bin/python`, no con
  `uv run`: el `PATH` de una unidad systemd no incluye `~/.local/bin`. Al vigilarla:
  `train/rechazos`, `train/rebobinados`, `train/iteracion_efectiva`, `train/T` lejos de 0,
  `qsm/fallo` (si fallan todas las evaluaciones QSM, la meseta no actúa y el lr queda en 4e-4
  hasta el techo), `qsm/nrmse_min_pct` (lo que eligen `best_qsm.pth` y la meseta),
  `qsm/iteraciones` y `meseta/*`; `qsm/estabilidad`, como diagnóstico.

## Lo que ya se sabe

Medido en 2D (BSDS400 color, σ = 25) salvo aviso; en 3D vale como supuesto.

- **El gradiente es exacto** (`tests/test_gradient.py`): los estallidos de 10⁸–10¹³ son reales y
  nacen en el backward, con la `loss` normal.
- **`cudnn.benchmark` rompe la reproducibilidad;** `referencia` lo apaga y repite bit a bit.
- **Evento, muerte y corrida perdida son tres cosas**: un evento es ‖g‖ > 10⁴; muere si la `loss`
  o `T` se vuelven `NaN` o `T` llega a 0; está perdida si el PSNR se hunde y no vuelve. El hazard
  de evento crece con la corrida.
- **`‖g‖` se lee contra la mediana móvil**, no en absoluto; el guard relativo es invariante de
  escala, y la cota absoluta en 3D se escala con `P³` (42 622 a 48³). En 3D, con razón 5 los
  escalones de ‖g‖ se volvían rebobinados cada ~90 iteraciones; con `--guard 10`, uno fuera del
  arranque en 3 103 (bitácora, «El disyuntor en 3D»).
- **`δ_Adam` va con el lr vivo** (`train.py:52` usa el inicial): el riesgo de que un pico lleve `T`
  a 0 está en la primera fase. Por eso `T_init` = 8·δ.
- **No hay piso en `T_init`, hay un costo:** a 10⁵, 2·δ da −0.011 dB y 1.99× de rechazos; 1.1·δ,
  −0.022 dB y 4.7×. Ninguno mueve el endpoint fuera del ruido entre semillas.
- **`T` sola es de gauge:** lo invariante es `T·‖∇R‖` y `λ·‖x − z‖`, no `T` ni `w = λ/T`.
- **`rapido` reproduce `referencia` a 10⁵** (una semilla: 31.546 contra 31.547 dB).
- **`λ` es casi plana en denoising** (apagarla, −0.033 dB); en QSM, sin medir.
- **Para un prior PnP el PSNR no elige:** cuatro redes 2D con 0.022 dB de diferencia expanden
  ×1.27 (el oficial) a ×23.9 en el punto QSM, y de ellas sólo el oficial hizo converger el ADMM;
  el ciego expande ×3.13 y converge igual (bitácora, «Lo que dejó el 2D»).
- **El solver 2D de QSM** (`../tdv/wTDV_QSM_torch.py`) usa la red entera, `vn(z·α, z·α)/α`, sin
  `T` ni `λ`; la evaluación 3D de `ddr/qsm3d.py` hace lo mismo, con α fijado para que la primera
  reconstrucción tenga desviación 0.06 en el cerebro (4.86 en COSMOS).
- **La red no es escala-equivariante** (15–28 %): la amplitud de χ que ve la red importa.
- **Recupera el rebobinado, no el rechazo;** cuesta lo que descarta (0.4 % en la réplica 2D).

## Lo que no sirve

Probado y cerrado en las fases anteriores; no se reabre sin un mecanismo nuevo.

- **Clip, a cualquier umbral:** los lotes que llevan a la cuenca mala tienen `‖g‖` ≤ 510, y adentro
  Adam cancela la escala.
- **lr propio de los escalares, piso `t_min` y warmup** (regla 1). **Atar `λ = T`:** Adam no es
  invariante.
- **`T_init` (2·δ–10·δ), `λ_init` o la precisión para estabilizar:** 0.9×–2.5× de hazard, y
  `T_init` no mueve el endpoint.
- **Más iteraciones tras la 50 000 en 2D:** +0.030 dB en 18.8 h.
- **Para el disyuntor:** otro detector sobre `‖g‖`, la cota absoluta como prevención, rechazar sin
  rebobinar, abortar tras N rechazos y tocar la ventana ante un rebobinado repetido (era un alias
  del respaldo de Adam, ya arreglado).
- **Como criterio de muerte o de corte:** `τ·L > 2` (el oficial opera a 4.8), `growth`, `amp₀`,
  `‖Δθ‖`, extrapolar `T` o el hazard; `σ_max` del adjunto separa a posteriori, pero no avisa.
- **Como causa de los estallidos:** el port, la inicialización, `torch.compile`, el checkpointing,
  DDP, TF32, un desborde de `v` en Adam, la estructura residual, la Student-t redescendente.
- **Para elegir un prior PnP:** el PSNR, `w = λ/T`, `T·‖K_N‖`, las eigenfunciones de `R`, subir
  `target_std` y el Rayleigh con signo como `λ_max`.

## Mejoras candidatas

- **Entrenar ciego al ruido**, con σ sorteado por parche (`--sigma-max 55`, ya es bandera en 3D):
  quitaría el reescalado α del ADMM-QSM. En 2D, `ciego2d-35k` bajó el número de estabilidad en QSM
  de 23.2 a 2.14 y el ADMM converge, pero a 33.18 % contra 27.34 % del oficial; su control con
  σ = 25 y el mismo calendario no se corrió. La corrida larga 3D va con σ = 25 fijo.
- **Baratas, con condiciones:** EMA (su estado viaja con el respaldo), `channels_last`, AdamW con
  `T` y `λ` fuera del weight decay (si no, los empuja a su pared; derivado, no medido).
- **Qué NO tocar.** ReLU, BatchNorm y autoatención: README, «El esquema». **LayerNorm:** su
  jacobiano crece como `1/√ε` donde la varianza se anula. Y lo que eligió el artículo: 32
  features, tres bloques y tres escalas; potencial final identidad; antialias binomial, no
  bilineal (`tests/test_conv.py`); sin sesgo, y `K₁` de media cero y norma ≤ 1.

## Lo que no está en un clon

`scripts/`, `.claude/` (skills `tanda`, `afirmar`, `huella`, `papers`), `CLAUDE.md`, `runs/` y las
imágenes de `datasets/` existen sólo en la máquina del autor.

- Un worktree se crea desde `HEAD` (`git worktree add -b <rama> <ruta> HEAD`; `origin/main` puede
  ir atrás) y se le enlazan `runs`, `datasets/bsds500/images` (sin ella, `test_reproduction.py` se
  saltea), `datasets/ldm100k3d/images`, `datasets/ldm100k3d/mascaras` y `datasets/qsm_cosmos` (sin
  ella, la evaluación QSM se apaga).
- `scripts/guardia_tanda.py`, el hook de la regla 2: con cualquier `train.py` o `train3d.py` vivo
  (también la suite, que lanza corridas cortas), bloquea escribir **cualquier** `train.py`,
  `train3d.py` o `runs/*/plan.sh`, también de un worktree o de `../QSM-TDV`.
- `scripts/` también: `poder_logrank.py` (regla 5), `resumen_ventana_3d.py`, `rachas_disyuntor_3d.py`
  y los scripts de la evaluación QSM (`admm_checkpoint_3d.py`, `corre_qsm_admm.py`).
