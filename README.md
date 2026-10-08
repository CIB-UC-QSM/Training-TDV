# Training-TDV — reentrenar Total Deep Variation desde cero

> E. Kobler, A. Effland, K. Kunisch, T. Pock — *Total Deep Variation: A Stable Regularizer for
> Inverse Problems*, [arXiv:2006.08789](https://arxiv.org/abs/2006.08789) (2020). Es la versión
> larga del paper de CVPR 2020 *Total Deep Variation for Linear Inverse Problems*; la receta que se
> replica acá es la de su sección 5.1.

*Total Deep Variation* (TDV) es un regularizador aprendido: una red que no devuelve la imagen
limpia, sino un número que mide cuán poco natural es una imagen. Para quitar el ruido se baja por
esa energía durante diez pasos. Este repositorio entrena TDV **desde inicialización aleatoria** para
denoising gaussiano en 2D. El destino es usarlo como prior en QSM 3D (mapeo cuantitativo de
susceptibilidad, en resonancia magnética), pero primero había que demostrar que el entrenamiento
está bien hecho, y la única prueba convincente era reproducir un número publicado con datos
públicos: 31.53 dB en CBSD68 con ruido de σ = 25.

Los autores publicaron el código de inferencia y dos checkpoints, no el entrenamiento. Acá hay dos
piezas. [`ddr/`](ddr/) es un port de [VLOGroup/tdv](https://github.com/VLOGroup/tdv) a PyTorch
puro, sin la biblioteca CUDA `optox` del original; conserva los nombres de clases, parámetros y
buffers, así que los checkpoints oficiales cargan con `strict=True` y reproducen los cuatro números
del paper. [`train.py`](train.py) es el entrenador, escrito desde la sección 5.1: sigue la receta
publicada y le agrega una sola pieza que el paper no tiene, un **disyuntor de lotes con
rebobinado**, porque sin él el entrenamiento estalla con frecuencia.

**El entrenamiento se replicó.** La corrida `t8delta-100k-s1`, entrenada desde cero con la receta
del paper, llegó a **31.547 dB** en CBSD68 en la iteración 100 000, contra 31.53 publicado. El
checkpoint oficial, evaluado con este mismo código y la misma realización del ruido, da 31.537. El
modelo entrenado está en
[`checkpoints/Entrenados/t8delta-100k-s1.pth`](checkpoints/Entrenados/t8delta-100k-s1.pth).

PSNR en CBSD68 (68 imágenes color). Todos los modelos se entrenaron sólo a σ = 25; las columnas
de σ = 15 y 50 usan el reescalado del paper.

| modelo | iteración | σ = 15 | σ = 25 | σ = 50 |
|---|---|---|---|---|
| publicado en el paper | — | 34.12 | 31.53 | 28.26 |
| checkpoint oficial, con este código | — | 34.122 | 31.537 | 28.264 |
| **`t8delta-100k-s1`** (nuestro) | 100 000 | **34.125** | **31.547** | **28.268** |
| `t8delta-100k-s0` (otra semilla, cortada a mano) | 64 000 | 34.110 | 31.526 | 28.246 |
| `replica-v4` (una corrida anterior, en bf16) | 68 000 | 34.060 | 31.478 | 28.186 |

Fuente: [`docs/presentacion/salida/metricas_sigma.md`](docs/presentacion/salida/metricas_sigma.md),
y los tres decimales en los `metricas_sigma_*.json` de la misma carpeta. Los +0.010 dB sobre el
oficial no son «mejor»: es una corrida contra un checkpoint, y en la iteración 64 000 las dos
semillas de la configuración final estaban en 31.5265 y 31.5257, una centésima *por debajo*. Lo
que la tabla muestra es que el entrenamiento reproduce el del paper, incluso fuera del σ con el que
se entrenó.

---

## Índice

- [Instalación y datos](#instalación-y-datos)
- [Uso](#uso)
  - [Evaluar un checkpoint](#evaluar-un-checkpoint)
  - [Entrenar](#entrenar)
  - [Todas las banderas de train.py](#todas-las-banderas-de-trainpy)
  - [Qué deja una corrida, y cómo leer el log](#qué-deja-una-corrida-y-cómo-leer-el-log)
  - [Tests](#tests)
- [Cómo funciona](#cómo-funciona)
  - El modelo: [el problema](#el-problema) · [una energía en vez de una predicción](#una-energía-en-vez-de-una-predicción) ·
    [la arquitectura](#la-arquitectura) · [la activación Student-t](#la-activación-student-t) ·
    [el esquema desenrollado](#el-esquema-desenrollado) · [T y λ](#los-dos-escalares-aprendidos-t-y-λ)
  - El entrenamiento: [la pérdida y los datos](#la-pérdida-y-los-datos) · [por qué el gradiente explota](#por-qué-el-gradiente-explota) ·
    [Adam y el clip](#por-qué-adam-lo-empeora-y-por-qué-el-clip-no-sirve) · [el disyuntor](#el-disyuntor-de-lotes) ·
    [el rebobinado](#el-rebobinado) · [los modos de ejecución](#los-modos-de-ejecución)
  - En la práctica: [qué se puede cambiar](#qué-se-puede-cambiar-y-qué-no) · [otro dataset](#entrenar-con-otro-dataset) ·
    [hardware](#hardware-y-presupuesto) · [divergencias del paper](#divergencias-deliberadas-respecto-del-paper) ·
    [de punta a punta](#el-entrenamiento-de-punta-a-punta)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Estado y lo que sigue](#estado-y-lo-que-sigue)
- [Licencia y citación](#licencia-y-citación)

---

## Instalación y datos

El proyecto se maneja con [`uv`](https://github.com/astral-sh/uv). Python 3.13 está fijado en
`.python-version` y las versiones exactas en `uv.lock`.

```bash
uv sync --group dev        # crea .venv con torch, numpy, pillow, ... y pytest
```

No instalar nada con `pip` dentro de `.venv`: se desincroniza del lockfile. Los checkpoints
oficiales y los entrenados ya vienen en el repositorio (1.6 MB cada uno).

**BSDS500.** Las imágenes no se versionan. Se bajan del
[Berkeley Segmentation Dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)
(`BSR_bsds500.tgz`); dentro del archivo, la carpeta `BSR/BSDS500/data/images/` es la que va en
`datasets/bsds500/images/`:

```text
datasets/bsds500/
├── images/
│   ├── train/   200 imágenes ─┐
│   ├── test/    200 imágenes ─┴─ las 400 de entrenamiento (BSDS400)
│   └── val/     100 imágenes ─── aquí viven las 68 de CBSD68
├── cbsd68.txt                    los 68 nombres de evaluación (versionado)
└── README.md
```

El split es fácil de equivocar. BSDS500 reorganizó BSDS300: el viejo `test` pasó a ser `val`, y
CBSD68 sale de ahí. Por eso se entrena con `train + test` y `val` queda reservado entero: entrenar
sobre `val` sería fuga de datos y volvería incomparable el 31.53. Detalle en
[`datasets/bsds500/README.md`](datasets/bsds500/README.md).

---

## Uso

### Evaluar un checkpoint

[`evaluate.py`](evaluate.py) mide cuánto limpia un checkpoint: toma imágenes completas, les suma
ruido gaussiano de σ dado, las pasa por los diez pasos del modelo y promedia el PSNR. Sirve con
cualquier checkpoint del formato del repo —los oficiales, los de `checkpoints/Entrenados/` y el
`best.pth` o `last.pth` de una corrida—, y para contrastar dos checkpoints se corre sobre cada uno.
El ruido de cada imagen sale de una semilla fija (1234 más el índice de la imagen), así que dos
checkpoints evaluados sobre el mismo conjunto ven exactamente el mismo ruido y la comparación es
pareada, imagen por imagen.

```bash
uv run python evaluate.py checkpoints/Entrenados/t8delta-100k-s1.pth \
    --file-list datasets/bsds500/cbsd68.txt
```

```text
checkpoint : checkpoints/Entrenados/t8delta-100k-s1.pth
conjunto   : val (68 imágenes)
sigma      : 25.0   S=10
PSNR ruido : 20.17 dB
PSNR salida: 31.55 dB
```

**Ojo con `--file-list`.** Sin ella se evalúa el split entero —las 100 imágenes de `val`—, que no
es CBSD68. Para comparar con el paper hay que pasar siempre `--file-list datasets/bsds500/cbsd68.txt`.

Los cuatro números del paper, con los checkpoints oficiales. Al lado va el valor publicado;
`evaluate.py` imprime dos decimales, así que el de σ = 25 sale 31.54 (mide 31.537):

```bash
F="--file-list datasets/bsds500/cbsd68.txt"
uv run python evaluate.py checkpoints/Oficiales/tdv3-3-25-f32-color.pth $F               # 31.53
uv run python evaluate.py checkpoints/Oficiales/tdv3-3-25-f32-color.pth $F --sigma 15    # 34.12
uv run python evaluate.py checkpoints/Oficiales/tdv3-3-25-f32-color.pth $F --sigma 50    # 28.26
uv run python evaluate.py checkpoints/Oficiales/tdv3-3-25-f32-gray.pth  $F --gray        # 29.37 (BSD68)
```

Un checkpoint entrenado en gris (`--channels 1`) se evalúa con `--gray`, y uno entrenado sobre otro
dataset, con el `--data` y el `--file-list` de ese dataset.

Para contrastar dos checkpoints conviene guardar el reporte por imagen con `--output` y comparar
los dos JSON. Cada uno trae los promedios y una lista `per_image` con el PSNR de cada imagen:

```bash
F="--file-list datasets/bsds500/cbsd68.txt"
uv run python evaluate.py checkpoints/Oficiales/tdv3-3-25-f32-color.pth $F --output oficial.json
uv run python evaluate.py checkpoints/Entrenados/t8delta-100k-s1.pth   $F --output mia.json
uv run python -c "
import json
a, b = (json.load(open(f))['per_image'] for f in ('oficial.json', 'mia.json'))
d = [y['psnr_denoised'] - x['psnr_denoised'] for x, y in zip(a, b)]
print(f'diferencia media {sum(d)/len(d):+.3f} dB, mejora en {sum(v > 0 for v in d)} de {len(d)}')"
```

| bandera | por defecto | qué hace |
|---|---|---|
| `checkpoint` | (obligatorio) | ruta al `.pth`: oficial, entrenado o de una corrida |
| `--data` | `datasets/bsds500` | raíz del dataset; las imágenes se leen de `<data>/images/<split>/` |
| `--split` | `val` | subcarpeta que se evalúa |
| `--file-list` | ninguna | archivo con nombres, uno por línea y sin extensión, que restringe el split |
| `--sigma` | `25.0` | nivel del ruido, en la escala de 0 a 255 |
| `--sigma-ref` | `25.0` | σ con el que se entrenó el checkpoint. Si difiere de `--sigma`, se aplica el reescalado del paper (sec. 5.2): la entrada se multiplica por `σ_ref/σ` y la salida se divide por lo mismo |
| `--gray` | apagada | lee las imágenes en gris (un canal). Va con el checkpoint gris |
| `--device` | `cuda` si hay GPU, si no `cpu` | dónde se corre |
| `--output` | ninguna | escribe el reporte completo en JSON, con el PSNR de cada imagen |

### Entrenar

El comando mínimo:

```bash
uv run python train.py --out runs/prueba
```

Eso corre **10 000 iteraciones** en modo `referencia`, con `T` arrancando en 8·δ_Adam ≈ 0.1012
—qué son `T` y δ_Adam se explica [más abajo](#los-dos-escalares-aprendidos-t-y-λ)—. A unos
1.35 s por iteración son ~3.8 h en una RTX 5070 Ti. **No es la receta completa**: el default de
iteraciones está pensado para experimentar, no para replicar.

Para replicar el 31.547 hay que pedir las 100 000 iteraciones del paper y la semilla de esa
corrida. Este es el comando exacto: copia el `config.json` de la corrida, donde todas las demás
banderas están en su default. El `--t-init` que se pasa es el default redondeado a diez decimales;
en float32, que es como se guarda `T`, son el mismo número.

```bash
uv run python train.py --out runs/replica --iterations 100000 --t-init 0.1011928851 --seed 1
```

Tarda ~38 h (la corrida original, 37.70 h) y usa ~10 GB de VRAM. En modo `referencia`, con la
misma semilla, el mismo `T_init`, las mismas versiones de `uv.lock` y la misma GPU, debería
terminar en el mismo 31.547. Es lo esperado, no lo medido para la corrida entera: lo medido es que
esta configuración, lanzada dos veces en la misma máquina, dio trazas idénticas bit a bit en las
18 000 iteraciones que llegó a correr la primera.

Por qué `--iterations` vale 10 000 por defecto (la constante `ITERACIONES_TANDA`) y no las 100 000
de la receta (`ITERATIONS`): para experimentar alcanzan. A las 10 000 iteraciones la réplica ya
estaba en 31.340 dB, a 0.21 dB de su valor final. Ojo que con 10 000 iteraciones no llega a haber
ningún halving del lr, que empieza en la 25 000.

Otros usos frecuentes:

```bash
# prueba rápida: bf16, unas 3 veces más rápido que referencia, no reproducible
uv run python train.py --out runs/humo --mode rapido --iterations 2000

# dos corridas a la vez, una por tarjeta (escalan 1.99x)
CUDA_VISIBLE_DEVICES=0 uv run python train.py --out runs/a --seed 0 &
CUDA_VISIBLE_DEVICES=1 uv run python train.py --out runs/b --seed 1 &

# sin disyuntor: apagar la razón, la cota absoluta y el rebobinado. Con sólo --guard inf la
# cota sigue activa, y los lotes con NaN o inf se rechazan igual
uv run python train.py --out runs/control --guard inf --guard-abs inf --guard-seguidos 0

# otro dataset, en gris: cortes de T1 con el calendario del lr comprimido
uv run python train.py --out runs/t1 --iterations 50000 --channels 1 \
    --data-root datasets/ldm100k --train-splits train --eval-split val \
    --eval-list datasets/ldm100k/lista_eval.txt \
    --lr-halve-start 25000 --lr-halve-every 5000 --lr-halvings 5
```

Para que una corrida larga sobreviva a una desconexión SSH:

```bash
tmux new -s tdv -d 'uv run python train.py --out runs/replica --iterations 100000 --seed 1'
tmux attach -t tdv        # Ctrl-b d para salir SIN matarla
```

La corrida **no es reanudable, a propósito**: retomarla no restauraría el estado de los generadores
de parches y de ruido, y en `referencia` eso destruiría la reproducibilidad. Si se corta, se relanza
desde cero. Lo ya evaluado no se pierde: `best.pth` se reescribe cada vez que una evaluación
mejora, así que un corte pierde a lo sumo las iteraciones desde la última evaluación.

### Todas las banderas de train.py

Lo que no es bandera es una constante al principio de [`train.py`](train.py), agrupada en receta
del paper, controles aprendidos, disyuntor, dataset y modos de ejecución. Las notas al final del
archivo explican de dónde sale cada valor.

**Básicas**

| bandera | por defecto | qué hace |
|---|---|---|
| `--out` | (obligatoria) | carpeta de salida; se crea si no existe |
| `--mode` | `referencia` | `referencia` (fp32 IEEE, repite bit a bit), `intermedio` (TF32) o `rapido` (bf16). Ver [Los modos de ejecución](#los-modos-de-ejecución) |
| `--iterations` | `10000` | iteraciones de entrenamiento, un lote cada una. La receta del paper son 100 000 |
| `--t-init` | 8·δ_Adam = `0.10119…` | valor inicial de `T`; el default es el de la réplica. Tiene que quedar por encima de 2·δ_Adam = 0.0253 |
| `--sigma` | `25.0` | nivel del ruido de entrenamiento, en la escala de 0 a 255. La evaluación usa el mismo |
| `--seed` | `0` | fija los pesos iniciales, el sorteo de parches y el del ruido |
| `--log-every` | `250` | cada cuántas iteraciones se imprime una línea de log |
| `--eval-every` | `2000` | cada cuántas iteraciones se evalúa y se guarda; también se evalúa en la última |

**Datos**

| bandera | por defecto | qué hace |
|---|---|---|
| `--data-root` | `datasets/bsds500` | raíz con el layout `<raíz>/images/<split>/` |
| `--train-splits` | `train,test` | splits de entrenamiento, separados por coma |
| `--eval-split` | `val` | split de evaluación |
| `--eval-list` | `datasets/bsds500/cbsd68.txt` | nombres a evaluar dentro de ese split. El archivo tiene que existir |
| `--channels` | `3` | canales de entrada; `1` para gris. Cambia la primera convolución (387 970 → 387 394 parámetros) |

**Calendario del lr.** El lr arranca en 4·10⁻⁴ y se divide a la mitad en cada hito. Con estas
tres banderas en su default y `--iterations 100000`, los hitos son 25 000, 50 000, 75 000 y
100 000, que es la receta; con las 10 000 iteraciones del default no hay ninguno.

| bandera | por defecto | qué hace |
|---|---|---|
| `--lr-halve-every` | `25000` | separación entre dos halvings |
| `--lr-halve-start` | `0` | iteración del primer halving; `0` lo pone en `--lr-halve-every` |
| `--lr-halvings` | `0` | cuántos halvings como máximo; `0` no pone tope |

Comprimir el calendario tiene un piso: Adam promedia sus momentos sobre ~1/(1−β₂) = 1 000
iteraciones, así que dos halvings tienen que quedar bastante más separados que eso. Con menos de
~2 500 iteraciones entre hitos el calendario deja de serlo; es un argumento, no una medición.

**Disyuntor de lotes.** Qué hace cada pieza y por qué cada número vale lo que vale, en
[El disyuntor de lotes](#el-disyuntor-de-lotes) y [El rebobinado](#el-rebobinado).

| bandera | por defecto | qué hace |
|---|---|---|
| `--guard` | `5.0` | rechaza el lote si `‖g‖` supera esta razón sobre la mediana móvil. `inf` apaga esta prueba |
| `--guard-abs` | `10000` | rechaza el lote si `‖g‖` supera esta cota absoluta. `inf` la apaga |
| `--guard-ventana` | `100` | largo de la mediana móvil; la razón se arma cuando la ventana se llena |
| `--guard-seguidos` | `25` | rechazos seguidos que disparan el rebobinado. `0` lo apaga |
| `--guard-respaldo` | `250` | iteraciones mínimas entre dos respaldos |
| `--guard-limpias` | `3` | iteraciones sanas seguidas que exige el respaldo antes de refrescarse |
| `--guard-amp` | `inf` (apagada) | rechaza también si la amplificación adjunta `‖p₀‖/‖p_S‖` supera este valor. Cuesta un vjp extra por micro-lote |

Los lotes con `‖g‖` no finito (`NaN`, `inf`) se rechazan siempre, aun con todo apagado.

**Diagnóstico y memoria**

| bandera | por defecto | qué hace |
|---|---|---|
| `--clip` | `inf` | norma máxima del gradiente. `inf` no recorta, y la norma se mide igual para el log y el disyuntor. Un valor finito es una desviación de la receta |
| `--checkpointing` | apagado | gradient checkpointing: recalcula el forward durante el backward. En el banco de `referencia`, de 10.1 a 2.2 GB de VRAM (−78 %) y de 1.00 a 1.31 s/it (+31 %); exacto sólo en `referencia` |

### Qué deja una corrida, y cómo leer el log

Todo va a la carpeta de `--out`:

| archivo | cuándo se escribe | qué tiene |
|---|---|---|
| `config.json` | al arrancar | las banderas con que se lanzó; alcanza para relanzarla igual |
| `history.json` | en cada evaluación | iteración, PSNR, `T` y `λ` |
| `best.pth` | cuando la evaluación mejora | **el entregable**: config, pesos, iteración, PSNR y el estado de Adam |
| `last.pth` | en cada evaluación | lo mismo, de la última evaluación |
| `traza.npy` | en cada evaluación | una fila por iteración: `loss, T, ∂L/∂T, λ, ∂L/∂λ, ‖g‖, centro, amp, rechazado, rebobinado` |
| `estado.json` | sólo al terminar | si terminó, mejor PSNR, PSNR final, rechazos, rebobinados, horas, VRAM pico |

Si `estado.json` no existe, la corrida no terminó. Los `.pth` de una corrida llevan además el
estado de Adam, y `evaluate.py` los lee igual. El detalle de cada columna está en las notas al
final de `train.py`.

La primera línea del log resume la configuración (parámetros, batch, modo, datos, hitos del lr,
`T_init`, disyuntor). Después, cada `--log-every` iteraciones (líneas reales de `t8delta-100k-s1`):

```text
[  2000/100000] loss=  20.890 lr=4.00e-04 T=0.10312 lambda=0.03665 |g|=52.4 max|g|=129 corr=0.952 rech=0 reb=0 (38.7h restantes)
    eval PSNR=31.003 dB (68 imgs)  *mejor*
[ 24773] LOTE RECHAZADO |g|=118589.4 mediana=17.4 amp=nan loss=21.295 (seguidos 2)
[ 24796] REBOBINADO a la it 24755 (1 en lo que va) |g|=3.33e+04
```

| campo | qué es |
|---|---|
| `loss` | `‖x_S − y‖²`, sumada sobre los píxeles del parche y promediada sobre el lote. No hacer nada vale **249.4** en color (83.1 en gris) |
| `lr` | el learning rate vigente |
| `T`, `lambda` | los dos escalares aprendidos. Si `T` toca 0, la corrida se apaga |
| `\|g\|` | norma del gradiente de todos los parámetros en esa iteración, antes de cualquier recorte |
| `max\|g\|` | la máxima desde la línea anterior, lotes rechazados incluidos. Sana anda en 10²–10³; un estallido llega a 10⁸–10¹³, y con el disyuntor puesto puede aparecer en una corrida viva que lo rechazó |
| `corr` | `‖x_S − x₀‖/‖ruido‖`: cuánto corrige la red, en unidades del ruido. ~0.95 es sano; `> 1` es sobre-corregir |
| `rech`, `reb` | lotes rechazados y rebobinados acumulados |

Cada parche tiene 93·93·3 = 25 947 valores, así que la `loss` se traduce a PSNR como
`−10·log₁₀(loss/25 947)`: `loss` 20 ≈ 31.1 dB y 249.4 ≈ 20.17 dB, que es devolver la imagen
ruidosa tal cual (en gris son 93·93 = 8 649 valores, y no hacer nada vale 83.1). Es el PSNR de
un solo lote de parches; el que cuenta es el de la línea `eval`.
Qué rango de cada número es sano está en [`docs/metricas.md`](docs/metricas.md).

### Tests

```bash
uv run pytest                                # todo
uv run pytest tests/test_reproduction.py     # sólo los números del paper
```

Los tests son el contrato del port: si un cambio los rompe, el cambio está mal.

- `test_conv.py` — la identidad adjunta `⟨Kx, y⟩ = ⟨x, Kᵀy⟩` para cada operador lineal (padding
  simétrico, convolución, bajada y subida de escala), y que el antialias sea el binomial.
- `test_gradient.py` — que `R.grad` sea el gradiente exacto de `R.energy` (contra autograd y por
  diferencias finitas), sus segundas derivadas en `x` y en los pesos, la proyección de `K₁` y que
  no haya un solo bias.
- `test_reproduction.py` — que los checkpoints oficiales carguen con `strict=True`, tengan 387 970
  parámetros y reproduzcan los cuatro PSNR publicados a 0.02 dB. Se salta si faltan los
  checkpoints o BSDS500, y usa la GPU si hay.
- `test_recipe.py` — que las constantes de `train.py` sigan siendo las del paper, que el lr se
  divida a la mitad, la regla 2-δ y que `T` y `λ` no estén atados.
- `test_disyuntor.py` — que el disyuntor rechace los picos sin tocar la corrida sana, que el
  rebobinado devuelva pesos y momentos exactos, y que el bucle no corte nunca.

---

## Cómo funciona

Esta parte no supone haber leído el paper. Repite cosas a propósito.

### El problema

Una imagen limpia `y`, con valores en [0, 1], se observa con ruido gaussiano:

```text
z = y + n,      n ~ N(0, σ²),      σ = 25/255
```

Denoising es recuperar `y` a partir de `z`. Se mide con el PSNR, `20·log₁₀(1/RMSE)` en dB: más es
mejor, y como es logarítmico +3 dB es la mitad del error cuadrático. La imagen ruidosa misma vale
20.17 dB; el paper llega a 31.53.

### Una energía en vez de una predicción

La mayoría de las redes de denoising aprenden el mapa «imagen sucia → imagen limpia». TDV hace otra
cosa: aprende una **energía** `R(x)`, un número que es bajo para las imágenes plausibles y alto
para las que no lo son. Limpiar es buscar una imagen que sea plausible y que a la vez se parezca a
lo medido, o sea bajar por

```text
E(x) = (λ/T) · ½‖x − z‖²  +  R(x)
       └── fidelidad ──┘    └ regularizador aprendido ┘
```

Para bajar no hace falta la altura `R(x)` sino la pendiente `∇R(x)`, una imagen del mismo tamaño
que dice hacia dónde empujar cada píxel. En el código son dos funciones distintas,
`R.energy(x)` y `R.grad(x)`, y el bucle sólo usa la segunda. La gracia de esta formulación es que
el término de datos se puede cambiar sin tocar `R`: en QSM pasa a ser `½‖Ax − z‖²`, con `A` la
convolución con el kernel del dipolo, y el mismo prior sigue sirviendo.

### La arquitectura

`R` es una red convolucional que devuelve una densidad por píxel, y la energía es su suma:

```text
ρ(x) = (1/m) · K_N( MB₃( MB₂( MB₁( K₁ x ) ) ) )        m = 32 features
R(x) = Σ_píxeles ρ(x)
```

- `K₁` es una convolución 3×3 de la imagen a 32 canales, con **media cero y norma ≤ 1 por
  filtro**, re-proyectadas después de cada paso del optimizador.
- Cada macro-bloque `MB` es una pequeña U-Net sobre **tres escalas**. En cada escala hay
  micro-bloques residuales `v = u + K_b φ(K_a u)` con convoluciones 3×3; para bajar de escala se
  convoluciona con stride 2 y un antialias binomial `[1,4,6,4,1]/16` incorporado al kernel, y para
  subir se usa la transpuesta de un kernel propio, independiente del de bajada. Son cinco
  micro-bloques por macro-bloque y tres macro-bloques (TDV33). El estado de cada escala persiste de
  un macro-bloque al siguiente.
- Los features **no se duplican** al bajar de escala (`multiplier: 1`), a diferencia de una U-Net
  típica: son 32 en las tres escalas.
- `K_N` es una convolución 1×1 de 32 canales a uno, y el potencial final es la identidad.
- **La media cero de `K₁` hace que `R` no vea el nivel medio.** Una imagen constante da cero en la
  primera capa, así que `R(x + c) = R(x)`: `R` mide cómo varía la imagen, no su brillo, y por eso
  es una *variación*. Además, **ninguna convolución tiene bias**.

En total son 387 970 parámetros en color, exactamente los del paper. Bias-free **no** vuelve a la
red escala-equivariante: eso exigiría una activación 1-homogénea, y la Student-t tiene escala
propia; medido sobre el oficial, el esquema completo se desvía 15–28 % de `VNet(cz) = c·VNet(z)`
([`docs/mediciones/escala-equivariancia.txt`](docs/mediciones/escala-equivariancia.txt)). Por eso
el reescalado entre σ de `evaluate.py` es una aproximación de dominio y no una identidad. Igual
funciona bien: sin él, el oficial a σ = 50 cae de 28.26 a 18.02 dB.

### La activación Student-t

La no linealidad de cada micro-bloque es

```text
φ(a)  = ½ · log(1 + a²)
φ′(a) = a / (1 + a²)              acotada: |φ′(a)| ≤ ½
φ″(a) = (1 − a²) / (1 + a²)²      negativa para |a| > 1
```

Viene de suponer que las respuestas de un filtro sobre imágenes naturales tienen colas pesadas
—muchos valores cerca de cero, pocos bordes fuertes—: salvo un factor ½ y una constante, `φ` es
menos el logaritmo de una densidad t de Student con un grado de libertad. Tiene dos propiedades que
importan acá.

Es **suave**, con todas sus derivadas continuas. Hace falta porque el entrenamiento deriva a
través de `∇R`, que ya es una derivada: el backward necesita las **segundas** derivadas de `R`. Una
ReLU no las tiene.

Y **no es convexa**: para `|a| > 1` la curvatura `φ″` es negativa. Entonces `R` tampoco es
convexa, y su hessiano `∇²R` tiene direcciones de curvatura negativa. En esas direcciones el paso
de gradiente no contrae: separa. Es la raíz de que el gradiente del entrenamiento pueda explotar,
y está en [Por qué el gradiente explota](#por-qué-el-gradiente-explota).

### El esquema desenrollado

Un algoritmo clásico iteraría hasta converger. TDV fija de antemano **S = 10 pasos** y trata esa
secuencia como una red de diez capas que se entrena de punta a punta. Arrancando de `x₀ = z`, cada
paso hace

```text
x̃       = x_s − (T/S) · ∇R(x_s)                 bajar por el regularizador
x_{s+1} = ( x̃ + (λ/S) · z ) / ( 1 + λ/S )        prox del término de datos
```

y la salida es `x_S`. El segundo renglón es el **operador proximal** del término de datos: el punto
que minimiza `½‖x − x̃‖² + (λ/S)·½‖x − z‖²`. En la práctica es un promedio ponderado entre lo que
propone el regularizador y lo que se midió. Todo el esquema es descenso de gradiente proximal sobre
`E(x)`, con paso `T/S`.

Lo que impide que bajar por `R` termine borrando la imagen no es tanto el prox como el número fijo
de pasos: el descenso dura un tiempo `T` y se corta ahí. Por eso, más abajo, apagar el término de
datos entero cuesta apenas 0.033 dB.

`T` está acoplada a `S`: cambiar el número de pasos en inferencia sin reajustar `T` degrada el
resultado.

### Los dos escalares aprendidos: T y λ

Además de los pesos de la red, el entrenamiento aprende dos números:

- **`T`, el tiempo de parada.** El esquema discretiza un flujo de gradiente que corre durante un
  tiempo `T`; cada paso avanza `T/S`. Es el tamaño total del descenso.
- **`λ`, el peso del término de datos.** Cuánto tira el prox hacia la imagen medida.

Que `T` se aprenda es la idea central del paper: el entrenamiento es un problema de **control
óptimo**, donde la imagen evoluciona diez pasos y los pesos y `T` son los controles que se ajustan
para que el estado final se parezca a la imagen limpia. Los dos escalares viven en `[0, 1000]`, y
`project_()` los devuelve a ese intervalo después de cada paso de Adam.

**`T → 0` mata la corrida.** Con `T = 0` el primer renglón del esquema no hace nada, la salida es
la propia imagen ruidosa —20.17 dB— y el gradiente de todos los pesos, que pasa por `T·∇R`, vale
cero: la red deja de aprender. Sólo `T` sigue recibiendo gradiente, y ninguna corrida que tocó esa
pared se recuperó del todo. **`λ → 0` no mata nada.** El prox se vuelve la identidad y el esquema
queda como descenso puro sobre `R`, que sigue aprendiendo; medido sobre el oficial, apagar el
término de datos entero cuesta 0.033 dB. Lo que sí hace daño es un `λ` grande:
`λ = 0.5` cuesta entre 0.6 y 1.8 dB
([`docs/mediciones/lambda-sensibilidad.txt`](docs/mediciones/lambda-sensibilidad.txt)).

**`T` no es un número a alcanzar.** El oficial terminó en `T* = 0.0247` y nuestra réplica en
0.0826, con el mismo PSNR. No es contradicción: escalar `T` por `c` y la última convolución `K_N`
por `1/c` da exactamente el mismo modelo
([`docs/mediciones/degeneracion-T-KN.txt`](docs/mediciones/degeneracion-T-KN.txt)), así que `T`
sola no está determinada. Lo que sí se conserva es el paso efectivo `T·‖∇R‖/‖n‖ ≈ 1.1–1.3`, igual
en el oficial y en las corridas sanas.

**Los valores iniciales no están en el paper.** El `config` guardado dentro del `.pth` oficial dice
`T = 0.001` y `λ = 0.01`, pero es un artefacto de inferencia, y arrancar `T` en 0.001 mató tres
corridas por el mecanismo de [la sección de Adam](#por-qué-adam-lo-empeora-y-por-qué-el-clip-no-sirve).
Acá `T_init` se mide en unidades de δ_Adam (8·δ, el default y el de la réplica) y `λ` arranca en
0.1, como constante independiente: no están atados.

### La pérdida y los datos

En cada iteración se sortean 32 parches de 93×93 de las 400 imágenes de entrenamiento, con
espejados y rotaciones de 90° al azar, y se les suma ruido **nuevo**: nunca se precalcula, así que
la red no ve dos veces el mismo par y el dataset es en la práctica infinito. La pérdida,
`L = ‖x_S − y‖²`, mira sólo el estado final. Se optimiza con Adam (β₁ = 0.9, β₂ = 0.999) y lr
4·10⁻⁴ dividido a la mitad cada 25 000 iteraciones, durante 100 000: la receta textual de la
sección 5.1. El lote se procesa como dos micro-lotes de 16 acumulados antes del paso, que da el
mismo gradiente con la mitad de memoria, porque ninguna capa acopla muestras.

### Por qué el gradiente explota

Para entrenar hay que llevar `∂L/∂x_S` hacia atrás por los diez pasos. Cada paso hacia atrás
multiplica por el transpuesto de su jacobiano,

```text
J_s = ( I − (T/S) · ∇²R(x_s) ) / ( 1 + λ/S )
```

que contiene el hessiano de `R`. Como `R` no es convexa, `∇²R` tiene autovalores negativos, y ahí
`I − (T/S)·∇²R` estira en vez de contraer. Diez estiramientos encadenados pueden amplificar mucho:
en el checkpoint oficial el producto de las normas de los diez jacobianos ya vale 166, aunque el
operador compuesto, en la práctica, amplifica a lo sumo 2.08. El entrenamiento puede construir
estados donde eso se descontrola —2.6·10⁵ medido en una corrida estallada
([`docs/mediciones/adjunta-sigma-max.txt`](docs/mediciones/adjunta-sigma-max.txt))— y entonces la
norma del gradiente salta de ~10³ a 10⁸–10¹³ en una o pocas iteraciones.

Eso **no es un bug**: el gradiente está verificado contra autograd en float64 sobre la arquitectura
real y los diez pasos, con errores relativos de 10⁻¹⁸ a 10⁻¹⁵. Es el gradiente verdadero de este
objetivo. Tampoco se puede predecir: una diferencia de un ULP de float32, 284 iteraciones antes,
basta para que un estallido no ocurra. Lo único que es propiedad de una configuración es la
**tasa** de estallidos, que se mide sobre varias semillas ([`docs/metricas.md`](docs/metricas.md),
sección 0, explica semilla, brazo y hazard sin fórmulas). Sin protección, cerca de la mitad de las corridas o más muere antes de las
4 000 iteraciones, de las 100 000 que pide la receta: 4 de 6 y 5 de 6 en dos grupos de semillas
con `T_init` 0.0253 y 0.1, y 6 de 12 y 5 de 12 en TF32 y en bf16. Una corrida de 10⁵ en
`referencia`, sin disyuntor, tocó 21.7 dB y terminó en −16.1.

### Por qué Adam lo empeora, y por qué el clip no sirve

Adam no da pasos proporcionales al gradiente: lo normaliza por la raíz de su segundo momento. Ante
un pico de tamaño `G`, el numerador y el denominador crecen juntos, y el desplazamiento total que
el pico le imprime a un parámetro, sumado sobre los pasos que siguen, vale

```text
δ_Adam = lr / √(1 − β₂) = 4·10⁻⁴ / √0.001 = 0.01265
```

**sin importar cuán grande sea el pico**: medido, un pico de 10⁴ y uno de 10⁹ mueven a `T` lo
mismo, hasta la sexta decimal
([`docs/mediciones/adam-respuesta-al-pico.txt`](docs/mediciones/adam-respuesta-al-pico.txt)). Es un
límite de régimen estacionario (vale para iteraciones `t ≫ 1/(1−β₂) = 1 000`) y se divide a la
mitad con cada halving del lr.

Para los 388 mil pesos, 0.0127 es poco. Para `T` es mucho si `T` vale del orden de δ_Adam: un solo
pico con el signo equivocado la manda a la pared en cero, donde la red deja de aprender. De ahí la
**regla 2-δ**: `T` tiene que arrancar por lo menos en `2·δ_Adam = 0.0253`. Es un piso, no un valor
recomendado: arrancar ahí es un régimen crónicamente inestable que pide 12–72 veces más
intervención del disyuntor. El default, `8·δ`, es el arranque de la réplica y deja margen de sobra:
en las dos corridas finales `T` nunca bajó de 6·δ. Por encima del piso, `T_init` no mueve el PSNR
—a iteración pareada, llevarlo de 2·δ a 10·δ cambió 0.026 dB a las 2 000 iteraciones y 0.015 dB a
las 10 000, debajo de lo detectable—; lo que cambia es cuánto tiene que intervenir el disyuntor.

El **clip** de norma global no sirve contra esto, y está probado dos veces. Recortar multiplica
todos los gradientes de ese paso por un mismo escalar, y el cociente `m̂/√v̂` de Adam cancela ese
escalar: el desplazamiento por paso no cambia. Por eso `--clip` existe sólo para experimentar y
viene en `inf`.

### El disyuntor de lotes

Si recortar el pico no sirve, la alternativa es **no dar el paso**. Después del backward y antes de
Adam, el disyuntor compara `‖g‖` contra la mediana de las últimas 100 normas aceptadas, y rechaza
el lote si:

- `‖g‖` no es finito, siempre;
- `‖g‖` supera la cota absoluta de 10⁴;
- `‖g‖` supera 5 veces la mediana, una vez que la ventana se llenó.

Un lote rechazado no llama a `opt.step()`: los pesos no se mueven y los momentos `m` y `v` de Adam
no ven el pico. Tampoco entra en la ventana, así que un pico no puede subir su propio umbral. El
calendario del lr sigue avanzando. No se pierde nada, porque los parches y el ruido se sortean de
nuevo en cada iteración: rechazar un lote es como haber sorteado otro.

Cada número salió de una medición: la calibración sobre las trazas de 81 corridas
([`docs/mediciones/disyuntor-calibracion.txt`](docs/mediciones/disyuntor-calibracion.txt)) y los
estallidos que se estudiaron después ([`docs/mediciones/disyuntor-v2.txt`](docs/mediciones/disyuntor-v2.txt)).

- **La razón 5** queda debajo de 7.70–8.97×, la razón de los tres lotes que, en una corrida
  medida, llevaron los pesos a una zona de la que no volvieron; un umbral de 31.62 los dejó pasar.
  Su trabajo es contener la secuela de un estallido, que va de 10 a 180 veces la mediana. Cuesta
  poco: rechaza el 0.1545 % de los lotes sanos (318 de 205 829), con 18 rechazos seguidos como
  máximo. Es una mediana y no un promedio exponencial porque `‖g‖` sube de ~26 a ~828 en las
  primeras doce iteraciones, y un promedio rechazaría ese arranque.
- **La cota absoluta de 10⁴** cubre las primeras 100 iteraciones, cuando la mediana todavía no se
  puede estimar. Cae en el hueco entre las excursiones que se resuelven solas (hasta 6 113) y las
  que necesitan intervención (desde 1.97·10⁵). En 226 000 iteraciones medidas no rechazó ningún
  lote que la razón no hubiera rechazado también
  ([`docs/mediciones/cota-absoluta-226k.txt`](docs/mediciones/cota-absoluta-226k.txt)).

Lo que el disyuntor **no** hace es prevenir. `‖g‖` se mide sobre los pesos de ahora, así que dice
dónde está la corrida, no a dónde la lleva el paso que falta dar: los estallidos medidos entraron
por lotes *aceptados*, a razón 3.58× y 3.67×
([`docs/mediciones/tinit-10k.txt`](docs/mediciones/tinit-10k.txt)), y recién después se vio el
salto. Todo detector sobre `‖g‖` llega un paso tarde. Para eso está el rebobinado.

### El rebobinado

Cuando el disyuntor rechaza **25 lotes seguidos**, la corrida ya está en la zona mala y rechazar no
la saca: congela los pesos ahí. Entonces vuelve al último **respaldo**: los pesos, los momentos de
Adam y la ventana de la mediana, los tres juntos, porque son un mismo estado. Desde ahí sigue con
lotes nuevos —el iterador de datos no retrocede—, así que no repite el camino que la llevó al
estallido.

El respaldo es una copia en RAM de ~4.5 MB. Se refresca cuando pasaron por lo menos 250
iteraciones desde el anterior, y sólo si las últimas **3 fueron sanas seguidas**: dentro de un
estallido aparecen lotes aceptados sueltos, y un respaldo tomado ahí guardaría pesos ya dañados. El
umbral de 25 queda por encima de los 18 rechazos seguidos que alcanza como máximo el tramo sano del
corpus, así que una corrida sana no debería rebobinar. Rebobinar no corta el bucle: la traza sigue
teniendo una fila por iteración.

Un detalle del código, comprobado con un optimizador de juguete y no en una corrida: el lr viaja
dentro del estado de Adam, así que rebobinar a un respaldo tomado antes de un halving deshace ese
halving, y el lr queda al doble del calendario hasta el final.

Es la pieza que recupera la corrida. En `t8delta-100k-s1` el disyuntor rechazó 163 lotes de
100 000 y rebobinó 4 veces (iteraciones 24 796, 27 123, 29 419 y 99 365), con una `‖g‖` máxima de
1.3·10¹⁰, y la corrida terminó en 31.547 dB. Ninguno de los cuatro cruzó un halving: el más
cercano volvió de la 24 796 a la 24 755, las dos antes del hito de la 25 000. La otra semilla, en
sus 66 000 iteraciones, rechazó 18 y no rebobinó nunca. Son dos semillas: muestran que el mecanismo
funciona, no miden cuánto baja la tasa de estallidos.

### Los modos de ejecución

Ninguno cambia la receta; cambian la precisión, la velocidad y si la corrida se puede repetir.

| modo | precisión | mantisa | determinista | `cudnn.benchmark` | repite bit a bit | s/it (banco) | VRAM | para qué |
|---|---|---|---|---|---|---|---|---|
| `rapido` | bf16 | 7 bits | no | sí | no | 0.28 | 5.1 GB | explorar |
| `intermedio` | fp32 + TF32 | 10 bits | no | sí | no | 0.70 | 10.0 GB | testear |
| `referencia` | fp32 IEEE | 23 bits | sí | no | **sí** | 1.00 | 10.1 GB | el número que se cita |

Los segundos por iteración salen de un banco de 400 iteraciones en una RTX 5070 Ti. Las corridas
largas de `referencia` midieron más, 1.35–1.39 s/it, y eso es lo que hay que presupuestar: ~38 h
por 10⁵ iteraciones. La Student-t y sus derivadas van en fp32 en los tres modos.

Lo que hace repetible a `referencia` es apagar `cudnn.benchmark`. Esa opción elige el algoritmo de
convolución cronometrándolo, la elección cambia de un proceso a otro, y dos corridas idénticas se
separan a nivel de ULP en la iteración 1 o 2; como el entrenamiento es caótico, a las 300
iteraciones ya son otras corridas. Además cuDNN trae TF32 encendido de fábrica, así que
`referencia` lo apaga a mano: si no, una corrida «fp32» no sería IEEE.

Cuándo usar cada uno: `referencia` para todo número que se vaya a citar y para cualquier
comparación entre dos corridas; `rapido` para explorar, sabiendo que no repite. bf16 no es más
estable que TF32: sobre 12 semillas por modo, sin disyuntor, murieron 5 en bf16 y 6 en TF32 antes
de las 4 000 iteraciones, y la diferencia en la tasa de muertes (1.31×, p = 0.61) no es
significativa ([`docs/mediciones/precision.txt`](docs/mediciones/precision.txt)). Se elige por
velocidad, no por seguridad.

### Qué se puede cambiar y qué no

**Libre.** `--out`, `--seed`, `--mode`, `--log-every`, `--eval-every` y `--checkpointing` no
cambian la receta; a lo sumo cambian la velocidad o la reproducibilidad.

**Cambia el experimento.** `--iterations` distinto de 100 000, `--t-init`, `--sigma`, el dataset,
el calendario del lr y los parámetros del disyuntor son legítimos para experimentar, pero el número
que salga ya no es comparable con el 31.53. `--t-init` tiene además un piso duro, `2·δ_Adam`.

**La receta del paper** son constantes al principio de `train.py`, a propósito sin bandera: parche
93, lote 32, lr inicial 4·10⁻⁴, β = (0.9, 0.999), S = 10 y TDV33 (32 features, 3 escalas, 3
macro-bloques). `tests/test_recipe.py` falla si alguna se mueve, y también si se mueve
`LR_HALVE_EVERY = 25 000`, que es el default de `--lr-halve-every`.

**Las restricciones matemáticas no se negocian**, porque romper cualquiera vuelve a TDV una CNN
cualquiera y deja sin efecto el análisis de estabilidad del paper: `energy` y `grad` separadas (es
una energía, no un predictor); ninguna convolución con bias; `K₁` con media cero y norma ≤ 1 por
filtro; la activación Student-t, dos veces derivable; el potencial final `ψ(x) = x` (ablación del
paper, Tabla 1); padding simétrico con su transpuesta exacta, que el `reflect` de PyTorch no da;
bajada de escala con el antialias binomial y no con interpolación bilineal; operadores de bajada y
de subida independientes; y el estado multiescala que persiste entre macro-bloques.

**Qué no conviene tocar.** Casi todo el instrumental moderno choca con la tesis del paper:
BatchNorm o LayerNorm introducen offsets, ReLU rompe la suavidad (GELU y SiLU son suaves: cambiar la
Student-t por una de ellas es un experimento, no algo prohibido), la atención con
softmax rompe la interpretación energética, y más features o bloques ya están ablacionados en el
paper y saturan. Dos arreglos que parecen obvios no sirven, y está medido: un lr distinto para `T`
y `λ` (crea un modo de fallo nuevo) y el clip (no rescata). Y no se entrena partiendo del
checkpoint oficial: el objetivo era reproducir el entrenamiento desde cero, y el oficial es contra
lo que se compara.

### Entrenar con otro dataset

El entrenamiento no sabe nada de fotografías: trabaja con arreglos `[C, H, W]` en float32. Hay tres
puntos de extensión, y ninguno toca el bucle:

1. **El formato de archivo.** Los cargadores viven en `ddr.data.IMAGE_LOADERS`, indexados por
   extensión; vienen `.png`, `.jpg`, `.jpeg`, `.npy` y `.npz`. Los PNG/JPG se normalizan a [0, 1];
   los `.npy` y `.npz` se usan tal cual.
2. **El layout.** `<raíz>/images/<split>/*`, con `--data-root`, `--train-splits`, `--eval-split` y
   una lista de evaluación en `--eval-list`. Cada imagen tiene que medir al menos 93×93.
3. **Canales y rango.** `--channels` para gris; las constantes `NOISE_SCALE = 255` y
   `DATA_RANGE = 1.0` fijan en qué escala se interpreta σ.

**El rango importa más de lo que parece.** `T` y `λ` se aprenden *para* una escala de datos, y σ se
lee relativo a ella: con datos en [0, 1], `--sigma 25` es ruido de 25/255. Entrenar en [0, 1] y
aplicar a datos en otras unidades sin normalizar es lo que después obliga a calibrar la escala de
entrada a mano.

El ejemplo armado es `datasets/ldm100k/`: cortes 2D de T1 sintéticas del dataset LDM_100k, un
corte por plano y por sujeto, normalizados a [0, 1] por el percentil 99.5 de los voxels positivos.
Lo arma [`datasets/cache_ldm100k.py`](datasets/cache_ldm100k.py) desde los NIfTI, con los sujetos
de evaluación disjuntos de los de entrenamiento. Son T1, **no QSM**: lo que se mida ahí es el cambio
de dominio de foto a cerebro, no a mapa de susceptibilidad. Detalle en
[`datasets/ldm100k/README.md`](datasets/ldm100k/README.md), y el comando de entrenamiento, en
[Entrenar](#entrenar).

### Hardware y presupuesto

Todo se midió en una máquina con 2× RTX 5070 Ti (16 GB, Blackwell `sm_120`, sin NVLink, límite de
potencia 270 W), un i9-14900KF y 62 GB de RAM, con torch 2.13.0+cu130 (driver 580.178.04 al cierre
de la fase 1).

**Una corrida, una GPU.** El lote de 32 se acumula como 2×16 y cabe sin checkpointing en los tres
modos. Para usar las dos tarjetas se lanzan dos corridas independientes, una por
`CUDA_VISIBLE_DEVICES`, y escalan 1.99× sin tocar nada
([`docs/mediciones/dos-gpu.txt`](docs/mediciones/dos-gpu.txt)); el pipeline de datos no es cuello
de botella porque las 400 imágenes viven en RAM. No hay paralelismo de datos (DDP), a propósito: no
es la causa de los estallidos ni de la irreproducibilidad, y agrega réplicas que se pueden
desincronizar sin aviso.

`torch.compile` se aplica sobre el **método** `R.grad`, no sobre el módulo —compilar el módulo es un
no-op silencioso, porque el bucle nunca llama a su `forward`— y da ~2×. Sin él no entra en memoria.
La réplica de 100 000 iteraciones en `referencia` son ~38 h de una tarjeta y 9.94 GB de VRAM pico.

### Divergencias deliberadas respecto del paper

- **El disyuntor y el rebobinado** no están en el paper. Mientras no rechazan, la corrida es bit a
  bit la de la receta.
- **`T_init` y `λ_init`** tampoco: el paper sólo da el intervalo admisible y el valor aprendido.
  Ver [T y λ](#los-dos-escalares-aprendidos-t-y-λ).
- **`T` y `λ` desacoplados.** La ec. (7) del paper usa el mismo `T/S` para el regularizador y el
  término de datos; el código publicado por los autores los separa en dos escalares aprendidos, y
  este port conserva eso. La ec. (7) se recupera con `λ = T`. Atarlos está medido y descartado:
  `|∂L/∂T|` es 24–99× mayor que `|∂L/∂λ|` y los dos gradientes tienen signo opuesto en el 93 % de
  las iteraciones ([`docs/mediciones/grad-T-vs-lambda.txt`](docs/mediciones/grad-T-vs-lambda.txt)).
- **Dos cambios matemáticamente equivalentes:** el lote de 32 como dos micro-lotes de 16, y el
  halving con `MultiStepLR` en vez de `StepLR`, que con los defaults da el mismo lr en cada
  iteración y sólo habilita comprimir el calendario.
- **`act_prime` como caché local.** En el original, `MicroBlock` lo guarda como estado mutable del
  módulo, inseguro con checkpointing y con llamadas repetidas.
- **Errata del paper.** La sección 5.2 dice `T* = 0.172` para el modelo gris; el checkpoint guarda
  0.0172, que es lo coherente con la Fig. 3. `tests/test_reproduction.py` lo fija.

Hay dos papers de TDV de los mismos autores y no hay que mezclarlos. El de CVPR 2020 es el
anterior y su receta es otra: parches de 96×96, 10⁶ iteraciones sin halving, otra pérdida y otra
activación. El que se replica es el de arXiv. Las notas destiladas de los dos están en
[`docs/papers/INDEX.md`](docs/papers/INDEX.md).

### El entrenamiento, de punta a punta

Una iteración de `train.py`, en orden:

1. Sortea 32 parches de 93×93 de BSDS400, con espejados y rotaciones, en dos micro-lotes de 16.
2. Les suma ruido gaussiano nuevo de σ = 25/255. La entrada es `x₀ = z`.
3. Corre los diez pasos: bajar por `R` con paso `T/10`, y el prox hacia `z` con peso `λ/10`.
4. Calcula `L = ‖x_10 − y‖²` sobre el estado final y hace el backward por los diez pasos, que
   involucra segundas derivadas de `R`.
5. Mide `‖g‖` y el disyuntor decide. Si el lote pasa, Adam da el paso con un único lr para todo:
   pesos, `T` y `λ`. Si no, no hay paso.
6. Avanza el calendario del lr, que se divide a la mitad cada 25 000 iteraciones, haya habido paso
   o no.
7. Re-proyecta las restricciones: media cero y norma ≤ 1 en `K₁`, `T` y `λ` en `[0, 1000]`.
8. Si van 25 rechazos seguidos, rebobina pesos, momentos y ventana al último respaldo. Si la
   iteración fue sana, la tercera seguida y a 250 del último respaldo, toma uno nuevo.

Cada 2 000 iteraciones evalúa en CBSD68 y guarda `last.pth` y, si mejoró, `best.pth`. Después de
100 000 iteraciones, ~38 h en `referencia`, el resultado es un regularizador que llega a 31.547 dB.

---

## Estructura del repositorio

```text
.
├── ddr/                        el modelo: port de VLOGroup/tdv
│   ├── conv.py                   operadores lineales con su adjunta exacta
│   ├── tdv.py                    TotalDeepVariation2D: la energía R y su gradiente
│   ├── model.py                  VNet2D: el esquema desenrollado, T y λ
│   └── data.py                   cargadores, parches, augmentación, ruido y PSNR
├── train.py                    el entrenador; las notas al final explican el código
├── evaluate.py                 PSNR de un checkpoint sobre imágenes completas
├── tests/                      el contrato del port y de la receta
├── checkpoints/
│   ├── Oficiales/                los dos .pth de los autores (color y gris) y su README
│   └── Entrenados/               t8delta-100k-s1, t8delta-100k-s0 y replica-v4, sin el estado de Adam
├── datasets/
│   ├── bsds500/                  imágenes locales; versionados cbsd68.txt y el README
│   ├── ldm100k/                  cortes de T1 locales; versionados lista_eval.txt y el README
│   └── cache_ldm100k.py          arma ldm100k desde los NIfTI de LDM_100k
├── docs/                       bitácora, métricas, mediciones, papers y la presentación
├── AGENTS.md                   contexto y reglas de trabajo del repositorio
├── pyproject.toml, uv.lock, .python-version
└── LICENSE

locales, sin versionar: runs/ (salidas de entrenamiento), scripts/, .claude/ y CLAUDE.md
```

Qué hay en `docs/` y por dónde empezar está en [`docs/README.md`](docs/README.md). Lo principal:
[`docs/bitacora.md`](docs/bitacora.md) es el registro cronológico de todo lo que se probó, lo que
falló y por qué; [`docs/metricas.md`](docs/metricas.md) dice qué significa cada número y en qué
rango es sano; [`docs/mediciones/`](docs/mediciones/) guarda la salida cruda de cada medición
citada en este README; y
[`docs/presentacion/pipeline-explicado.txt`](docs/presentacion/pipeline-explicado.txt) explica el
pipeline desde cero, sin dar nada por sabido.

`ddr/` es un **port**, no una reimplementación: traslada el mismo cálculo, con los operadores de
[`optox`](https://github.com/VLOGroup/optox) (C++/CUDA, a compilar aparte) reescritos en PyTorch
nativo, lo que además deja actuar a `torch.compile`. `train.py`, `evaluate.py`, `ddr/data.py` y los
tests son nuevos.

### tests/ y scripts/ no son lo mismo

**`tests/` es el contrato, y se versiona.** Son binarios —pasan o fallan— y verifican el código: las
adjuntas, el gradiente contra autograd, la reproducción de los números del paper con los
checkpoints oficiales, la receta y el disyuntor. Corren en cada cambio; con un entrenamiento en
vuelo, en CPU y sin los cuatro PSNR del paper, para no cargar su GPU:
`CUDA_VISIBLE_DEVICES="" uv run pytest -k "not published_psnr"`.

**`scripts/` eran las herramientas de la construcción, y quedan locales.** Mediciones, veredictos
de tandas y hooks: producían números que interpreta una persona, y sirvieron mientras se buscaba
el entrenador. Con la fase 1 cerrada no hacen falta para usar el repositorio. Lo que midieron no se
pierde: su salida está congelada en [`docs/mediciones/`](docs/mediciones/) y su lectura, en la
bitácora.

---

## Estado y lo que sigue

**La fase 1 está cerrada.** El entrenador reproduce el número publicado (31.547 contra 31.53 dB) y
el checkpoint queda en `checkpoints/Entrenados/`. El recorrido completo —el mecanismo de muerte por
`T → 0`, el gradiente exacto, la irreproducibilidad por `cudnn.benchmark`, las hipótesis
descartadas y el disyuntor— está en [`docs/bitacora.md`](docs/bitacora.md).

Lo siguiente es la tanda **`t1-ldm100k`**, pre-registrada en la bitácora y sin lanzar: dos
corridas de 50 000 iteraciones sobre cortes de T1, con `T_init` en 8·δ y en 2·δ y el calendario
del lr comprimido. Es el primer cambio de dominio.

Después viene la **fase 2**: llevar el prior a QSM 3D, en el repositorio hermano `QSM-TDV`,
entrenando desde cero con datos QSM. Lo que cambia, más allá de la dimensión: el término de datos
deja de ser la identidad y pasa a ser la convolución con el kernel del dipolo, que se anula en un
cono a 54.7° del campo principal, así que el prior tiene que rellenar un nulo del operador y no
sólo quitar ruido. `T` y `λ` se reaprenden porque dependen del término de datos; las unidades son
ppm y no [0, 1]; la augmentación deja de ser libre, porque el eje del campo no es intercambiable con
los otros dos; y las métricas pasan a ser RMSE, HFEN y SSIM contra COSMOS. La mejora candidata de
más valor para QSM es entrenar a ciegas del nivel de ruido, con σ sorteado en cada lote, que
eliminaría el reescalado entre σ ([`AGENTS.md`](AGENTS.md), «Mejoras candidatas»).

---

## Licencia y citación

MIT. El regularizador de `ddr/` deriva de [VLOGroup/tdv](https://github.com/VLOGroup/tdv), también
MIT; su nota de copyright se conserva en [`LICENSE`](LICENSE).

```bibtex
@article{kobler2020tdv,
  title   = {Total Deep Variation: A Stable Regularizer for Inverse Problems},
  author  = {Kobler, Erich and Effland, Alexander and Kunisch, Karl and Pock, Thomas},
  journal = {arXiv:2006.08789},
  year    = {2020}
}
```
