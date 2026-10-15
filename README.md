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
  - [Entrenar en 3D](#entrenar-en-3d)
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

#### Cuántas iteraciones hacen falta de verdad

Las 100 000 son las del paper, pero **el 99 % del PSNR está mucho antes**, y está medido sobre las
cuatro corridas de 10⁵ que tiene el repo. Primera iteración a partir de la cual el PSNR ya **no
vuelve a bajar** del endpoint menos la tolerancia:

| si tolerás perder… | cortá en | ahorro | `referencia` | `rapido` |
|---|---|---|---|---|
| **0.10 dB** | **it 22 000** | **78 %** | 8.3 h | **2.8 h** |
| 0.05 dB | it 36 000–48 000 | 52–64 % | 13.6 h | 6.1 h |
| 0.20 dB | it 12 000 | 88 % | 4.5 h | 1.5 h |

El reparto por fase de lr explica por qué: la **primera** fase (it 1–25 000, lr 4e-4) gana
**+11.283 dB** de los +11.377 totales, y las otras tres son el **76 % del costo** por **+0.094 dB**
(`docs/mediciones/calendario-lr.txt`, figura en
[`docs/presentacion/salida/img_evolucion_4corridas.png`](docs/presentacion/salida/img_evolucion_4corridas.png)).

Dos avisos antes de recortar:

- **Un corte deja el modelo sin templar.** Parar en la 22 000 lo deja con el lr todavía en 4e-4.
  Para un número citado o para usarlo de prior en un lazo PnP, conviene que haya pasado al menos un
  hito: el control truncado en la it 35 000 está a −0.050 dB y en la 40 000 a −0.042, ya con un
  halving hecho.
- **Comprimir el calendario no es gratis.** Medido en `runs/t1-rapido-30k`: el primer halving paga
  como siempre, pero **el segundo encimado 5 000 iteraciones después no pagó nada** y la corrida se
  congeló. El piso defendible para un tramo es **2 500** iteraciones, que son 2.5 memorias del
  segundo momento de Adam (`1/(1−β₂) = 1 000`).

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
| `--t-init` | 8·δ_Adam = `0.10119…` | valor inicial de `T`; el default es el de la réplica. No hay piso: bajarlo no mueve el PSNR y sube los rechazos (2·δ: 1.99×; 1.1·δ: 4.7×, a 10⁵), y nada verifica lo que se pase. La «regla 2-δ» se retiró el 2026-10-12 (ver [Por qué Adam lo empeora](#por-qué-adam-lo-empeora-y-por-qué-el-clip-no-sirve)) |
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
  divida a la mitad, que los init de `T` y `λ` no arranquen debajo de `δ_Adam` y que no estén
  atados.
- `test_disyuntor.py` — que el disyuntor rechace los picos sin tocar la corrida sana, que el
  rebobinado devuelva pesos y momentos exactos, y que el bucle no corte nunca.

### Entrenar en 3D

Viene de la rama `3D-Training`, integrada en `main` el 2026-10-15. `train3d.py` es `train.py`
llevado a 3D: la receta (lr, betas, σ, S, TDV33, inits, disyuntor, modos) se importa de `train.py`,
y lo que la tercera dimensión obliga a cambiar tiene su procedencia en el bloque final del archivo.
El dataset es [`datasets/ldm100k3d`](datasets/ldm100k3d/README.md): T1 sintética **con cráneo**,
no QSM.

**Lote y memoria para la tarjeta de destino, una RTX PRO 6000 Blackwell Max-Q de 96 GB** (fijados
el 2026-10-09 para 48 GB, llevados a 96 el 2026-10-15):

| | valor | qué es |
|---|---|---|
| lote (`--lote`) | **32** parches de 48³ por paso de Adam | receta: igual en cualquier tarjeta |
| micro-lote (`--micro-lote auto`) | **16** en 96 GiB → 2 acumulaciones por paso; 8 en 48 GiB; 2 en una 5070 Ti | hardware: el gradiente no cambia |
| tope de VRAM (`--vram-gb`) | **96** (GiB, como los reporta torch; se recorta a la tarjeta) | PyTorch no pasa de ahí aunque la tarjeta tenga más |
| pico predicho | **57.1 GiB** con 16; 28.8 con 8; 113.5 con 32, que no entra | extrapolado desde la 5070 Ti: lo mide `--sonda-vram` |
| `--checkpointing` | no hace falta | baja la memoria ~5× por +22–27 % de tiempo |
| antes de lanzar en la tarjeta nueva | `--sonda-vram`, y una corrida corta para medir s/it | el costo en esa tarjeta no está medido |

Por qué 16 y no más: tiene que dividir a 32, y 32 no entra. **Si 16 es más rápido que 8 no está
medido.** En la 5070 Ti, doblar el micro-lote bajaba el costo por parche sólo 2–6 %, porque lo que
da velocidad es saturar el cómputo, no llenar la memoria. La Blackwell tiene más cómputo y puede
saturar más tarde. La sonda lo dice en la tarjeta: si 16 no baja claramente los s por parche
frente a 8, `--micro-lote 8` rinde igual con 28.8 GiB.

```bash
# el dataset, una vez: las listas están versionadas, así que basta con cachear (~20 GB, 1 min).
# Escanear y volver a seleccionar está en datasets/ldm100k3d/README.md, y pisaría las listas.
python datasets/cache_ldm100k3d.py cachea --lista lista_eval.txt --split val
python datasets/cache_ldm100k3d.py cachea --lista lista_train_2000.txt --split train

# una vez en cada tarjeta nueva: el pico medido de cada micro-lote contra el predicho, y los s
# por parche; no entrena
uv run train3d.py --out runs/sonda --sonda-vram

# el tronco del piloto: lr constante, un estado completo cada 10 000
uv run train3d.py --out runs/tronco3d --hitos --eval-every 1000

# lo mismo en una tarjeta compartida de la que tocan 48 GB (decimales): 44.7 GiB
uv run train3d.py --out runs/tronco3d --hitos --eval-every 1000 --vram-gb 44.7

# una rama desde la 10 000 con lr/2 a partir de la 10 001
uv run train3d.py --out runs/rama3d-10k --reanudar runs/tronco3d/estado-010000.pth \
    --iterations 15000 --hitos 10000 --eval-every 1000

# seguir después de un corte, en el mismo --out
uv run train3d.py --out runs/tronco3d --reanudar runs/tronco3d/estado-020000.pth --hitos
```

Lo que cambia respecto de `train.py`, en una línea cada uno:

- **Modo `rapido` por defecto.** En 3D `referencia` cuesta ~21× más
  ([`docs/mediciones/costo-tdv3d.txt`](docs/mediciones/costo-tdv3d.txt)): queda para tests cortos,
  y de una corrida en `rapido` se lee la tasa sobre semillas, nunca una corrida contra otra. La
  excepción es una rama contra su tronco desde el mismo estado, que ven los mismos parches y el
  mismo ruido: ese par sí se compara, y sólo por encima de la dispersión entre evaluaciones vecinas
  del tronco.
- **Parches de 48³ y lote 32** (`--patch`, `--lote`): el mismo número de parches por paso que el
  paper en 2D («mini batch size of 32», sec. 5.1), no el mismo contenido: 32 × 48³ = 3.54 M
  vóxeles, 4.3× los 830 k valores de BSDS color. El lote es receta y no depende de la tarjeta. En
  una 5070 Ti el tronco de 40 000 iteraciones son 89 h con `--micro-lote 4` y 94 h con el 2 de
  `auto` (derivado de 0.250 y 0.265 s por parche); en la tarjeta de destino no está medido.
- **El micro-lote lo elige la memoria** (`--micro-lote auto`, el default): el mayor divisor del
  lote cuyo pico predicho entra en `--vram-gb` con 10 % de margen. Con acumulación el gradiente es
  el mismo para cualquier partición, porque ninguna capa acopla muestras, así que el micro-lote es
  hardware: 16 en 96 GiB (57.1 GiB predichos), 8 en 48 (28.8), 2 en una 5070 Ti. `--vram-gb`
  (default 96) va en
  GiB, como los reporta torch, que da algo menos que la etiqueta: la 5070 Ti de «16 GB» reporta
  15.45, y 44.7 vale sólo para una porción pactada en GB decimales. Se recorta a la memoria de la
  tarjeta, y si queda por debajo PyTorch no pasa de ahí, para no pisar a otro en una tarjeta
  compartida. El modelo de memoria se ajustó en la 5070 Ti y en otra tarjeta es una
  extrapolación: **`--sonda-vram` mide el pico real** de cada micro-lote, lo deja en
  `sonda-vram.json` y sale sin entrenar; si admite más que `auto`, se pasa `--micro-lote`
  explícito. La columna `s/parche` es el tiempo por parche de tres iteraciones, tras una que
  compila y no cuenta, y la sonda dice cuál micro-lote es el más rápido: si uno mayor no baja los
  s por parche, usar más memoria no acelera (en la 5070 Ti, de 48³ × 2 a × 4 bajaba 6 %).
  `--checkpointing` baja la memoria ~5× por +27 % de tiempo: el primer paso del esquema queda
  sin checkpoint, porque la entrada no lleva gradiente; con `--guard-amp`, que se lo da, son ~9×.
- **La RAM no depende de la VRAM**: la ocupa el dataset (`--train-list`; los 2 000 volúmenes son
  ~19 GB). La corrida de humo usó 3.6 GB de VRAM y 26.8 GB de RAM, el pico del servicio según
  systemd ([`docs/mediciones/humo-train3d.txt`](docs/mediciones/humo-train3d.txt)).
- **El lr es una función de la iteración** (`--hitos`): sin la bandera, el calendario del paper;
  sola, lr constante. Se fija en cada iteración, así que un rebobinado no lo toca.
- **Reanudación exacta** (`--guardar-cada`, `--reanudar`): el estado completo viaja en
  `estado-<it>.pth`. Seguir en el mismo `--out` no puede cambiar la receta; una rama (otro `--out`)
  sólo puede cambiar además `--hitos`. Seguir en otra tarjeta con otro micro-lote está permitido y
  avisa: el gradiente es el mismo salvo el orden de suma (con `--sigma-max` cambia además el ruido
  de cada parche, y con `--guard-amp` la amplificación que mira), y desde ahí la corrida no es bit
  a bit.
- **La cota absoluta escala con el parche**: `GUARD_ABS · P³ / (93²·3)`, 4.26·10⁴ a 48³. El arranque
  sano en 3D cruza 1e4 ([`docs/mediciones/arranque-g-3d.txt`](docs/mediciones/arranque-g-3d.txt)).
- **Catorce columnas en la traza**: las diez de `train.py` y, al final, `‖g‖` del primer micro-lote
  (con la 5 estima la escala de ruido del gradiente), el lr, la iteración efectiva y σ.
- **Candidatas apagadas por defecto**: `--sigma-max` (ruido ciego), `--aug-signo`, `--aug-ganancia`
  y `--mascara-cerebro` (abajo).
- **La validación mide más, y se sigue en W&B**: NRMSE, PSNR en el cerebro, la reconstrucción
  QSM de COSMOS con el modelo como denoiser y el número de estabilidad del esquema, con métricas
  e imágenes en Weights & Biases (abajo, «Seguimiento y evaluación QSM»). No tocan la aritmética
  del entrenamiento. El checkpoint para QSM es `best_qsm.pth`, el de menor NRMSE QSM entre los
  estados estables, no el `best.pth` de mejor PSNR.

El porqué de cada cosa, y qué cambia respecto del paper, está en la bitácora (entrada «Fase 2 en
3D: qué cambia respecto del paper»).

#### Cubos 100 % cerebro (`--mascara-cerebro`)

Viene apagada; el autor la eligió para la corrida larga (2026-10-15). LDM_100k tiene cráneo: con el filtro de tejido,
sólo el 4.0 % de los parches de 48³ cae entero en el cerebro, y el parche medio es 60 % cerebro.
Con `--mascara-cerebro` cada parche es un cubo entero dentro del cerebro, a 1 vóxel o más de su
borde, según una máscara por volumen: la intersección de deepbet 1.0.2 y HD-BET 2.0.1 sin TTA. El
precio es la periferia: a 48³ los cubos cubren el 66 % del cerebro y el 18.5 % de la franja
cortical (≤ 6 mm del borde); a 32³, el 80 % y el 38 % (bitácora, «Cubos 100 % cerebro»). La
evaluación no cambia, así que una corrida con la bandera y otra sin ella se miden sobre los mismos
recortes. Es receta: no se cambia al reanudar.

Las máscaras se generan una vez por lista con
[`datasets/mascaras_ldm100k3d.py`](datasets/mascaras_ldm100k3d.py), en tres pasos. Cada
herramienta corre en su propio entorno de Python, no en el del proyecto: cómo armarlos con uv, con
las versiones exactas, está en el docstring del script. `$H` es la carpeta de esos entornos.

```bash
# 1. deepbet: 0.16 s por cerebro en GPU (5070 Ti), 0.48 s en CPU
CUDA_VISIBLE_DEVICES=0 $H/deepbet/bin/python -I datasets/mascaras_ldm100k3d.py deepbet \
    --lista lista_train_2000.txt
# 2. HD-BET sin TTA: 0.51 s por cerebro en GPU, 21.8 s en CPU
CUDA_VISIBLE_DEVICES=0 $H/hdbet/bin/python -I datasets/mascaras_ldm100k3d.py hdbet \
    --lista lista_train_2000.txt --herramientas $H
# 3. en el .venv: intersección, recorte con la caja de la cache y verificación contra ella
.venv/bin/python datasets/mascaras_ldm100k3d.py combina --lista lista_train_2000.txt --split train
```

- **Cuánto tarda:** con GPU, ~0.67 s por cerebro, ~22 min para los 2 000 y ~1.5 h para 8 000; en
  CPU, ~12 h para los 2 000 (derivado de los tiempos medidos el 2026-10-09).
- **Dónde correrlo:** donde haya una GPU libre, en esta máquina sin tanda en vuelo o en la tarjeta
  de destino antes de entrenar. El paso 3 tiene que correr donde está la cache (`images/train`),
  porque verifica cada máscara contra el volumen cacheado.
- **Qué deja:** los pasos 1 y 2, las máscaras en el espacio del NIfTI, comprimidas (~55 KB), en
  `datasets/ldm100k3d/mascaras-nifti/`; un sujeto ya hecho se saltea, así que cortar y relanzar
  sigue donde quedó. El paso 3, `datasets/ldm100k3d/mascaras/train/<sujeto>.npy` (bool, 5.2 MB),
  que es lo que lee `train3d.py`. Sin ellas, `--mascara-cerebro` se niega y da estos comandos.
- **Al arrancar**, `train3d` lee las máscaras una por vez y se queda con las esquinas válidas
  (~0.2 MB por volumen, 0.06 s cada uno), y el log dice cuántos volúmenes tienen cubos y la
  mediana de esquinas: ~55 000 a 48³.

#### Seguimiento y evaluación QSM

`train3d` no tiene épocas: sortea parches. La «época» del seguimiento es la **validación**, cada
`--eval-every` iteraciones (2 000 por defecto) y en la última. En cada una mide el denoising
sobre los 12 recortes de evaluación y, si toca, evalúa el modelo **como denoiser dentro de la
reconstrucción QSM**, que es su destino. Todo va al log, a `--out` y a
[Weights & Biases](https://wandb.ai). W&B es una vista: lo que registra está también en
`traza.npy`, `history.json`, `qsm.jsonl` y `estado.json`.

**Qué se registra en W&B** (el paso es la iteración):

| cuándo | métricas | imagen |
|---|---|---|
| cada `--log-every` | `train/loss`, `train/T`, `train/lambda`, sus derivadas, `train/grad_norm` y `train/grad_norm_max` (el `max\|g\|` del log), la mediana del disyuntor y `train/grad_razon`, `train/lr`, la iteración efectiva, `train/rechazos`, `train/rebobinados`, `train/s_por_it` y, con GPU, `train/vram_pico_gib` | — |
| cada validación | `val/psnr`, `val/psnr_tejido`, `val/psnr_cerebro`, `val/nrmse_pct`, `val/nrmse_pct_tejido` | `val/denoising`: limpia, ruidosa, salida y 5·\|error\| del primer volumen |
| cada evaluación QSM | `qsm/nrmse_final_pct`, `qsm/nrmse_min_pct`, `qsm/iter_min`, `qsm/iteraciones`, `qsm/alpha`, `qsm/segundos`, `qsm/fallo` | `qsm/reconstruccion`: GT, reconstrucción y error en ±0.1 ppm |
| con la cadencia de QSM, haya COSMOS o no | `val/estabilidad` y `val/lambda_max`; con COSMOS, también `qsm/estabilidad` y `qsm/lambda_max` | — |
| al final | el resumen: lo de `estado.json`, con el mejor PSNR y el último y el menor NRMSE QSM con su iteración | — |

- **NRMSE** = 100·‖x̂ − x‖/‖x‖, en %. En denoising, por volumen y promediado como el PSNR. En QSM es
  la métrica del autor, contra `chi_cosmos` dentro de su máscara.
- **`val/psnr_cerebro`** se mide sólo si están las máscaras de los 12 de evaluación en
  `datasets/ldm100k3d/mascaras/val/`. Se generan como las de entrenamiento, con
  `--lista lista_eval.txt --split val`. Mide la región que ve el entrenamiento con
  `--mascara-cerebro`; el recorte entero tiene cráneo y aire. Agregarla no cambia los PSNR de
  antes.

**W&B**, con `--seguimiento` (default `auto`) y `--wandb-proyecto` (default `training-tdv3d`):

- `auto`: sube en vivo si hay credencial y, si no, guarda offline. Para tener credencial, una vez
  por máquina: `wandb login`, o `WANDB_API_KEY` en el entorno.
- `offline`: guarda en `<out>/wandb/` sin red. Se sube después con
  `wandb sync <out>/wandb/offline-run-*`; el log de arranque da el comando exacto.
- `no`: sin W&B. Ni siquiera importa wandb.

El seguimiento nunca corta el entrenamiento. Si wandb falla, imprime el error una vez y la
corrida sigue sin él. Reanudar en el mismo `--out` sigue la misma corrida de W&B sólo si no se
registró nada después del estado: al extender una terminada, o si el corte cayó justo en un
estado. Tras un corte común, W&B ya tiene pasos posteriores al último estado, así que se abre
otra corrida con el mismo nombre, y el id de la vieja queda en `descartadas` de
`<out>/seguimiento.json`. Una rama es otra corrida, en el grupo del tronco.

**La evaluación QSM** es el protocolo de `../tdv/wTDV_QSM_torch_3d.py` en
[`ddr/qsm3d.py`](ddr/qsm3d.py): ADMM plug-and-play con la red 3D sobre los 160³ de COSMOS, con
μ = 0.008, μ₂ = 1.0, target_std = 0.06 y hasta 100 iteraciones con su criterio de parada.
Reproduce el 26.89 % de NRMSE del autor con su red 2D.

- **Los datos** van en `datasets/qsm_cosmos/`: `params.mat` y `chi_cosmos.mat` de
  `../tdv/data_mat` (40 y 7.7 MB). En esta máquina son enlaces simbólicos, que no viajan. **En
  otra máquina hay que copiar los dos archivos**:

  ```bash
  mkdir -p datasets/qsm_cosmos
  cp <tdv>/data_mat/params.mat <tdv>/data_mat/chi_cosmos.mat datasets/qsm_cosmos/
  ```

  Si falta alguno, el log de arranque avisa cuál y de dónde copiarlo, la evaluación QSM queda
  apagada y el entrenamiento sigue. `--qsm-datos <carpeta>` usa otra carpeta, y
  `--qsm-datos ''` la apaga sin aviso.
- **Cuándo:** `--eval-qsm-cada 0` (default) la corre en cada validación. Con `N > 0`, sólo en
  las validaciones de iteración múltiplo de N y en la última. `--qsm-max-iter` cambia el tope de
  iteraciones del ADMM.
- **Cuánto cuesta**, medido en la 5070 Ti
  ([`docs/mediciones/qsm-eval-gpu.txt`](docs/mediciones/qsm-eval-gpu.txt)):

  | modo | s por iteración de ADMM | pico de VRAM |
  |---|---|---|
  | `rapido` | 3.40 (44 la primera, que compila) | 5.19 GiB |
  | fp32 sin compilar | 25.95 | 6.92 GiB |

  Una evaluación típica tiene ~27 iteraciones: ~1.5 min en `rapido`. Eso es el 0.6–2 % de un
  intervalo de 2 000 iteraciones (derivado). Una red que no converge corre las 100 iteraciones,
  ~5.7 min. En la tarjeta de destino no está medido.
- **No puede matar el entrenamiento.** Un error, también un OOM, se imprime, queda como
  `qsm/fallo` = 1 y en `qsm.jsonl`, y la corrida sigue.
- **No toca el entrenamiento:** no usa el generador del ruido ni el de los parches. Con y sin
  ella, la traza sale bit a bit igual (test en `referencia` y CPU).

`estado.json` suma el resumen QSM (`qsm_nrmse_ultimo_pct`, `qsm_nrmse_mejor_pct` y su
iteración, `qsm_fallos`). `qsm_nrmse_mejor_pct` es el menor de todas las evaluaciones, estables o
no; el del checkpoint para QSM está en `best_qsm` (abajo).

**El número de estabilidad**, `(T/S)·λ_max(∇²R)`, dice si el esquema desenrollado amplifica en un
punto. Cada paso es descenso de gradiente explícito sobre `T·R`, y un paso explícito es estable
mientras `(T/S)·λ_max < 2`, con `λ_max` el autovalor mayor de la Hessiana del regularizador en ese
punto. Por encima de 2 el paso amplifica, y los S = 10 pasos componen la amplificación. Como en
`main`, `λ_max` se mide como `‖Hv‖` de la iteración de potencia: el módulo del autovalor dominante,
que nunca es menor que el mayor positivo. Si el número pasa de 2, el paso amplifica cualquiera sea
el signo del dominante: con uno negativo, `|1 − (T/S)·μ| ≥ 3`.

- **Por qué importa:** el PSNR de la validación no lo ve. En `main`, el 2026-10-12, los
  checkpoints 2D propios rendían peor que el oficial como prior del ADMM de QSM y no convergían.
  A σ = 25, donde entrenan, los cuatro daban 1.38–2.01 y sus PSNR coincidían en 0.027 dB. A la
  amplitud de uso de QSM, el oficial daba 0.99 y los propios 2.8–23.2, en el mismo orden en que se
  degradaban (bitácora de `main`, «POR QUÉ los checkpoints propios rinden peor como prior PnP de
  QSM»).
- **Dónde se mide:** en dos puntos, en un cubo de `--estabilidad-lado`³ (96³ por defecto). `val`
  es el centro del primer recorte de evaluación, con el ruido de la validación: es el punto de
  entrenamiento. `qsm`, sólo con COSMOS, es la entrada del primer denoiser del ADMM, la
  estimación inicial llevada a std 0.06: es el punto de uso. El número depende del lado, así que
  se compara al mismo lado.
- **Cuándo y dónde queda:** con la cadencia de la evaluación QSM (`--eval-qsm-cada`), haya COSMOS
  o no. Va al log, a `history.json` (`val`) y a `qsm.jsonl` (`qsm`), con las claves
  `estabilidad`, `lambda_max` y `estabilidad_cambio`, y a W&B. Un `*INESTABLE*` en el log marca un
  número de 2 o más.
- **Cómo leerlo:** `estabilidad_cambio` es el cambio relativo de `λ_max` en la última de las 40
  iteraciones de la potencia. Si es grande, las iteraciones no alcanzaron, y el número queda por
  debajo del verdadero; uno chico no lo garantiza. En una red de juguete, el error fue de 1.7 a
  5.3 veces el cambio, y 41 veces con los dos mayores módulos a 0.98 uno del otro (0.9 % contra
  2.2e-4: [`docs/mediciones/convergencia-estabilidad.txt`](docs/mediciones/convergencia-estabilidad.txt)).
  El log da al lado el cociente de Rayleigh `vᵀHv`, que da el signo: ≈ `+λ_max` si el dominante
  es positivo y ≈ `−λ_max` si es negativo. No es el número porque, con dos dominantes de signo
  opuesto y módulo parecido, no converge a ninguno de los dos.
- **Cuánto cuesta:** en GPU a 96³, 19–36 s por punto y 39–73 s por validación, derivado de lo
  medido en CPU ([`docs/mediciones/costo-estabilidad.txt`](docs/mediciones/costo-estabilidad.txt)).
  Pide ~9 GB de memoria en float32, también derivado. Si no entra, el OOM queda como
  `estabilidad_fallo` y el entrenamiento sigue; `--estabilidad-lado 64` la baja ~3.4×.
  `--estabilidad-lado 0` apaga la medición. Con ella y sin ella, la traza sale bit a bit igual
  (test en `referencia` y CPU).

**`best_qsm.pth` es el checkpoint que se pasa a QSM**, no `best.pth`. En `main`, el 2026-10-12, el
PSNR de denoising no servía para elegir un prior del ADMM: el criterio es el NRMSE QSM entre los
estados estables. `train3d` lo escribe tras una validación en la que la evaluación QSM terminó sin
fallo y la estabilidad en COSMOS se midió sin fallo y dio menos de 2, si su NRMSE final es el menor
hasta ahora. El log lo dice con una línea `best_qsm.pth: NRMSE QSM …`.

- **Qué lleva:** lo de `best.pth` salvo el estado de Adam (`config`, `model`, `iteration`, `psnr`),
  más `nrmse_final_pct`, `nrmse_min_pct`, `iteraciones`, `motivo` y `estabilidad` de esa
  validación. Se carga como `best.pth`. Para seguir entrenando desde ahí está el `estado-<it>.pth`.
- **Cuándo no hay:** sin COSMOS o con `--estabilidad-lado 0` no se busca, y el log de arranque lo
  dice. Tampoco hay si ningún estado estable tuvo evaluación QSM. `estado.json` lo resume en
  `best_qsm` (`iteration`, `nrmse_final_pct` y `estabilidad`), o `null`.
- **Al reanudar** viaja en el estado y se trata como `best.pth`: uno de otro linaje se aparta como
  `best_qsm-descartada-<it>.pth` y se repone el del estado. Una rama hereda el del tronco.
- **Estable no es usable: mirá su `nrmse_final_pct` antes de pasarlo a QSM.** La estabilidad se
  mide en un solo punto, la entrada del primer denoiser del ADMM. En `v12-control` dio 1.54 en la
  iteración 200 y el ADMM igual divergió, a 144 %; ese estado quedó como `best_qsm.pth` porque era
  el único estable (bitácora, RESULTADO de `ventana-3d-12h`). Qué condición agregar está abierto.
- **Se compara a criterio fijo:** el NRMSE y el número de estabilidad dependen de COSMOS,
  `--qsm-max-iter` y `--estabilidad-lado`. Son libres al reanudar, pero cambiarlos a mitad de
  corrida mezcla criterios en la misma elección, y nada lo verifica.

El porqué de cada decisión está en el bloque final de `train3d.py`, entradas SEGUIMIENTO,
EVALUACIÓN QSM y ESTABILIDAD, y el método, en el de `ddr/qsm3d.py`.

#### Llevar el entrenamiento 3D a otra máquina

Nada de esto está probado en la tarjeta de destino: los costos son de la RTX 5070 Ti (16 GB),
medidos el 2026-10-13 (bitácora, `ventana-3d-12h`).

**1. Qué copiar.** El repositorio en `main` (o en la rama `3D-Training`, que se integró en `main` el
2026-10-15), y aparte los datos, que no se versionan. Copiá archivos, no enlaces: un enlace a otra
carpeta de esta máquina no viaja.

| carpeta | qué | tamaño |
|---|---|---|
| `datasets/ldm100k3d/images/train/` | los 2 000 volúmenes cacheados de `lista_train_2000.txt` (float16) | 20 GB |
| `datasets/ldm100k3d/images/val/` | los 12 de evaluación | 119 MB |
| `datasets/ldm100k3d/mascaras/val/` | máscaras de cerebro de los 12: dan `val/psnr_cerebro` | 60 MB |
| `datasets/ldm100k3d/mascaras/train/` | las de los 2 000 (`.npy` bool), sólo para `--mascara-cerebro` | 9.8 GB |
| `datasets/qsm_cosmos/` | `params.mat` y `chi_cosmos.mat` de `../tdv/data_mat`: la evaluación QSM | 48 MB |

Las listas (`datasets/ldm100k3d/lista_*.txt`) están versionadas. Con los volúmenes cacheados no
hacen falta ni el LDM_100k crudo ni las herramientas de máscara. Las máscaras de train se
generaron el 2026-10-13 (deepbet ∩ HD-BET sin TTA, 2 000 de 2 000 verificadas contra su volumen;
mediana 1 387 ml de cerebro, 27 % del recorte).

**2. Entorno.** `uv sync` (torch 2.13 con CUDA 13, según `uv.lock`; si el driver pide otra CUDA,
el índice de torch que le corresponda) y `uv run python -c "import torch;
print(torch.cuda.get_device_name())"`. Para W&B en línea, `uv run wandb login` o `WANDB_API_KEY`:
sin credencial, `--seguimiento auto` queda offline en `<out>/wandb/` y se sube después con
`wandb sync`. La RAM la ocupa el dataset: ~27 GB con los 2 000.

**3. Antes de lanzar, en orden:**

1. `uv run pytest -k "not published_psnr"`: todo verde.
2. `uv run train3d.py --out runs/sonda --sonda-vram`: el micro-lote que entra y los s por parche.
   En 96 GB, `auto` elige 16 (~57 GiB). Si su s/parche no baja claramente frente al 8, sumá
   `--micro-lote 8` al lanzar (~29 GiB).
3. `uv run train3d.py --out runs/prueba --iterations 40 --eval-every 20`: de punta a punta, con dos
   validaciones completas. En la 5070 Ti tardó 21 min (2026-10-13, `runs/v12-humo`).

**4. Cuánto cuesta.** En la 5070 Ti, con lote 32 de 48³ y micro-lote 2: 0.280–0.291 s por parche,
9.0–9.3 s/it, según la sonda. Cada validación suma tres cosas:
- el PSNR de los 12, en segundos;
- la estabilidad a 96³, ~31–40 s por punto, dos puntos;
- el ADMM de COSMOS, 3.4–3.8 s por iteración. Corta en ~27 iteraciones si la red es buena (~1.5 min) y
  llega a 100 si no (~6 min, como en las primeras validaciones).

Las iteraciones para un presupuesto de horas son N ≈ horas · 3 600 / (s/it), menos el costo de
las validaciones. N y el calendario (`--hitos`) los decide el autor con el s/it de la tarjeta
nueva.

**5. Lanzar** en segundo plano, con una unidad o `tmux`, nunca encadenado:

```bash
systemd-run --user --unit tdv-3d-largo -p WorkingDirectory=$PWD bash -c \
    'uv run train3d.py --out runs/3d-largo --iterations N --hitos <hitos> --mascara-cerebro \
     --eval-every 2000 --guardar-cada 5000 > runs/3d-largo.log 2>&1'
```

Lo que queda fijo para la corrida larga (bitácora, 2026-10-15):
- **`T_init = 8·δ_Adam`**: es el default, no se pasa.
- **Los 2 000 cerebros de `lista_train_2000.txt`**: el default de `--train-list`.
- **`--mascara-cerebro`**: no cuesta tiempo por iteración, y al arrancar suma ~2 min de leer
  máscaras. Su precio es la periferia: los cubos cubren el 66 % del cerebro y el 18.5 % de la
  franja cortical.
- **`--aug-ganancia` apagada** (el default).

Si se corta, `--reanudar runs/3d-largo/estado-<it>.pth --out runs/3d-largo` sigue exacto.

**6. Qué se pasa a QSM-TDV: `best_qsm.pth`**, no `best.pth`: el estado de menor NRMSE QSM entre los
que tienen estabilidad < 2 (arriba). Si `estado.json` dice `"best_qsm": null`, ningún estado fue
estable y no hay checkpoint para QSM. Antes de usarlo, mirá su `nrmse_final_pct`: estable no
garantiza que el ADMM converja. Durante la corrida conviene mirar:
- `train/rechazos` y `train/rebobinados`: el disyuntor. En el humo y en `v12-control` hubo eventos en las
  primeras 50 iteraciones, contenidos con rebobinados.
- `train/T` lejos de 0.
- `val/psnr` y `qsm/nrmse_final_pct`.
- `qsm/iteraciones` < 100: el ADMM corta.
- `qsm/estabilidad` < 2.

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

#### El paso es explícito, y eso le pone una condición de estabilidad

El primer renglón es descenso de gradiente **explícito**, y un paso explícito sólo baja mientras

```text
(T/S) · λ_max(∇²R) < 2
```

Por encima de ese umbral el paso amplifica en vez de bajar, y con `S = 10` pasos encadenados la
amplificación se compone diez veces.

El entrenamiento lo mantiene bajo control **a la amplitud a la que entrena**, porque si lo cruzara
entrenaría mal: los cuatro checkpoints de 10⁵ iteraciones caen en **1.38–2.01** a σ = 25, pegados
al umbral y por debajo. Lo que pasa a **otra** amplitud no lo controla nada. Medido a `std = 0.06`,
que es donde corre la red el solver de QSM (`../tdv/wTDV_QSM_torch.py`), el checkpoint oficial
**baja** a 0.99 y los tres nuestros **suben** a 2.79, 5.76 y 23.2 — con los cuatro coincidiendo en
**0.027 dB** de PSNR a σ = 25.

Dentro del ADMM eso se ve entero **en el punto de operación**: con `target_std = 0.06` el oficial
—el único estable ahí— converge en **33** iteraciones con 27.34 % de RMSE, y ninguno de los tres
nuestros converge en 100 (28.36–28.50 %). Cuatro de cuatro.

Fuera de ese punto la correspondencia **no** se sostiene, y conviene decirlo: el oficial a
`target_std = 0.12` es estable (0.95) y aun así no converge en 100, y el de 1.1·δ a 0.12 es
inestable y converge en 30. Mover `target_std` cambia a la vez la estabilidad **y** la fuerza del
prior, y la convergencia del ADMM depende además de su propio acoplamiento (`μ`). O sea: el número
explica el comportamiento **donde se lo usa**, no es una teoría completa del ADMM.

La salida no está en el solver sino en el entrenamiento, y hay una demostración a mano:
`../tdv/checkpoints/checkpoint_qsm_rapido.pth`, reentrenado 30 000 iteraciones sobre QSM, da
**1.14** en ese mismo punto — estable. Y no por tener la `T` más chica: la suya es 0.0956, casi la
del de 8·δ (0.0826). Lo que baja es `λ_max(∇²R)`, 119 contra 2 809. Reentrenar en el dominio y a la
amplitud de uso cambia el regularizador, no la coordenada.

**Si el checkpoint va a usarse como prior Plug-and-Play, la PSNR de denoising no alcanza para
elegirlo.** El número de arriba es barato, sólo necesita el checkpoint, y se mide con
`scripts/diag_qsm_estabilidad.py`. Bitácora del 2026-10-12.

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
pico con el signo equivocado la manda a la pared en cero, donde la red deja de aprender. `T_INIT =
0.001` mató tres corridas por esa vía.

> **Acá hubo una «regla 2-δ» y se retiró el 2026-10-12.** Decía que `T` tenía que arrancar por lo
> menos en `2·δ_Adam = 0.0253`. `δ_Adam` es real y está derivado y medido; **el factor 2 era una
> elección**, y nunca predijo nada: `tinit/B-larga` arrancó **en** 2·δ y murió con `T = 0`,
> `t1p1delta-rapido-100k` arrancó en **1.1·δ** y terminó las 100 000, y el mismo razonamiento ya
> estaba refutado para `λ` (hacía falta 2.2·δ y 10 de 12 seguían tocando el clamp). Lo que queda es
> un **costo continuo**, medido, y el default de 8·δ, que es el de la réplica.

Lo que sí está medido, a 10⁵ iteraciones y misma semilla
([`docs/mediciones/modo-y-tinit-100k-s1.txt`](docs/mediciones/modo-y-tinit-100k-s1.txt),
[`t1p1delta-100k.txt`](docs/mediciones/t1p1delta-100k.txt)):

| `T_init` | PSNR @100 000 | contra el control | rechazos | rebobinados | % del costo en la 1ª fase de lr |
|---|---|---|---|---|---|
| `8·δ` (default, el de la réplica) | 31.547 | — | 163 | 4 | 2 % |
| `2·δ` | 31.536 | **−0.011 dB** | 325 (**1.99×**) | 5 | 96 % |
| `1.1·δ` | 31.525 | **−0.022 dB** | 378 (**4.7×** en `rapido`) | 8 | 91 % |

O sea: **`T_init` no mueve el endpoint y sí mueve la exposición**, de forma continua y sin umbral a
la vista. Y el riesgo tiene fecha de vencimiento, porque **`δ_Adam` no es constante: se parte a la
mitad en cada hito del lr**. `train.py` lo calcula una vez desde el `LR` inicial, pero el
desplazamiento real va con el lr vivo, así que la reserva `T/δ_vivo` se duplica sola en cada hito
—la corrida de 1.1·δ fue 1.65 → 3.84 → 8.46 → 17.80 → 35.66— y por eso >90 % del costo de las dos
corridas de `T_init` bajo cae antes del primer hito.

`T` tampoco tiene destino: las cuatro corridas terminan ordenadas por su `T_init` (1.1 → 2.23·δ,
2 → 2.76·δ, 8 → 6.53·δ), o sea que `T*` **no** es un atractor. Es lo que pide el gauge
`(T, K_N) → (cT, K_N/c)`, que deja el modelo idéntico: **`T` sola no es observable**, sólo el
producto `T·∇R`.

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

Dos detalles del código, corregidos el 2026-10-08 en la rama `3D-Training` y el 10-09 en `main`, con
el mismo código. `opt.load_state_dict` no copia los tensores que recibe, los adopta: sin una copia,
los pasos siguientes mutaban el respaldo y el segundo rebobinado al mismo respaldo devolvía los
pesos pero no Adam, que es lo que trabó `t1-guard-2delta-5k`
(`docs/mediciones/alias-respaldo-adam.txt`). Y el lr viaja dentro del estado de Adam, así que
rebobinar a un respaldo tomado antes de un halving lo deshacía. Ahora `rebobina()` restaura una
copia y conserva el lr vigente.

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
comparación entre dos corridas; `rapido` para explorar, sabiendo que no repite. Para **entrenar**
sirve igual, y está medido a iteración pareada: a 10⁵ iteraciones con la misma semilla `rapido` da
**31.546 dB** contra **31.547** de `referencia`, con 49 de 50 puntos pareados dentro de 0.05 dB y
**2.98×** de velocidad ([`docs/mediciones/rapido-vs-referencia-100k.txt`](docs/mediciones/rapido-vs-referencia-100k.txt)).
Lo que no da es repetibilidad bit a bit, ni el mismo `λ`: queda **4.7× más baja**. Medido de forma
invariante de gauge, eso es un término de dato **4.8× más débil** (`ρ = λ‖x−z‖ / T‖∇R‖` da
0.031–0.039 en `rapido` contra 0.15–0.18 en `referencia`), con el mismo regularizador en los dos
(coseno 0.98) y los denoisers coincidiendo a 43–46 dB
([`docs/mediciones/gauge-4-corridas.txt`](docs/mediciones/gauge-4-corridas.txt)). bf16 no es más
estable que TF32: sobre 12 semillas por modo, sin disyuntor, murieron 5 en bf16 y 6 en TF32 antes
de las 4 000 iteraciones, y la diferencia en la tasa de muertes (1.31×, p = 0.61) no es
significativa ([`docs/mediciones/precision.txt`](docs/mediciones/precision.txt)). Se elige por
velocidad, no por seguridad.

### Qué se puede cambiar y qué no

**Libre.** `--out`, `--seed`, `--mode`, `--log-every`, `--eval-every` y `--checkpointing` no
cambian la receta; a lo sumo cambian la velocidad o la reproducibilidad.

**Cambia el experimento.** `--iterations` distinto de 100 000, `--t-init`, `--sigma`, el dataset,
el calendario del lr y los parámetros del disyuntor son legítimos para experimentar, pero el número
que salga ya no es comparable con el 31.53. `--t-init` no tiene piso —la «regla 2-δ» se retiró el
2026-10-12—: bajarlo cuesta exposición al estallido, no PSNR.

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
│   ├── model.py                  VNet2D y VNet3D: el esquema desenrollado, T y λ
│   ├── data.py                   cargadores, parches, augmentación, ruido y PSNR
│   ├── conv3d.py, tdv3d.py       los mismos operadores y la misma energía, en 3D
│   ├── data3d.py                 parches 3D con las 48 simetrías del cubo, recortes de evaluación
│   ├── qsm3d.py                  la evaluación QSM: el ADMM del autor con el denoiser, sobre COSMOS
│   └── seguimiento.py            W&B para train3d.py, que nunca corta el entrenamiento
├── train.py                    el entrenador; las notas al final explican el código
├── train3d.py                  el entrenador 3D: la receta de train.py, con reanudación y ramas
├── evaluate.py                 PSNR de un checkpoint sobre imágenes completas (y recortes 3D)
├── tests/                      el contrato del port y de la receta
├── checkpoints/
│   ├── Oficiales/                los dos .pth de los autores (color y gris) y su README
│   └── Entrenados/               t8delta-100k-s1, t8delta-100k-s0 y replica-v4, sin el estado de Adam
├── datasets/
│   ├── bsds500/                  imágenes locales; versionados cbsd68.txt y el README
│   ├── ldm100k/                  cortes de T1 locales; versionados lista_eval.txt y el README
│   ├── ldm100k3d/                volúmenes de T1 locales; versionadas las listas y el README
│   ├── qsm_cosmos/               local: params.mat y chi_cosmos.mat de ../tdv/data_mat
│   ├── cache_ldm100k.py          arma ldm100k desde los NIfTI de LDM_100k
│   ├── cache_ldm100k3d.py        arma ldm100k3d: escaneo, selección de máxima variedad y cache
│   └── mascaras_ldm100k3d.py     máscaras de cerebro de ldm100k3d, para --mascara-cerebro
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

Después viene la **fase 2**: llevar el prior a QSM 3D. El primer paso, TDV 3D sobre volúmenes de T1
con la receta del paper, es `train3d.py` ([Entrenar en 3D](#entrenar-en-3d)), que vino de la rama
`3D-Training`; el siguiente es entrenar con datos QSM. Lo que cambia, más allá de la dimensión: el
término de datos deja de ser la identidad y pasa a ser la convolución con el kernel del dipolo, que
se anula en un cono a 54.7° del campo principal, así que el prior tiene que rellenar un nulo del
operador y no sólo quitar ruido. `T` y `λ` se reaprenden porque dependen del término de datos; las
unidades son ppm y no [0, 1]; la augmentación deja de ser libre, porque el eje del campo no es
intercambiable con los otros dos; y las métricas pasan a ser RMSE, HFEN y SSIM contra COSMOS. La
mejora candidata de más valor para QSM es entrenar a ciegas del nivel de ruido, con σ sorteado en
cada lote, que eliminaría el reescalado entre σ ([`AGENTS.md`](AGENTS.md), «Mejoras candidatas»).

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
