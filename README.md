# Training-TDV: entrenamiento de Total Deep Variation en 2D y 3D

Este repositorio entrena desde cero el regularizador *Total Deep Variation* (TDV; Kobler et al.,
2022) para eliminar ruido gaussiano. Los autores publicaron la inferencia y dos checkpoints, no el
entrenamiento, que aquí se reconstruye del artículo: con su receta y un disyuntor propio de lotes
con gradiente anómalo, una ejecución alcanza 31.547 dB en CBSD68 a σ = 25 (31.53 publicados). La
versión 3D lleva la receta a volúmenes T1 de resonancia magnética, con reanudación exacta y una
validación de mapeo cuantitativo de susceptibilidad (QSM), hacia un prior para QSM.

## Estructura del repositorio

```text
.
├── ddr/                  el modelo, port de VLOGroup/tdv en 2D y 3D: R y el esquema (tdv*, model),
│                         convoluciones con adjunta exacta, datos, QSM y estabilidad, W&B
├── train.py, train3d.py  entrenamiento 2D y 3D
├── evaluate.py           PSNR de un checkpoint 2D sobre un conjunto
├── a_qsm_tdv.py          conversor de un checkpoint 3D al formato de ../QSM-TDV (hecho con IA)
├── checkpoints/          Oficiales/ (de los autores) y Entrenados/ (de aquí, 2D en color)
├── datasets/             listas y scripts de preparación de BSDS500 y LDM_100k
├── tests/                pruebas del port, de la receta y de los entrenadores
├── docs/                 bitácora del 3D y notas de los artículos (papers/)
└── AGENTS.md             contexto operativo para agentes de IA
```

No se versionan los datos (sí sus listas) ni las salidas (`runs/`). El estado del 3D y lo que lo
sostiene está en la [bitácora](docs/bitacora.md); el porqué de cada constante del código, en
[`docs/porque-del-codigo.md`](docs/porque-del-codigo.md). `train3d.py` se lee desde `main`, al
final: arma la corrida en pasos con nombre y llama a `entrena`, el bucle.

## Instalación y datos

```bash
uv sync        # crea .venv con las versiones exactas de uv.lock (torch 2.13, CUDA 13)
```

Instalar con `pip` desincroniza el entorno del lockfile. Entrenar requiere una GPU NVIDIA; en CPU,
más lentos, corren la evaluación, los tests y entrenamientos de prueba (en 3D, con
`--sin-compilar`).

- **BSDS500 (2D).** `BSR/BSDS500/data/images/` del
  [Berkeley Segmentation Dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)
  se copia a `datasets/bsds500/images/`. Se entrena con `train` + `test` (400 imágenes) y se evalúa
  en CBSD68, 68 imágenes de `val` ([`datasets/bsds500/`](datasets/bsds500/README.md)).
- **LDM_100k (T1, 3D).** Volúmenes T1 sintéticos de cerebro, con cráneo (Pinaya et al., 2022), que
  prepara [`datasets/ldm100k3d/`](datasets/ldm100k3d/README.md).
- **COSMOS (opcional, 3D).** Un sujeto del challenge QSM 2016, con susceptibilidad de referencia
  calculada de varias orientaciones de la cabeza: `params.mat` y `chi_cosmos.mat`, de `data_mat/` en
  [CIB-UC-QSM/TDV](https://github.com/CIB-UC-QSM/TDV), van en `datasets/qsm_cosmos/` (o en
  `--qsm-datos`). Sin ellos, el entrenamiento avisa y sigue sin la evaluación QSM; con `--meseta`,
  que decide con ella, sale con error al arrancar.

## Uso

### Arranque rápido

```bash
uv run pytest -k "not published_psnr"   # minutos; sin BSDS500 se omiten los de reproducción
# sin descargar nada: el checkpoint oficial en gris sobre PNG propios en datasets/mio/images/test/
uv run python evaluate.py checkpoints/Oficiales/tdv3-3-25-f32-gray.pth --data datasets/mio --split test --gray
# con BSDS500: el checkpoint entrenado aquí, sobre CBSD68 (31.55 dB)
uv run python evaluate.py checkpoints/Entrenados/t8delta-100k-s1.pth --file-list datasets/bsds500/cbsd68.txt
uv run python train.py --out runs/prueba --mode rapido --iterations 2000   # entrenamiento corto en BSDS500
```

### Aplicar un checkpoint

[`evaluate.py`](evaluate.py) mide el PSNR medio de un checkpoint 2D sobre
`<--data>/images/<--split>/` o los nombres de `--file-list`, con ruido de semilla fija por imagen.
`--gray` debe coincidir con el checkpoint; con `--sigma` ≠ 25 se aplica el reescalado del artículo,
aproximado.

Con datos propios, 2D o 3D, se usa el modelo directamente, con cualquier `.pth` (también los
`estado-<it>.pth`). Un script fuera de la raíz necesita `PYTHONPATH=<raíz>` (o
`sys.path.insert(0, "<raíz>")` antes del import) aunque se ejecute desde ella, porque Python busca
`ddr` en la carpeta del script.

```python
import torch
from ddr.model import VNet2D, VNet3D

ck = torch.load("checkpoints/Entrenados/t8delta-100k-s1.pth", map_location="cpu", weights_only=False)
cfg = ck["config"]
model = (VNet3D if cfg["R"]["type"] == "tdv3d" else VNet2D)(cfg).eval()
model.load_state_dict(ck["model"])
C = cfg["R"]["config"]["in_channels"]          # 3 en color, 1 en gris y en 3D

y = torch.linspace(0, 1, 96).repeat(1, C, 96, 1)   # aquí, la imagen limpia en [0, 1]; en 3D, [1, 1, D, H, W]
sigma = 25.0                                   # desvío del ruido, escala 0–255
z = y + sigma / 255 * torch.randn_like(y)      # la observación ruidosa
c = 25.0 / sigma                               # a la σ de entrenamiento (1 si fue ciego)
with torch.no_grad():                          # en GPU: model.cuda() y z.cuda()
    x = model(c * z, c * z)[-1] / c            # S + 1 estados; el último es la salida
```

Un checkpoint de color espera `C = 3`; el único gris versionado es
`checkpoints/Oficiales/tdv3-3-25-f32-gray.pth`. Las dimensiones no necesitan ser múltiplos de 4.

**En ../QSM-TDV.** Ese repositorio (de Mathías Lambert) tiene su propio TDV 3D y reconstruye QSM con
pasos de gradiente explícitos, sin ADMM. Un checkpoint de `train3d.py` no carga ahí tal cual:
`python a_qsm_tdv.py runs/<corrida>/best_qsm.pth <salida>/` lo convierte a su `checkpoint.pt` +
`report.json` (renombra 44 pesos y rehace 12 buffers). La conversión no es exacta en el borde del
volumen: la bajada de escala no es el mismo operador junto a las caras, y ∇R difiere 1.2–3.6 %. Lo
decidido el 2026-10-18 es una bandera en QSM-TDV que use nuestra bajada y subida (opción a, sin
escribir: el repositorio es de otra persona). Falta además la escala de χ, que nuestro ADMM
multiplica por α ≈ 4.86 antes de la red y QSM-TDV no toca, y calibrar allí los pasos y el número de
iteraciones.

### Entrenar en 2D

```bash
uv run python train.py --out runs/prueba   # 10 000 iteraciones, referencia
uv run python train.py --out runs/replica --iterations 100000 --seed 1   # la receta completa del artículo
```

La receta son 100 000 iteraciones, con el lr a la mitad cada 25 000; la réplica estaba en 31.340 dB
en la 10 000 y en 31.547 en la 100 000 ([`checkpoints/Entrenados/`](checkpoints/Entrenados/README.md)).
En `referencia` pide ~10 GB de VRAM (9.94 GB; bitácora, «Lo que dejó el 2D»); en una tarjeta menor,
`--checkpointing`.

### Entrenar en 3D

```bash
R=/ruta/a/LDM_100k/rawdata
uv run python datasets/cache_ldm100k3d.py cachea --lista lista_eval.txt --split val --rawdata $R
uv run python datasets/cache_ldm100k3d.py cachea --lista lista_train_2000.txt --split train --rawdata $R
uv run python train3d.py --out runs/sonda --sonda-vram   # opcional: memoria y costo, sin entrenar
uv run python train3d.py --out runs/humo --iterations 40 --eval-every 20 --seguimiento no   # ¿corre todo?
uv run python train3d.py --out runs/prueba3d --iterations 400 --eval-every 200   # ¿aprende?
uv run python train3d.py --out runs/sobreajuste --sobreajuste --lote 4 --hitos --iterations 1500 \
    --eval-every 250 --mascara-cerebro --qsm-datos '' --estabilidad-lado 0   # ¿memoriza un lote?
uv run python train3d.py --out runs/t1-3d
```

Por defecto, 40 000 iteraciones y el lr a la mitad cada 25 000 (`--hitos` sola lo deja constante;
`--hitos 30000` lo divide desde la 30 001).

**El arranque puede ser inestable.** La prueba de 40 iteraciones sólo comprueba que todo corre (con
`--log-every` en 250, no imprime líneas de avance): la de este repositorio rechazó 25 lotes y
terminó con 6 iteraciones efectivas y su `best.pth` marcado `*mejor*`; la de 400 llegó a 331. Hasta
juntar 100 normas aceptadas, el disyuntor sólo aplica su cota absoluta, y hasta la iteración ~250 el
único respaldo es el de la 3: rebobinar equivale a reiniciar. Si `ef` no avanza en unos cientos de
iteraciones, se relanza con otra semilla.

**Memoria.** `auto` elige el micro-lote (parches por backward; el lote se acumula) con un modelo del
entrenamiento y un margen del 10 %, supuesto suficiente para la validación; el pico real queda en
`estado.json`. Sirve cualquier entero: el lote se parte en trozos casi iguales (32 con
`--micro-lote 12` da 11 + 11 + 10). Lo que cuesta tiempo son las pasadas por iteración, no la
memoria, y `auto` toma el que deja menos pasadas.

**Prueba de sobreajuste.** `--sobreajuste` repite en cada iteración el mismo lote con el mismo
ruido, sin las dos cotas del disyuntor (con datos fijos, un rechazo se repetiría para siempre;
siguen el rechazo de un gradiente no finito y el rebobinado), y la validación agrega el PSNR de ese
lote. En la 5070 Ti (2026-10-18; bitácora, «Lo que mostraron las sondas»), 1 500 iteraciones (0.53
h) llevaron el lote fijo a 33.59 dB y su pérdida de 1 072.7 a 48.4, mientras su validación en
cerebro quedaba en 31.63; un control con datos frescos llegó a 32.18. Desde la 500, +1.50 dB en el
lote, +0.37 en su validación y +0.79 en la del control (`runs/prep3d-sobreajuste` y
`runs/prep3d-control`).

**lr/2 por meseta.** `--meseta N` mira el NRMSE QSM mínimo de cada evaluación, el de la mejor
iteración del ADMM: si en N seguidas no baja del mejor, el lr se divide por 2 desde la iteración
siguiente, hasta `--meseta-mitades` veces (5); la meseta siguiente corta la corrida, y `estado.json`
lo dice (`corte`). Mira el mínimo y no el final porque, mientras el ADMM no converge, el final salta
decenas de puntos entre validaciones (144 → 103 % en `ventana-3d-12h`) y el mínimo menos de uno. Una
evaluación sin NRMSE finito no cuenta, y una que falla tampoco: si fallan todas, la meseta no actúa
y el lr queda constante; lo avisan «qsm FALLÓ» en el log y `qsm/fallo` en W&B. Con `--meseta-desde
IT` sólo cuentan las evaluaciones desde esa iteración, y la primera fija el mejor. Con
`--meseta-psnr` la mitad pide además que el PSNR de la validación (el que elige `best.pth`, en el
cerebro con `--mascara-cerebro`) no haya hecho récord en esas N. La paciencia que importa es N por
la cadencia de la evaluación QSM, en iteraciones (por qué, en la bitácora, «La meseta»).

**El ADMM de la evaluación** corre hasta `--qsm-max-iter` iteraciones, 25 por defecto, sin el corte
por umbral del script del autor (sus umbrales se ajustaron para su red 2D en el pseudo-3D), y para
antes si el NRMSE sube dos iteraciones seguidas. Pare como pare, devuelve la mejor reconstrucción
que vio, así que su NRMSE final es el mínimo de lo recorrido (decisión del autor, 2026-10-19). Dos y
no una porque muchos ADMM 2D suben un solo paso y siguen bajando: sobre 26 trayectorias medidas,
parar en la primera subida da el mínimo en 19 y parar en la segunda, en 25, con una iteración más en
las que divergen. Con 25, el orden entre redes ya es casi el del final (en 2D, Spearman +0.98 contra
el final; con 1 o 2 iteraciones, +0.07 y +0.82), y el pseudo-3D oficial, que baja en las 25, da
26.08 %. Esta parada mira la referencia: sirve para comparar checkpoints de una misma corrida, que
es lo que hace la meseta. El ~26 % es una referencia para reportar y no decide nada. `qsm.jsonl`
guarda el NRMSE y el update de cada iteración, también los que hicieron parar.

### Entrenar con un dataset propio

**2D.** `train.py` lee `<raíz>/images/<split>/`: PNG y JPG de 8 bits (uno de 16 bits se satura sin
aviso), o `.npy`/`.npz` (`[C, H, W]` o `[H, W]`) en [0, 1], de al menos 93×93. Entrena con todas las
imágenes de `--train-splits` (separados por comas; por defecto, `train,test`) y valida, cada
`--eval-every` iteraciones y en la última, sobre la lista de `--eval-list` (un nombre por línea,
como `datasets/bsds500/cbsd68.txt`), que con otro dataset hay que pasar.

```bash
uv run python train.py --out runs/mio --channels 1 --iterations 50000 \
    --data-root datasets/mio --train-splits train --eval-split val \
    --eval-list datasets/mio/lista_val.txt
uv run python evaluate.py runs/mio/best.pth --data datasets/mio --split test \
    --file-list datasets/mio/lista_test.txt --gray
```

`--channels 1` es para gris. Como la validación elige `best.pth`, el PSNR se informa en otro
conjunto, y no se compara entre datasets: un fondo plano lo infla. Otro formato se agrega en
`ddr.data.IMAGE_LOADERS`.

**3D.** Cada volumen es un `.npy` `[D, H, W]` en [0, 1] (basta float16; el rango no se comprueba),
con vóxeles isotrópicos, porque el aumento permuta los ejes. Los de `--train-list` se buscan en
`images/train/` y miden al menos `--patch` (48) por eje; los de `--eval-list`, en `images/val/`, con
al menos `--eval-recorte` (128). Las listas llevan un nombre por línea, sin extensión. Las máscaras,
opcionales, van en `mascaras/{train,val}/<nombre>.npy` (bool, misma forma): las de `train`, para
`--mascara-cerebro`; las de `val`, para el PSNR en cerebro.

```bash
uv run python train3d.py --out runs/mio3d --data-root datasets/mio3d \
    --train-list datasets/mio3d/lista_train.txt --eval-list datasets/mio3d/lista_eval.txt --qsm-datos ''
```

Sin máscaras, un parche se acepta si al menos el 35 % de sus vóxeles supera 0.05, lo que supone un
fondo cercano a cero. Los volúmenes se cargan en RAM (2 000 de LDM_100k ocupan ~19 GB).
`--qsm-datos ''` apaga la evaluación QSM sin aviso.

### Qué deja una ejecución y cómo leerla

| archivo en `--out` | qué contiene |
|---|---|
| `config.json`, `history.json` | las banderas; una entrada por validación |
| `best.pth`, `last.pth` | el estado de mejor PSNR de validación (en 3D, en el cerebro con `--mascara-cerebro` y en el recorte entero sin ella) y el último |
| `traza.npy`, `estado.json` | una fila por iteración; el resumen final (terminó si dice `"terminada": true`). En 3D con `--mascara-cerebro`, su `mejor_psnr` es el PSNR en el cerebro, el criterio de `best.pth`, y `psnr_final`, el del recorte entero en la última validación: no se comparan entre sí. El resumen de W&B lleva lo mismo |
| `estado-<it>.pth` | 3D: lo necesario para reanudar, cada `--guardar-cada` (10 000) y en la última |
| `best_qsm.pth`, `qsm.jsonl` | 3D con COSMOS: el checkpoint para QSM y una línea por evaluación QSM |

El log escribe una línea cada `--log-every` iteraciones y otra por validación y por rebobinado; de
una racha de rechazos, las tres primeras y una cada 1 000.
En 3D (prueba de 400):

```text
[    13] LOTE RECHAZADO |g|=48297.3 mediana=1355.7 amp=nan loss=1384.601 (seguidos 1)
[    72] REBOBINADO a la it 3 (2 en lo que va) |g|=6.95e+04
    eval PSNR=30.223 dB, en tejido 29.951 dB, en cerebro 31.859 dB | NRMSE 9.13 %, … *mejor*
[   250/400] loss=  97.103 lr=4.00e-04 T=0.10946 lambda=0.08285 |g|=291.5 max|g|=8.13e+04 corr=0.952 rech=50 reb=2 ef=181 (…)
```

| campo | qué es |
|---|---|
| `loss` | `‖x_S − y‖²` por parche, media del lote; no hacer nada vale N·(σ/255)² (N valores por parche): 249.4 en color, 83.1 en gris, 1 063 a 48³ |
| `\|g\|`, `max\|g\|` | norma del gradiente y la máxima desde la línea anterior |
| `corr` | `‖x_S − x_0‖/‖x_0 − y‖`: sube desde cerca de 0 a ~0.95 si está sana; > 1 sobrecorrige |
| `rech`, `reb`, `ef` | rechazos, rebobinados e iteración efectiva (3D), sin lo descartado al rebobinar |
| en tejido, en cerebro | en los vóxeles limpios > 0.05 o en la máscara; NRMSE = 100·‖x − y‖/‖y‖ |
| estabilidad | `(T/S)·ρ` en la entrada de la validación o del ADMM (`qsm`), con ρ el mayor módulo de autovalor del hessiano de `R` (`λ_max`; el cociente de Rayleigh da su signo): un índice del primer paso, no una garantía. `*INESTABLE*` (≥ 2) no dice nada del PSNR, que se mide aparte |

- **Qué checkpoint usar.** Para eliminar ruido, `best.pth`: en 3D con `--mascara-cerebro` ya es el
  de mejor PSNR en el cerebro; sin ella, si no lo acompaña el PSNR en tejido, que el fondo no infla,
  el `estado-<it>.pth` de mejor `psnr_tejido` en `history.json`. Como prior de QSM, `best_qsm.pth`:
  el de menor NRMSE mínimo del ADMM en COSMOS, el de su mejor iteración (la estabilidad < 2 no
  predice la convergencia). El ADMM de la evaluación para mirando la susceptibilidad de referencia:
  antes de usarlo en un sujeto sin ella, mirar en `qsm.jsonl` su `motivo` y su trayectoria, porque
  si el ADMM se aleja después del mínimo hace falta una parada calibrada para esa red. El PSNR no lo
  sustituye: de cuatro redes 2D con 31.525–31.547 dB, sólo la oficial hizo converger el ADMM (n =
  4). Y el número no lo predice: una quinta, entrenada ciega al ruido, da 2.14 y converge, a 33.18 %
  de RMSE.
- **Cuándo relanzar.** Una ejecución muere si la pérdida o `T` se vuelven `NaN` o `T` llega a 0 (la
  salida es la entrada, 20.17 dB), y está perdida si el PSNR se hunde y no vuelve. Se relanza con
  otra semilla: el fallo es una tasa de la configuración. Son normales los rechazos y rebobinados
  aislados: 80–378 y 2–8 en cuatro ejecuciones 2D de 100 000 iteraciones.

### Reanudar

`train.py` no reanuda, porque perdería la repetibilidad de los generadores de parches y de ruido.
`train3d.py` reanuda de forma exacta: se repite el comando original entero y se le agrega
`--reanudar`.

```bash
D=datasets/mio3d   # A: todas las banderas del comando original
A="--data-root $D --train-list $D/lista_train.txt --eval-list $D/lista_eval.txt --mascara-cerebro \
   --iterations 40000 --hitos 30000 35000"
uv run python train3d.py --out runs/mio3d $A --reanudar runs/mio3d/estado-020000.pth  # corte
uv run python train3d.py --out runs/mio3d-lr2 $A --reanudar runs/mio3d/estado-010000.pth \
    --iterations 15000 --hitos 10000   # rama
```

Sólo pueden cambiar las banderas de `LIBRES` en `train3d.py` (iteraciones, cadencias, memoria,
seguimiento, QSM) y, en una rama (otro `--out`), el calendario: `--hitos`, `--meseta` y
`--meseta-mitades`; si otra difiere, `--reanudar` se niega
y la nombra. De una bandera repetida vale la última. Otro micro-lote cambia el orden de suma (el
log lo avisa).

### Llevar el entrenamiento 3D a otra máquina

1. **Requisitos del sistema.** Driver NVIDIA ≥ 580, el que pide el CUDA 13 de `uv.lock` (en
   Blackwell, el paquete `-open`): con uno menor, torch no ve la GPU y `train3d.py` sale con error
   (en CPU corre sólo con `--sin-compilar`, para probar).
   `torch.compile` pide un compilador C (`sudo apt install build-essential`). RAM: la corrida
   carga los 2 000 volúmenes de `lista_train_2000.txt` (~19 GB), más los workers, el compilador y
   la evaluación. Medido con esa lista en la 5070 Ti (2026-10-18, `runs/prep3d-punta`): 25.7 GiB
   de pico al compilar la primera evaluación, y 23.4–24.9 entrenando. Hacen falta más de 32 GB, y
   mejor 48: con 32 y un escritorio quedan 1–2 GiB, y sin swap eso es un corte por memoria. Disco:
   ~45 GB (estimado): 31.7 de datos, ~5 del entorno, la caché de uv y unos 2 de estados a
   100 000 iteraciones.
2. **Código y entorno.**
   - **`main` en GitHub tiene que tener el código de este README antes de clonar:** el `grep` de
     `QSM_MAX_ITER` de abajo lo comprueba. Una versión anterior corre sin error, pero con otras
     reglas de evaluación y de selección de checkpoints.
   - El repositorio es público: con `https` no hace falta clave SSH.
   - El clon va antes de copiar los datos (paso 3): `rsync` crea la carpeta, y `git clone` no
     clona en una que no está vacía.
   - `--locked` no deja que `uv sync` reescriba `uv.lock`. Si lo hiciera, el commit de
     `config.json` quedaría con `+cambios` (`commit_git` en `train3d.py`).
   - `--managed-python` instala el Python 3.13 de `.python-version` con uv, no el del sistema:
     Triton compila contra `Python.h`, que un Python del sistema puede no traer.

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh     # una vez, si no está uv
   cat /proc/driver/nvidia/version                     # driver ≥ 580
   git clone https://github.com/CIB-UC-QSM/Training-TDV.git && cd Training-TDV
   grep -n '^QSM_MAX_ITER = 25' train3d.py   # tiene que encontrar la línea
   uv sync --locked --managed-python && mkdir -p runs
   uv run python -c "import torch; print(torch.cuda.get_device_name())"
   ```

3. **Datos.** Son 4 026 archivos y 31.7 GB: los volúmenes cacheados (21 GB), las máscaras (10.5 GB;
   `--mascara-cerebro` y el PSNR en cerebro) y COSMOS (48 MB). Se copian desde la raíz del
   repositorio de origen, con el clon ya hecho en el destino (paso 2). `-L` copia archivos, no
   enlaces. Fuera de una LAN gigabit conviene `-z`: las máscaras comprimen ~90–100× y los
   volúmenes ~2× (zlib, medido en tres de cada uno, 2026-10-19).

   ```bash
   rsync -aRL --info=progress2 datasets/ldm100k3d/images datasets/ldm100k3d/mascaras \
       datasets/qsm_cosmos usuario@destino:Training-TDV/      # -azRL fuera de una LAN gigabit
   # en el destino, desde Training-TDV/: tiene que dar 4026
   find -L datasets/ldm100k3d/images datasets/ldm100k3d/mascaras datasets/qsm_cosmos -type f | wc -l
   ```

4. **Seguimiento.** Sin credencial, `--seguimiento auto` (el default) registra en `<out>/wandb/`, y
   un fallo de W&B no detiene el entrenamiento. `uv run wandb login` deja la credencial en
   `~/.netrc`, que ve también una ejecución lanzada como servicio (una `WANDB_API_KEY` exportada en
   la terminal, no); con `--seguimiento online`, el log dice al arrancar `seguimiento: W&B online`,
   y si falta wandb o la credencial, o W&B no abre, sale con error antes de entrenar.
   El proyecto `training-tdv3d` va público (decisión del autor, 2026-10-18), para que el supervisor
   lo siga con el link: lo crea la primera corrida online, y la visibilidad se cambia a «Public» en
   la configuración del proyecto en W&B. Tras un corte, reanudar abre otra corrida con el mismo
   nombre: lo registrado después del último estado no se mezcla (`ddr/seguimiento.py`,
   «Identidad»). Por eso al supervisor se le da el link del proyecto, no el de una corrida.
   Qué vigilar: `train/rechazos`, `train/rebobinados`,
   `train/iteracion_efectiva` (si no avanza en unos cientos de iteraciones, el arranque se trabó:
   relanzar con otra semilla, «El arranque puede ser inestable»; bitácora, «El disyuntor
   en 3D»), `train/T` lejos de 0, `qsm/fallo` (si fallan todas las evaluaciones QSM, la meseta no actúa y el
   lr queda constante hasta el techo), `qsm/nrmse_min_pct` (lo que eligen `best_qsm.pth` y la
   meseta), `qsm/nrmse_final_pct`, `qsm/iteraciones` y `meseta/*`.
5. **Comprobaciones.**
   - **Los tests:** el `pytest` de «Arranque rápido».
   - **Del bloque de «Entrenar en 3D»** se saltean las dos líneas `cachea`: piden el `rawdata` de
     LDM_100k, y los volúmenes ya llegan cacheados.
   - **La sonda:** `uv run python train3d.py --out runs/sonda --sonda-vram`.
     - Va en tmux o en una unidad, porque un corte del ssh la mata.
     - Compila y mide los 11 micro-lotes que cambian las pasadas: de 1 a 8, 11, 16 y 32. El 32 no
       entra en 96 GB: da OOM, la sonda lo ataja y sigue.
     - En esa tarjeta no se midió cuánto tarda (una estimación, sin medir: 15–30 min).
     - El resultado queda en `runs/sonda/sonda-vram.json`. `filas[]` trae, por micro-lote,
       `predicho_gib`, `medido_gib`, `s_por_parche` y `entra`; `auto_predicho` es lo que elegiría
       `auto` (en 96 GB, 16, ~57 GiB predichos), y `mas_rapido_por_parche`, el de menor tiempo
       por parche. El s/it es ~32 veces el `s_por_parche`.
     - Si el de 16 no baja claramente el `s_por_parche` del de 8, se pasa `--micro-lote 8`.
   - **La prueba de 40 iteraciones.** La de «Entrenar en 3D» va con `--seguimiento no`, sin
     `--mascara-cerebro` ni `--meseta` y fuera de systemd. No prueba la credencial de
     `~/.netrc` vista desde una unidad, ni linger, ni las 2 012 máscaras. En el destino conviene
     correr el comando del paso 6, con las banderas finales, cambiando cinco cosas: otro
     `--unit`, otro `--out`, `--wandb-proyecto training-tdv3d-pruebas` (para no ensuciar el
     proyecto público), `--guardar-cada 20` y `--eval-every 20 --eval-qsm-cada 20`, más
     `--iterations 40`. Deja estados en la 20 y la 40 y evalúa (PSNR y QSM) en las dos: la de la
     20 compila, y la de la 40 da el costo de una evaluación de régimen en esa tarjeta. Después se
     reanuda una vez desde `estado-000020.pth`, con `--reanudar` y `>>`. Eso prueba la
     reanudación y que W&B abre otra corrida.

     ```bash
     systemd-run --user --collect --unit tdv-3d-prueba -p WorkingDirectory=$PWD bash -c \
         ".venv/bin/python -u train3d.py <todas las banderas del paso 6> --out runs/3d-prueba \
          --iterations 40 --guardar-cada 20 --eval-every 20 --eval-qsm-cada 20 \
          --wandb-proyecto training-tdv3d-pruebas \
          > runs/3d-prueba.log 2>&1"   # al final, porque de una bandera repetida vale la última
     # cuando termine (systemctl --user status tdv-3d-prueba), la misma línea con
     # --reanudar runs/3d-prueba/estado-000020.pth y >> runs/3d-prueba.log
     ```

6. **Iteraciones, calendario y lanzamiento.** Decisiones del autor (2026-10-19): el ADMM de la
   evaluación con techo de 25 iteraciones, sin corte por umbral y parando si el NRMSE sube dos
   iteraciones seguidas (el default), la meseta sólo desde la
   10 000 (`--meseta-desde 10000`), la mitad del lr sólo si se estancan a la vez el NRMSE QSM y el
   PSNR (`--meseta-psnr`), `--guard 10`, y ~26 % de NRMSE como referencia para reportar, no para decidir («lr/2 por meseta» y «El
   ADMM de la evaluación» en «Entrenar en 3D»). `--hitos` sola deja el lr constante hasta que la
   meseta decida, y `--iterations 100000` es sólo el techo: la corrida puede terminar antes por sí
   sola. Como servicio se usa `.venv/bin/python` y no `uv run`: el `PATH` de una unidad de systemd
   no incluye `~/.local/bin`, donde el instalador deja `uv`.

   **La cadencia y la paciencia: recomendación de la sesión coordinadora sobre la propuesta del
   autor (2026-10-19); decide el autor.** El autor propuso evaluar cada 100 con `--meseta 10`. El
   comando de abajo lleva la validación de PSNR cada 100 (`--eval-every 100`), la evaluación QSM
   cada 500 (`--eval-qsm-cada 500`) y `--meseta 10`: 10 evaluaciones QSM, 5 000 iteraciones.
   - **La paciencia es en iteraciones, N × la cadencia QSM.** 100 × 10 son 1 000: con la tendencia
     subiendo 1σ cada 1 000 iteraciones, parte el lr por error en 5 000 iteraciones el 80 % de las
     veces (Monte Carlo); 500 × 10 o 100 × 50, el 0 %. En 2D, 2 000 iteraciones de paciencia habrían
     partido el lr en 4 de 8 corridas mientras el PSNR todavía subía.
   - **El costo** (5070 Ti, 8.77 s/it): QSM y estabilidad cada 100 suman un 19.4 % al entrenamiento,
     ~47 h en 100 000; PSNR cada 100 con QSM cada 500, un 5.9 %; todo cada 500, un 3.9 %. La tarjeta
     de destino no está medida.
   - **Para evaluar el QSM cada 100** igual: `--eval-every 100 --meseta 50` (la misma paciencia,
     más caro), sin `--eval-qsm-cada`.

   **`--guard 10`** (recomendación del 2026-10-19 de la sesión coordinadora y la sesión 3D, que el
   autor mantiene). Con la razón 5, los escalones de ‖g‖ (5–12× la mediana, con la pérdida normal)
   se volvían rebobinados, una vez cada ~93 iteraciones con el relativo armado, y las corridas de
   1 000 conservaron el 32 y el 22 % de sus iteraciones. Con 10 hubo un solo rebobinado fuera del
   arranque en 3 103 iteraciones con el relativo armado, y se conservó el 69–97 %. De 20 lotes
   aceptados a 5–10×, uno solo precedió algo que rebobinó. Es de la bitácora, «El disyuntor en 3D»
   (sondas `prep3d-mil`, `prep3d-g10` y `prep3d-g10-s2-2500`). `--guard` no se cambia al reanudar: una
   corrida lanzada con la razón 5 se empieza de cero.

   **La cota absoluta queda en 42 622** (`--guard-abs` sin pasar). Es GUARD_ABS = 10⁴ del 2D, que en
   BSDS cae en el hueco medido entre excursiones que se resuelven solas (≤ 6 113) y las que no (≥
   1.97·10⁵), llevada al cubo de 48³ por el cociente de valores sumados por parche (×4.26). Sólo
   actúa antes de que la mediana tenga 100 normas aceptadas: después, con `--guard 10`, la razón
   rechaza todo lo que pase de ~3·10³. Y en las trazas 3D queda por encima de todos los lotes con
   los que el arranque salió adelante (1.4–4.0·10⁴): con 2·10⁴ se habrían rechazado 4 de esas 6
   salidas y con 10⁴ las 6.

   Si nada mejorara nunca desde la 10 000 (ni el NRMSE ni el PSNR):

   | cadencia QSM × `--meseta` | mitades del lr (iteración) | corte |
   |---|---|---|
   | 500 × 10 (el comando de abajo) | 15 000, 20 000, 25 000, 30 000, 35 000 | 40 000 |
   | 100 × 10 (la propuesta) | 11 000, 12 000, 13 000, 14 000, 15 000 | 16 000 |

   ```bash
   sudo loginctl enable-linger $USER   # una vez: la unidad sigue al cerrar la sesión ssh
   loginctl show-user $USER -p Linger  # tiene que decir Linger=yes
   systemd-run --user --collect --unit tdv-3d-largo -p WorkingDirectory=$PWD bash -c \
       ".venv/bin/python -u train3d.py --out runs/3d-largo --iterations 100000 --hitos \
        --meseta 10 --meseta-desde 10000 --meseta-psnr --guard 10 --mascara-cerebro \
        --seguimiento online --eval-every 100 --eval-qsm-cada 500 --guardar-cada 2500 \
        > runs/3d-largo.log 2>&1"
   systemctl --user status tdv-3d-largo   # ¿corre?
   tail -f runs/3d-largo.log
   systemctl --user stop tdv-3d-largo     # la para: se pierde lo posterior al último estado-<it>.pth
   ```

   `--collect` va siempre. Sin él, una unidad que falla queda «failed»: por ejemplo, con un corte
   por memoria o con el error de `--seguimiento online`. Relanzar con el mismo `--unit` falla
   hasta `systemctl --user reset-failed tdv-3d-largo`, justo al reanudar.

   Cada 2 500 iteraciones queda un `estado-<it>.pth` (~47 MB; uno cada 100 serían ~53 GB), y
   `--guardar-cada` no toca la aritmética. Tras un corte se lanza el mismo comando con `--reanudar`
   y `>>`, para no pisar el log, con TODAS las banderas: las de la receta (`--guard`) y las del
   calendario (`--meseta*`) se exigen iguales, y las libres (`--eval-every`, `--eval-qsm-cada`,
   `--qsm-max-iter`) vuelven a su default si faltan. La meseta sigue con su cuenta y sus mitades,
   que viajan en el estado. Si una mitad llegara antes de tiempo, se rehace con una rama (otro
   `--out`) desde el estado anterior a ella, con otra `--meseta`:

   ```bash
   systemd-run --user --collect --unit tdv-3d-largo -p WorkingDirectory=$PWD bash -c \
       ".venv/bin/python -u train3d.py --out runs/3d-largo --iterations 100000 --hitos \
        --meseta 10 --meseta-desde 10000 --meseta-psnr --guard 10 --mascara-cerebro \
        --seguimiento online --eval-every 100 --eval-qsm-cada 500 --guardar-cada 2500 \
        --reanudar runs/3d-largo/estado-<it>.pth >> runs/3d-largo.log 2>&1"
   ```

   En `qsm.jsonl`, `nrmse_final_pct` y `nrmse_min_pct` son el mismo número, el del χ que devuelve
   el ADMM (salvo un ADMM no finito). El ~26 % sirve para reportar; lo que decide es la corrida
   contra sí misma, en la meseta. `best_qsm.pth` se elige por ese número entre todas las
   evaluaciones, incluidas las de antes de la 10 000, y con 200 evaluaciones es optimista: COSMOS
   lo elige y lo mide.

   La alternativa son hitos fijos, `--hitos N−15000 N−10000 N−5000`, con N ≈ horas disponibles ·
   3 600 / (s/it), descontadas las evaluaciones; con hitos y meseta a la vez, las mitades se suman.

### Modos y costo

Ningún modo cambia la receta. `referencia` (fp32 IEEE, determinista, sin `cudnn.benchmark`) repite
bit a bit: es el modo para un número que se vaya a citar. `intermedio` (TF32) y `rapido` (bf16) no
repiten y se comparan por tasas sobre semillas. Con la misma semilla y 100 000 iteraciones, `rapido`
dio 31.546 dB y `referencia` 31.547. La receta 2D completa tardó 12.6 h en `rapido` en una RTX 5070
Ti.

### Banderas

La arquitectura, el lr inicial, β y S no tienen bandera: están al principio de `train.py`,
justificadas en sus notas finales. `--sigma` (25, en 0–255) fija también la σ de validación. En 3D,
`--sigma-max` > 0 sortea σ ~ U(`--sigma-min`, `--sigma-max`) por parche (ruido ciego), y
`--eval-qsm-cada N` limita QSM y estabilidad a las validaciones múltiplo de N (y la última).
`--qsm-max-iter` (25) es el techo de iteraciones del ADMM, que para antes si el NRMSE sube dos
iteraciones seguidas. `--meseta`,
`--meseta-mitades`, `--meseta-desde`, `--meseta-psnr` y `--sobreajuste` vienen apagadas (arriba,
«Entrenar en 3D»), y
`train.py` tiene también `--sigma-min/--sigma-max`. El resto, con sus valores por defecto, está en
`--help`.

## Diferencias entre 2D y 3D

| | 2D (`train.py`) | 3D (`train3d.py`) |
|---|---|---|
| parche y lote | 93×93, 32 (fijo, 2 × 16) | 48³, 32 (`--patch`, `--lote`; micro-lote `auto`) |
| modo por defecto | `referencia` | `rapido` |
| costo de `referencia` / `rapido` | ~3× (2.98× en 100 000 iteraciones) | 16–21× por vóxel (estimado) |
| costo de una iteración en `rapido` | 1 | ~20 (derivado): 40 000 iteraciones ≈ 8 recetas 2D |
| `--checkpointing` | −78 % de memoria, +31 % de tiempo | ~5× menos memoria (derivado), +27 % de tiempo |
| cota absoluta del disyuntor | 10⁴ | 10⁴·P³/(93²·3): 4.26·10⁴ a 48³ |

Costos: bitácora, «Costo y memoria» y «Lo que dejó el 2D», en el mismo modelo de tarjeta que la 2D.

- **Cota absoluta.** Se escala con `P³` porque la pérdida suma sobre el parche y ‖g‖ crece con los
  valores sumados (un supuesto, apoyado en lo medido entre 1 y 3 canales); sin escalar, el arranque
  la habría cruzado.
- **Máscara de cerebro.** LDM_100k conserva el cráneo: sólo el 4 % de los parches de 48³ que pasan
  el filtro de tejido cae entero en el cerebro. `--mascara-cerebro` usa sólo cubos dentro de
  máscaras (deepbet ∩ HD-BET, [`mascaras_ldm100k3d.py`](datasets/mascaras_ldm100k3d.py)), que cubren
  el 66 % del cerebro.
- **Validación QSM.** Reconstruye el sujeto de COSMOS con un ADMM plug-and-play (Venkatakrishnan et
  al., 2013) con la red como denoiser y el protocolo de Milovic et al. (2021) en CIB-UC-QSM/TDV,
  tres cortes como RGB, extendido a los tres ejes. Desde el 2026-10-18 lleva las máscaras de ese
  pipeline (el solver de `../tdv`): la entrada y la salida de la red pasan por la máscara del
  cerebro en cada iteración, y el mapa final también. Con ellas y la red 2D oficial reproduce su
  25.94 % (25.9393 %); sin ellas reproducía su 26.89 %.

## Cómo funciona

### El esquema

El denoiser es un descenso por gradiente con el regularizador TDV. A partir de la observación
ruidosa `z`, da `S = 10` pasos sobre la energía `E(x) = (λ/T)·½‖x − z‖² + R(x)` (artículo, ec. 1,
en las unidades del código), explícitos en el regularizador aprendido `R` e implícitos en el término
de datos:

```text
x_0 = z,     x_{s+1} = ( x_s − (T/S)·∇R(x_s) + (λ/S)·z ) / ( 1 + λ/S ),     salida: x_S
```

`T`, la longitud total del descenso, y `λ`, el peso acumulado del dato, se aprenden con `R`. La
salida no es un minimizador de `E`, sino el estado del flujo detenido en el tiempo aprendido `T`
(Effland et al., 2020); seguir descendiendo sobresuaviza. Los diez pasos se entrenan de punta a
punta, con error cuadrático sobre `x_S` y ruido nuevo en cada iteración.

`R` es una red convolucional cuya salida por píxel se suma. Su primera convolución tiene filtros de
media cero y ninguna capa tiene sesgo: `R` no ve el nivel medio, sino cómo varía la imagen. Siguen
tres U-Net de tres escalas con bloques residuales y la activación `φ(a) = ½·log(1 + a²)`. El
análisis del artículo pide un `R` dos veces diferenciable, con derivadas acotadas y que dependa sólo
de `x`: por eso no hay ReLU (GELU o SiLU las cumplen), BatchNorm, que depende del lote, ni
autoatención por producto escalar, que no es globalmente Lipschitz (Kim et al., 2021).

### Qué se sabe de la inestabilidad

Sin protección, 5 de 12 semillas en bf16 y 6 de 12 en TF32 tuvieron un *evento* (‖g‖ > 10⁴) antes de
la iteración 4 000; después, la ejecución se recupera, queda perdida o muere. Lo que sigue es un
mecanismo posible y lo medido; la causa de un evento no está identificada.

- **Amplificación en el backward.** El gradiente recorre los pasos hacia atrás multiplicado en cada
  uno por `J_s = (I − (T/S)·∇²R(x_s)) / (1 + λ/S)`: una dirección propia de `∇²R` con autovalor `μ`
  se amplifica si el paso es largo, `(T/S)·μ > 2 + λ/S`, o si la curvatura es negativa, `μ < −λ/T`
  (`R` no es convexa). El operador compuesto mide 2.08 en el oficial y 2.6·10⁵ en una ejecución
  rota; no se sabe si esa amplificación precede al evento o lo sigue.
- **No es un error de implementación** (el gradiente coincide con autograd en float64 a ~10⁻¹⁶) **ni
  es atribuible a un estado**: una diferencia de redondeo en la iteración 3 141 borró el evento de
  la 3 429. Lo que caracteriza a una configuración es su tasa sobre semillas.
- **Adam y la pared de `T`.** Adam divide cada paso por la raíz de su segundo momento: cuando un
  pico domina ese momento, el desplazamiento que causa no depende de su tamaño (es igual con 10⁴ que
  con 10⁹) y es a lo sumo δ_Adam = lr/√(1 − β₂), 0.0127 con el lr inicial. Para un `T` de ese orden,
  un pico de signo adverso lo lleva a 0, una pared: el esquema devuelve `z` y los gradientes de los
  pesos se anulan. Recortar la norma del gradiente no lo evita, porque el valor recortado sigue
  dominando el segundo momento.

### Cómo lo maneja el repositorio

- **`T` arranca en 8·δ_Adam ≈ 0.101**, lejos de la pared. El artículo no publica valores iniciales;
  el `config` de los checkpoints oficiales trae `T = 0.001`, desde donde murieron las tres
  ejecuciones en fp32 (las tres en bf16 pasaron la 5 000; medido en 2D).
- **Disyuntor de lotes.** Entre el backward y el paso de Adam, descarta el lote si ‖g‖ no es finita
  o supera la cota absoluta o 5 veces la mediana de las últimas 100 normas aceptadas; el lote
  descartado no mueve pesos ni momentos. Mientras no rechaza, la ejecución es bit a bit la de la
  receta. Los umbrales se calibraron sobre las trazas de 69 trayectorias distintas.
- **Rebobinado.** El rechazo llega un paso tarde: ‖g‖ se mide en pesos que ya movieron los lotes
  aceptados. Tras 25 rechazos seguidos, la ejecución vuelve al último respaldo en memoria (pesos,
  momentos de Adam y mediana) y sigue con lotes nuevos, sin retroceder el lr. Hay un respaldo cada
  250 iteraciones o más; el primero, en la 3.

## Resultados y alcance

| PSNR en CBSD68 (dB) | σ = 15 | σ = 25 | σ = 50 |
|---|---|---|---|
| publicado (Kobler et al.) | 34.12 | 31.53 | 28.26 |
| checkpoint oficial, evaluado con este código | 34.122 | 31.537 | 28.264 |
| **`t8delta-100k-s1`**, entrenado aquí | **34.125** | **31.547** | **28.268** |

La fila propia es una ejecución (semilla 1, 100 000 iteraciones, `referencia`) con la receta de la
sección 5.1 más el disyuntor, que no está en el artículo: descartó 163 lotes y rebobinó 4 veces
([`checkpoints/Entrenados/`](checkpoints/Entrenados/README.md)). Sin disyuntor, ninguna de las
ejecuciones de 100 000 iteraciones llegó al valor publicado. σ = 15 y 50 usan el reescalado del
artículo.

- **Alcance.** Una ejecución completa, más la semilla 0, cortada en la 66 000 con 31.526 dB (igual
  que la 1 en la 64 000); evaluadas sólo en CBSD68, que también elige `best.pth` (en la 1, el
  último). El efecto del disyuntor sobre la tasa de fallos no está medido. El 3D sólo tiene pruebas
  cortas: la más larga, la rama `prep3d-g10-s2-2500`, de 2 500 iteraciones.
- **QSM.** Un solo sujeto, que también ajusta el solver y elige `best_qsm.pth`: su NRMSE es
  optimista. El prior se entrena sobre magnitud T1 y se aplica a `χ`, con signo y otra amplitud, y
  la red entera se usa como denoiser, no `R` como regularizador: las garantías del artículo no se
  trasladan.
- **Diferencias con el artículo.** Los valores iniciales de `T` y `λ`, que no publica, se eligieron
  aquí; `λ` separado de `T` y la cota de norma de la primera convolución salen del código de los
  autores. El artículo da `T* = 0.172` para el modelo gris; el checkpoint guarda 0.0172.

## Referencias y licencia

- Kobler, Effland, Kunisch, Pock. *IEEE TPAMI* 44(12):9163–9180, 2022,
  [doi:10.1109/TPAMI.2021.3124086](https://doi.org/10.1109/TPAMI.2021.3124086); secciones y
  ecuaciones según [arXiv:2006.08789v1](https://arxiv.org/abs/2006.08789v1).
- Effland, Kobler, Kunisch, Pock. *J. Math. Imaging Vis.* 62(3):396–416, 2020 (parada aprendida).
- Kingma, Ba. *ICLR* 2015 (Adam). Zhang et al. *IEEE TIP* 26(7), 2017 (DnCNN; protocolo CBSD68).
  Kim et al. *ICML* 2021 (Lipschitz de la autoatención).
- Milovic et al. *ISMRM* 2021 (TDV-QSM). Langkammer et al. *Magn. Reson. Med.* 79(3), 2018
  (challenge QSM 2016). Venkatakrishnan et al. *IEEE GlobalSIP* 2013 (plug-and-play).
- Arbeláez et al. *IEEE TPAMI* 33(5), 2011 (BSDS500). Pinaya et al. *DGM4MICCAI* 2022 (LDM_100k).
  Isensee et al. *Hum. Brain Mapp.* 40(17), 2019 (HD-BET). Fisch et al. *Comput. Biol. Med.* 179,
  2024 (deepbet).

Licencia MIT ([`LICENSE`](LICENSE)). El regularizador de `ddr/` deriva de
[VLOGroup/tdv](https://github.com/VLOGroup/tdv), la inferencia oficial, también MIT.
