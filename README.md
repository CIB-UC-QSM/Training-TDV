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
├── checkpoints/          Oficiales/ (de los autores) y Entrenados/ (de aquí, 2D en color)
├── datasets/             listas y scripts de preparación de BSDS500 y LDM_100k
├── tests/                pruebas del port, de la receta y de los entrenadores
├── docs/                 bitácora, métricas, mediciones crudas y notas de artículos
└── AGENTS.md             contexto operativo para agentes de IA
```

No se versionan los datos (sí sus listas) ni las salidas (`runs/`). Las mediciones citadas están en
[`docs/mediciones/`](docs/mediciones/), y la historia de cada experimento, en la
[bitácora](docs/bitacora.md).

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
- **LDM_100k (T1, 2D y 3D).** Volúmenes T1 sintéticos de cerebro, con cráneo (Pinaya et al.,
  2022), que preparan [`datasets/ldm100k/`](datasets/ldm100k/README.md) (cortes 2D) y
  [`datasets/ldm100k3d/`](datasets/ldm100k3d/README.md) (volúmenes).
- **COSMOS (opcional, 3D).** Un sujeto del challenge QSM 2016, con susceptibilidad de referencia
  calculada de varias orientaciones de la cabeza: `params.mat` y `chi_cosmos.mat`, de `data_mat/` en
  [CIB-UC-QSM/TDV](https://github.com/CIB-UC-QSM/TDV), van en `datasets/qsm_cosmos/` (o en
  `--qsm-datos`). Sin ellos, el entrenamiento avisa y sigue.

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

[`evaluate.py`](evaluate.py) mide el PSNR medio de un checkpoint 2D sobre `<--data>/images/<--split>/`
o los nombres de `--file-list`, con ruido de semilla fija por imagen. `--gray` debe coincidir con el
checkpoint; con `--sigma` ≠ 25 se aplica el reescalado del artículo, aproximado
([`escala-equivariancia.txt`](docs/mediciones/escala-equivariancia.txt)).

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

### Entrenar en 2D

```bash
uv run python train.py --out runs/prueba   # 10 000 iteraciones, referencia
uv run python train.py --out runs/replica --iterations 100000 --seed 1   # la receta completa del artículo
```

La receta son 100 000 iteraciones, con el lr a la mitad cada 25 000; la réplica estaba en 31.340 dB
en la 10 000 y en 31.547 en la 100 000 ([`t8delta-100k-s1.md`](docs/resultados/t8delta-100k-s1.md)).
En `referencia` pide ~10 GB de VRAM (9.94 GB, bitácora 2026-10-08); en una tarjeta menor,
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
terminó con 6 iteraciones efectivas y su `best.pth` marcado `*mejor*`; la de 400 llegó a 331
([`ventana-3d-12h.txt`](docs/mediciones/ventana-3d-12h.txt)). Hasta juntar 100 normas aceptadas, el
disyuntor sólo aplica su cota absoluta, y hasta la iteración ~250 el único respaldo es el de la 3:
rebobinar equivale a reiniciar. Si `ef` no avanza en unos cientos de iteraciones, se relanza con
otra semilla.

**Memoria.** `auto` elige el micro-lote (parches por backward; el lote se acumula) con un modelo del
entrenamiento y un margen del 10 %, supuesto suficiente para la validación; el pico real queda en
`estado.json`. Sirve cualquier entero: el lote se parte en trozos casi iguales (32 con
`--micro-lote 12` da 11 + 11 + 10). Lo que cuesta tiempo son las pasadas por iteración, no la
memoria, y `auto` toma el que deja menos pasadas.

**Prueba de sobreajuste.** `--sobreajuste` repite en cada iteración el mismo lote con el mismo
ruido, sin disyuntor (con datos fijos, un rechazo se repetiría para siempre), y la validación
agrega el PSNR de ese lote. En la 5070 Ti, 1 500 iteraciones (0.54 h) llevaron el lote a 33.29 dB
mientras su validación en cerebro quedaba en 31.38; un control con datos frescos iba en 32.17 en
la 1 000 (bitácora, 2026-10-16).

**lr/2 por meseta.** `--meseta N` mira el NRMSE QSM de cada evaluación: si en N seguidas no baja del
mejor, el lr se divide por 2 desde la iteración siguiente, hasta `--meseta-mitades` veces (5); la
meseta siguiente corta la corrida, y `estado.json` lo dice (`corte`). Mientras el ADMM no
converge el NRMSE salta mucho entre validaciones, y con N = 1 una sola peor ya parte el lr.

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
| `best.pth`, `last.pth` | el estado de mejor PSNR de validación (en 3D, el del recorte entero) y el último |
| `traza.npy`, `estado.json` | una fila por iteración; el resumen final (terminó si dice `"terminada": true`) |
| `estado-<it>.pth` | 3D: lo necesario para reanudar, cada `--guardar-cada` (10 000) y en la última |
| `best_qsm.pth`, `qsm.jsonl` | 3D con COSMOS: el checkpoint para QSM y una línea por evaluación QSM |

El log escribe una línea cada `--log-every` iteraciones y otra por validación, rechazo o rebobinado.
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
| `corr` | `‖x_S − x_0‖/‖x_0 − y‖`: sube desde cerca de 0 a ~0.95 si está sana; > 1 sobrecorrige ([`metricas.md`](docs/metricas.md)) |
| `rech`, `reb`, `ef` | rechazos, rebobinados e iteración efectiva (3D), sin lo descartado al rebobinar |
| en tejido, en cerebro | en los vóxeles limpios > 0.05 o en la máscara; NRMSE = 100·‖x − y‖/‖y‖ |
| estabilidad | `(T/S)·ρ` en la entrada de la validación o del ADMM (`qsm`), con ρ el mayor módulo de autovalor del hessiano de `R` (`λ_max`; el cociente de Rayleigh da su signo): un índice del primer paso, no una garantía. `*INESTABLE*` (≥ 2) no dice nada del PSNR, que se mide aparte |

- **Qué checkpoint usar.** Para eliminar ruido, `best.pth`; en 3D, si no lo acompaña el PSNR en
  tejido o en cerebro, que el fondo no infla, el `estado-<it>.pth` de mejor `psnr_tejido` o
  `psnr_cerebro` en `history.json`. Como prior de QSM, `best_qsm.pth` (menor NRMSE en COSMOS con
  estabilidad < 2), tras revisar su `nrmse_final_pct` en `qsm.jsonl`: estable no implica
  convergente. El PSNR no lo sustituye: de cuatro redes 2D con 31.525–31.547 dB, sólo la oficial
  hizo converger el ADMM (n = 4;
  [`qsm-prior-4corridas.txt`](docs/mediciones/qsm-prior-4corridas.txt)). Y el número no lo
  predice: una quinta, entrenada ciega al ruido, da 2.14 y converge, a 33.18 % de RMSE
  ([`qsm-prior-ciego2d.txt`](docs/mediciones/qsm-prior-ciego2d.txt)).
- **Cuándo relanzar.** Una ejecución muere si la pérdida o `T` se vuelven `NaN` o `T` llega a 0 (la
  salida es la entrada, 20.17 dB), y está perdida si el PSNR se hunde y no vuelve. Se relanza con
  otra semilla: el fallo es una tasa de la configuración. Son normales los rechazos y rebobinados
  aislados: 80–378 y 2–8 en cuatro ejecuciones 2D de 100 000 iteraciones
  ([`modo-y-tinit-100k-s1.txt`](docs/mediciones/modo-y-tinit-100k-s1.txt)).

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
   `torch.compile` pide un compilador C (`sudo apt install build-essential`).
2. **Código y entorno.** `uv sync` instala el Python 3.13 que fija `.python-version`, y no conviene
   forzar otro.

   ```bash
   git clone git@github.com:CIB-UC-QSM/Training-TDV.git && cd Training-TDV
   uv sync && mkdir -p runs
   uv run python -c "import torch; print(torch.cuda.get_device_name())"
   ```

3. **Datos.** Desde la raíz del repositorio de origen, por la red local, 4 026 archivos y 31.7 GB:
   los volúmenes cacheados (21 GB), las máscaras (10.5 GB; `--mascara-cerebro` y el PSNR en
   cerebro) y COSMOS (48 MB). `-L` copia archivos, no enlaces.

   ```bash
   rsync -aRL --info=progress2 datasets/ldm100k3d/images datasets/ldm100k3d/mascaras \
       datasets/qsm_cosmos usuario@destino:Training-TDV/
   ```

4. **Seguimiento.** Sin credencial, `--seguimiento auto` (el default) registra en `<out>/wandb/`, y
   un fallo de W&B no detiene el entrenamiento. `uv run wandb login` deja la credencial en
   `~/.netrc`, que ve también una ejecución lanzada como servicio (una `WANDB_API_KEY` exportada en
   la terminal, no); con `--seguimiento online`, el log dice al arrancar `seguimiento: W&B online`,
   y si falta wandb o la credencial, o W&B no abre, sale con error antes de entrenar.
   El proyecto `training-tdv3d` va público (decisión del autor, 2026-10-18), para que el supervisor
   lo siga con el link: lo crea la primera corrida online, y la visibilidad se cambia a «Public» en
   la configuración del proyecto en W&B.
5. **Comprobaciones.** Los tests, la sonda y la prueba de 40 iteraciones de «Entrenar en 3D». La
   sonda mide los micro-lotes que cambian las pasadas (1–8, 11, 16 y 32) y da los s por parche (el
   s/it es ~32 veces eso): en 96 GB, `auto` elige micro-lote 16
   (~57 GiB predichos), y si 16 no baja claramente el tiempo por parche frente a 8, se pasa
   `--micro-lote 8`. La prueba da el costo de una validación.
6. **Iteraciones, calendario y lanzamiento.** Decisión del autor (2026-10-18): el NRMSE QSM decide
   el lr. `--hitos` sola deja el lr constante, y `--meseta 2` lo divide a la mitad cada vez que el
   NRMSE no baja del mejor en dos evaluaciones seguidas, hasta 5 veces; la meseta siguiente corta
   la corrida («lr/2 por meseta» en «Entrenar en 3D»). `--iterations 100000` es sólo el techo: la
   corrida puede terminar antes por sí sola. Como servicio se usa `.venv/bin/python` y no `uv run`:
   el `PATH` de una unidad de systemd no incluye `~/.local/bin`, donde el instalador deja `uv`.

   ```bash
   sudo loginctl enable-linger $USER   # una vez: la unidad sigue al cerrar la sesión ssh
   systemd-run --user --unit tdv-3d-largo -p WorkingDirectory=$PWD bash -c \
       ".venv/bin/python -u train3d.py --out runs/3d-largo --iterations 100000 --hitos --meseta 2 \
        --mascara-cerebro --seguimiento online --eval-every 2500 --guardar-cada 2500 \
        > runs/3d-largo.log 2>&1"
   ```

   Con una evaluación cada 2 500 iteraciones, dos sin mejora son 5 000 de meseta. Cada validación
   QSM deja su `estado-<it>.pth`, y si `best_qsm.pth` no convence se puede elegir otro a mano; cada
   uno ocupa ~51 MB en la 40 000 (derivado, bitácora 2026-10-13), y `--guardar-cada` no toca la
   aritmética. Tras un corte, el mismo comando con `--reanudar runs/3d-largo/estado-<it>.pth`, y
   `>>` para no pisar el log: la meseta sigue con su cuenta y sus mitades, que viajan en el estado.

   La alternativa son hitos fijos, `--hitos N−15000 N−10000 N−5000`, con N ≈ horas disponibles ·
   3 600 / (s/it), descontada una validación cada 2 500; con hitos y meseta a la vez, las mitades
   se suman.

### Modos y costo

Ningún modo cambia la receta. `referencia` (fp32 IEEE, determinista, sin `cudnn.benchmark`) repite
bit a bit: es el modo para un número que se vaya a citar. `intermedio` (TF32) y `rapido` (bf16) no
repiten y se comparan por tasas sobre semillas. Con la misma semilla y 100 000 iteraciones,
`rapido` dio 31.546 dB y `referencia` 31.547
([`rapido-vs-referencia-100k.txt`](docs/mediciones/rapido-vs-referencia-100k.txt)).
La receta 2D completa tardó 12.6 h en `rapido` en una RTX 5070 Ti.

### Banderas

La arquitectura, el lr inicial, β y S no tienen bandera: están al principio de `train.py`,
justificadas en sus notas finales. `--sigma` (25, en 0–255) fija también la σ de validación. En 3D,
`--sigma-max` > 0 sortea σ ~ U(`--sigma-min`, `--sigma-max`) por parche (ruido ciego), y
`--eval-qsm-cada N` limita QSM y estabilidad a las validaciones múltiplo de N (y la última).
`--meseta`, `--meseta-mitades` y `--sobreajuste` vienen apagadas (arriba, «Entrenar en 3D»), y
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

Costos: [`modo-y-tinit-100k-s1.txt`](docs/mediciones/modo-y-tinit-100k-s1.txt),
[`costo-tdv3d.txt`](docs/mediciones/costo-tdv3d.txt) y la sonda 3D (bitácora, 2026-10-13), en el
mismo modelo de tarjeta que la 2D.

- **Cota absoluta.** Se escala con `P³` porque la pérdida suma sobre el parche y ‖g‖ crece con los
  valores sumados (un supuesto, apoyado en lo medido entre 1 y 3 canales); sin escalar, el arranque
  la habría cruzado ([`ventana-3d-12h.txt`](docs/mediciones/ventana-3d-12h.txt)).
- **Máscara de cerebro.** LDM_100k conserva el cráneo: sólo el 4 % de los parches de 48³ que pasan el
  filtro de tejido cae entero en el cerebro. `--mascara-cerebro` usa sólo cubos dentro de máscaras
  (deepbet ∩ HD-BET, [`mascaras_ldm100k3d.py`](datasets/mascaras_ldm100k3d.py)), que cubren el 66 %
  del cerebro ([`cobertura-cubos-cerebro.txt`](docs/mediciones/cobertura-cubos-cerebro.txt)).
- **Validación QSM.** Reconstruye el sujeto de COSMOS con un ADMM plug-and-play (Venkatakrishnan et
  al., 2013) con la red como denoiser y el protocolo de Milovic et al. (2021) en CIB-UC-QSM/TDV,
  tres cortes como RGB, extendido a los tres ejes. Con la red 2D oficial reproduce su 26.89 % de
  NRMSE ([`qsm-eval-cosmos.txt`](docs/mediciones/qsm-eval-cosmos.txt)).

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

Sin protección, 5 de 12 semillas en bf16 y 6 de 12 en TF32 tuvieron un *evento* (‖g‖ > 10⁴) antes
de la iteración 4 000 ([`precision.txt`](docs/mediciones/precision.txt)); después, la ejecución se
recupera, queda perdida o muere. Lo que sigue es un mecanismo posible y lo medido; la causa de un
evento no está identificada.

- **Amplificación en el backward.** El gradiente recorre los pasos hacia atrás multiplicado en cada
  uno por `J_s = (I − (T/S)·∇²R(x_s)) / (1 + λ/S)`: una dirección propia de `∇²R` con autovalor `μ`
  se amplifica si el paso es largo, `(T/S)·μ > 2 + λ/S`, o si la curvatura es negativa,
  `μ < −λ/T` (`R` no es convexa). El operador compuesto mide 2.08 en el oficial y 2.6·10⁵ en una
  ejecución rota ([`adjunta-sigma-max.txt`](docs/mediciones/adjunta-sigma-max.txt)); no se
  sabe si esa amplificación precede al evento o lo sigue.
- **No es un error de implementación** (el gradiente coincide con autograd en float64 a ~10⁻¹⁶)
  **ni es atribuible a un estado**: una diferencia de redondeo en la iteración 3 141 borró el evento
  de la 3 429 ([`replay-ulp.txt`](docs/mediciones/replay-ulp.txt)). Lo que caracteriza a una
  configuración es su tasa sobre semillas.
- **Adam y la pared de `T`.** Adam divide cada paso por la raíz de su segundo momento: cuando un pico
  domina ese momento, el desplazamiento que causa no depende de su tamaño (es igual con 10⁴ que con
  10⁹; [`adam-respuesta-al-pico.txt`](docs/mediciones/adam-respuesta-al-pico.txt)) y es a lo sumo
  δ_Adam = lr/√(1 − β₂), 0.0127 con el lr inicial. Para un `T` de ese orden, un pico de signo
  adverso lo lleva a 0, una pared: el esquema devuelve `z` y los gradientes de los pesos se anulan.
  Recortar la norma del gradiente no lo evita, porque el valor recortado sigue dominando el segundo
  momento.

### Cómo lo maneja el repositorio

- **`T` arranca en 8·δ_Adam ≈ 0.101**, lejos de la pared. El artículo no publica valores iniciales;
  el `config` de los checkpoints oficiales trae `T = 0.001`, desde donde murieron las tres
  ejecuciones en fp32 (las tres en bf16 pasaron la 5 000; bitácora, 2026-09-02).
- **Disyuntor de lotes.** Entre el backward y el paso de Adam, descarta el lote si ‖g‖ no es
  finita o supera la cota absoluta o 5 veces la mediana de las últimas 100 normas aceptadas; el
  lote descartado no mueve pesos ni momentos. Mientras no rechaza, la ejecución es bit a bit la de
  la receta (bitácora, 2026-10-03). Los umbrales se calibraron sobre las trazas de 69 trayectorias
  distintas ([`disyuntor-calibracion.txt`](docs/mediciones/disyuntor-calibracion.txt)).
- **Rebobinado.** El rechazo llega un paso tarde: ‖g‖ se mide en pesos que ya movieron los lotes
  aceptados ([`tinit-10k.txt`](docs/mediciones/tinit-10k.txt)). Tras 25 rechazos seguidos, la
  ejecución vuelve al último respaldo en memoria (pesos, momentos de Adam y mediana) y sigue con
  lotes nuevos, sin retroceder el lr. Hay un respaldo cada 250 iteraciones o más; el primero, en la 3.

## Resultados y alcance

| PSNR en CBSD68 (dB) | σ = 15 | σ = 25 | σ = 50 |
|---|---|---|---|
| publicado (Kobler et al.) | 34.12 | 31.53 | 28.26 |
| checkpoint oficial, evaluado con este código | 34.122 | 31.537 | 28.264 |
| **`t8delta-100k-s1`**, entrenado aquí | **34.125** | **31.547** | **28.268** |

La fila propia es una ejecución (semilla 1, 100 000 iteraciones, `referencia`) con la receta de la
sección 5.1 más el disyuntor, que no está en el artículo: descartó 163 lotes y rebobinó 4 veces
([`checkpoints/Entrenados/`](checkpoints/Entrenados/README.md)). Sin disyuntor, ninguna de las
cuatro ejecuciones de 100 000 iteraciones (tres con `T` inicial 0.001) llegó al valor publicado; la
mejor quedó en 31.478 dB (bitácora, «Inventario de corridas»). σ = 15 y 50 usan el reescalado del
artículo.

- **Alcance.** Una ejecución completa, más la semilla 0, cortada en la 66 000 con 31.526 dB (igual
  que la 1 en la 64 000); evaluadas sólo en CBSD68, que también elige `best.pth` (en la 1, el
  último). El efecto del disyuntor sobre la tasa de fallos no está medido. El 3D sólo tiene pruebas
  de hasta 400 iteraciones.
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
