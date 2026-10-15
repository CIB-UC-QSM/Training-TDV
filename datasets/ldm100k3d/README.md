# ldm100k3d

Volúmenes 3D de T1w sintéticas sacados de `LDM_100k` (Pinaya et al., 2022): 100 000 volúmenes de
cerebro generados por un modelo de difusión latente entrenado sobre UK Biobank, en BIDS,
`sub-*/anat/sub-*_T1w.nii.gz`, uint16, 160×224×160 a 1 mm isotrópico y alineados entre sí.
`participants.tsv` viene vacío: no hay edad, sexo ni volúmenes por sujeto.

**No es QSM.** Es T1w: sirve para medir el paso de TDV de 2D a 3D sobre un dominio de cerebro, no
«imagen natural → mapa de susceptibilidad». Nada de lo que se entrene acá se traslada a QSM sin
volver a medirlo.

**Tiene cráneo.** LDM_100k no está skull-stripped (sí tiene defacing): el «tejido» de este
dataset (> 0.05 de la norma) es la cabeza entera, con cuero cabelludo y hueso, no sólo cerebro. La
cabeza llena el campo de visión, así que el recorte a la caja del tejido ahorra poco.

Las imágenes y el escaneo **no se versionan**. Lo versionado de esta carpeta es este archivo y las
listas `lista_*.txt`; el script que arma todo está un nivel arriba, en
[`../cache_ldm100k3d.py`](../cache_ldm100k3d.py).

## Contenido

| ruta | qué es |
|---|---|
| `escaneo/miniaturas.npy` | float16 `[100 000, 20, 28, 20]` (2.2 GB): cada volumen normalizado, promediado en bloques de 8³ |
| `escaneo/escalares.npz` | por sujeto, percentil 99.5 crudo, fracción de tejido, fracción oscura y cuantiles 10/50/90 en la caja central, vóxeles saturados; `ok` marca los que se leyeron |
| `seleccion.npz` | la PCA de las miniaturas y los índices de la selección (control de calidad, evaluación, orden del k-center voraz); queda en la `--salida` de `selecciona` |
| `lista_eval.txt` | 12 sujetos, uno por línea: el más cercano a cada centroide de un k-means en el espacio PCA |
| `lista_train_<N>.txt` | N sujetos de entrenamiento (2 000 y 8 000), disjuntos de los de evaluación |
| `images/<split>/sub-XXXXXX.npy` | float16 `[D, H, W]` en [0, 1], recortado a la caja del tejido con 4 vóxeles de margen |
| `mascaras/<split>/sub-XXXXXX.npy` | bool, de la misma forma que su `images/<split>/sub-XXXXXX.npy` y alineada vóxel a vóxel: el cerebro según deepbet ∩ HD-BET sin TTA. La lee `train3d.py --mascara-cerebro` |
| `mascaras-nifti/<herramienta>/sub-XXXXXX.npz` | la máscara de cada herramienta en el espacio del NIfTI (160×224×160), comprimida: el paso intermedio de las de arriba |

Cada `.npy` de `images/` está normalizado como los cortes de [`../ldm100k`](../ldm100k/README.md):
dividido por el percentil 99.5 de los vóxeles positivos de su volumen y recortado a [0, 1].

Medido el 2026-10-08 sobre 200 volúmenes sorteados (crudo en
`docs/mediciones/tejido-ldm100k3d.txt`): la caja del tejido mide 144–160 × 195–223 × 144–154 y en
el eje 2 toca el borde del campo en los 200; el 38–51 % (p1–p99) de los vóxeles crudos vale exactamente 0.
Un volumen cacheado pesa 9.9 MB de mediana (10.9 MB entero), así que 2 000 son ~19 GB en RAM y
8 000, ~78 GB.

## Cómo se elige

1. **Control de calidad.** Fuera los sujetos con algún escalar a más de 10 desviaciones robustas
   (MAD) de la mediana, o con la distancia de su miniatura al centro del espacio PCA a más de 10
   desviaciones robustas de la mediana de esas distancias: un volumen generado mal es lo primero que
   elegiría un muestreo por distancia máxima. De los 100 000, todos legibles, ninguno pasa de
   z = 8.1, así que no sale ninguno. A z = 6 salían 31; los 8 que se miraron (cortes medios) tenían
   anatomía plausible, casi todos con ventrículos muy grandes, que es la cola que se busca.
2. **PCA** de las miniaturas (32 componentes, sin blanquear), ajustada sobre 20 000 sujetos.
3. **Evaluación:** el más cercano a cada centroide de un k-means de 12 grupos; representativos de
   los modos, no extremos.
4. **Entrenamiento, regla `mezcla`:** la mitad de cada lista por k-center voraz (cubre las colas)
   y la otra mitad sorteada entre el resto (sigue la densidad).

Cobertura, la distancia en el espacio PCA de cada sujeto del corpus a su elegido más cercano,
medida el 2026-10-08 sobre los 100 000 con los defaults (`--z 10 --n 2000 8000`), que son los
que arman las listas de esta carpeta (crudo en `docs/mediciones/seleccion-ldm100k3d.txt`):

| N | regla | mediana | p99 | máx |
|---|---|---|---|---|
| 2 000 | voraz | 4.621 | 5.363 | 5.529 |
| 2 000 | sorteo | 3.841 | 5.593 | 8.305 |
| 2 000 | **mezcla** | 3.996 | 5.374 | 5.999 |
| 8 000 | voraz | 4.024 | 4.597 | 4.654 |
| 8 000 | sorteo | 3.505 | 5.021 | 8.378 |
| 8 000 | **mezcla** | 3.606 | 4.734 | 5.081 |

La mezcla queda cerca del sorteo en la mediana y cerca del voraz en la cola: la de 2 000 cubre las
colas mejor que un sorteo de 8 000 (máximo 5.999 contra 8.378). Con `--z 6 --n 500 2000 8000`,
como se corrió `scripts/selecciona_ldm100k.py` (`docs/mediciones/variedad-ldm100k.txt`), las
listas son otras y la cobertura casi la misma: máximo 5.987 la mezcla de 2 000 contra 7.921 el
sorteo de 8 000.

Todo sale de un generador con semilla fija (`--seed 20261008`) consumido en orden, así que la lista
de un N depende de la lista `--n` completa: `--n 2000 8000` y `--n 500 2000 8000` dan listas de
2 000 distintas. Las listas de esta carpeta salen de los defaults de `selecciona` (paso 2 de
abajo). Las de 2 000 y de 8 000 no están anidadas: comparten la mitad voraz, no la sorteada.

**La que vale es la lista versionada, no el comando.** Repetida el 2026-10-08 con BLAS en 16 y en
32 hilos, la selección da la misma lista de evaluación y la misma de 2 000; la de 8 000 tiene los
mismos sujetos, pero dos pares de vecinos del voraz en otro orden (sus coordenadas PCA difieren en
≤ 2.9e-6 según los hilos, y el voraz desempata distinto). El orden importa en `train3d.py`, que arma
`PatchDataset3D` con los archivos en el orden de `--train-list` y sortea el volumen por posición:
con la lista permutada una corrida ve otros parches. Por eso para
entrenar basta el paso 3 de abajo; los pasos 1 y 2 rehacen la selección, y el 2 va a otra
`--salida`.

## Cómo se arma

```bash
# 1 y 2, sólo para rehacer la selección: las listas ya están versionadas
# 1. 25.5 min con 8 procesos
python datasets/cache_ldm100k3d.py escanea --salida datasets/ldm100k3d/escaneo
# 2. 9 min con BLAS en 32 hilos; a otra --salida: la de defecto es esta y pisaría las listas
python datasets/cache_ldm100k3d.py selecciona --escaneo datasets/ldm100k3d/escaneo \
    --n 2000 8000 --n-eval 12 --salida /tmp/seleccion-ldm100k3d
# 3. volúmenes normalizados y recortados, de las listas versionadas
python datasets/cache_ldm100k3d.py cachea --lista lista_eval.txt --split val
python datasets/cache_ldm100k3d.py cachea --lista lista_train_2000.txt --split train
```

`--rawdata` apunta por defecto a la copia de LDM_100k en la máquina del autor. Necesita `nibabel`,
que ya está en las dependencias.

### Máscaras de cerebro

Sólo para `train3d.py --mascara-cerebro`, que viene apagada. Las arma
[`../mascaras_ldm100k3d.py`](../mascaras_ldm100k3d.py) en tres pasos por lista, después de
cachearla: `deepbet` y `hdbet`, cada uno en el Python de su propio entorno (cómo armarlos, en el
docstring del script), dejan su máscara en `mascaras-nifti/`; `combina`, en el `.venv`, las
interseca, las recorta con la misma caja que la cache y escribe `mascaras/<split>/`, verificando
que cada volumen cacheado sea, bit a bit, el que sale del NIfTI con esa caja. Los comandos y los
tiempos (0.67 s por cerebro en GPU), en la sección «Entrenar en 3D» del
[README principal](../../README.md). Las de `val` no las usa el entrenamiento: se generaron el
2026-10-12 para verificar la alineación (`docs/mediciones/mascaras-val-ldm100k3d.txt`).

## Cómo se entrena sobre esto

`train3d.py` arma `PatchDataset3D(archivos, ...)` con los archivos en el orden de `--train-list`
(por defecto `lista_train_2000.txt`): sortea parches de 48³ con al menos 35 % de tejido (con
`--mascara-cerebro`, cubos enteros dentro del cerebro según `mascaras/train/`) y una de las 48
simetrías del cubo. `PatchDataset3D.from_root` también existe, pero ordena por nombre e
ignora el orden de la lista. Y
`VolumeDataset3D.from_root("datasets/ldm100k3d", "val", names=<lista_eval.txt>)` da un recorte
fijo de 128³ centrado en la cabeza por volumen de evaluación. El entrenador es `train3d.py`.
