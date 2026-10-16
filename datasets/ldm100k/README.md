# ldm100k

Cortes 2D de T1w sintéticas sacados de `LDM_100k` (Pinaya et al., 2022): 100 000 volúmenes de
cerebro generados por un modelo de difusión latente entrenado sobre UK Biobank, en BIDS,
`sub-*/anat/sub-*_T1w.nii.gz`, uint16, 160×224×160 a 1 mm isotrópico.

**No es QSM.** Es T1w: sirve para medir el cambio de dominio «imagen natural → corte de cerebro»,
no «imagen natural → mapa de susceptibilidad». Nada de lo que se entrene acá se traslada a QSM sin
volver a medirlo. El checkpoint `../tdv/checkpoints/checkpoint_qsm_rapido.pth` salió de acá, pese
al nombre: es `runs/t1-rapido-30k/best.pth`.

Las imágenes **no se versionan** (3.1 GB). Lo versionado de esta carpeta es `lista_eval.txt` y
este archivo; el script que arma el cache está un nivel arriba, en
[`../cache_ldm100k.py`](../cache_ldm100k.py).

## Contenido

| ruta | qué es |
|---|---|
| `images/train/*.npy` | 24 000 cortes: 8 000 sujetos × un corte por plano (`_sag`, `_cor`, `_axi`) |
| `images/val/*.npy` | 68 cortes axiales de 68 sujetos disjuntos de los de entrenamiento |
| `lista_eval.txt` | los 68 nombres de `val`, uno por línea: lo que consume `train.py --eval-list` |

Cada `.npy` es float32 en [0, 1], normalizado por el percentil 99.5 de los voxels positivos de su
volumen. Los cortes salen del 60 % central del eje, con al menos 35 % de tejido, y todos miden
93 píxeles o más por lado (`PATCH_SIZE` de `train.py`). 68 cortes de evaluación igualan el
cardinal de CBSD68, así que la evaluación cuesta lo mismo que en BSDS.

## Cómo se arma

```bash
uv run python datasets/cache_ldm100k.py --rawdata <LDM_100k>/rawdata --salida datasets/ldm100k \
    --sujetos-train 8000
```

El default de `--sujetos-train` es 4 000; el cache actual tiene 8 000 sujetos de entrenamiento. La
semilla del sorteo de sujetos es fija (`--seed 20261007`), así que el mismo comando da los mismos
cortes. Necesita `nibabel`, que ya está en las dependencias.

## Cómo se entrena sobre esto

En gris (`--channels 1`), con los splits y la lista de este dataset:

```bash
uv run train.py --out runs/t1 --channels 1 \
    --data-root datasets/ldm100k --train-splits train --eval-split val \
    --eval-list datasets/ldm100k/lista_eval.txt
```
