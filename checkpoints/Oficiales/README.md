# Checkpoints oficiales

Pesos publicados por los autores de TDV en [VLOGroup/tdv](https://github.com/VLOGroup/tdv). Se
versionan (1.6 MB cada uno) porque los tests de reproducción dependen de ellos: sin el oráculo no
hay forma de verificar que este port es equivalente al original. Los entrenados en este
repositorio están en [`../Entrenados/`](../Entrenados/README.md).

| archivo | modelo | `T` aprendido | `λ` aprendido | PSNR con este port, σ = 25 |
|---|---|---|---|---|
| `tdv3-3-25-f32-color.pth` | TDV33, RGB | 0.024727 | 0.104522 | 31.537 dB en CBSD68 (paper 31.53) |
| `tdv3-3-25-f32-gray.pth` | TDV33, gris | 0.017223 | 0.216907 | sin número archivado (paper 29.37 en BSD68) |

El 31.537 sale de `docs/presentacion/salida/metricas_sigma_oficial.json`. Del gris sólo queda el
test: `tests/test_reproduction.py` exige el 29.37 publicado con tolerancia 0.02 dB.

Cada archivo trae `config` (la arquitectura, con la que `evaluate.load_vnet` reconstruye el modelo)
y `model` (el `state_dict`). Además de los parámetros, el `state_dict` trae 44 buffers `mask` y 12
`blur` que no se entrenan: por eso sumar sus elementos no da los 387 970 parámetros del paper
(los del color; el gris tiene 387 394). Los de `../Entrenados/` y los que `train.py` escribe en
`runs/` tienen las mismas dos claves más `iteration` y `psnr` —y los de `runs/`, `optimizer`— y
cargan igual.
