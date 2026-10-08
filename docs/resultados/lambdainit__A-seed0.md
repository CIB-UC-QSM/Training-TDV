# lambdainit/A-seed0

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.0252982 |
| `sigma` | 25 |
| `seed` | 0 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 30.992 dB** en la iteración 2,000
- Última medida: 21.459 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.724 | 0.02913 | 0.00044 |
| 2,000 | 30.992 | 0.02825 | 0.00465 |
| 3,000 | 18.686 | 0.00423 | 0.07829 |
| 4,000 | 21.459 | 0.00412 | 0.07844 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **171.9**, máxima aceptada 7.586e+08
- `‖g‖` máxima vista: 7.586e+08
- Por encima de 1e+04: **552 de 4,000 (13.80 %)**
- Primera vez por encima de 1e+04: iteración **2,660**
- No finitas (NaN o inf): 0
- `T` final 0.00412 · `lambda` final 0.07845

