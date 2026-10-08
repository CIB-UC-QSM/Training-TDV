# lambdainit/A-seed5

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.0252982 |
| `sigma` | 25 |
| `seed` | 5 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 30.993 dB** en la iteración 2,000
- Última medida: 15.484 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.589 | 0.02700 | 0.00000 |
| 2,000 | 30.993 | 0.02661 | 0.00617 |
| 3,000 | 14.486 | 0.01718 | 0.01663 |
| 4,000 | 15.484 | 0.01723 | 0.01688 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **292.4**, máxima aceptada 4.06e+10
- `‖g‖` máxima vista: 4.06e+10
- Por encima de 1e+04: **1,659 de 4,000 (41.48 %)**
- Primera vez por encima de 1e+04: iteración **2,338**
- No finitas (NaN o inf): 0
- `T` final 0.01723 · `lambda` final 0.01688

