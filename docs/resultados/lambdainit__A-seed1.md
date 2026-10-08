# lambdainit/A-seed1

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.0252982 |
| `sigma` | 25 |
| `seed` | 1 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 30.728 dB** en la iteración 1,000
- Última medida: 30.605 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.728 | 0.02873 | 0.00058 |
| 2,000 | 29.633 | 0.01095 | 0.00443 |
| 3,000 | 30.411 | 0.01383 | 0.00000 |
| 4,000 | 30.605 | 0.01388 | 0.00000 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **152.9**, máxima aceptada 1.864e+05
- `‖g‖` máxima vista: 1.864e+05
- Por encima de 1e+04: **5 de 4,000 (0.12 %)**
- Primera vez por encima de 1e+04: iteración **1,729**
- No finitas (NaN o inf): 0
- `T` final 0.01389 · `lambda` final 0.00000

