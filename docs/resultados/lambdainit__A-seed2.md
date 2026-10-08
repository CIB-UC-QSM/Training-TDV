# lambdainit/A-seed2

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.0252982 |
| `sigma` | 25 |
| `seed` | 2 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.003 dB** en la iteración 4,000
- Última medida: 31.003 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.664 | 0.02982 | 0.00122 |
| 2,000 | 30.716 | 0.02198 | 0.00000 |
| 3,000 | 30.954 | 0.02339 | 0.00172 |
| 4,000 | 31.003 | 0.02323 | 0.01013 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **94.36**, máxima aceptada 1.252e+04
- `‖g‖` máxima vista: 1.252e+04
- Por encima de 1e+04: **1 de 4,000 (0.03 %)**
- Primera vez por encima de 1e+04: iteración **1,769**
- No finitas (NaN o inf): 0
- `T` final 0.02325 · `lambda` final 0.01008

