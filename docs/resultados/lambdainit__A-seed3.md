# lambdainit/A-seed3

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.0252982 |
| `sigma` | 25 |
| `seed` | 3 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 30.991 dB** en la iteración 2,000
- Última medida: nan dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.708 | 0.02969 | 0.00057 |
| 2,000 | 30.991 | 0.02905 | 0.00767 |
| 3,000 | nan | nan | nan |
| 4,000 | nan | nan | nan |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **nan**, máxima aceptada nan
- `‖g‖` máxima vista: nan
- Por encima de 1e+04: **546 de 4,000 (13.65 %)**
- Primera vez por encima de 1e+04: iteración **2,007**
- No finitas (NaN o inf): 1313
- `T` final nan · `lambda` final nan

