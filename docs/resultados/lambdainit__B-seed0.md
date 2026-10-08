# lambdainit/B-seed0

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 0 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 30.738 dB** en la iteración 1,000
- Última medida: -30.190 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.738 | 0.10365 | 0.05291 |
| 2,000 | 9.053 | 0.10817 | 0.04545 |
| 3,000 | -29.662 | 0.07697 | 0.10929 |
| 4,000 | -30.190 | 0.07725 | 0.11041 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **3.652e+06**, máxima aceptada 2.454e+16
- `‖g‖` máxima vista: 2.454e+16
- Por encima de 1e+04: **2,042 de 4,000 (51.05 %)**
- Primera vez por encima de 1e+04: iteración **1,958**
- No finitas (NaN o inf): 0
- `T` final 0.07725 · `lambda` final 0.11041

