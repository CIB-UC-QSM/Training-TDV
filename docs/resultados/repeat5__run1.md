# repeat5/run1

**Estado:** terminada (deducido)  
**De dónde se sabe:** 10000 filas = las 10000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 10000 |
| `t_init` | 0.0632456 |
| `sigma` | 25 |
| `seed` | 7 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 30.395 dB** en la iteración 2,000
- Última medida: 27.756 dB en la 10,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 2,000 | 30.395 | 0.05382 | 0.04071 |
| 4,000 | 29.652 | 0.03821 | 0.04168 |
| 6,000 | 23.870 | 0.02441 | 0.09636 |
| 8,000 | 25.806 | 0.02390 | 0.09724 |
| 10,000 | 27.756 | 0.01192 | 0.10033 |

## Salud del gradiente

- Iteraciones con traza: **10,000**
- `‖g‖` mediana de las aceptadas: **528.7**, máxima aceptada 5.635e+07
- `‖g‖` máxima vista: 5.635e+07
- Por encima de 1e+04: **1,629 de 10,000 (16.29 %)**
- Primera vez por encima de 1e+04: iteración **1,390**
- No finitas (NaN o inf): 0
- `T` final 0.01203 · `lambda` final 0.10031

