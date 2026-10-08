# nodDP/seed1

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

## PSNR en CBSD68

- **Mejor: 30.726 dB** en la iteración 1,000
- Última medida: 20.744 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.726 | 0.02870 | 0.00100 |
| 2,000 | 30.105 | 0.01576 | 0.00568 |
| 3,000 | 20.980 | 0.00004 | 0.04478 |
| 4,000 | 20.744 | 0.00013 | 0.06971 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **527.4**, máxima aceptada 6.997e+08
- `‖g‖` máxima vista: 6.997e+08
- Por encima de 1e+04: **1,618 de 4,000 (40.45 %)**
- Primera vez por encima de 1e+04: iteración **1,448**
- No finitas (NaN o inf): 0
- `T` final 0.00013 · `lambda` final 0.06971

