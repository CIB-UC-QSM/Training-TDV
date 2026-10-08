# clip/base-seed1

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
- Última medida: 4.525 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.728 | 0.02869 | 0.00057 |
| 2,000 | 4.522 | 0.03558 | 0.04722 |
| 3,000 | 4.528 | 0.03558 | 0.04722 |
| 4,000 | 4.525 | 0.03558 | 0.04722 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **1.85e+06**, máxima aceptada 2.176e+16
- `‖g‖` máxima vista: 2.176e+16
- Por encima de 1e+04: **2,272 de 4,000 (56.80 %)**
- Primera vez por encima de 1e+04: iteración **1,726**
- No finitas (NaN o inf): 0
- `T` final 0.03558 · `lambda` final 0.04722

