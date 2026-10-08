# precision/intermedio-seed11

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | intermedio |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 11 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 30.742 dB** en la iteración 1,000
- Última medida: -33.806 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.742 | 0.10442 | 0.05617 |
| 2,000 | -33.896 | 0.09872 | 0.07455 |
| 3,000 | -33.912 | 0.09872 | 0.07478 |
| 4,000 | -33.806 | 0.09861 | 0.07530 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **1.662e+10**, máxima aceptada 2.234e+17
- `‖g‖` máxima vista: 2.234e+17
- Por encima de 1e+04: **2,446 de 4,000 (61.15 %)**
- Primera vez por encima de 1e+04: iteración **1,376**
- No finitas (NaN o inf): 0
- `T` final 0.09861 · `lambda` final 0.07530

