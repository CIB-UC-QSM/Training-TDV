# precision/intermedio-seed6

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | intermedio |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 6 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 30.943 dB** en la iteración 2,000
- Última medida: 15.530 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.594 | 0.10455 | 0.04935 |
| 2,000 | 30.943 | 0.10317 | 0.03208 |
| 3,000 | 11.682 | 0.11141 | 0.04485 |
| 4,000 | 15.530 | 0.10719 | 0.06700 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **230.5**, máxima aceptada 2.058e+09
- `‖g‖` máxima vista: 2.058e+09
- Por encima de 1e+04: **1,229 de 4,000 (30.73 %)**
- Primera vez por encima de 1e+04: iteración **2,379**
- No finitas (NaN o inf): 0
- `T` final 0.10719 · `lambda` final 0.06699

