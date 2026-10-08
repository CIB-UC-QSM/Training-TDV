# precision/intermedio-seed3

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | intermedio |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 3 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.178 dB** en la iteración 4,000
- Última medida: 31.178 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.689 | 0.10451 | 0.05383 |
| 2,000 | 30.965 | 0.10289 | 0.03167 |
| 3,000 | 31.100 | 0.10167 | 0.02603 |
| 4,000 | 31.178 | 0.10089 | 0.02629 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **53.91**, máxima aceptada 1016
- `‖g‖` máxima vista: 1016
- Por encima de 1e+04: **0 de 4,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.10087 · `lambda` final 0.02631

