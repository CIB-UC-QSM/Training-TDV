# precision/rapido-seed7

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | rapido |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 7 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.139 dB** en la iteración 4,000
- Última medida: 31.139 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.693 | 0.10418 | 0.04306 |
| 2,000 | 30.965 | 0.10241 | 0.02168 |
| 3,000 | 31.047 | 0.10051 | 0.01635 |
| 4,000 | 31.139 | 0.09881 | 0.01963 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **56.3**, máxima aceptada 3079
- `‖g‖` máxima vista: 3079
- Por encima de 1e+04: **0 de 4,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.09881 · `lambda` final 0.01961

