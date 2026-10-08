# precision/rapido-seed6

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | rapido |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 6 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.155 dB** en la iteración 4,000
- Última medida: 31.155 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.607 | 0.10485 | 0.04278 |
| 2,000 | 30.962 | 0.10337 | 0.01714 |
| 3,000 | 31.051 | 0.10221 | 0.01117 |
| 4,000 | 31.155 | 0.10075 | 0.01247 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **57.88**, máxima aceptada 1188
- `‖g‖` máxima vista: 1188
- Por encima de 1e+04: **0 de 4,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.10073 · `lambda` final 0.01251

