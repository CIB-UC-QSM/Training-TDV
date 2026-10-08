# precision/intermedio-seed7

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | intermedio |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 7 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.127 dB** en la iteración 4,000
- Última medida: 31.127 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.710 | 0.10390 | 0.05307 |
| 2,000 | 30.963 | 0.10219 | 0.03792 |
| 3,000 | 31.066 | 0.10050 | 0.03538 |
| 4,000 | 31.127 | 0.09863 | 0.03811 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **61.68**, máxima aceptada 1937
- `‖g‖` máxima vista: 1937
- Por encima de 1e+04: **0 de 4,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.09864 · `lambda` final 0.03809

