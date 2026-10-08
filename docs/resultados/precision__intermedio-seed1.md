# precision/intermedio-seed1

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | intermedio |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 1 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.161 dB** en la iteración 4,000
- Última medida: 31.161 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.749 | 0.10383 | 0.05726 |
| 2,000 | 31.001 | 0.10207 | 0.03560 |
| 3,000 | 31.078 | 0.10065 | 0.02830 |
| 4,000 | 31.161 | 0.09938 | 0.03064 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **53.51**, máxima aceptada 1003
- `‖g‖` máxima vista: 1003
- Por encima de 1e+04: **0 de 4,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.09934 · `lambda` final 0.03067

