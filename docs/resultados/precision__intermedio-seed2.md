# precision/intermedio-seed2

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | intermedio |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 2 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.157 dB** en la iteración 4,000
- Última medida: 31.157 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.680 | 0.10273 | 0.05069 |
| 2,000 | 30.971 | 0.10171 | 0.04047 |
| 3,000 | 31.080 | 0.10062 | 0.03958 |
| 4,000 | 31.157 | 0.09946 | 0.04107 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **52.06**, máxima aceptada 2825
- `‖g‖` máxima vista: 2825
- Por encima de 1e+04: **0 de 4,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.09944 · `lambda` final 0.04108

