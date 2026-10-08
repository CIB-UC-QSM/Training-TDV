# precision/rapido-seed0

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | rapido |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 0 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.190 dB** en la iteración 4,000
- Última medida: 31.190 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.708 | 0.10364 | 0.04024 |
| 2,000 | 30.987 | 0.10206 | 0.01363 |
| 3,000 | 31.121 | 0.10037 | 0.00924 |
| 4,000 | 31.190 | 0.09855 | 0.01287 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **47.16**, máxima aceptada 824.2
- `‖g‖` máxima vista: 824.2
- Por encima de 1e+04: **0 de 4,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.09855 · `lambda` final 0.01285

