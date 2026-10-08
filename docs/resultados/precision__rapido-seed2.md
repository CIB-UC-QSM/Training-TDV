# precision/rapido-seed2

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | rapido |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 2 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.005 dB** en la iteración 2,000
- Última medida: 30.937 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.746 | 0.10323 | 0.04301 |
| 2,000 | 31.005 | 0.10194 | 0.02352 |
| 3,000 | 30.639 | 0.07760 | 0.00914 |
| 4,000 | 30.937 | 0.07632 | 0.00657 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **59.63**, máxima aceptada 6113
- `‖g‖` máxima vista: 6113
- Por encima de 1e+04: **0 de 4,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.07632 · `lambda` final 0.00656

