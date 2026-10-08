# precision/rapido-seed8

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | rapido |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 8 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 30.972 dB** en la iteración 2,000
- Última medida: 9.835 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.733 | 0.10556 | 0.04668 |
| 2,000 | 30.972 | 0.10456 | 0.01790 |
| 3,000 | 2.063 | 0.10600 | 0.03596 |
| 4,000 | 9.835 | 0.10533 | 0.04178 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **161.5**, máxima aceptada 1.139e+11
- `‖g‖` máxima vista: 1.139e+11
- Por encima de 1e+04: **1,644 de 4,000 (41.10 %)**
- Primera vez por encima de 1e+04: iteración **2,356**
- No finitas (NaN o inf): 0
- `T` final 0.10533 · `lambda` final 0.04178

