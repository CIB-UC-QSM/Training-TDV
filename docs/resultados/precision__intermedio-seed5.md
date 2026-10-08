# precision/intermedio-seed5

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | intermedio |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 5 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 30.619 dB** en la iteración 1,000
- Última medida: 6.427 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.619 | 0.10339 | 0.05193 |
| 2,000 | 1.499 | 0.09317 | 0.04545 |
| 3,000 | 3.353 | 0.09311 | 0.04615 |
| 4,000 | 6.427 | 0.09305 | 0.04680 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **3.069e+05**, máxima aceptada 1.092e+12
- `‖g‖` máxima vista: 1.092e+12
- Por encima de 1e+04: **2,142 de 4,000 (53.55 %)**
- Primera vez por encima de 1e+04: iteración **1,857**
- No finitas (NaN o inf): 0
- `T` final 0.09305 · `lambda` final 0.04680

