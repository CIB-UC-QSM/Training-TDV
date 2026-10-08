# lambdainit/B-seed2

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 2 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 30.985 dB** en la iteración 2,000
- Última medida: 12.780 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.744 | 0.10272 | 0.05459 |
| 2,000 | 30.985 | 0.10170 | 0.04534 |
| 3,000 | 12.014 | 0.10528 | 0.05795 |
| 4,000 | 12.780 | 0.10520 | 0.05825 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **185.4**, máxima aceptada 1.177e+10
- `‖g‖` máxima vista: 1.177e+10
- Por encima de 1e+04: **1,647 de 4,000 (41.17 %)**
- Primera vez por encima de 1e+04: iteración **2,350**
- No finitas (NaN o inf): 0
- `T` final 0.10520 · `lambda` final 0.05825

