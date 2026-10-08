# precision/intermedio-seed10

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | intermedio |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 10 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.158 dB** en la iteración 4,000
- Última medida: 31.158 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.652 | 0.10528 | 0.05152 |
| 2,000 | 30.937 | 0.10307 | 0.02982 |
| 3,000 | 31.080 | 0.10104 | 0.02580 |
| 4,000 | 31.158 | 0.09897 | 0.02870 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **56.4**, máxima aceptada 1025
- `‖g‖` máxima vista: 1025
- Por encima de 1e+04: **0 de 4,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.09897 · `lambda` final 0.02871

