# precision/rapido-seed4

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | rapido |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 4 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.078 dB** en la iteración 4,000
- Última medida: 31.078 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.116 | 0.10015 | 0.03993 |
| 2,000 | 30.901 | 0.10018 | 0.01382 |
| 3,000 | 31.013 | 0.09873 | 0.00794 |
| 4,000 | 31.078 | 0.09728 | 0.00791 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **57.51**, máxima aceptada 5473
- `‖g‖` máxima vista: 5473
- Por encima de 1e+04: **0 de 4,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.09726 · `lambda` final 0.00794

