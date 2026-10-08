# precision/rapido-seed9

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | rapido |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 9 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 30.695 dB** en la iteración 1,000
- Última medida: nan dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.695 | 0.10557 | 0.03676 |
| 2,000 | nan | nan | nan |
| 3,000 | nan | nan | nan |
| 4,000 | nan | nan | nan |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **nan**, máxima aceptada nan
- `‖g‖` máxima vista: nan
- Por encima de 1e+04: **260 de 4,000 (6.50 %)**
- Primera vez por encima de 1e+04: iteración **1,634**
- No finitas (NaN o inf): 2106
- `T` final nan · `lambda` final nan

