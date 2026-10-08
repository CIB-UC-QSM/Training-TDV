# precision/rapido-seed1

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | rapido |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 1 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.002 dB** en la iteración 2,000
- Última medida: nan dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.749 | 0.10407 | 0.04699 |
| 2,000 | 31.002 | 0.10222 | 0.02040 |
| 3,000 | nan | nan | nan |
| 4,000 | nan | nan | nan |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **nan**, máxima aceptada nan
- `‖g‖` máxima vista: nan
- Por encima de 1e+04: **123 de 4,000 (3.08 %)**
- Primera vez por encima de 1e+04: iteración **2,149**
- No finitas (NaN o inf): 1676
- `T` final nan · `lambda` final nan

