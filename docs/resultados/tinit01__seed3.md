# tinit01/seed3

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 3 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 30.982 dB** en la iteración 2,000
- Última medida: 3.147 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.711 | 0.10393 | 0.00000 |
| 2,000 | 30.982 | 0.10277 | 0.00302 |
| 3,000 | 8.646 | 0.08308 | 0.03746 |
| 4,000 | 3.147 | 0.08254 | 0.04037 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **225.7**, máxima aceptada 2.938e+11
- `‖g‖` máxima vista: 2.938e+11
- Por encima de 1e+04: **1,649 de 4,000 (41.23 %)**
- Primera vez por encima de 1e+04: iteración **2,349**
- No finitas (NaN o inf): 0
- `T` final 0.08254 · `lambda` final 0.04037

