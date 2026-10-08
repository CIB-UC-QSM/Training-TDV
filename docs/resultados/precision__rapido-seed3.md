# precision/rapido-seed3

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | rapido |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 3 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 30.982 dB** en la iteración 2,000
- Última medida: 16.713 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.692 | 0.10440 | 0.04187 |
| 2,000 | 30.982 | 0.10291 | 0.01243 |
| 3,000 | 15.032 | 0.06106 | 0.05808 |
| 4,000 | 16.713 | 0.06086 | 0.05917 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **182.7**, máxima aceptada 1.451e+10
- `‖g‖` máxima vista: 1.451e+10
- Por encima de 1e+04: **1,606 de 4,000 (40.15 %)**
- Primera vez por encima de 1e+04: iteración **2,349**
- No finitas (NaN o inf): 0
- `T` final 0.06086 · `lambda` final 0.05917

