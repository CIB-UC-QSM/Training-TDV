# precision/rapido-seed5

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | rapido |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 5 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.152 dB** en la iteración 4,000
- Última medida: 31.152 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.736 | 0.10613 | 0.03817 |
| 2,000 | 30.995 | 0.10487 | 0.01102 |
| 3,000 | 31.068 | 0.10309 | 0.00765 |
| 4,000 | 31.152 | 0.10158 | 0.01007 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **51.83**, máxima aceptada 1170
- `‖g‖` máxima vista: 1170
- Por encima de 1e+04: **0 de 4,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.10155 · `lambda` final 0.01009

