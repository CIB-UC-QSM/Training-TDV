# lambdainit/B-seed1

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 1 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.165 dB** en la iteración 4,000
- Última medida: 31.165 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.747 | 0.10371 | 0.05932 |
| 2,000 | 31.000 | 0.10191 | 0.03771 |
| 3,000 | 31.071 | 0.10047 | 0.03103 |
| 4,000 | 31.165 | 0.09932 | 0.03463 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **53.91**, máxima aceptada 1004
- `‖g‖` máxima vista: 1004
- Por encima de 1e+04: **0 de 4,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.09929 · `lambda` final 0.03464

