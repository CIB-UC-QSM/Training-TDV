# clip/base-seed3

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.0252982 |
| `sigma` | 25 |
| `seed` | 3 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.070 dB** en la iteración 4,000
- Última medida: 31.070 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.451 | 0.02600 | 0.00000 |
| 2,000 | 30.794 | 0.02746 | 0.00572 |
| 3,000 | 30.963 | 0.02759 | 0.01666 |
| 4,000 | 31.070 | 0.02764 | 0.03054 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **87.65**, máxima aceptada 9334
- `‖g‖` máxima vista: 9334
- Por encima de 1e+04: **0 de 4,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.02764 · `lambda` final 0.03052

