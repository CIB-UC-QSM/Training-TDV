# nodDP/seed5

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.0252982 |
| `sigma` | 25 |
| `seed` | 5 |

## PSNR en CBSD68

- **Mejor: 31.163 dB** en la iteración 4,000
- Última medida: 31.163 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.716 | 0.02827 | 0.00211 |
| 2,000 | 30.993 | 0.02744 | 0.01053 |
| 3,000 | 31.104 | 0.02684 | 0.01730 |
| 4,000 | 31.163 | 0.02619 | 0.02153 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **78.67**, máxima aceptada 2204
- `‖g‖` máxima vista: 2204
- Por encima de 1e+04: **0 de 4,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.02615 · `lambda` final 0.02155

