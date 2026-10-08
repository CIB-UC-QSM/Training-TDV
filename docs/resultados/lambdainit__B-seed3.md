# lambdainit/B-seed3

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

- **Mejor: 31.167 dB** en la iteración 4,000
- Última medida: 31.167 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.659 | 0.10429 | 0.05631 |
| 2,000 | 30.980 | 0.10288 | 0.03241 |
| 3,000 | 31.100 | 0.10145 | 0.02491 |
| 4,000 | 31.167 | 0.10051 | 0.02696 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **56.28**, máxima aceptada 4099
- `‖g‖` máxima vista: 4099
- Por encima de 1e+04: **0 de 4,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.10047 · `lambda` final 0.02699

