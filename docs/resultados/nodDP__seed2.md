# nodDP/seed2

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.0252982 |
| `sigma` | 25 |
| `seed` | 2 |

## PSNR en CBSD68

- **Mejor: 31.175 dB** en la iteración 4,000
- Última medida: 31.175 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.708 | 0.02958 | 0.00091 |
| 2,000 | 30.989 | 0.02939 | 0.00844 |
| 3,000 | 31.090 | 0.02904 | 0.01826 |
| 4,000 | 31.175 | 0.02892 | 0.02738 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **79.96**, máxima aceptada 2556
- `‖g‖` máxima vista: 2556
- Por encima de 1e+04: **0 de 4,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.02890 · `lambda` final 0.02738

