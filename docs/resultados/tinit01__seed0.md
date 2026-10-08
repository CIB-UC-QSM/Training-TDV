# tinit01/seed0

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 0 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.188 dB** en la iteración 4,000
- Última medida: 31.188 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.730 | 0.10377 | 0.00000 |
| 2,000 | 30.984 | 0.10216 | 0.00229 |
| 3,000 | 31.111 | 0.10010 | 0.00899 |
| 4,000 | 31.188 | 0.09843 | 0.02123 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **52.71**, máxima aceptada 835.1
- `‖g‖` máxima vista: 835.1
- Por encima de 1e+04: **0 de 4,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.09844 · `lambda` final 0.02121

