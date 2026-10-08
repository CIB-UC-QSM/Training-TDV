# clip/clip-seed1

**Estado:** incompleta (deducido) — en vuelo o matada, por filas no se distingue  
**De dónde se sabe:** 1000 filas de 4000

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.0252982 |
| `sigma` | 25 |
| `seed` | 1 |
| `clip` | 10000 |

## PSNR en CBSD68

- **Mejor: 30.728 dB** en la iteración 1,000
- Última medida: 30.728 dB en la 1,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.728 | 0.02869 | 0.00057 |

## Salud del gradiente

- Iteraciones con traza: **1,000**
- `‖g‖` mediana de las aceptadas: **172**, máxima aceptada 1950
- `‖g‖` máxima vista: 1950
- Por encima de 1e+04: **0 de 1,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.02869 · `lambda` final 0.00056

