# precision/rapido-seed11

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | rapido |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 11 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.138 dB** en la iteración 4,000
- Última medida: 31.138 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.734 | 0.10442 | 0.04747 |
| 2,000 | 30.934 | 0.09787 | 0.01141 |
| 3,000 | 31.051 | 0.09722 | 0.00571 |
| 4,000 | 31.138 | 0.09534 | 0.01018 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **51.06**, máxima aceptada 2990
- `‖g‖` máxima vista: 2990
- Por encima de 1e+04: **0 de 4,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.09532 · `lambda` final 0.01020

