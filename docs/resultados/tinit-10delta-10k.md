# tinit-10delta-10k

**Estado:** terminada  
**De dónde se sabe:** `estado.json`

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 10000 |
| `t_init` | 0.126491 |
| `sigma` | 25 |
| `seed` | 0 |
| `clip` | inf |
| `guard` | 5 |
| `guard_abs` | 10000 |
| `guard_ventana` | 100 |
| `guard_seguidos` | 25 |
| `guard_amp` | inf |

## PSNR en CBSD68

- **Mejor: 31.352 dB** en la iteración 10,000
- Última medida: 31.352 dB en la 10,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 2,000 | 30.998 | 0.12986 | 0.02687 |
| 4,000 | 31.176 | 0.12616 | 0.02693 |
| 6,000 | 31.257 | 0.12221 | 0.03167 |
| 8,000 | 31.301 | 0.11885 | 0.03712 |
| 10,000 | 31.352 | 0.11605 | 0.04294 |

## Salud del gradiente

- Iteraciones con traza: **10,000**
- `‖g‖` mediana de las aceptadas: **30.89**, máxima aceptada 868.1
- `‖g‖` máxima vista: 4428
- Por encima de 1e+04: **0 de 10,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.11602 · `lambda` final 0.04296

## Disyuntor de lotes

- Lotes rechazados: **4** de 10,000 (0.040 %)
- Rechazos seguidos, máximo: **1**
- Tandas de rechazo: 4
- Rebobinados: **0**

Las primeras tandas:

| desde la it | largo |
|---|---|
| 2,184 | 1 |
| 2,473 | 1 |
| 5,960 | 1 |
| 6,725 | 1 |

