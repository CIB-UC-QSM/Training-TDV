# t8delta-100k-corte-s0

**Estado:** MATADA — sin motivo anotado  
**De dónde se sabe:** `estado.json`

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 100000 |
| `t_init` | 0.101193 |
| `sigma` | 25 |
| `seed` | 0 |
| `clip` | inf |
| `guard` | 5 |
| `guard_abs` | 10000 |
| `guard_ventana` | 100 |
| `guard_seguidos` | 25 |
| `guard_amp` | inf |

## PSNR en CBSD68

- **Mejor: 31.419 dB** en la iteración 16,000
- Última medida: 31.393 dB en la 18,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 2,000 | 30.982 | 0.10312 | 0.02871 |
| 4,000 | 31.191 | 0.09987 | 0.02696 |
| 6,000 | 31.271 | 0.09632 | 0.03590 |
| 8,000 | 31.314 | 0.09307 | 0.04469 |
| 10,000 | 31.337 | 0.09021 | 0.05400 |
| 12,000 | 31.376 | 0.08788 | 0.05962 |
| 14,000 | 31.401 | 0.08578 | 0.06544 |
| 16,000 | 31.419 | 0.08387 | 0.06948 |
| 18,000 | 31.393 | 0.08199 | 0.07068 |

## Salud del gradiente

- Iteraciones con traza: **18,000**
- `‖g‖` mediana de las aceptadas: **24.44**, máxima aceptada 906.9
- `‖g‖` máxima vista: 3569
- Por encima de 1e+04: **0 de 18,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.08199 · `lambda` final 0.07070

## Disyuntor de lotes

- Lotes rechazados: **11** de 18,000 (0.061 %)
- Rechazos seguidos, máximo: **6**
- Tandas de rechazo: 6
- Rebobinados: **0**

Las primeras tandas:

| desde la it | largo |
|---|---|
| 764 | 6 |
| 5,254 | 1 |
| 9,156 | 1 |
| 9,405 | 1 |
| 11,568 | 1 |
| 17,494 | 1 |

