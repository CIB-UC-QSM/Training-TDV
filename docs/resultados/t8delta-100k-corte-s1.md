# t8delta-100k-corte-s1

**Estado:** MATADA — sin motivo anotado  
**De dónde se sabe:** `estado.json`

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 100000 |
| `t_init` | 0.101193 |
| `sigma` | 25 |
| `seed` | 1 |
| `clip` | inf |
| `guard` | 5 |
| `guard_abs` | 10000 |
| `guard_ventana` | 100 |
| `guard_seguidos` | 25 |
| `guard_amp` | inf |

## PSNR en CBSD68

- **Mejor: 31.419 dB** en la iteración 18,000
- Última medida: 31.419 dB en la 18,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 2,000 | 31.003 | 0.10312 | 0.03665 |
| 4,000 | 31.172 | 0.10068 | 0.03401 |
| 6,000 | 31.005 | 0.09772 | 0.03825 |
| 8,000 | 31.310 | 0.09568 | 0.04223 |
| 10,000 | 31.340 | 0.09355 | 0.04699 |
| 12,000 | 31.377 | 0.09155 | 0.05151 |
| 14,000 | 31.401 | 0.08957 | 0.05811 |
| 16,000 | 31.411 | 0.08782 | 0.06385 |
| 18,000 | 31.419 | 0.08615 | 0.07138 |

## Salud del gradiente

- Iteraciones con traza: **18,000**
- `‖g‖` mediana de las aceptadas: **25.44**, máxima aceptada 1002
- `‖g‖` máxima vista: 1002
- Por encima de 1e+04: **0 de 18,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.08620 · `lambda` final 0.07133

## Disyuntor de lotes

- Lotes rechazados: **4** de 18,000 (0.022 %)
- Rechazos seguidos, máximo: **3**
- Tandas de rechazo: 2
- Rebobinados: **0**

Las primeras tandas:

| desde la it | largo |
|---|---|
| 6,000 | 3 |
| 15,656 | 1 |

