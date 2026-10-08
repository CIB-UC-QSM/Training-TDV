# tinit-7p9delta-10k

**Estado:** terminada  
**De dónde se sabe:** `estado.json`

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 10000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 0 |
| `clip` | inf |
| `guard` | 5 |
| `guard_abs` | 10000 |
| `guard_ventana` | 100 |
| `guard_seguidos` | 25 |
| `guard_amp` | inf |

## PSNR en CBSD68

- **Mejor: 31.343 dB** en la iteración 10,000
- Última medida: 31.343 dB en la 10,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 2,000 | 30.993 | 0.10190 | 0.02912 |
| 4,000 | 31.190 | 0.09848 | 0.02676 |
| 6,000 | 31.264 | 0.09547 | 0.03502 |
| 8,000 | 31.310 | 0.09230 | 0.04321 |
| 10,000 | 31.343 | 0.08977 | 0.05189 |

## Salud del gradiente

- Iteraciones con traza: **10,000**
- `‖g‖` mediana de las aceptadas: **32.02**, máxima aceptada 869.1
- `‖g‖` máxima vista: 1.971e+05
- Por encima de 1e+04: **18 de 10,000 (0.18 %)**
- Primera vez por encima de 1e+04: iteración **5,845**
- No finitas (NaN o inf): 0
- `T` final 0.08978 · `lambda` final 0.05189

## Disyuntor de lotes

- Lotes rechazados: **93** de 10,000 (0.930 %)
- Rechazos seguidos, máximo: **25**
- Tandas de rechazo: 13
- Rebobinados: **2**

Las primeras tandas:

| desde la it | largo |
|---|---|
| 1,953 | 1 |
| 2,274 | 25 |
| 2,938 | 1 |
| 2,942 | 1 |
| 4,389 | 1 |
| 5,485 | 1 |
| 5,837 | 3 |
| 5,841 | 11 |
| 5,853 | 6 |
| 5,860 | 16 |

