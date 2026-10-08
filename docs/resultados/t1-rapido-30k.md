# t1-rapido-30k

**Estado:** incompleta (deducido) — en vuelo o matada, por filas no se distingue  
**De dónde se sabe:** 4000 filas de 30000

## Configuración

| | |
|---|---|
| `mode` | rapido |
| `iterations` | 30000 |
| `t_init` | 0.101193 |
| `sigma` | 25 |
| `seed` | 0 |
| `channels` | 1 |
| `data_root` | /home/santicien/Documents/Training-TDV/datasets/ldm100k |
| `eval_split` | val |
| `clip` | inf |
| `guard` | 5 |
| `guard_abs` | 10000 |
| `guard_ventana` | 100 |
| `guard_seguidos` | 25 |
| `guard_amp` | inf |

## PSNR en `ldm100k/val`

- **Mejor: 32.233 dB** en la iteración 4,000
- Última medida: 32.233 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 2,000 | 31.884 | 0.10954 | 0.06095 |
| 4,000 | 32.233 | 0.10723 | 0.07167 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **10.3**, máxima aceptada 205.4
- `‖g‖` máxima vista: 5514
- Por encima de 1e+04: **0 de 4,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.10725 · `lambda` final 0.07164

## Disyuntor de lotes

- Lotes rechazados: **69** de 4,000 (1.725 %)
- Rechazos seguidos, máximo: **25**
- Tandas de rechazo: 7
- Rebobinados: **2**

Las primeras tandas:

| desde la it | largo |
|---|---|
| 209 | 1 |
| 214 | 2 |
| 217 | 25 |
| 507 | 2 |
| 752 | 13 |
| 1,595 | 25 |
| 3,323 | 1 |

