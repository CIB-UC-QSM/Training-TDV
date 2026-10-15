# t1-rapido-30k

**Estado:** terminada  
**De dónde se sabe:** `estado.json`

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

- **Mejor: 32.766 dB** en la iteración 30,000
- Última medida: 32.766 dB en la 30,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 2,000 | 31.884 | 0.10954 | 0.06095 |
| 4,000 | 32.233 | 0.10723 | 0.07167 |
| 6,000 | 32.371 | 0.10522 | 0.09311 |
| 8,000 | 32.461 | 0.10350 | 0.10149 |
| 10,000 | 32.545 | 0.10144 | 0.10772 |
| 12,000 | 32.585 | 0.09983 | 0.11083 |
| 14,000 | 32.621 | 0.09806 | 0.11213 |
| 16,000 | 32.641 | 0.09670 | 0.11312 |
| 18,000 | 32.654 | 0.09488 | 0.11386 |
| 20,000 | 32.684 | 0.09365 | 0.11565 |
| 22,000 | 32.715 | 0.09435 | 0.12159 |
| 24,000 | 32.736 | 0.09437 | 0.12404 |
| 26,000 | 32.759 | 0.09482 | 0.12617 |
| 28,000 | 32.761 | 0.09535 | 0.12808 |
| 30,000 | 32.766 | 0.09561 | 0.12978 |

## Salud del gradiente

- Iteraciones con traza: **30,000**
- `‖g‖` mediana de las aceptadas: **6.465**, máxima aceptada 205.4
- `‖g‖` máxima vista: 5514
- Por encima de 1e+04: **0 de 30,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.09560 · `lambda` final 0.12980

## Disyuntor de lotes

- Lotes rechazados: **155** de 30,000 (0.517 %)
- Rechazos seguidos, máximo: **25**
- Tandas de rechazo: 17
- Rebobinados: **5**

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
| 6,313 | 25 |
| 7,177 | 4 |
| 7,182 | 25 |

