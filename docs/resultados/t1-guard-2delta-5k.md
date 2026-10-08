# t1-guard-2delta-5k

**Estado:** incompleta (deducido) — en vuelo o matada, por filas no se distingue  
**De dónde se sabe:** 1000 filas de 5000

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 5000 |
| `t_init` | 0.0252982 |
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

- **Mejor: 30.087 dB** en la iteración 1,000
- Última medida: 30.087 dB en la 1,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.087 | 0.02743 | 0.08757 |

## Salud del gradiente

- Iteraciones con traza: **1,000**
- `‖g‖` mediana de las aceptadas: **50.65**, máxima aceptada 668.4
- `‖g‖` máxima vista: 9.572e+04
- Por encima de 1e+04: **18 de 1,000 (1.80 %)**
- Primera vez por encima de 1e+04: iteración **720**
- No finitas (NaN o inf): 0
- `T` final 0.02742 · `lambda` final 0.08759

## Disyuntor de lotes

- Lotes rechazados: **93** de 1,000 (9.300 %)
- Rechazos seguidos, máximo: **25**
- Tandas de rechazo: 15
- Rebobinados: **3**

Las primeras tandas:

| desde la it | largo |
|---|---|
| 449 | 1 |
| 471 | 2 |
| 475 | 25 |
| 699 | 1 |
| 702 | 1 |
| 709 | 1 |
| 712 | 1 |
| 716 | 1 |
| 718 | 3 |
| 722 | 3 |

