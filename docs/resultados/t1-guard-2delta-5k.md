# t1-guard-2delta-5k

**Estado:** MATADA — a mano 2026-10-08: entro en el lazo del disyuntor v2 descrito en docs/bitacora.md y docs/mediciones/lazo-disyuntor-t1.txt — rebobinaba al respaldo de la it 1756 cada 26 iteraciones (GUARD_SEGUIDOS+1) con la mediana movil congelada en 18.2, sin salida posible por dentro del mecanismo. Se dejo llegar al eval de la it 3000 para que la traza del lazo quedara en disco. No murio: T quedo a 2.16x de su pared (0.02736; antes decia 2.14x, mal). CAUSA CORREGIDA el mismo dia: no era la ventana sino que el segundo rebobinado al mismo respaldo no restaura Adam (opt.load_state_dict adopta los tensores del respaldo); docs/mediciones/alias-respaldo-adam.txt.  
**De dónde se sabe:** `estado.json`

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

- **Mejor: 30.477 dB** en la iteración 2,000
- Última medida: 30.471 dB en la 3,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.087 | 0.02743 | 0.08757 |
| 2,000 | 30.477 | 0.02746 | 0.07398 |
| 3,000 | 30.471 | 0.02736 | 0.07392 |

## Salud del gradiente

- Iteraciones con traza: **3,000**
- `‖g‖` mediana de las aceptadas: **33.65**, máxima aceptada 668.4
- `‖g‖` máxima vista: 9.572e+04
- Por encima de 1e+04: **28 de 3,000 (0.93 %)**
- Primera vez por encima de 1e+04: iteración **720**
- No finitas (NaN o inf): 0
- `T` final 0.02736 · `lambda` final 0.07392

## Disyuntor de lotes

- Lotes rechazados: **1,192** de 3,000 (39.733 %)
- Rechazos seguidos, máximo: **25**
- Tandas de rechazo: 73
- Rebobinados: **44**

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

> **SE TRABÓ EN LA ITERACIÓN 2,978.** Desde ahí no se dio ningún paso: los pesos quedaron congelados y la corrida siguió gastando GPU sin que nada pudiera cambiar.

