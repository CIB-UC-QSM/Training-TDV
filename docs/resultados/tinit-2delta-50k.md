# tinit-2delta-50k

**Estado:** MATADA — a mano 2026-10-07: se reasigno la GPU a la tanda t1-ldm100k sobre cortes de T1. Se dejo llegar al eval de la it 4000 para que la traza del pico de 3.77e9 de la it 2171 quedara en disco  
**De dónde se sabe:** `estado.json`

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 50000 |
| `t_init` | 0.0252982 |
| `sigma` | 25 |
| `seed` | 0 |
| `clip` | inf |
| `guard` | 5 |
| `guard_abs` | 10000 |
| `guard_ventana` | 100 |
| `guard_seguidos` | 25 |
| `guard_amp` | inf |

## PSNR en CBSD68

- **Mejor: 31.155 dB** en la iteración 4,000
- Última medida: 31.155 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 2,000 | 31.008 | 0.02795 | 0.02912 |
| 4,000 | 31.155 | 0.02713 | 0.02576 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **84.75**, máxima aceptada 1678
- `‖g‖` máxima vista: 1.042e+10
- Por encima de 1e+04: **50 de 4,000 (1.25 %)**
- Primera vez por encima de 1e+04: iteración **2,171**
- No finitas (NaN o inf): 0
- `T` final 0.02710 · `lambda` final 0.02578

## Disyuntor de lotes

- Lotes rechazados: **79** de 4,000 (1.975 %)
- Rechazos seguidos, máximo: **25**
- Tandas de rechazo: 7
- Rebobinados: **3**

Las primeras tandas:

| desde la it | largo |
|---|---|
| 1,996 | 1 |
| 2,093 | 1 |
| 2,171 | 25 |
| 2,308 | 1 |
| 2,440 | 1 |
| 2,547 | 25 |
| 3,498 | 25 |

