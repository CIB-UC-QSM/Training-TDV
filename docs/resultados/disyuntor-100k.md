# disyuntor-100k

**Estado:** MATADA — matada a mano el 2026-10-04 12:14: trabada desde la it 4416 sin dar un solo paso (el ultimo fue en la 4415)  
**De dónde se sabe:** `estado.json`

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 100000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 0 |
| `clip` | inf |
| `guard` | 31.6228 |
| `guard_ventana` | 100 |
| `guard_amp` | inf |

## PSNR en CBSD68

- **Mejor: 31.182 dB** en la iteración 4,000
- Última medida: 27.631 dB en la 50,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 2,000 | 30.993 | 0.10190 | 0.02912 |
| 4,000 | 31.182 | 0.09813 | 0.02763 |
| 6,000 | 27.631 | 0.09810 | 0.02931 |
| 8,000 | 27.631 | 0.09810 | 0.02931 |
| 10,000 | 27.631 | 0.09810 | 0.02931 |
| 12,000 | 27.631 | 0.09810 | 0.02931 |
| 14,000 | 27.631 | 0.09810 | 0.02931 |
| 16,000 | 27.631 | 0.09810 | 0.02931 |
| 18,000 | 27.631 | 0.09810 | 0.02931 |
| 20,000 | 27.631 | 0.09810 | 0.02931 |
| 22,000 | 27.631 | 0.09810 | 0.02931 |
| 24,000 | 27.631 | 0.09810 | 0.02931 |
| 26,000 | 27.631 | 0.09810 | 0.02931 |
| 28,000 | 27.631 | 0.09810 | 0.02931 |
| 30,000 | 27.631 | 0.09810 | 0.02931 |
| 32,000 | 27.631 | 0.09810 | 0.02931 |
| 34,000 | 27.631 | 0.09810 | 0.02931 |
| 36,000 | 27.631 | 0.09810 | 0.02931 |
| 38,000 | 27.631 | 0.09810 | 0.02931 |
| 40,000 | 27.631 | 0.09810 | 0.02931 |
| 42,000 | 27.631 | 0.09810 | 0.02931 |
| 44,000 | 27.631 | 0.09810 | 0.02931 |
| 46,000 | 27.631 | 0.09810 | 0.02931 |
| 48,000 | 27.631 | 0.09810 | 0.02931 |
| 50,000 | 27.631 | 0.09810 | 0.02931 |

## Salud del gradiente

- Iteraciones con traza: **50,000**
- `‖g‖` mediana de las aceptadas: **52.09**, máxima aceptada 962
- `‖g‖` máxima vista: 6.907e+12
- Por encima de 1e+04: **45,904 de 50,000 (91.81 %)**
- Primera vez por encima de 1e+04: iteración **4,075**
- No finitas (NaN o inf): 0
- `T` final 0.09810 · `lambda` final 0.02931

## Disyuntor de lotes

- Lotes rechazados: **45,925** de 50,000 (91.850 %)
- Rechazos seguidos, máximo: **45,585**
- Tandas de rechazo: 4

Las primeras tandas:

| desde la it | largo |
|---|---|
| 1,953 | 1 |
| 4,075 | 10 |
| 4,086 | 329 |
| 4,416 | 45,585 |

> **SE TRABÓ EN LA ITERACIÓN 4,416.** Desde ahí no se dio ningún paso: los pesos quedaron congelados y la corrida siguió gastando GPU sin que nada pudiera cambiar.

