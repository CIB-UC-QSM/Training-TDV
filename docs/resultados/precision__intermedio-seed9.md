# precision/intermedio-seed9

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | intermedio |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 9 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 10.467 dB** en la iteración 4,000
- Última medida: 10.467 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 1.184 | 0.09820 | 0.06463 |
| 2,000 | 6.256 | 0.09773 | 0.06669 |
| 3,000 | 9.219 | 0.09741 | 0.06803 |
| 4,000 | 10.467 | 0.09719 | 0.06885 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **4.791e+05**, máxima aceptada 9.281e+10
- `‖g‖` máxima vista: 9.281e+10
- Por encima de 1e+04: **3,220 de 4,000 (80.50 %)**
- Primera vez por encima de 1e+04: iteración **781**
- No finitas (NaN o inf): 0
- `T` final 0.09720 · `lambda` final 0.06885

