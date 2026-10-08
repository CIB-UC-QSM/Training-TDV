# precision/intermedio-seed0

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | intermedio |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 0 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 30.993 dB** en la iteración 2,000
- Última medida: 12.956 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.746 | 0.10361 | 0.05103 |
| 2,000 | 30.993 | 0.10198 | 0.02640 |
| 3,000 | 10.431 | 0.11652 | 0.02748 |
| 4,000 | 12.956 | 0.11648 | 0.02777 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **273**, máxima aceptada 1.866e+10
- `‖g‖` máxima vista: 1.866e+10
- Por encima de 1e+04: **1,814 de 4,000 (45.35 %)**
- Primera vez por encima de 1e+04: iteración **2,185**
- No finitas (NaN o inf): 0
- `T` final 0.11648 · `lambda` final 0.02777

