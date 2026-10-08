# precision/intermedio-seed4

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | intermedio |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 4 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.159 dB** en la iteración 4,000
- Última medida: 31.159 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.716 | 0.10604 | 0.05577 |
| 2,000 | 30.981 | 0.10493 | 0.03127 |
| 3,000 | 31.105 | 0.10363 | 0.02274 |
| 4,000 | 31.159 | 0.10151 | 0.02156 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **55.63**, máxima aceptada 890.3
- `‖g‖` máxima vista: 890.3
- Por encima de 1e+04: **0 de 4,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.10148 · `lambda` final 0.02158

