# lambdainit/B-seed4

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 4 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 30.717 dB** en la iteración 1,000
- Última medida: 17.541 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.717 | 0.10598 | 0.05638 |
| 2,000 | 15.163 | 0.07830 | 0.07317 |
| 3,000 | 12.994 | 0.07817 | 0.07488 |
| 4,000 | 17.541 | 0.07801 | 0.07620 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **1.164e+04**, máxima aceptada 2.644e+10
- `‖g‖` máxima vista: 2.644e+10
- Por encima de 1e+04: **2,194 de 4,000 (54.85 %)**
- Primera vez por encima de 1e+04: iteración **1,403**
- No finitas (NaN o inf): 0
- `T` final 0.07801 · `lambda` final 0.07620

