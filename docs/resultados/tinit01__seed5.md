# tinit01/seed5

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 5 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.095 dB** en la iteración 3,000
- Última medida: -4.106 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.743 | 0.10607 | 0.00000 |
| 2,000 | 30.979 | 0.10459 | 0.00321 |
| 3,000 | 31.095 | 0.10303 | 0.01376 |
| 4,000 | -4.106 | 0.06432 | 0.06729 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **67.14**, máxima aceptada 1.469e+15
- `‖g‖` máxima vista: 1.469e+15
- Por encima de 1e+04: **366 de 4,000 (9.15 %)**
- Primera vez por encima de 1e+04: iteración **3,619**
- No finitas (NaN o inf): 0
- `T` final 0.06432 · `lambda` final 0.06729

