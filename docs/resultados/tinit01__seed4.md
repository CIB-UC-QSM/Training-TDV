# tinit01/seed4

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

- **Mejor: 30.716 dB** en la iteración 1,000
- Última medida: 17.225 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.716 | 0.10583 | 0.00000 |
| 2,000 | 15.198 | 0.09295 | 0.02598 |
| 3,000 | 14.905 | 0.09231 | 0.03097 |
| 4,000 | 17.225 | 0.09155 | 0.03800 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **1.825e+05**, máxima aceptada 8.851e+08
- `‖g‖` máxima vista: 8.851e+08
- Por encima de 1e+04: **2,891 de 4,000 (72.28 %)**
- Primera vez por encima de 1e+04: iteración **1,106**
- No finitas (NaN o inf): 0
- `T` final 0.09155 · `lambda` final 0.03798

