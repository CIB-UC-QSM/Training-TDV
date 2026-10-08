# nodDP/seed0

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.0252982 |
| `sigma` | 25 |
| `seed` | 0 |

## PSNR en CBSD68

- **Mejor: 30.989 dB** en la iteración 2,000
- Última medida: 25.012 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.715 | 0.02900 | 0.00072 |
| 2,000 | 30.989 | 0.02810 | 0.00538 |
| 3,000 | 23.197 | 0.00726 | 0.07050 |
| 4,000 | 25.012 | 0.00714 | 0.07110 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **297.3**, máxima aceptada 1.287e+08
- `‖g‖` máxima vista: 1.287e+08
- Por encima de 1e+04: **379 de 4,000 (9.47 %)**
- Primera vez por encima de 1e+04: iteración **2,184**
- No finitas (NaN o inf): 0
- `T` final 0.00714 · `lambda` final 0.07110

