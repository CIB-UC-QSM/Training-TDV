# lambdainit/B-seed5

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

- **Mejor: 22.085 dB** en la iteración 3,000
- Última medida: 21.909 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 16.628 | 0.10604 | 0.09923 |
| 2,000 | 17.219 | 0.10476 | 0.10862 |
| 3,000 | 22.085 | 0.09122 | 0.15204 |
| 4,000 | 21.909 | 0.07815 | 0.16591 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **1.472e+04**, máxima aceptada 1.084e+08
- `‖g‖` máxima vista: 1.084e+08
- Por encima de 1e+04: **2,321 de 4,000 (58.02 %)**
- Primera vez por encima de 1e+04: iteración **289**
- No finitas (NaN o inf): 0
- `T` final 0.07815 · `lambda` final 0.16591

