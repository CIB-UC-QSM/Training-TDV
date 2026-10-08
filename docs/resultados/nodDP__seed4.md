# nodDP/seed4

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.0252982 |
| `sigma` | 25 |
| `seed` | 4 |

## PSNR en CBSD68

- **Mejor: 31.005 dB** en la iteración 2,000
- Última medida: -1.601 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.750 | 0.02874 | 0.00043 |
| 2,000 | 31.005 | 0.02767 | 0.00613 |
| 3,000 | -3.395 | 0.03805 | 0.00567 |
| 4,000 | -1.601 | 0.03800 | 0.00647 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **156.1**, máxima aceptada 2.107e+13
- `‖g‖` máxima vista: 2.107e+13
- Por encima de 1e+04: **1,284 de 4,000 (32.10 %)**
- Primera vez por encima de 1e+04: iteración **2,717**
- No finitas (NaN o inf): 0
- `T` final 0.03800 · `lambda` final 0.00647

