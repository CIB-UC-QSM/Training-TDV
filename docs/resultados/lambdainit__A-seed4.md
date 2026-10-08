# lambdainit/A-seed4

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
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 30.748 dB** en la iteración 1,000
- Última medida: 20.131 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.748 | 0.02870 | 0.00107 |
| 2,000 | 24.361 | 0.00934 | 0.02472 |
| 3,000 | 26.764 | 0.00922 | 0.02303 |
| 4,000 | 20.131 | 0.00423 | 0.02701 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **489.3**, máxima aceptada 2.787e+16
- `‖g‖` máxima vista: 2.787e+16
- Por encima de 1e+04: **97 de 4,000 (2.42 %)**
- Primera vez por encima de 1e+04: iteración **1,246**
- No finitas (NaN o inf): 0
- `T` final 0.00497 · `lambda` final 0.02627

