# clip/base-seed4

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

- **Mejor: 31.114 dB** en la iteración 3,000
- Última medida: 9.956 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.740 | 0.02867 | 0.00035 |
| 2,000 | 31.004 | 0.02763 | 0.00600 |
| 3,000 | 31.114 | 0.02678 | 0.01302 |
| 4,000 | 9.956 | 0.03361 | 0.07909 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **114.4**, máxima aceptada 2.107e+10
- `‖g‖` máxima vista: 2.107e+10
- Por encima de 1e+04: **783 de 4,000 (19.57 %)**
- Primera vez por encima de 1e+04: iteración **3,215**
- No finitas (NaN o inf): 0
- `T` final 0.03362 · `lambda` final 0.07898

