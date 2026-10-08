# traza/seed4

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

- **Mejor: 31.108 dB** en la iteración 3,000
- Última medida: 12.850 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 500 | 30.359 | 0.02917 | 0.00000 |
| 1,000 | 30.739 | 0.02873 | 0.00097 |
| 1,500 | 30.906 | 0.02792 | 0.00152 |
| 2,000 | 30.990 | 0.02749 | 0.00645 |
| 2,500 | 31.028 | 0.02694 | 0.01069 |
| 3,000 | 31.108 | 0.02659 | 0.01515 |
| 3,500 | 11.781 | 0.02555 | 0.00193 |
| 4,000 | 12.850 | 0.02547 | 0.00283 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **116**, máxima aceptada 3.204e+09
- `‖g‖` máxima vista: 3.204e+09
- Por encima de 1e+04: **705 de 4,000 (17.62 %)**
- Primera vez por encima de 1e+04: iteración **3,251**
- No finitas (NaN o inf): 0
- `T` final 0.02547 · `lambda` final 0.00282

