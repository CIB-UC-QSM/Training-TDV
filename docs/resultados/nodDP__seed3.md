# nodDP/seed3

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.0252982 |
| `sigma` | 25 |
| `seed` | 3 |

## PSNR en CBSD68

- **Mejor: 30.690 dB** en la iteración 1,000
- Última medida: 20.583 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.690 | 0.02964 | 0.00060 |
| 2,000 | 20.211 | 0.00001 | 0.03127 |
| 3,000 | 20.380 | 0.00004 | 0.03127 |
| 4,000 | 20.583 | 0.00008 | 0.03127 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **1.917e+05**, máxima aceptada 1.707e+11
- `‖g‖` máxima vista: 1.707e+11
- Por encima de 1e+04: **2,400 de 4,000 (60.00 %)**
- Primera vez por encima de 1e+04: iteración **1,568**
- No finitas (NaN o inf): 0
- `T` final 0.00008 · `lambda` final 0.03127

