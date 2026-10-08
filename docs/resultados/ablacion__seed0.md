# ablacion/seed0

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

- **Mejor: 31.127 dB** en la iteración 3,000
- Última medida: 17.392 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.750 | 0.02904 | 0.00060 |
| 2,000 | 31.027 | 0.02806 | 0.01177 |
| 3,000 | 31.127 | 0.02732 | 0.02339 |
| 4,000 | 17.392 | 0.00869 | 0.04799 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **110.9**, máxima aceptada 1.538e+10
- `‖g‖` máxima vista: 1.538e+10
- Por encima de 1e+04: **752 de 4,000 (18.80 %)**
- Primera vez por encima de 1e+04: iteración **3,218**
- No finitas (NaN o inf): 0
- `T` final 0.00869 · `lambda` final 0.04799

