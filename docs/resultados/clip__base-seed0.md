# clip/base-seed0

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
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.099 dB** en la iteración 3,000
- Última medida: 22.462 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.730 | 0.02893 | 0.00063 |
| 2,000 | 30.974 | 0.02777 | 0.00506 |
| 3,000 | 31.099 | 0.02692 | 0.01522 |
| 4,000 | 22.462 | 0.00017 | 0.04498 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **104**, máxima aceptada 2.476e+10
- `‖g‖` máxima vista: 2.476e+10
- Por encima de 1e+04: **568 de 4,000 (14.20 %)**
- Primera vez por encima de 1e+04: iteración **3,429**
- No finitas (NaN o inf): 0
- `T` final 0.00017 · `lambda` final 0.04498

