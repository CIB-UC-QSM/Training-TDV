# traza/seed2

**Estado:** terminada (deducido)  
**De dónde se sabe:** 2600 filas = las 2600 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 2600 |
| `t_init` | 0.0252982 |
| `sigma` | 25 |
| `seed` | 2 |

## PSNR en CBSD68

- **Mejor: 30.868 dB** en la iteración 1,500
- Última medida: 30.681 dB en la 2,600

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 500 | 30.303 | 0.02994 | 0.00000 |
| 1,000 | 30.735 | 0.03020 | 0.00076 |
| 1,500 | 30.868 | 0.03032 | 0.00506 |
| 2,000 | 30.239 | 0.01737 | 0.01190 |
| 2,500 | 30.660 | 0.01899 | 0.00000 |
| 2,600 | 30.681 | 0.01909 | 0.00000 |

## Salud del gradiente

- Iteraciones con traza: **2,600**
- `‖g‖` mediana de las aceptadas: **124.9**, máxima aceptada 2.546e+04
- `‖g‖` máxima vista: 2.546e+04
- Por encima de 1e+04: **2 de 2,600 (0.08 %)**
- Primera vez por encima de 1e+04: iteración **1,876**
- No finitas (NaN o inf): 0
- `T` final 0.01909 · `lambda` final 0.00000

