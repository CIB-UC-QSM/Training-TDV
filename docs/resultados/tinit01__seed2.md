# tinit01/seed2

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 2 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.105 dB** en la iteración 3,000
- Última medida: -7.750 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.718 | 0.10284 | 0.00112 |
| 2,000 | 30.996 | 0.10154 | 0.01483 |
| 3,000 | 31.105 | 0.10017 | 0.02556 |
| 4,000 | -7.750 | 0.07962 | 0.06428 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **59.71**, máxima aceptada 2.947e+10
- `‖g‖` máxima vista: 2.947e+10
- Por encima de 1e+04: **113 de 4,000 (2.83 %)**
- Primera vez por encima de 1e+04: iteración **3,887**
- No finitas (NaN o inf): 0
- `T` final 0.07963 · `lambda` final 0.06421

