# tinit01/seed1

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 1 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.004 dB** en la iteración 2,000
- Última medida: -5.433 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.752 | 0.10369 | 0.00092 |
| 2,000 | 31.004 | 0.10185 | 0.00769 |
| 3,000 | 1.199 | 0.09234 | 0.02925 |
| 4,000 | -5.433 | 0.09829 | 0.02331 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **85.56**, máxima aceptada 4.609e+15
- `‖g‖` máxima vista: 4.609e+15
- Por encima de 1e+04: **1,015 de 4,000 (25.38 %)**
- Primera vez por encima de 1e+04: iteración **2,978**
- No finitas (NaN o inf): 0
- `T` final 0.09829 · `lambda` final 0.02331

