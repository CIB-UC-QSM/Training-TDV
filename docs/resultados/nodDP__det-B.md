# nodDP/det-B

**Estado:** terminada (deducido)  
**De dónde se sabe:** 300 filas = las 300 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 300 |
| `t_init` | 0.0252982 |
| `sigma` | 25 |
| `seed` | 7 |

## PSNR en CBSD68

- **Mejor: 30.022 dB** en la iteración 300
- Última medida: 30.022 dB en la 300

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 300 | 30.022 | 0.02916 | 0.00000 |

## Salud del gradiente

- Iteraciones con traza: **300**
- `‖g‖` mediana de las aceptadas: **480.4**, máxima aceptada 1818
- `‖g‖` máxima vista: 1818
- Por encima de 1e+04: **0 de 300 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.02915 · `lambda` final 0.00000

