# clip/base-seed2

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.0252982 |
| `sigma` | 25 |
| `seed` | 2 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.177 dB** en la iteración 4,000
- Última medida: 31.177 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.729 | 0.02971 | 0.00068 |
| 2,000 | 31.005 | 0.02941 | 0.00914 |
| 3,000 | 31.100 | 0.02894 | 0.01995 |
| 4,000 | 31.177 | 0.02872 | 0.02751 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **75.2**, máxima aceptada 2037
- `‖g‖` máxima vista: 2037
- Por encima de 1e+04: **0 de 4,000 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.02870 · `lambda` final 0.02751

