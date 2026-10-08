# precision/intermedio-seed8

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | intermedio |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 8 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.077 dB** en la iteración 3,000
- Última medida: 20.806 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.717 | 0.10573 | 0.05627 |
| 2,000 | 30.956 | 0.10449 | 0.03177 |
| 3,000 | 31.077 | 0.10331 | 0.02707 |
| 4,000 | 20.806 | 0.10026 | 0.05902 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **68.33**, máxima aceptada 3.764e+07
- `‖g‖` máxima vista: 3.764e+07
- Por encima de 1e+04: **259 de 4,000 (6.47 %)**
- Primera vez por encima de 1e+04: iteración **3,720**
- No finitas (NaN o inf): 0
- `T` final 0.10027 · `lambda` final 0.05902

