# clip/base-seed5

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 4000 |
| `t_init` | 0.0252982 |
| `sigma` | 25 |
| `seed` | 5 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.057 dB** en la iteración 3,000
- Última medida: 20.173 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.710 | 0.02827 | 0.00183 |
| 2,000 | 30.969 | 0.02740 | 0.01115 |
| 3,000 | 31.057 | 0.02589 | 0.01388 |
| 4,000 | 20.173 | 0.00000 | 0.04317 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **128.4**, máxima aceptada 1.988e+09
- `‖g‖` máxima vista: 1.988e+09
- Por encima de 1e+04: **310 de 4,000 (7.75 %)**
- Primera vez por encima de 1e+04: iteración **3,075**
- No finitas (NaN o inf): 0
- `T` final 0.00000 · `lambda` final 0.04317

