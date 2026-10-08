# precision/rapido-seed10

**Estado:** terminada (deducido)  
**De dónde se sabe:** 4000 filas = las 4000 pedidas

## Configuración

| | |
|---|---|
| `mode` | rapido |
| `iterations` | 4000 |
| `t_init` | 0.1 |
| `sigma` | 25 |
| `seed` | 10 |
| `clip` | inf |

## PSNR en CBSD68

- **Mejor: 31.056 dB** en la iteración 3,000
- Última medida: -2.567 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.652 | 0.10537 | 0.04080 |
| 2,000 | 30.932 | 0.10323 | 0.01619 |
| 3,000 | 31.056 | 0.10097 | 0.01236 |
| 4,000 | -2.567 | 0.07226 | 0.14993 |

## Salud del gradiente

- Iteraciones con traza: **4,000**
- `‖g‖` mediana de las aceptadas: **83.11**, máxima aceptada 5.511e+17
- `‖g‖` máxima vista: 5.511e+17
- Por encima de 1e+04: **976 de 4,000 (24.40 %)**
- Primera vez por encima de 1e+04: iteración **3,019**
- No finitas (NaN o inf): 0
- `T` final 0.07226 · `lambda` final 0.14993

