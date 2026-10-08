# traza/seed1

**Estado:** terminada (deducido)  
**De dónde se sabe:** 1500 filas = las 1500 pedidas

## Configuración

| | |
|---|---|
| `mode` | referencia |
| `iterations` | 1500 |
| `t_init` | 0.0252982 |
| `sigma` | 25 |
| `seed` | 1 |

## PSNR en CBSD68

- **Mejor: 30.894 dB** en la iteración 1,500
- Última medida: 30.894 dB en la 1,500

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 500 | 30.323 | 0.02849 | 0.00001 |
| 1,000 | 30.670 | 0.02844 | 0.00315 |
| 1,500 | 30.894 | 0.02810 | 0.00606 |

## Salud del gradiente

- Iteraciones con traza: **1,500**
- `‖g‖` mediana de las aceptadas: **131**, máxima aceptada 1690
- `‖g‖` máxima vista: 1690
- Por encima de 1e+04: **0 de 1,500 (0.00 %)**
- Primera vez por encima de 1e+04: nunca
- No finitas (NaN o inf): 0
- `T` final 0.02808 · `lambda` final 0.00605

