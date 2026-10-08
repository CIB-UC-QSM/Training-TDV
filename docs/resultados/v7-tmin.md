# v7-tmin

**Estado:** antigua, solo history.json  
**De dónde se sabe:** no hay `traza.npy` ni `estado.json`

## Configuración

| | |
|---|---|
| `mode` | rapido |
| `iterations` | 20000 |
| `t_init` | 0.001 |
| `sigma` | 25 |
| `seed` | 0 |

## PSNR en CBSD68

- **Mejor: 30.935 dB** en la iteración 2,000
- Última medida: 26.569 dB en la 4,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 2,000 | 30.935 | 0.00923 | 0.00323 |
| 4,000 | 26.569 | 0.00291 | 0.02051 |

