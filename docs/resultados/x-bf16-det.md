# x-bf16-det

**Estado:** antigua, solo history.json  
**De dónde se sabe:** no hay `traza.npy` ni `estado.json`

## Configuración

| | |
|---|---|
| `mode` | rapido |
| `iterations` | 5000 |
| `t_init` | 0.001 |
| `sigma` | 25 |
| `seed` | 0 |

## PSNR en CBSD68

- **Mejor: 31.194 dB** en la iteración 5,000
- Última medida: 31.194 dB en la 5,000

| iteración | PSNR (dB) | T | lambda |
|---|---|---|---|
| 1,000 | 30.682 | 0.00852 | 0.00000 |
| 2,000 | 30.935 | 0.00908 | 0.00313 |
| 3,000 | 31.079 | 0.00971 | 0.00673 |
| 4,000 | 31.158 | 0.01002 | 0.01135 |
| 5,000 | 31.194 | 0.01031 | 0.01674 |

