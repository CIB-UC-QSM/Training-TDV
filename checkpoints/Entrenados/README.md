# Checkpoints entrenados

Pesos entrenados en este repositorio desde inicialización aleatoria, con el oficial color al lado
como referencia. Común a las cuatro columnas: TDV33 (`S` = 10, 32 features, 3 escalas, 3
macro-bloques; **387 970 parámetros entrenables**, más 44 buffers `mask` y 12 `blur` que no se
entrenan), color, σ = 25, BSDS400 (`train` + `test` de BSDS500, 400 imágenes), parches 93×93,
batch 32, Adam β = (0.9, 0.999), lr 4·10⁻⁴ a la mitad en la 25 000, 50 000 y 75 000. Se evalúa
sobre CBSD68 (68 imágenes de `val`).

| | `t8delta-100k-s1.pth` | `t8delta-100k-s0.pth` | `replica-v4.pth` | oficial color |
|---|---|---|---|---|
| origen | `runs/t8delta-100k-s1/best.pth` | `runs/t8delta-100k-s0/best.pth` | `runs/replica-v4/checkpoint.pth` | `../Oficiales/tdv3-3-25-f32-color.pth` |
| log | `salida.log`, `history.json`, `traza.npy` | ídem | `train.log`, `history.json` | — |
| modo | `referencia`: fp32 IEEE, determinista, sin `cudnn.benchmark`; 1 GPU, 2×16 | ídem | bf16, `cudnn.benchmark`, `torch.compile`, sin checkpointing; DDP en 2 GPU, 16 c/u | — |
| semilla | 1 | 0 | 0 | — |
| `T_init` | 0.10119 (8 δ) | 0.10119 (8 δ) | 0.001 | 0.001 (`config` del `.pth`) |
| `λ_init` | 0.1 | 0.1 | 0.01 | 0.01 (`config` del `.pth`) |
| disyuntor / clip | disyuntor v2: 5× la mediana de 100, cota abs. 10⁴, rebobina a los 25 seguidos; respaldo cada 250, tras 3 limpias. Sin clip | ídem | sin disyuntor; clip de norma 35 000, ningún paso recortado hasta la 68 000 | — |
| iteraciones | 100 000, terminada | 66 000 guardadas (el log llega a la 67 750): cortada a mano el 2026-10-06, PSNR estancada | 100 000; evento en la 69 250, no volvió al pico | 100 000 (paper) |
| iteración del checkpoint | 100 000 (la mejor) | 64 000 (la mejor) | 68 000 (la mejor) | — |
| horas de GPU | 37.70 h | — | — | — |
| lotes rechazados / rebobinados | 163 / 4 | 18 / 0 | — | — |

| CBSD68 | `t8delta-100k-s1.pth` | `t8delta-100k-s0.pth` | `replica-v4.pth` | oficial color |
|---|---|---|---|---|
| PSNR, σ = 15 | 34.125 dB | 34.110 dB | 34.060 dB | 34.122 dB (paper 34.12) |
| **PSNR, σ = 25** | **31.547 dB** | 31.526 dB | 31.478 dB | 31.537 dB (paper 31.53) |
| PSNR, σ = 50 | 28.268 dB | 28.246 dB | 28.186 dB | 28.264 dB (paper 28.26) |
| SNR de salida, σ = 25 | 24.92 dB | 24.89 dB | 24.85 dB | 24.91 dB |
| NRMSE, σ = 25 | 6.00 % | 6.02 % | 6.05 % | 6.01 % |
| `T` del checkpoint | 0.08264 (6.53 δ) | 0.07956 (6.29 δ) | 0.02211 (1.75 δ) | 0.02473 (1.95 δ) |
| `λ` del checkpoint | 0.12909 | 0.10926 | 0.03118 | 0.10452 |

- δ = δ_Adam = 4·10⁻⁴/√(1 − 0.999) = 0.012649. σ = 15 y 50 salen del reescalado del paper sobre
  la red entrenada a σ = 25; la SNR de entrada a σ = 25 es 13.54 dB. PSNR, SNR y NRMSE salen de
  los `metricas_sigma_*.json` de
  [`docs/presentacion/salida/`](../../docs/presentacion/salida/metricas_sigma.md); `T` y `λ`, del
  `model` de cada `.pth`.
- Son copias **sin el estado del optimizador** (claves `config`, `model`, `iteration`, `psnr`).
  Para continuar un entrenamiento hacen falta los `.pth` completos de `runs/`, que no se versiona;
  de v4 lo tiene `last.pth` (it 100 000, posterior al evento), no `checkpoint.pth`.
- v4 es anterior a los modos y al disyuntor de `train.py`: sus argumentos están en el dict `args`
  de `runs/replica-v4/checkpoint.pth`. Ese dict dice `weight_decay = 1e-4`, pero los dos grupos de
  Adam en `last.pth` tienen `weight_decay = 0`: Adam no lo aplicó.
- Cargan con `evaluate.load_vnet`, igual que los oficiales:
  `uv run python evaluate.py checkpoints/Entrenados/<archivo> --file-list datasets/bsds500/cbsd68.txt`.
