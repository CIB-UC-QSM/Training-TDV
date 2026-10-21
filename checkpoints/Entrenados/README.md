# Checkpoints entrenados

Pesos 2D entrenados en este repositorio desde inicialización aleatoria, con el oficial color al
lado como referencia. Común a todas las columnas: TDV33 (`S` = 10, 32 features, 3 escalas, 3
macro-bloques; **387 970 parámetros entrenables**, más 44 buffers `mask` y 12 `blur` que no se
entrenan), color, BSDS400 (`train` + `test` de BSDS500, 400 imágenes), parches 93×93, batch 32,
Adam β = (0.9, 0.999), lr inicial 4·10⁻⁴, `T_init` = 8·δ = 0.10119, `λ_init` = 0.1, disyuntor
(5× la mediana de 100, cota absoluta 10⁴, rebobina a los 25 rechazos seguidos; respaldo cada 250)
y sin clip. Modo `referencia` (fp32 IEEE, determinista, sin `cudnn.benchmark`), 1 GPU, 2×16.
Se evalúa sobre CBSD68 (68 imágenes de `val`).

| | `t8delta-100k-s1.pth` | `t8delta-100k-s0.pth` | `ciego2d-35k.pth` | oficial color |
|---|---|---|---|---|
| origen | `runs/t8delta-100k-s1/best.pth` | `runs/t8delta-100k-s0/best.pth` | `runs/ciego2d-35k/last.pth` | `../Oficiales/tdv3-3-25-f32-color.pth` |
| semilla | 1 | 0 | 1 | — |
| ruido de entrenamiento | σ = 25 | σ = 25 | σ ~ U(0, 55) por parche (`--sigma-min 0 --sigma-max 55`) | σ = 25 |
| calendario del lr | a la mitad en la 25 000, 50 000 y 75 000 | ídem | a la mitad en la 25 000 y la 30 000 | paper |
| iteraciones | 100 000, terminada | 66 000 guardadas: cortada a mano el 2026-10-06, PSNR estancada | 35 000, terminada | 100 000 (paper) |
| iteración del checkpoint | 100 000 (la mejor) | 64 000 (la mejor) | 35 000 (`best.pth`, la 34 000, da 31.42811 contra 31.42800) | — |
| horas de GPU | 37.70 h | — | 13.09 h | — |
| lotes rechazados / rebobinados | 163 / 4 | 18 / 0 | 202 / 7 | — |

| CBSD68 | `t8delta-100k-s1.pth` | `t8delta-100k-s0.pth` | `ciego2d-35k.pth` | oficial color |
|---|---|---|---|---|
| PSNR, σ = 15 | 34.125 dB | 34.110 dB | sin medir | 34.122 dB (paper 34.12) |
| **PSNR, σ = 25** | **31.547 dB** | 31.526 dB | 31.428 dB | 31.537 dB (paper 31.53) |
| PSNR, σ = 50 | 28.268 dB | 28.246 dB | sin medir | 28.264 dB (paper 28.26) |
| `T` del checkpoint | 0.08264 (6.53 δ) | 0.07956 (6.29 δ) | 0.10002 (7.91 δ) | 0.02473 (1.95 δ) |
| `λ` del checkpoint | 0.12909 | 0.10926 | 0.02519 | 0.10452 |

- δ = δ_Adam = 4·10⁻⁴/√(1 − 0.999) = 0.012649. σ = 15 y 50 salen del reescalado del paper sobre
  la red entrenada a σ = 25 (en un modelo ciego se medirían sin reescalado).
- Son copias **sin el estado del optimizador** (claves `config`, `model`, `iteration`, `psnr`).
  Para continuar un entrenamiento hacen falta los `.pth` completos de `runs/`, que no se versiona.
- Cargan con `evaluate.load_vnet`, igual que los oficiales:
  `uv run python evaluate.py checkpoints/Entrenados/<archivo> --file-list datasets/bsds500/cbsd68.txt`.
- **Como prior de QSM 2D** (ADMM del autor sobre COSMOS, 2026-10-17): el oficial converge a
  27.34 % de RMSE; `t8delta-100k-s1` y los otros de σ = 25 quedan en 28.4–28.5 % sin converger;
  `ciego2d-35k` converge en 33 iteraciones, a 33.18 %. En el punto del solver, (T/S)·ρ(∇²R) es 2.14
  en el ciego, con el dominante negativo (el paso expande ×3.13), y en el oficial el paso expande
  ×1.27. El PSNR no ordena estos priors: bitácora, «Lo que dejó el 2D».
