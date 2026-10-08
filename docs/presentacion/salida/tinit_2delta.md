# T₀ = 2δ contra 8δ (`tinit-2delta-50k`, cortada a mano en la it 4 000)

| 0–4 000 it | 2δ (s0) | 8δ final s0 | 8δ final s1 |
|---|---|---|---|
| PSNR @2 000 | 31.008 | 30.982 | 31.003 |
| PSNR @4 000 | 31.155 | 31.191 | 31.172 |
| lotes rechazados | 79 | 6 | 0 |
| rebobinados | 3 (2195, 2571, 3522) | 0 | 0 |
| eventos ‖g‖ > 10⁴ | 50 | 0 | 0 |
| máx ‖g‖ | 1e+10 | 9.1e+02 | 1e+03 |
| mediana ‖g‖ aceptado | 85 | 52 | 53 |
| T/δ mínima | 1.97 | 7.89 | 7.95 |
| T @4 000 (en δ) | 0.0271 (2.14 δ) | 0.0999 (7.90 δ) | 0.1006 (7.96 δ) |

Estado @4 000 (16 recortes de CBSD68; paso efectivo como en dentro.py):

| modelo | T | ‖∇R‖ | paso T·‖∇R‖/‖ruido‖ |
|---|---|---|---|
| 2δ @4k | 0.0271 | 2889 | 1.202 |
| ≈8δ @4k
(misma semilla) | 0.0981 | 784 | 1.181 |
| final s1
@100k | 0.0826 | 970 | 1.230 |
| oficial | 0.0247 | 2844 | 1.079 |

## Predicciones registradas (docs/bitacora.md:3733) contra lo medido

| # | predicción | medido | veredicto |
|---|---|---|---|
| 1 | T no toca cero en las 50 000 | mín 0.0249 en 4 000 it | sin resultado (cortada en la 4 000; no tocó en ese tramo) |
| 2 | PSNR @4 000 en [31.05, 31.25] | 31.155 | ✔ |
| 3 | T @4 000 en [0.023, 0.028] | 0.0271 | ✔ |
| 4 | PSNR @16 000 ≥ 31.09 | — | sin resultado |
| 5 | PSNR @50 000 en [31.48, 31.58] | — | sin resultado |
| 6 | T @50 000 > 0.0300 | — | sin resultado |
| 7 | rechazos > 18 en las 50 000 | 79 en 4 000 | ✔ (ya en la 4 000) |
