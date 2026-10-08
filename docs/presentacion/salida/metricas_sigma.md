# Métricas a distintos niveles de ruido (CBSD68, 68 imágenes, sin re-entrenar)

La red se entrenó a σ = 25; los otros σ usan el reescalado del paper (z·25/σ).

| modelo | it | σ = 15: PSNR · NRMSE · SNR salida | σ = 25: PSNR · NRMSE · SNR salida | σ = 50: PSNR · NRMSE · SNR salida |
|---|---|---|---|---|
| oficial | — | 34.12 dB · 4.43 % · 27.49 dB | 31.54 dB · 6.01 % · 24.91 dB | 28.26 dB · 8.82 % · 21.63 dB |
| v4 | 68000 | 34.06 dB · 4.46 % · 27.43 dB | 31.48 dB · 6.05 % · 24.85 dB | 28.19 dB · 8.89 % · 21.55 dB |
| final s0 | 64000 | 34.11 dB · 4.44 % · 27.48 dB | 31.53 dB · 6.02 % · 24.89 dB | 28.25 dB · 8.83 % · 21.62 dB |
| final s1 | 100000 | 34.12 dB · 4.43 % · 27.49 dB | 31.55 dB · 6.00 % · 24.92 dB | 28.27 dB · 8.82 % · 21.64 dB |
| publicado (paper) | | 34.12 dB | 31.53 dB | 28.26 dB |

SNR de entrada (imagen ruidosa): σ = 15 → 17.98 dB, σ = 25 → 13.54 dB, σ = 50 → 7.52 dB

NRMSE = ‖x̂ − y‖/‖y‖ por imagen, promediado. SNR de salida = −20·log10(NRMSE) por imagen.
