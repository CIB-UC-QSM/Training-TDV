# Números de las corridas finales

Checkpoint oficial: **31.534 dB** (`runs/oracle-cbsd68.json`); paper 31.53.

| | t8delta-100k-s0 | t8delta-100k-s1 |
|---|---|---|
| estado | cortada en la 66k (a mano 2026-10-06) | terminada |
| iteraciones en la traza | 66,000 | 100,000 |
| mejor PSNR (dB) | **31.526** | **31.547** |
| en la it | 64,000 | 100,000 |
| PSNR de la última evaluación | 31.524 | 31.547 |
| T al final | 0.07962 | 0.08263 |
| T mínima | 0.07848 | 0.07963 |
| λ al final | 0.10924 | 0.12910 |
| lotes rechazados | 18 | 163 |
| rebobinados | 0 | 4 |
| racha máx. de rechazos | 6 | 25 |
| lotes agarrados SÓLO por la cota 10⁴ | 0 | 0 |
| horas | — | 37.7 |

## Predicciones registradas antes de correr

| # | predicción | medido (s0, s1) | veredicto |
|---|---|---|---|
| 1 | T a la it 100 000 en [0.018, 0.032] | 0.0826 | ✘ reemplazada por E1 antes de terminar (los parciales ya la falsificaban) |
| 2 | T nunca baja de la pared δ = 0.01265 | mín 0.0785, mín 0.0796 | ✔ se cumple — NO INFORMATIVA (E2): no podía fallar |
| 3 | mejor PSNR ≥ 31.40 dB | 31.526, 31.547 | ✔ se cumple |
| 4 | ninguna se traba: racha de rechazos < 1 000 | 6, 25 | ✔ se cumple |
| 5 | las dos semillas a < 0.33 dB (mejor PSNR) | 0.021 dB | ✔ se cumple — ojo: s0: cortada en la 66k (a mano 2026-10-06), su mejor es de la it 64,000 |
| 6 | la cota 10⁴ nunca agarra un lote sola | 0, 0 | ✔ se cumple |
| E1 | T a la it 100 000 en [0.050, 0.075] | 0.0826 | ✘ falla (sólo s1; s0: cortada en la 66k (a mano 2026-10-06)) |
| E2 | ningún brazo llega a T < δ | = la 2 | no informativa por diseño (lo dice la enmienda) |
| E3 | s1 reproduce su trayectoria cortada hasta la it 18 000 | traza idéntica en 18,000 filas: True; PSNR idénticas: True | ✔ se cumple (bit a bit) |
| E4 | s0 ídem | traza idéntica en 18,000 filas: True; PSNR idénticas: True | ✔ se cumple (bit a bit) |
| E5 | PSNR a la it 100 000 en [31.45, 31.60] | 31.547 | ✔ se cumple (sólo s1; s0: cortada en la 66k (a mano 2026-10-06)) |
| E6 | tras rebobinar, ‖g‖/mediana restaurada finita y en la banda de v1 [0.72, 1.04] | 0.819, 1.082, 0.716, 0.928 | ✔ finitas; 2 de 4 dentro de la banda |

1–6: pre-registro original. E1–E6: enmienda del 10-05, escrita antes de relanzar; E1 reemplaza a la 1 y las demás son nuevas (la 5 original sigue vigente). 1, E1 y E5 hablan de la it 100 000 y se leen sólo sobre las corridas que llegaron (la s0 se cortó a mano en la 66k).
