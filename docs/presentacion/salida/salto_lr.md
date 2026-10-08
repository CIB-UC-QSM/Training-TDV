# El salto en cada halving del lr

Ventanas: «antes» = mediana de las 1 000 it previas; «después» = de las 300 siguientes.

| halving | semilla | PSNR antes (it H o H−1k) | PSNR después (H+2k o H+1k) | ‖g‖ antes → después | pérdida antes → después | T antes → después |
|---|---|---|---|---|---|---|
| 25,000 | s0 | 31.4432 | 31.4765 | 17.75 → 13.46 (-24 %) | 21.040 → 20.566 | 0.07869 → 0.07880 |
| 25,000 | s1 | 31.4531 | 31.4826 | 17.24 → 12.99 (-25 %) | 20.849 → 20.690 | 0.08106 → 0.08112 |
| 50,000 | s0 | 31.5080 | 31.5178 | 16.30 → 14.04 (-14 %) | 20.671 → 20.245 | 0.07855 → 0.07869 |
| 50,000 | s1 | 31.5169 | 31.5245 | 14.72 → 12.38 (-16 %) | 20.432 → 20.531 | 0.07967 → 0.07980 |
| 75,000 | s1 | 31.5348 | 31.5423 | 14.06 → 11.66 (-17 %) | 20.204 → 20.224 | 0.08112 → 0.08119 |

Con pesos (it 75,000): todavía faltan instantáneas; se completa solo al re-correr.
