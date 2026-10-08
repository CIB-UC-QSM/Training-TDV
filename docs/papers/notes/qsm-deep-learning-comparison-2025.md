# A Fair Comparison in Deep Learning-Based QSM Reconstruction

**Oh, Jung, Kim, Jeong, Lee (SNU) — IEEE Access 13 (2025), doi:10.1109/ACCESS.2025.3620427, 14 pp.**
PDF: `../qsm-deep-learning-comparison-2025.pdf`

## Tesis

Comparar redes de QSM entre sí es **injusto por defecto**, y casi toda la literatura lo hace mal.

## Hallazgos

- No existe un dataset público estándar de QSM (costo + regulación), así que cada grupo entrena y
  evalúa en su propio dataset privado, con características distintas (resolución, orientación, SNR).
- Las dos opciones habituales de comparación **fallan las dos**:
  1. **Comparar contra una red preentrenada** → la red ajena sufre por datos fuera de distribución,
     no por ser peor arquitectura.
  2. **Reentrenar la red ajena** → el reentrenamiento rara vez alcanza el desempeño original
     reportado, así que se la subestima.
- Peor: hay **problemas de reproducibilidad** (misma arquitectura, mismo dataset, distintas semillas)
  que ya bastan para invalidar comparaciones de márgenes pequeños.
- Las redes son sensibles a perturbaciones mínimas de entrada y a diferencias de resolución; los
  pasos de preprocesamiento ocultos inflan resultados.

## Recomendación del paper

Para una comparación justa hacen falta **un dataset público** y **la red entrenada publicada con
información suficiente para reproducirla**.

## Relevancia para el proyecto — alta, y metodológica

Este paper es la razón por la que **la validación en RGB no es opcional**. Define el estándar contra
el que se va a juzgar cualquier número que reportemos:

- Reproducir primero el resultado publicado en un dominio con dataset público (BSDS400/CBSDS68)
  y métrica verificable (PSNR 31.53 dB) demuestra que el pipeline es correcto **antes** de que
  entren en juego los datos privados de QSM.
- Cuando comparemos contra TV/TGV/otras redes, hay que documentar: preprocesamiento exacto,
  semilla, resolución, y publicar el checkpoint.
- Un margen pequeño de mejora en QSM **no es concluyente** si no se controla la varianza entre
  semillas. Correr varias semillas y reportar la dispersión.

---
Ver también: [[tdv-qsm-iterative-2022]].
