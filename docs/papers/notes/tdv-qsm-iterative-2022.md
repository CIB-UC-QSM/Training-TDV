# TDV Regularization for Improved Iterative QSM (TDV-QSM)

**Milovic, Larrain, Shmueli — ISMRM 2021, abstract de 4 pp.** PDF: `../tdv-qsm-iterative-2022.pdf`

> Este es el paper que **origina el proyecto**. Define el punto de partida y, en su sección de
> Discusión, plantea literalmente el trabajo que estamos haciendo.

## Idea central

Meter el TDV **preentrenado en color** (`VLOGroup/tdv`, BSDS400, ruido gaussiano 2D) como paso
proximal dentro de un solver ADMM de QSM. **Sin reentrenar nada.**

Funcional QSM con splitting `z = χ`:
```
argmin_{χ,z}  F(Dχ, φ) + α R(z) + (μ/2)‖χ − z + s‖²₂
```
- Subproblema **χ**: como en FANSI (solución cerrada en Fourier).
- Subproblema **z**: es exactamente un problema de denoising → ahí entra la red TDV preentrenada.
- `s`: multiplicador de Lagrange, `s_{k+1} = s_k + χ_{k+1} − z_{k+1}`.
- `α` actúa como **parámetro de escala** que adapta la red al nivel de ruido de la entrada.
  (Esto es el `target_std = 0.06` / calibración de `alpha` del repo `../tdv`.)

## El truco 2D→3D

> "The pre-trained TDV network for 2D color images was extended to 3D images by inputting groups of
> three consecutive slices as red, green and blue channels."

**Los 3 canales RGB se reutilizan como 3 cortes axiales consecutivos.** Por eso el checkpoint que
importa es el de **color** (`in_channels=3`), no el de gris — y por eso reproducir el entrenamiento
RGB es el paso de validación correcto antes de pasar a 3D real.

## Evaluación

- RC1: simulación forward SNR=100 desde COSMOS (2016 QSM Challenge).
- RC2: campo SIM2SNR1.
- In vivo: voluntaria sana, 3T Philips Achieva, 32 canales, 3D GRE 5 ecos,
  TE1/ΔTE/TR = 3/5.4/29 ms, flip 20°, matriz 240×240×144, 1 mm³ isotrópico.
- Pipeline: ajuste no lineal multi-eco → SEGUE (unwrapping) → LBV + WH-QSM (background) → máscara por R2*.
- Métricas: RMSE, **XSIM** (SSIM adaptado a QSM), métricas RC2.

## Resultados

TDV supera a FANSI-TV y FANSI-TGV en RMSE y XSIM, con menos iteraciones y mayor estabilidad.
Mejor cortical, menos *staircasing* y menos *streaking*. Las mejoras son "promising, albeit subtle",
a costa de más tiempo de cómputo.

## Trabajo futuro que declara el propio paper

> "Our implementation is not optimal, and may be improved [...] by **retraining the TDV network with
> 3D data**. Retraining may also target suppression of QSM streaking artifacts."

Eso es precisamente el objetivo de `QSM-TDV` + este repo.

---
Ver también: [tdv-stable-regularizer-2020](tdv-stable-regularizer-2020.md),
[qsm-deep-learning-comparison-2025](qsm-deep-learning-comparison-2025.md).
