# Índice de papers

PDFs en `docs/papers/`, notas destiladas en `docs/papers/notes/`.
**Leer la nota primero**; abrir el PDF solo si la nota no alcanza.

## Núcleo — leer antes de tocar el código

| Paper | Nota | De qué sirve |
|---|---|---|
| **TDV: A Stable Regularizer for Inverse Problems** (Kobler et al., 2020) | [notes/tdv-stable-regularizer-2020.md](notes/tdv-stable-regularizer-2020.md) | **El paper de referencia.** Arquitectura, formulación de control óptimo, y la receta de entrenamiento exacta (sec. 5.1), reproducida en 2D: 31.547 dB en CBSD68 a σ = 25, contra 31.53 publicados |
| **TDV for Linear Inverse Problems** (Kobler et al., CVPR 2020) | [notes/tdv-linear-inverse-problems-2020.md](notes/tdv-linear-inverse-problems-2020.md) | El paper **anterior** de los mismos autores. Su receta **difiere**: 96×96, 10⁶ pasos sin schedule, ν=9. Leerla antes de citar cualquier número, para no confundir las dos |
| **TDV Regularization for Improved Iterative QSM** (Milovic et al., ISMRM 2021) | [notes/tdv-qsm-iterative-2022.md](notes/tdv-qsm-iterative-2022.md) | Origen del proyecto: TDV color preentrenado dentro de ADMM-QSM, con los 3 canales RGB usados como 3 cortes |
| **A Fair Comparison in DL-Based QSM Reconstruction** (Oh et al., 2025) | [notes/qsm-deep-learning-comparison-2025.md](notes/qsm-deep-learning-comparison-2025.md) | Estándar metodológico: por qué la validación RGB no es opcional y cómo reportar resultados sin engañarse |

## Marco teórico

| Paper | Nota | De qué sirve |
|---|---|---|
| An Optimal Control Approach to Deep Learning (Li & Hao, 2018) | [notes/optimal-control-deep-learning-2018.md](notes/optimal-control-deep-learning-2018.md) | PMP discreto y MSA; el vocabulario estado/adjunto/control que usa TDV |
| DL as Optimal Control Problems (Benning et al., 2019) | [notes/deep-learning-optimal-control-2019.md](notes/deep-learning-optimal-control-2019.md) | Discretize-then-optimize; aprender el paso temporal (justifica `T` y `λ` aprendidos) |
| Data-driven Approaches to Inverse Problems (Schönlieb & Shumaylov, 2025) | [notes/data-driven-inverse-problems-2025.md](notes/data-driven-inverse-problems-2025.md) | Mapa del territorio: variacional aprendido vs plug-and-play vs unrolled |

## Exploratorio — nada de esto está en el pipeline actual

| Paper | Nota | De qué sirve |
|---|---|---|
| Step-by-Step Diffusion (Nakkiran et al., 2024) | [notes/diffusion-tutorial-2024.md](notes/diffusion-tutorial-2024.md) | Entrada barata a difusión, por si se evalúan priors generativos |
| Flow Matching and Diffusion Models (Holderrieth & Erives, 2026) | [notes/flow-matching-diffusion-2026.md](notes/flow-matching-diffusion-2026.md) | Versión larga y formal; la sección de guidance es la relevante a problemas inversos |
| Practitioner's Guide to KANs (Noorizadegan et al., 2026) | [notes/kan-practitioners-guide-2026.md](notes/kan-practitioners-guide-2026.md) | Especulativo: reemplazar la activación Student-t por una base aprendida (ojo con la restricción de suavidad C²) |
