# An Optimal Control Approach to Deep Learning

**Qianxiao Li, Shuji Hao — ICML 2018, arXiv:1803.01299 (20 pp).**
PDF: `../optimal-control-deep-learning-2018.pdf`

## Por qué está en esta carpeta

Es una de las fuentes del marco teórico que TDV adopta en sus secciones 2-3. Explica **por qué**
tiene sentido escribir el entrenamiento como control óptimo en vez de como minimización plana.

## Contenido

- Formula el deep learning como un problema de **control óptimo en tiempo discreto**: las capas son
  pasos temporales, los pesos son controles.
- Introduce el **Principio del Máximo de Pontryagin (PMP)** discreto como conjunto alternativo de
  condiciones necesarias de optimalidad — distinto de `∇_θ J = 0`.
- Propone el **Método de Aproximaciones Sucesivas (MSA)** discreto: un algoritmo de entrenamiento
  que **no usa gradientes respecto a los parámetros**. Itera: forward del estado → backward del
  adjunto → maximización del Hamiltoniano respecto al control.
- Da una cota de error rigurosa para MSA que explica su dinámica y cómo estabilizarlo.
- Aplicación: redes con **pesos discretos** (binarios/ternarios), donde el gradiente no existe.
  Obtiene redes ternarias muy dispersas.

## Relevancia para el proyecto

**Conceptual, no de implementación.** TDV usa el marco (estado, adjunto, Hamiltoniano) pero entrena
con Adam y backprop estándar, no con MSA. Lo útil de acá es el vocabulario: *estado*, *adjunto*,
*control*, *condición de optimalidad del tiempo de parada*.

Si alguna vez el entrenamiento de TDV se vuelve inestable por diferenciar a través de S pasos, MSA
es la alternativa teórica que este paper ofrece — pero es un cambio grande y no está justificado
salvo que backprop falle de verdad.

---
Ver también: [deep-learning-optimal-control-2019](deep-learning-optimal-control-2019.md),
[tdv-stable-regularizer-2020](tdv-stable-regularizer-2020.md).
