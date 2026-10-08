# Deep Learning as Optimal Control Problems: Models and Numerical Methods

**Benning, Celledoni, Ehrhardt, Owren, Schönlieb — arXiv:1904.05657 (2019), 34 pp.**
PDF: `../deep-learning-optimal-control-2019.pdf`

## Contenido

- Revisa las redes profundas como **discretizaciones de un problema de control óptimo sujeto a una
  EDO** (línea Haber–Ruthotto, LeCun 1988).
- Revisa las **condiciones de primer orden** para optimalidad y, crucialmente, las condiciones que
  garantizan que la optimalidad **se preserva tras discretizar**.
- De ahí deriva una clase de algoritmos que satisfacen las condiciones necesarias discretas
  (integradores simplécticos / métodos que conmutan discretización y optimización).
- Explora **aprender la discretización temporal misma** (los pasos `h_s`) como parámetro adicional,
  con restricciones naturales (p. ej. los pasos viven en un símplex).
- Comparación numérica en clasificación binaria: flujo inducido y capacidad de generalización.

## Relevancia para el proyecto

Dos cosas concretas:

1. **"Discretize-then-optimize" vs "optimize-then-discretize".** TDV hace lo primero: fija S = 10 y
   optimiza el problema ya discretizado. Este paper explica cuándo eso es consistente. Importa si
   alguna vez se cambia S entre entrenamiento e inferencia — algo que en TDV **no** es gratis,
   porque `T` está aprendido conjuntamente con S.
2. **Aprender el paso temporal.** TDV aprende `T` (y el código aprende `λ`) exactamente en este
   espíritu, con proyección a un intervalo acotado. Ver `reset_scalar` + `p.proj()` en
   `../tdv/ddr/model.py`, y la parametrización `T = T_max·σ(α_T)` en QSM-TDV.

Nivel de uso: **teórico de fondo.** No hay nada que copiar directamente.

---
Ver también: [[optimal-control-deep-learning-2018]], [[tdv-stable-regularizer-2020]].
