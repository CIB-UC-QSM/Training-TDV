# Data-driven Approaches to Inverse Problems

**Schönlieb & Shumaylov (Cambridge) — CIME 2023 lecture notes, arXiv:2506.11732 (2025), 71 pp.**
PDF: `../data-driven-inverse-problems-2025.pdf`

## Qué es

Apuntes de curso, no un paper de resultados. **Es el mapa del territorio**: dónde encaja TDV dentro
del panorama completo de métodos data-driven para problemas inversos.

## Cubre

- Problemas inversos mal planteados y por qué necesitan tratamiento matemático (regularización,
  bien-planteamiento, convergencia).
- **Modelos variacionales data-driven** ← *aquí vive TDV*: el regularizador se aprende, pero la
  estructura variacional (fidelidad + regularizador) se conserva.
- **Plug-and-play (PnP)**: sustituir el operador proximal por un denoiser genérico ← *aquí vive el
  uso original de TDV en QSM* ([tdv-qsm-iterative-2022](tdv-qsm-iterative-2022.md)), que mete un
  denoiser preentrenado como paso proximal de ADMM.
- **Learned iterative / unrolled schemes** ← *aquí vive el entrenamiento end-to-end de TDV*, con S
  pasos desenrollados.
- Enfoques generativos para problemas inversos.

## Por qué importa acá

Da el vocabulario preciso para situar las tres cosas distintas que este proyecto mezcla:

| Enfoque | Dónde está en el proyecto |
|---|---|
| PnP con denoiser preentrenado | `../tdv` — ADMM + red BSDS congelada, escala `alpha` calibrada |
| Regularizador variacional aprendido | El objeto TDV en sí: `R(x,θ)` con energía y fuerza |
| Esquema iterativo desenrollado | El entrenamiento end-to-end con S=10 y pérdida en `x_S` |

Útil también para justificar decisiones en un paper o tesis: acá están las referencias canónicas
y las garantías de convergencia asociadas a cada familia.

**Uso: consulta, no implementación.** Leer el capítulo relevante cuando haya que argumentar una
elección metodológica.

---
Ver también: [tdv-stable-regularizer-2020](tdv-stable-regularizer-2020.md),
[tdv-qsm-iterative-2022](tdv-qsm-iterative-2022.md).
