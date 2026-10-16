# Step-by-Step Diffusion: An Elementary Tutorial

**Nakkiran, Bradley, Zhou, Advani (Apple / Mila) — arXiv:2406.08929 (2024), 51 pp.**
PDF: `../diffusion-tutorial-2024.pdf`

## Qué es

Curso introductorio a modelos de difusión y flow matching, con las matemáticas simplificadas pero
suficientes para derivar algoritmos correctos. Asume cero experiencia previa en difusión.

## Contenido

1. Fundamentos de difusión (difusión gaussiana, discretización).
2. **Muestreo estocástico: DDPM** — corrección, algoritmos, reducción de varianza prediciendo `x₀`,
   difusiones como SDEs.
3. **Muestreo determinista: DDIM** — campos de velocidad, la ODE de flujo de probabilidad,
   DDPM vs DDIM.
4. Flow matching.
5. Aspectos prácticos de entrenamiento y guidance.

## Relevancia para el proyecto

**Tangencial, exploratoria.** No hay difusión en TDV ni en el pipeline de QSM actual.

Está en la carpeta porque los priors generativos son la alternativa moderna al regularizador
aprendido tipo TDV para problemas inversos: en vez de aprender `R(x,θ)` y bajar por su gradiente,
se aprende el score `∇log p(x)` y se muestrea condicionando en la medición. Si en algún momento se
evalúa un prior de difusión para QSM, este es el punto de entrada más barato.

Empezar por acá y solo después ir a [flow-matching-diffusion-2026](flow-matching-diffusion-2026.md),
que es más largo y formal.

---
Ver también: [flow-matching-diffusion-2026](flow-matching-diffusion-2026.md),
[data-driven-inverse-problems-2025](data-driven-inverse-problems-2025.md).
