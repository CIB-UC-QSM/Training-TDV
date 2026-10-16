# An Introduction to Flow Matching and Diffusion Models

**Holderrieth & Erives (MIT) — arXiv:2506.02070v3 (2026), 84 pp.**
Sitio: https://diffusion.csail.mit.edu/ · PDF: `../flow-matching-diffusion-2026.pdf`

## Qué es

Apuntes del curso de MIT. Más largo, más formal y más completo que
[diffusion-tutorial-2024](diffusion-tutorial-2024.md); cubre desde la teoría hasta arquitecturas de
escala real.

## Estructura

1. Modelado generativo como muestreo.
2. **Modelos de flujo y difusión** (ODEs y SDEs).
3. **Flow matching**: caminos de probabilidad condicionales y marginales, campos vectoriales,
   aprender el campo marginal.
4. **Score matching**: funciones de score condicionales y marginales, muestreo con SDEs.
5. **Guidance**: vanilla y classifier-free.
6. Generadores de imagen/video a escala: arquitecturas, espacio latente (VAEs),
   casos de estudio (Stable Diffusion 3, Meta Movie Gen).
7. Difusión discreta (CTMC) para modelos de lenguaje.

## Relevancia para el proyecto

**Tangencial, exploratoria** — igual que el tutorial de Apple. Es el material de referencia si se
decide explorar priors generativos para QSM en vez de (o junto a) TDV.

La sección de **guidance** es la más pertinente a problemas inversos: condicionar la generación en
una medición es formalmente el mismo problema que resolver `b = Aχ + η` con un prior aprendido.

---
Ver también: [diffusion-tutorial-2024](diffusion-tutorial-2024.md),
[data-driven-inverse-problems-2025](data-driven-inverse-problems-2025.md).
