# A Practitioner's Guide to Kolmogorov–Arnold Networks

**Noorizadegan, Wang, Ling, Dominguez-Morales — arXiv:2510.25781v5 (2026), 75 pp.**
Repo: https://github.com/AmirNoori68/kan-review · PDF: `../kan-practitioners-guide-2026.pdf`

## Qué es

Review sistemático de la literatura de KANs, redes con funciones de activación **aprendidas sobre
las aristas** en vez de pesos lineales + activación fija.

## Organización

1. Relación entre KANs y el teorema de superposición de Kolmogorov (inspiración, **no** derivación
   estricta — el paper insiste en esto).
2. Relación con MLPs y con métodos de kernel clásicos.
3. **Funciones base como eje central de diseño**: B-splines, y alternativas (Chebyshev, wavelets, RBF…).
4. Avances en precisión, eficiencia, regularización y convergencia.
5. Guía práctica "Choose-Your-KAN" + problemas abiertos.

## Relevancia para el proyecto

**Especulativa.** Ninguna parte del pipeline actual usa KANs.

El punto de contacto real, si lo hay, es la **activación Student-t** de TDV:
`φ(a) = ½log(1+a²)`. Es una no linealidad fija, escalar, elegida por sus propiedades de suavidad
(la energía debe ser C² para que existan la fuerza y el estado adjunto). Una KAN reemplazaría esa
función fija por una base aprendida.

⚠️ Antes de intentarlo, notar la restricción dura: **cualquier activación candidata debe tener
derivada primera y segunda continuas y acotadas**, o se rompe el análisis de estabilidad de
[tdv-stable-regularizer-2020](tdv-stable-regularizer-2020.md) y la fuerza deja de ser el gradiente
exacto de una energía. Las B-splines de grado ≥ 3 califican; muchas otras bases no. Además, TDV ya
mostró (Tabla 1) que la elección del potencial mueve el PSNR en ~0.01 dB — el margen de mejora por
esa vía es minúsculo.

**Uso: idea de largo plazo, baja prioridad.**

---
Ver también: [tdv-stable-regularizer-2020](tdv-stable-regularizer-2020.md).
