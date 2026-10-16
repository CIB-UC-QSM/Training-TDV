# Total Deep Variation for Linear Inverse Problems

Kobler, Effland, Kunisch, Pock — arXiv:2001.05005v2, 17-feb-2020 (CVPR 2020).
PDF: `docs/papers/tdv-linear-inverse-problems-2020.pdf` (+ `-supp.pdf`).

**Es el paper *anterior* de los mismos autores.** El de referencia de este repo es
[tdv-stable-regularizer-2020](tdv-stable-regularizer-2020.md), que lo extiende. Esta nota existe
para dejar por escrito **en qué se diferencian las dos recetas**, porque las diferencias no son
cosméticas y es fácil citar la equivocada.

## Por qué importa

Los checkpoints oficiales que usamos como oráculo (`tdv3-3-25-f32-color.pth`) reproducen los números
del **paper extendido**, no los de éste. Todo lo que sigue del CVPR es contexto histórico, **no** la
receta a replicar.

## Las diferencias, lado a lado

| | **CVPR 2020** (este) | **Extendido 2020** (el que seguimos) |
|---|---|---|
| parche | **96 × 96** | **93 × 93** |
| iteraciones | **10⁶** | **10⁵** |
| schedule del lr | no menciona ninguno | **dividido a la mitad cada 25 000** |
| pérdida (denoising) | `½‖x‖²` | `‖x‖²` |
| activación | `φ(x) = 1/(2ν)·log(1+νx²)` con **ν = 9** | `φ(x) = ½log(1+x²)`, o sea **ν = 1** |
| batch, `m`, `S` | 32, 32, 10 | idénticos |
| dataset, augmentación | BSDS400, flips + rot 90° | idénticos |
| macro-bloques | `l` consecutivos, 5 micro-bloques en 3 escalas | idéntico |

Cita textual de la sec. 4.1, para que no haya que volver al PDF:

> *"[…] on batches of size 32 with a patch size 96 × 96. The zero-mean constraint of the kernel K is
> enforced by a projection after each iteration step. For image denoising, we incorporate the squared
> ℓ₂-loss function l(x) = ½‖x‖₂² and the learning rate 4·10⁻⁴ […]. The first and second order momentum
> variables of the ADAM optimizer are set to 0.9 and 0.999 and **10⁶ training steps** are performed.
> Throughout all experiments we set **ν = 9**, the number of feature channels m = 32, and S = 10."*

## Qué se hace con cada diferencia

1. **`ν = 9` vs `ν = 1`: no tocar.** `ν` es el parámetro de la log-student-t, `φ(x) = 1/(2ν)·log(1+νx²)`
   — en el código es `student_t(x, alpha)`, con `alpha = 1.0` fijo. El extendido usa
   `φ(x) = ½log(1+x²)`, o sea `ν = 1`, y **es lo que reproducen los checkpoints oficiales al
   centésimo de dB**. Nuestra implementación ya coincide. `ν` no cambia `φ'(0)=0` ni `φ''(0)=1`:
   controla cuán rápido satura la no convexidad.
2. **`½‖x‖²` vs `‖x‖²`: irrelevante.** Adam es invariante de escala —su paso es `m/(√v+ε)`— así que
   un factor global sobre la pérdida no cambia nada salvo por `ε=10⁻⁸`, despreciable frente a
   gradientes de orden 10²–10⁴. No hay nada que decidir acá.
3. **96 vs 93 píxeles: irrelevante para la receta**, y el extendido es el que manda. Vale saberlo
   sólo si alguna vez se compara contra números del CVPR.
4. **10⁶ vs 10⁵ iteraciones: ésta sí importa, y es la más interesante.**

## El presupuesto de iteraciones se redujo 10× entre los dos papers

El CVPR entrena **10⁶ pasos sin schedule**; el extendido, **10⁵ con halving cada 25 000**. O sea que
los autores **bajaron el presupuesto un orden de magnitud** y en el mismo movimiento agregaron un
schedule que no estaba.

Dos lecturas posibles, y no tenemos con qué elegir entre ellas:

- **Encontraron que 10⁵ alcanzaba** —con el schedule— y publicaron la receta más barata.
- **El modelo cambió** entre los dos papers (`ν` de 9 a 1 es un cambio real de la activación) y con
  él el presupuesto necesario.

Lo que sí es un dato para nosotros: **10⁵ no es un límite natural del problema, es una elección**, y
los mismos autores llegaron a entrenar 10× más. Eso pesa sobre la pregunta abierta *"¿hacían falta
las 10⁵ iteraciones?"* que se hacía entonces el README — la respuesta del CVPR sería que hacían
falta diez veces más.

> ⚠️ **No convertir esto en una excusa para entrenar más.** La réplica sigue siendo la del paper
> extendido: 10⁵ iteraciones con halving. Entrenar 10⁶ es una desviación, y hoy ni siquiera podemos
> llegar sanos a 10⁵.

## Lo que este paper aporta y el extendido no

- **Teorema 3.1 — condición de optimalidad** en la forma directamente evaluable
  `−(1/N)·ΣΣ⟨pⁱ_{s+1}, (Id + (T/S)AᵀA)⁻¹(xⁱ_{s+1} − xⁱ_s)⟩ = 0`. Es la versión operativa de lo que
  el README listaba como pendiente en la fase 1 ("condición de optimalidad de `T`").
- **Teorema 3.2 — sensibilidad al dataset de entrenamiento**: cota computable de `‖x_s − x̃_s‖` para
  dos pares de controles obtenidos de datasets distintos, en términos de la constante de Lipschitz
  `L_R` de `R`. Relevante para la fase 2, donde el dataset cambia por completo.

Ver también: [tdv-stable-regularizer-2020](tdv-stable-regularizer-2020.md) (el de referencia),
[optimal-control-deep-learning-2018](optimal-control-deep-learning-2018.md) y
[deep-learning-optimal-control-2019](deep-learning-optimal-control-2019.md) (el marco).
