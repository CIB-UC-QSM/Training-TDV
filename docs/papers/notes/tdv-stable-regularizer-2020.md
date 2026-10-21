# TDV: A Stable Regularizer for Inverse Problems

**Kobler, Effland, Kunisch, Pock — arXiv:2006.08789 (2020).** Versión larga del paper CVPR 2020
*Total Deep Variation for Linear Inverse Problems*. PDF: `../tdv-stable-regularizer-2020.pdf` (30 pp).

> **Este es EL paper de referencia del repo.** La receta de entrenamiento de la sección 5.1 es
> exactamente lo que hay que reproducir en RGB antes de tocar QSM.

---

## 1. Qué es TDV (y qué no es)

TDV **no es un denoiser**. Es un **regularizador energético** `R(x, θ)` aprendido, que se usa dentro
de un esquema variacional. La red nunca predice la imagen limpia: predice un **escalar de energía**
por muestra, y lo que entra al esquema de reconstrucción es su **gradiente** `D₁R(x, θ)`.

```
min_x  D(Ax, z) + R(x, θ)      ← formulación variacional clásica
                 └── aquí vive la red
```

Consecuencia práctica: en el código, `R.energy(x)` y `R.grad(x)` son APIs distintas, y la que se
usa en el bucle es `grad`. Ver `ddr/tdv.py:_transformation` / `_transformation_T`.

## 2. Formulación de control óptimo

El entrenamiento es un **problema de control óptimo discretizado** (sec. 2-3), no un ajuste
supervisado plano. Los controles son `(T, θ)`: el tiempo de parada `T` y los pesos `θ`.

Discretización semi-implícita del estado (ec. 7):

```
x_{s+1} = x_s − (T/S) Aᵀ(A x_{s+1} − z) − (T/S) D₁R(x_s, θ)
```

equivalente a la forma cerrada

```
x_{s+1} = (Id + (T/S) AᵀA)⁻¹ ( x_s + (T/S)(Aᵀz − D₁R(x_s, θ)) )
```

con estado inicial `x₀ = A_init z`. El costo discreto es

```
J_S(T, θ) = E_{(y,ξ)} [ ℓ( x̂_S(y, ξ, T, θ) − y ) ]
```

o sea: **la pérdida se evalúa solo en el estado final `x_S`**, y el gradiente se propaga a través de
las S iteraciones completas. Eso hace que entrenar TDV sea diferenciar a través de un gradiente
(derivada segunda de la energía) — ver `force` vs `force_autograd_reference` en QSM-TDV.

**Estado adjunto** (Teorema 3.2) — útil si alguna vez se implementa backprop manual:
```
p_s = (Id − (T/S) D₁₁R(x_s, θ)) (Id + (T/S) AᵀA)⁻¹ p_{s+1},    p_S = −Dℓ(x_S − y)
```

**Condición de optimalidad del tiempo de parada** (Teorema 3.3), evaluable numéricamente:
```
E [ Σ_{s=0}^{S-1} ⟨ p_{s+1}, (Id + (T/S)AᵀA)⁻¹ (x_{s+1} − x_s) ⟩ ] = 0
```

> ⚠️ **Desviación conocida del código respecto al paper.** La ec. (7) acopla ambos términos al mismo
> `T/S`. Tanto `tdv/ddr/model.py` (VNet) como QSM-TDV desacoplan en dos coeficientes aprendidos
> independientes, `T` y `λ`:
> ```
> x_{s+1} = x_s − (T/S)·grad_R − (λ/S)·D.grad(x_s, z)      # rama explícita
> x_{s+1} = D.prox(x_s − (T/S)·grad_R, z, λ/S)             # rama proximal (use_prox=True)
> ```
> Para denoising L2 la rama proximal es `(x + τz)/(1+τ)`, que **sí** recupera la ec. (7) cuando
> `λ = T`. Los checkpoints originales usan `use_prox=True` con `λ ≠ T`. No "corregir" esto pensando
> que es un bug.
>
> ⚠️ **Corregido el 2026-09-06.** Acá decía que ese `λ ≠ T` prueba que «el código generaliza el
> paper». **`λ/T` no es invariante de gauge**: `(T, K_N) → (cT, K_N/c)` deja el esquema idéntico y no
> toca `λ`, así que con `c = λ*/T* = 4.227` el checkpoint oficial se escribe exactamente con
> `λ = T = 0.1045`, salida idéntica a 5·10⁻⁸. El checkpoint publicado **es** la ec. (7) en otras
> unidades.
>
> ⚠️ **Ampliado el 2026-09-06.** Verificado contra `VLOGroup/tdv` (rama `master`), que es lo único
> publicado por los autores —y es **sólo inferencia**: no hay entrenamiento ni optimizador—: su
> `model.py` define `T` y `lmbda` como dos `Parameter` independientes, con los únicos modos `fixed`
> y `learned`, **sin ningún modo atado**; y el `config` serializado dentro de los dos `.pth` dice
> `learned` en los dos. **Atarlos está descartado**: Adam no es invariante (AGENTS.md, «Lo que no
> sirve»).

## 3. Arquitectura

```
ρ(x) = (1/m) · K_N ∘ M_L ∘ ... ∘ M_1 ∘ K_1 (x)          m = num_features
R(x) = Σ_pixels ρ(x)
```

> **Procedencia, corregida el 2026-09-07.** El paper (sec. 2, p.5) escribe `r(x,θ) = Ψ(w N(Kx))`:
> **el `1/m` no está**, y de `K` sólo enuncia la restricción de media cero (`Σ_i K_{j,i} = 0`), que
> además comenta como origen de la *shift-invariance* espacial y radiométrica. La **cota de norma
> `‖·‖₂ ≤ 1` por filtro no aparece en el PDF ni en el CVPR 2020**, que sólo dice «The zero-mean
> constraint of the kernel K is enforced by a projection». Las dos cosas salen del **código de los
> autores**, no del texto, y las dos son correctas —los pesos publicados tienen K₁ con media
> 8.8·10⁻⁹ y normas por filtro entre 0.991972 y 1.000000, o sea proyectado a la esfera y no a la
> bola—, pero la fuente es el código. Es exactamente la clase de confusión que produjo el
> `T_init = 0.001` letal.

- **K₁**: convolución de análisis 3×3 aprendida — *bias-free*, **media cero** (del paper) y
  **norma ℓ₂ ≤ 1** por filtro de salida (del código de los autores, no del paper). Se re-proyecta
  después de **cada** paso del optimizador.
- **M_ℓ**: macro-bloque multiescala (3 escalas). Con `num_scales=3` son exactamente **5 micro-bloques**:
  uno antes de bajar y uno después de subir en las escalas 0 y 1, más uno en la escala más gruesa.
  El estado multiescala **persiste entre macro-bloques** (`[x₀, x₁, x₂]`, inicializado `[x, None, None]`).
- **K_N**: cabeza 1×1 de un canal, bias-free.

**Micro-bloque** (residual, bias-free): `v = u + K₂ φ(K₁ u)`

**Activación Student-t** con α = 1 — *no ReLU*:
```
φ(a)  = ½ log(1 + a²)
φ'(a) = a / (1 + a²)
```
La suavidad es un requisito matemático, no una preferencia: la energía debe ser C² para que el
análisis de estabilidad y el estado adjunto existan.

Jacobiano transpuesto del micro-bloque:
```
J(u)ᵀ q = q + K₁ᵀ [ φ'(K₁u) ⊙ K₂ᵀ q ]
```

- **Padding**: simétrico con su transpuesta exacta (`symmetric_pad2d` / `symmetric_pad2d_transpose`
  en `ddr/conv.py`). La reflexión de PyTorch **no** es equivalente.
- **Downsampling**: convolución escalada aprendida + antialias binomial separable `[1,4,6,4,1]/16`
  antes del submuestreo stride-2. Los operadores down y up son **independientes** (no comparten pesos);
  el up llama a la adjunta de su propio operador.

### Función potencial ψ

Ablación (Tabla 1, gris σ=25): `ln cosh(x)` → 29.3596 · `½log(1+x²)` → 29.3662 · **`x` → 29.3722**.
Gana la identidad. Por eso en `ddr/tdv.py`:
```python
def _potential(self, x):  return x / self.num_features
def _activation(self, x): return torch.ones_like(x) / self.num_features   # ψ'(x) = 1
```

## 4. Receta de entrenamiento (sección 5.1) — LA QUE HAY QUE REPRODUCIR

| Ítem | Valor |
|---|---|
| Dataset de entrenamiento | **BSDS400** |
| Test | BSDS68 (gris) / **CBSDS68** (color), Set12, Urban100, Kodak24, McMaster |
| Tamaño de parche | **93 × 93** |
| Augmentación | flips horizontales/verticales aleatorios + rotaciones múltiplo de 90° |
| Optimizador | **Adam**, β₁ = 0.9, β₂ = 0.999 |
| Mini-batch | **32** |
| Iteraciones | **10⁵** |
| Learning rate | **4·10⁻⁴**, se **divide a la mitad cada 25 000 iteraciones** |
| Pérdida | `ℓ(x) = ‖x‖₂²` (denoising) — evaluada solo en `x_S` |
| Ruido | `ξ ~ N(0, σ²)`, **re-muestreado en cada iteración** (no precomputado) |
| S (pasos desenrollados) | **10** |
| Arquitectura | **TDV33** = 3 macro-bloques, 3 escalas, 32 features |
| σ principal | **25** (también 15 y 50) |

Ablaciones que ya están hechas — no repetirlas:
- El PSNR satura en **S ≈ 10**; más allá no mejora.
- El PSNR satura en 3 bloques / 3 escalas; más parámetros no compensan.
- ψ(x) = x gana.

**Transferencia a otros σ sin reentrenar**: reescalar la entrada `x_init = (25/σ)·z`, aplicar el
esquema, y devolver `x_S ← (σ/25)·x_S`.

## 5. Resultados de referencia (objetivo de la réplica)

**Color, σ=25, TDV33,25 — Tabla 3:**

| Dataset | σ=15 | σ=25 | σ=50 |
|---|---|---|---|
| **CBSDS68** | 34.12 | **31.53** | 28.26 |
| Kodak24 | 35.01 | 32.59 | 29.44 |
| McMaster | 34.55 | 32.47 | 29.41 |

Parámetros: **387 970** (color) / 387 394 (gris). Compárese con FOCNet: 53.5 M para +0.01 dB.

**Gris, σ=25:** BSDS68 29.37 · Set12 30.68 · Urban100 30.38.

**Tiempo de parada óptimo aprendido:**
- gris σ=25 → `T* = 0.0172`
- color σ=25 → `T* = 0.0247`

> ⚠️ **Errata del paper.** El texto de la sección 5.2 dice "T\* = 0.172 for gray-scale denoising".
> Es un error tipográfico: el eje de la Figura 3 va de 0.01 a 0.05, y el checkpoint oficial
> `tdv3-3-25-f32-gray.pth` guarda `T = 0.017222747`. El valor correcto es **0.0172**.
> El de color (0.0247) sí coincide exacto con el checkpoint (0.024726812). Verificado leyendo
> los `.pth` en `../tdv/checkpoints/`.

## 6. Otras aplicaciones demostradas (sin reentrenar)

El mismo TDV33,25 entrenado *solo* para denoising gris se aplica tal cual a **CT**, **MRI** y
**super-resolución** cambiando únicamente el operador `A`. Esto es el argumento central de
"general-purpose regularizer" — y la justificación de todo el proyecto QSM: el prior es reutilizable
porque no está atado al operador directo.

## 7. Estabilidad (secs. 4, 5.7-5.9)

Cotas de Lipschitz `L_x`, `L_θ` estimadas empíricamente sobre 10⁵ parches 128×128 de BSDS400.
Incluye ataques adversariales y una cota empírica del error de generalización. Es el material que
justifica la palabra "Stable" del título; relevante si hay que defender robustez, no para implementar.

## 8. Análisis de eigenfunciones

Sec. 5.6: se buscan `x` con `D₁R(x,θ) ≈ λx` para visualizar qué estructuras "prefiere" el
regularizador. Implementado en `../tdv/ddr/eigenfunctions.py`. Es un diagnóstico útil para
comparar un TDV reentrenado contra el original.

---

Ver también: [tdv-qsm-iterative-2022](tdv-qsm-iterative-2022.md) (aplicación a QSM),
[optimal-control-deep-learning-2018](optimal-control-deep-learning-2018.md) y
[deep-learning-optimal-control-2019](deep-learning-optimal-control-2019.md) (el marco de control
óptimo), [data-driven-inverse-problems-2025](data-driven-inverse-problems-2025.md) (dónde encaja TDV
en el panorama).

## El conjunto admisible incluye `T = 0`, y la condición de optimalidad no lo cubre

Anotado el 2026-09-02, releyendo la sec. 3 a raíz de las divergencias del entrenamiento.

- **Teorema 3.1**: el mínimo de `J_S` **se alcanza** sobre el compacto `[0,Tmax] × Θ`. Ese conjunto
  **incluye `T = 0`**, o sea que el modelo identidad es un punto admisible del problema.
- **Teorema 3.3 presupone `T* ≠ 0`.** La demostración deriva la condición de optimalidad respecto de
  `T` dividiendo por `T*`: *"the optimality condition of L w.r.t. T\* reads as
  E[−(1/T\*)Σ⟨p*_{s+1}, (Id + (T*/S)A*A)⁻¹(x*_{s+1} − x*_s)⟩] = 0"*. En `T = 0` no dice nada.

Conclusión (nuestra, no del paper): cuando el entrenamiento colapsa a `T = 0` **no sale del conjunto
admisible** — cae en una esquina que la condición de optimalidad no caracteriza. Medido sobre los
pesos finales de una corrida colapsada: `‖∇θ‖ = 0` exacto y `∂L/∂T = +3.2·10⁴`, o sea un punto KKT
de frontera, no un punto estacionario.

**Consecuencia práctica:** evaluar numéricamente la condición del Teorema 3.3 sólo tiene sentido
mientras `T > 0`.

## El adjunto discreto es la retropropagación

`p_s = (Id − (T/S)·D₁²R(x_s,θ))·(Id + (T/S)AᵀA)⁻¹·p_{s+1}`, con `p_S = −Dℓ(x_S − y)` (Teorema 3.2).

Es *discretize-then-optimize*: se discretiza el flujo primero (ec. 7) y se optimiza el problema
discreto, así que **el adjunto coincide exactamente con lo que calcula autograd** por el esquema
desenrollado. El teorema prueba que el gradiente existe y lo da en forma cerrada; **no** describe un
algoritmo alternativo. La sec. 5.1 confirma que la optimización es Adam a secas, y la intro lo dice
en una línea: *"learning is based on unrolling a finite number of iterations of a gradient descent
algorithm"*. No hay MSA.

Detalle que vale: **`D₁²R` —el Hessiano del regularizador— aparece explícito en el adjunto.** Es la
confirmación del propio paper de que entrenar exige segundas derivadas de la energía, y por lo tanto
que `φ` tiene que ser C².

