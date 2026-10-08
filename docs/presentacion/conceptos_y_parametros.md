# Conceptos y parámetros de la fase 1

Hoja de referencia para la presentación: qué significa cada término y qué valores se usaron en
cada corrida. Los parámetros salen de `train.py` (constantes) y de los `config` guardados dentro de
cada checkpoint, no de memoria. La explicación sin fórmulas de semilla, brazo y hazard está también
en [`docs/metricas.md`](../metricas.md), sección 0.

---

## 1. Conceptos

### El modelo

| término | qué es |
|---|---|
| **TDV** | *Total Deep Variation* (Kobler et al., 2020): un regularizador aprendido. Una red `R(x)` que devuelve un número: cuán «poco natural» es la imagen `x`. |
| **energía** | `E(x) = (λ/T)·½‖x − z‖² + R(x)`: parecerse a la imagen ruidosa `z` + parecer natural. Quitar el ruido = bajar por esta energía. |
| **S** | número de pasos de descenso: **10**, fijo. La salida es `x_S`. |
| **T** | «tiempo de parada»: el tamaño total del descenso (cada paso es `T/S`). Se **aprende**. |
| **λ** | peso del término de datos (cuánto se respeta `z`). Se **aprende**. |
| **w = λ/T** | el peso del dato en la energía. **Depende del gauge**: `(T, K_N) → (cT, K_N/c)` no toca `λ`, así que `w` cambia sin que el modelo cambie (el oficial se puede escribir con `w = 1`). No sirve para comparar modelos distintos (`docs/mediciones/gauge-lambda-T.txt`); sí para seguir una misma corrida en el tiempo. |
| **φ** | la no linealidad de cada bloque: `φ(a) = ½·log(1 + a²)`. Suave, con derivada acotada. Fija. |
| **∇R** | el gradiente del regularizador: hacia dónde y cuánto «empuja» la imagen en cada paso. |
| **paso efectivo** | `T·‖∇R‖/‖ruido‖`: cuánto se mueve la imagen en total, en unidades del ruido. Es el **invariante**: ≈1.1–1.3 en el oficial y en toda corrida sana. `T` sola no está determinada —`(T, K_N) → (cT, K_N/c)` da el mismo modelo—; el producto sí. |
| **`corr`** | `‖x_S − z‖/‖ruido‖`: cuánto corrige la red. Sano ≈ 0.95; en un estallido llega a cientos. |
| **checkpoint oficial / oráculo** | el modelo publicado por los autores (`checkpoints/Oficiales/tdv3-3-25-f32-color.pth`): 31.534 dB con nuestro código. Es contra lo que se compara; **nunca** se entrena partiendo de él. |

### El entrenamiento

| término | qué es |
|---|---|
| **iteración** | un paso de Adam con un lote de 32 parches de 93×93, ruido nuevo cada vez. La receta son 100 000. |
| **semilla** | el número que fija todo el azar de una corrida: init de los pesos, orden de los parches, ruido. Misma semilla + modo `referencia` = misma corrida bit a bit. |
| **brazo** | un grupo de corridas con la misma configuración y distintas semillas. |
| **modo** | `referencia`: fp32 IEEE, determinista, **bit-reproducible** (es el de las finales). `intermedio`: TF32. `rapido`: bf16. Los dos últimos usan `cudnn.benchmark` y no repiten. |
| **ULP** | la distancia entre un float32 y el siguiente representable (relativa: ~6·10⁻⁸ del valor). Un ULP de diferencia 284 it antes cambia si una corrida estalla o no: por eso una muerte concreta **no es atribuible** a nada. |
| **δ_Adam** | `lr/√(1 − β₂) = 4·10⁻⁴/√0.001 = 0.01265`: cuánto mueve a un parámetro un pico de gradiente bajo Adam, sea del tamaño que sea (en régimen estacionario, t ≫ 1 000). Con el lr a la mitad, δ también. |
| **pared de T** | `T < δ`: un solo pico puede llevar `T` a 0. Por eso `T_init` se mide en unidades de δ («8δ»). |
| **halving** | el lr se divide por 2 cada 25 000 it (25k, 50k, 75k). En las finales cada uno dio un escalón de PSNR (+0.03, +0.01 y +0.0075 dB) y una caída de ‖g‖ de 14–25 % (`salida/salto_lr.md`). |
| **meseta** | el tramo final donde la PSNR casi no sube (< 0.01 dB entre la 52k y la 60k). |

### Cómo falla

| término | qué es |
|---|---|
| **‖g‖** | la norma del gradiente de todos los parámetros en una iteración. Sana: ~15–100. |
| **evento** | la primera iteración con `‖g‖ > 10⁴`. El umbral cae en el hueco entre lo que se resuelve solo (hasta 6 113) y lo que necesita intervención (desde 1.97·10⁵). |
| **muerte por T → 0** | el modo de los `T_init` bajos: la receta vieja (0.001 < δ) y también 2δ = 0.0253 (`tinit/A-seed4`, `tinit/B-larga`, `clip/base-seed5`). `T` llega al clamp en 0 —por un pico si arranca cerca de δ, o por una deriva hacia la pared, con ‖g‖ normal, en corridas largas (`## Descartado` de la bitácora)—, la salida es la imagen ruidosa (20.17 dB) y los pesos dejan de recibir gradiente. Para los pesos es absorbente (v5 quedó así 97 500 it, de la 2 500 a la 100 000 según su log; bitácora :1009 dice 92 000); `T` misma puede despegar según el signo de ∂L/∂T (v1, bitácora :211), pero ninguna corrida se recuperó del todo. |
| **estallido** | el modo actual: el gradiente que vuelve por los 10 pasos se amplifica (hasta 10¹⁹). Con `T` sana. |
| **evento ≠ muerte ≠ corrida perdida** | una corrida puede estallar y seguir sin NaN pero inservible (`referencia-100k`: 98 000 it, −16 dB). El criterio es la PSNR, no «no muere». |
| **hazard / tasa** | estallidos por iteración (~3·10⁻⁴, y crece con la corrida). Es lo único que es propiedad de una configuración. |
| **σ_max (amplificador adjunto)** | la máxima amplificación del operador que lleva el gradiente hacia atrás por los 10 pasos. Mediana 1.8–2.1 en los nueve estados sanos medidos (máximo por parche 1.5–5.1; 61 en `tinit01-s3` antes de su evento); en `referencia-100k`, mediana 176 (it 38k) y 3.3·10⁴ (it 100k), máximo 2.3·10³ y 2.6·10⁵. |

### El disyuntor (*gradient rejection* + rebobinado)

| término | qué es |
|---|---|
| **rechazo** | no dar el paso de Adam con ese lote. Así los momentos `m`, `v` no se contaminan con el pico. |
| **razón (prueba relativa)** | rechazar si `‖g‖ > 5 × mediana` de las últimas 100 normas aceptadas. Se arma cuando hay 100. |
| **cota absoluta** | rechazar si `‖g‖ > 10⁴`, haya o no mediana. |
| **rebobinado** | con 25 rechazos seguidos, volver al último respaldo: pesos + `m` + `v` (+ la ventana de la mediana, desde v2). |
| **respaldo** | copia en RAM del estado, cada 250 it sanas; desde v2 exige 3 limpias seguidas. |
| **trabarse** | rechazar todo para siempre: la mediana deja de actualizarse y la corrida no da pasos (le pasó al primer disyuntor: 45 585 it quieta). |

### Las métricas

| término | qué es |
|---|---|
| **PSNR** | `20·log10(1/RMSE)` en dB, rango de datos 1, promedio sobre las 68 imágenes de CBSD68. Más es mejor; +0.1 dB ya se nota. |
| **NRMSE** | `‖x̂ − y‖/‖y‖`: error relativo a la energía de cada imagen (en %). Menos es mejor. Es la métrica habitual en QSM. |
| **SNR** | `10·log10(‖y‖²/‖error‖²)` en dB. De entrada (imagen ruidosa) baja al subir σ; de salida es `−20·log10(NRMSE)`. «Evaluar a distintos SNR» = evaluar a distintos σ. |
| **reescalado de σ** | la red se entrenó a σ = 25; para otro σ se denoisa `z·25/σ` y se reescala la salida (paper, sec. 5.2). Aproximación: la red no es escala-equivariante. |
| **eigenfunción no lineal** | una imagen `x` con `∇R(x) = Λx`: lo que R considera más «natural». Se busca por descenso desde un recorte. **Residuo** bajo (≲ 0.04) = convergió; alto = R está roto. |

---

## 2. Parámetros fijos: la receta del paper (`train.py:30-48`)

| parámetro | valor | nota |
|---|---|---|
| optimizador | Adam, β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸ | un solo grupo, un solo lr |
| lr | 4·10⁻⁴, ÷ 2 cada 25 000 it | δ_Adam: 0.01265 → 0.00632 → 0.00316 → 0.00158 |
| iteraciones | 100 000 | las tandas de prueba corren 4 000 o 10 000 |
| lote | 32 (2 micro-lotes de 16 acumulados) | una GPU por corrida |
| parche | 93×93, BSDS400 (train + test) | CBSD68 queda fuera, para evaluar |
| ruido | σ = 25 (en 0–255), nuevo en cada iteración | |
| S | 10 pasos | |
| arquitectura | TDV 3-3: 32 features, 3 escalas, 3 macro-bloques | 387 970 parámetros |
| clamp de T y λ | [0, 1000] | |
| weight decay, warmup, EMA | no | |

## 3. El disyuntor: valores (`train.py:60-66`)

| parámetro | 1er disyuntor (`disyuntor-100k`, 3 oct) | v1 (`tinit-*-10k`, cortadas 18k) | **v2 (finales)** | de dónde sale el valor |
|---|---|---|---|---|
| razón sobre la mediana | 31.62 = 1/√(1−β₂) | **5** | **5** | debajo de la banda 7.70–8.97× que metió al primero; cuesta 0.15 % de los lotes sanos |
| cota absoluta | — | 10⁴ | **10⁴** | el hueco [6 113; 1.97·10⁵] entre lo que se resuelve solo y lo que no |
| ventana de la mediana | 100 | 100 | **100** | normas aceptadas |
| rechazos seguidos para rebobinar | — (no rebobinaba) | 25 | **25** | el máximo sano medido es 18 |
| respaldo cada | — | 250 it sanas | **250 it sanas** | |
| limpias exigidas al respaldo | — | — | **3** | dentro de un evento hay lotes aceptados sueltos (hueco 1) |
| la ventana viaja con el respaldo | — | no (se borraba) | **sí** | borrarla dejaba la razón desarmada 100 it justo al salir del evento |

Costo en las finales: s0 (cortada en la 66k) rechazó 18 lotes y no rebobinó; s1 (100k) rechazó
163 y rebobinó 4 veces (it 24 796, 27 123, 29 419 y 99 365). Ningún lote lo agarró sólo la cota.
`salida/numeros.md`.

## 4. T y λ iniciales, por familia de corridas (del `config` dentro de cada checkpoint)

| corridas | fecha | `T_init` | en δ | `λ_init` | modo / precisión |
|---|---|---|---|---|---|
| v1, v4, v5, v6, v7 (receta vieja) | 31 ago – 2 sep | 0.001 | 0.08 δ | 0.01 | v1/v4/v7 bf16; v5 TF32; v6 fp32 IEEE |
| `tinit/A`, `traza`, `nodDP`, `clip`, `ablacion` | 2–4 sep | 0.0253 | 2 δ | 0.01 | varios |
| `tinit01` | 4 sep | 0.1 | 7.9 δ | 0.01 | referencia |
| `lambdainit/A` | 6 sep | 0.0253 | 2 δ | 0.0253 (atado a T) | referencia |
| `lambdainit/B`, `precision`, `referencia-100k`, `traza-adjunta`, `caza-adjunta`, `contrafactual`, `disyuntor-100k`, `tinit-7p9delta-10k` | 6 sep – 4 oct | 0.1 | 7.9 δ | 0.1 | referencia (`precision`: rápido/intermedio) |
| `repeat5` (5 copias bit a bit) | 25–26 sep | 0.0632 | 5 δ | 0.1 | referencia |
| `tinit-10delta-10k` | 4 oct | 0.1265 | 10 δ | 0.1 | referencia |
| **finales** `t8delta-100k-s0/s1` (y sus cortadas) | 5–7 oct | **0.10119** | **8 δ** | **0.1** | **referencia (fp32 IEEE)** |
| `tinit-2delta-50k` (pre-registro: bitácora :3733) | 6–7 oct | 0.0253 | 2 δ | 0.1 | referencia, disyuntor v2; cortada a mano en la it 4 000 (`salida/tinit_2delta.md`) |

El default actual de `train.py` es `T_INIT = 10·δ = 0.1265` y `LAMBDA_INIT = 0.1`; las finales se
lanzaron con `--t-init 0.1011928851`.

## 5. A dónde llegan T y λ

| modelo | T | λ | paso efectivo T·‖∇R‖/‖ruido‖ | PSNR |
|---|---|---|---|---|
| checkpoint oficial | 0.02473 | 0.1045 | 1.08 | 31.534 |
| v4, it 68 000 | 0.02211 | 0.03118 | — | 31.478 |
| final s0, cortada en la 66k (mejor en la 64k) | 0.0796 | 0.109 | 1.22 | 31.526 |
| final s1, terminada (it 100 000) | 0.0826 | 0.129 | 1.23 | **31.547** |

`T` y `λ` de las finales no tienen por qué coincidir con las del oficial: son modelos distintos
(salieron de otros inits; su salida difiere ~1.4 niveles de gris), y además `T` sola no es
identificable —cada modelo tiene su órbita `(cT, K_N/c)` de representaciones equivalentes—. Lo que
sí se compara es el paso efectivo, que es invariante. `w = λ/T` no se compara entre modelos (ver
sección 1). Fuente de los valores finales: `salida/numeros.md`.
