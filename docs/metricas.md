# Qué se mide, y qué valor es sano

Referencia rápida de todo lo que este repositorio mide, para qué sirve cada número y en qué rango
está bien. Los rangos salen de las 12 corridas de 4 000 iteraciones del 2026-09-04/06 en modo
`referencia` y del checkpoint oficial usado como oráculo.

**Definición de «evento»:** la primera iteración con `‖g‖ > 10⁴`. Es el corte que usa
`scripts/hazard_fase1.py`. Sin disyuntor separa sin solapamiento: las corridas sanas no pasan de
~10⁴ (máximo del corpus, 9 397) y las que estallan llegan a 10⁸–10¹³. Con el disyuntor puesto,
un evento puede ser un lote rechazado en una corrida que sigue sana.
Un evento no siempre mata la corrida — puede quedarse produciendo basura sin `NaN`.

---

## 0. Cómo se corre un experimento acá, en palabras simples

Esta sección no tiene fórmulas. Es para entender de qué se está hablando cuando abajo se dice
«brazo», «semilla» o «hazard».

**El problema de fondo.** Entrenar esta red es como tirar una moneda cargada: la misma receta, con
la misma configuración, a veces llega al final y a veces explota a mitad de camino. No es un bug —
está verificado que el cálculo es exacto. Es que el objetivo mismo es inestable. Así que **de una
sola corrida no se aprende nada**: si sobrevive puede haber sido suerte, y si muere también.

**La semilla es el número que fija el azar.** El programa necesita azar en tres lugares: qué pesos
aleatorios usa al empezar, en qué orden mira las 400 imágenes, y qué ruido les inventa encima. La
semilla es un número entero que fija los tres. Misma semilla = mismos pesos iniciales, mismo orden,
mismo ruido. Semilla 0 y semilla 1 son **dos tiradas distintas de la misma moneda**.

Por eso los experimentos se corren con **12 semillas** (0 a 11) y no con una: lo que se mide no es
si una corrida sobrevivió, sino **qué fracción** sobrevive. Esa fracción sí es una propiedad de la
configuración; una corrida suelta, no.

> **Aviso importante y contraintuitivo.** Repetir la *misma* semilla **no** da la misma corrida,
> salvo en el modo `referencia`. En los otros dos modos la GPU elige sus algoritmos sobre la marcha
> y dos corridas idénticas se separan en la primera iteración. Y aun en `referencia`, está medido
> que una diferencia del último dígito decimal, 284 iteraciones antes, hace que una explosión **no
> ocurra**. Por eso nunca se re-corre una semilla para entender por qué murió: la respuesta sería
> distinta cada vez.

**«Brazo» y «control».** Un experimento compara dos grupos de corridas que difieren en **una sola
cosa**. Cada grupo es un *brazo*. Si uno de los dos es la configuración vieja, se le dice *control*.
Ejemplo real, el que está corriendo ahora: doce corridas en el modo `rapido` contra doce en el modo
`intermedio`, y la única diferencia entre los dos modos es cuántos decimales usa la GPU. Todo lo
demás —receta, semillas, iteraciones— es idéntico. Si algo cambia, fue eso.

Si se cambian dos cosas a la vez, el experimento no sirve, y acá ya pasó: hay conclusiones que
hubo que retirar porque el brazo difería en tres cosas y no en una.

**«Hazard» es la tasa de explosiones.** En vez de contar «cuántas murieron», que depende de cuánto
rato se las miró, se cuenta **cuántas explosiones hubo por cada iteración de entrenamiento
observada**. Es lo mismo que decir «tantas fallas por cada mil horas de vuelo» en vez de «tantos
aviones se cayeron». Sirve para comparar grupos que no corrieron lo mismo, y para las que se
cortaron todavía vivas.

**«Se predice antes de correr».** Antes de lanzar, se escribe en la bitácora qué se espera que pase
y con qué números. Suena burocrático pero tiene una razón concreta: si el resultado se mira primero
y se explica después, cualquier resultado tiene explicación. Escribirlo antes es lo que permite que
una predicción **falle**, y una predicción fallada es lo que más enseña. Ya pasó dos veces acá.

---

## 1. Lo que imprime el log, cada 250 iteraciones

```
[  2000/100000] loss=  25.412 lr=4.00e-04 T=0.02939 lambda=0.00406 |g|=701.1 max|g|=1.69e+03 corr=0.875
```

| campo | qué es | sano | alarma |
|---|---|---|---|
| `loss` | `‖x_S − y‖²` por parche. El modelo identidad vale **249.4** (83.1 en gris) | 20–30 a las 4 000 iteraciones | `> 1 000` |
| `lr` | learning rate. Se divide por 2 cada 25 000 | 4·10⁻⁴ hasta la 25 000 | — |
| `T` | tiempo de parada aprendido. Multiplica todo el gradiente de los 388 k pesos | se mueve poco: arrancando en 0.1 baja a ~0.098 en 4 000 iteraciones | **`T → 0`** mata la corrida |
| `lambda` | peso del término de datos | cualquier cosa entre 0 y ~0.1 | `> 0.5` degrada 0.6–1.8 dB |
| `\|g\|` | norma del gradiente de esa iteración | mediana 79–80 | `> 10⁴` es un evento |
| `max\|g\|` | máxima de la ventana, rechazados incluidos | 10²–10³; picos sanos hasta ~10⁴ | 10⁸–10¹³ sin disyuntor; con él, mirar los aceptados |
| `corr` | `‖x_S − x_0‖/‖ruido‖`: cuánto corrige la red, en unidades del ruido | **~0.95** | `> 1` sobre-corrige; en un evento llega a 400 |

`PSNR = −10·log₁₀(loss / 25 947)` (en gris, `/ 8 649`). Así `loss` 20 ≈ 31.1 dB, y `loss` 249.4 = 20.2 dB, que es no
hacer nada. El objetivo publicado es **31.53 dB**; lo mejor de este repo es **31.547**
(`t8delta-100k-s1`, it 100 000, `referencia`), o sea que el objetivo está cumplido desde el
2026-10-07. El 31.478 que figuraba acá era de v4 en bf16.

**`lambda = 0.00000` no es un problema.** Apagar el término de datos entero cuesta 0.033 dB en el
oráculo. Está medido; no gastar tiempo en eso.

---

## 2. Lo que se mide después, sobre una corrida

`traza.npy` guarda una fila por iteración: `loss, T, ∂L/∂T, λ, ∂L/∂λ, ‖g‖`. Con eso y los
checkpoints se calculan:

| cantidad | qué es | sano |
|---|---|---|
| `‖∇R‖` | fuerza del regularizador aprendido | **crece 736×** durante el entrenamiento (3.8 → 2 800). Crecer es normal |
| `T·‖∇R‖/‖n‖` | el paso del esquema en unidades del ruido. **El invariante real** | 1.1–1.3, igual en el oráculo y en las supervivientes |
| `‖K_N‖` | norma de la última conv de la energía | 0.98 ± 0.24, y **no se mueve** (±1 %) |
| `‖p₀‖/‖p_S‖` | amplificación de la cadena adjunta (los 10 pasos del backward) | 0.67–0.72: **contrae**. No amplifica |
| `\|∂L/∂T\| / ‖g‖` | qué fracción del gradiente total es el escalar `T` | 0.70–0.86 |

---

## 3. Lo que se mide sobre un conjunto de corridas

Una sola corrida no dice nada: el entrenamiento es caótico a nivel ULP y dos corridas idénticas
divergen. Lo único que es propiedad de una configuración es la **tasa**.

| cantidad | qué es | valor actual |
|---|---|---|
| **hazard** | eventos por iteración de exposición | **3.01·10⁻⁴** promedio en `referencia`, sobre 24 corridas |
| vida mediana | la iteración donde la mitad ya explotó, por Kaplan-Meier (sin suponer ningún modelo) | 2 660 iteraciones |
| mortalidad a 4 000 | fracción de semillas con evento antes de la 4 000 | 19 de 24 |

**El hazard es la unidad de la fase 1**, no la mortalidad: cuenta bien las corridas que se cortan
vivas, y no depende de cuánto tiempo se las miró.

**Pero no es constante** (medido el 2026-09-06, reconfirmado sobre las 24 corridas el 2026-09-07):
sube de 4.3·10⁻⁵ en las primeras 1 000 iteraciones a 6.9·10⁻⁴ entre la 3 000 y la 4 000 —un factor
16—, forma de Weibull `k = 2.07`, p = 0.0014. Consecuencias:

- Dos configuraciones se comparan **sólo sobre la misma ventana de iteraciones**.
- Ningún hazard medido a 4 000 se extrapola a 100 000.
- **La vida mediana no es `ln 2 / hazard`.** Esa fórmula supone hazard constante, que es
  justamente lo que acá no vale: da 2 303 contra las 2 660 que da Kaplan-Meier sobre los mismos
  datos, un 13 % de subestimación. Se usa Kaplan-Meier, que no supone nada.

**Y para comparar dos configuraciones, el test es el log-rank sobre los tiempos de evento**, no la
mortalidad. Con 6 semillas por brazo las 924 particiones se enumeran enteras, así que el p-valor es
exacto y no hace falta la aproximación normal. Medido el 2026-09-07 sobre los mismos datos: la
mortalidad (6/6 contra 4/6) da Fisher p = 0.45 y el conteo de Poisson p = 0.19, mientras el log-rank
da p = 0.022. `scripts/veredicto_lambdainit.py` es la plantilla; `scripts/poder_experimento.py`
calcula sobre mortalidad y por eso **subestima** el poder real.

Rango sano de la razón de hazard entre un brazo y su control: **1× es el null**. Lo medido en la
fase 1 va de 0.89× a 2.5×, y **eso vale para todo lo que se probó**: `T_init`, `λ_init`, el clipping
y —desde el 2026-09-07— también la precisión, que da 1.31× (IC95 % [0.33, 5.41]). Acá decía que la
precisión era el único efecto grande, ~15×: **retirado**, era una comparación confundida. Hoy no hay
ningún lever medido por encima de 2.5×, y la fase 1 necesita ~15×.

---

## 4. Lo que NO significa nada

Cinco números que parecen informativos y no lo son. Cada uno costó horas antes de quedar escrito.

1. **`T` sola.** Existe una degeneración exacta `(T, K_N) → (cT, K_N/c)` que deja el modelo
   idéntico, así que `T` es una elección de unidades. Comparar la `T` de dos corridas no compara
   nada. El invariante es `T·‖∇R‖/‖n‖`.
2. **`λ/T`.** Por lo mismo: la degeneración no toca `λ`, así que el cociente se lleva a cualquier
   valor eligiendo `c`. El checkpoint oficial, transportado, tiene `λ = T` exactamente.
3. **`λ*` y `T*` del checkpoint oficial como destino.** Son las coordenadas de *ese* punto sobre su
   órbita. Llevar nuestros checkpoints a `λ* = 0.1045` los **empeora**.
4. **`max‖g‖` entre configuraciones con `T_init` distinto.** `|∂L/∂T|` escala como `1/T`, así que
   subir `T_init` baja `max‖g‖` sin alejar la corrida de la muerte. Dentro de un mismo `T_init` sí
   separa vivas de muertas. **Acotado el 2026-10-05: la ley vive en `|∂L/∂T|`, no en `‖g‖`, y el
   efecto se satura.** El 70–86 % de la norma es a `T_init = 0.0253`; a `T_init ≈ 0.1` esa
   componente es el **27–36 %**. Medido entre 0.1 y 10·δ: la razón de `max|∂L/∂T|` es **0.7655**
   —dentro de la banda predicha [0.74, 0.85]— pero la de `max‖g‖` sobre lotes sanos es **0.9988**.
   Son la misma medición: escalar 0.364 por el 4× de `T` da 0.839.
5. **Cualquier propiedad del estado en el que murió una corrida.** Reanudar desde el checkpoint
   anterior da una corrida bit-idéntica 140 iteraciones que después no muere. La muerte no es
   atribuible al estado, ni al lote, ni al checkpoint.

---

## 5. Los tres modos de ejecución

| modo | precisión | mantisa | `cudnn.benchmark` | ¿repite bit a bit? | para qué |
|---|---|---|---|---|---|
| `rapido` | bf16 | 7 bits | **encendido** | no | explorar |
| `intermedio` | fp32 + TF32 | 10 bits | **encendido** | no | testear |
| `referencia` | fp32 IEEE | 24 bits | apagado | **sí** | el número que se cita |

**Toda la campaña de la fase 1 corrió en `referencia`.**

**RETIRADO el 2026-10-07:** acá decía que «`referencia` es, medido, el modo que menos entrena: los
tres mejores resultados del repositorio (31.478, 31.194, 31.155 dB) son todos bf16, o sea
`rapido`». Se da vuelta. Los dos mejores resultados del repo son ahora **31.547** y **31.5257**,
los dos en **`referencia`** (`t8delta-100k-s1` @100 000 y `-s0` @64 000, cortada a mano en la
66 000), y el 31.478 de bf16 queda tercero. Lo que el corpus viejo medía era un sesgo de **largo de
corrida**, no de modo: ninguna corrida en `referencia` había pasado de 18 000 iteraciones cuando se
escribió esa frase. La comparación de modos sigue sin estar hecha a iteración pareada y larga: es
el punto 3 de la cola vieja, «¿`referencia` es de verdad peor?», que quedó abierto al cerrar la
fase 1 (`docs/bitacora.md`, entrada del 2026-10-08).
