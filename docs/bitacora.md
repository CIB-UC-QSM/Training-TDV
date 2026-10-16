# Bitácora

Registro cronológico de los experimentos de la **fase 1**: reproducir el entrenamiento de TDV
(Kobler et al., 2020) sobre BSDS400 con ruido gaussiano σ=25, desde inicialización aleatoria, hasta
alcanzar ≈31.5 dB en CBSD68.

Se anotan también las hipótesis descartadas y los errores de medición propios: sin ellos el registro
no permite distinguir lo verificado de lo supuesto.

---

## Estado

*Al 2026-10-12.*

| | |
|---|---|
| **Fase 1 CERRADA (10-08)** | el autor da por terminada la construcción del entrenador con **31.547 dB** (`t8delta-100k-s1`, it 100 000, `referencia`) contra 31.53 publicado. La cola de corridas ya no existe: lo que quedó abierto está en la entrada del 2026-10-08, al final. Las filas de abajo son el historial y no se actualizan más |
| **Nuevo (10-12): los cuatro checkpoints son el mismo denoiser y priors PnP distintos — el paso desenrollado se vuelve INESTABLE 4× abajo de donde entrena** | corrido el ADMM de `../tdv/wTDV_QSM_torch.py` con los cuatro y ajustes idénticos: el oficial da **27.34 %** de RMSE y converge en **33** iteraciones; `8δ`, `1.1δ` y `2δ` dan 28.36–28.50 % y **ninguno converge en 100**. La causa, medida por iteración de potencia sobre el Hessiano: el paso es descenso **explícito**, estable sólo si `(T/S)·λ_max(∇²R) < 2`, y a σ = 25 los cuatro caen en **1.38–2.01** —el entrenamiento lo fija ahí— pero a std = 0.06 el oficial baja a **0.99** y los nuestros suben a **2.79 / 5.76 / 23.2**. **La PSNR de denoising es ciega a esto** porque se mide justo donde el entrenamiento fijó la estabilidad. No es calibración: `target_std = 0.06` es el óptimo de todos (`8δ` a 0.25 da 31.21 %, mi predicción FALLÓ) |
| **Nuevo (10-12): retiradas la «regla 2-δ» y el papel de `w` — las cuatro corridas aprendieron el MISMO regularizador** | el piso de `2·δ` no era un umbral medido sino el factor 2 de una elección, y nunca predijo (`B-larga` murió **en** 2·δ, `1.1·δ` terminó las 100 000): queda un costo continuo. Y `w = λ/T` es una coordenada de **gauge** que el solver de QSM **no usa** —usa `vn(z·α, z·α)/α`—. Midiendo lo invariante: `w` abre **12.2×** pero `T·‖∇R‖` queda en **1.13×** y los cuatro `∇R` dan coseno **0.978–0.987**, con los denoisers a **42–46 dB** entre sí. Lo que queda real en denoising es la fuerza del dato, `ρ` = 0.15–0.18 (`referencia`) contra 0.031–0.039 (`rapido`), **4.8× y por modo, no por `T_init`**. **Corregido el 10-12 (más abajo): «lo único» era de más** — `(T/S)·λ_max(∇²R)` a amplitud de QSM separa por `T_init` y va de 0.99 a 23.2 |
| **Nuevo (10-11): tanda `modo-y-tinit-100k-s1` CERRADA — 7 CONFIRMADAS, 3 FALLAS, y ninguna perilla mueve el endpoint** | las dos terminaron. El modo cuesta **−0.001 dB** y `T_init` (×4, de 8·δ a 2·δ) **−0.011 dB**, contra un piso de detección de 0.33: 31.546075 y 31.535652 contra 31.547008. Las seis predicciones del brazo B se confirman, cinco con umbral flojo a propósito. Lo que **sí** se mueve es la exposición —el brazo B hace **1.99×** los rechazos del control y el **96 %** en la primera fase de lr— y `w = λ/T`, que las dos perillas mueven en direcciones opuestas y un factor **2.5–5** sin tocar el PSNR: 3.87 (B) > 1.56 (control) > 1.07 (1.1·δ) > 0.32 (A). **`T*` no es un atractor** y el arreglo del alias del respaldo **sigue sin prueba en vivo**: los 19 rebobinados de las cuatro corridas van a respaldos distintos |
| **Nuevo (10-11): `T_init = 1.1·δ` sobrevive las 100 000, y `δ_Adam` no es constante** | `t1p1delta-rapido-100k` termina en **31.525 dB** contra 31.547 del control, con `T` **nunca** cerca de 0: mínimo 1.437·δ, final **2.228·δ**. La predicción registrada falló y lo que se retira es el método —estimar la reserva exigida con un replay de Adam sobre `∂L/∂T` de otra corrida es una traslación—. Sobrevive porque **`δ_Adam` se parte a la mitad en cada hito del lr**: la reserva `T/δ_vivo` va 1.65 → 3.84 → 8.46 → 17.80 → 35.66, o sea que **el piso de 2·δ sólo ata en la primera fase de lr**. Ahí cae el costo: **345 de 378 rechazos (91 %) y los 8 rebobinados**, 4.7× el control, con +0.3 % de reloj. Y `w` queda en 0.69× del de `referencia` contra 0.20× de `rapido` a 8·δ |
| **Nuevo (10-10): `rapido` reproduce `referencia` a 10⁵ — y `λ` no** | `rapido-100k-s1` da **31.546075 dB** contra 31.547008 del control con la misma semilla: **50 de 50** puntos pareados dentro del piso de 0.33 dB, **49 de 50** dentro de 0.05, mediana de |Δ| **0.0022 dB**, y **2.980×** de velocidad (12.648 h contra 37.697). Las imágenes son indistinguibles (48 dB entre las dos salidas) y `∇R` apunta igual a **0.982** en el régimen con ruido. Lo que **sí** se separa es `λ`: **4.7× más baja** (0.0275 contra 0.1291) y `w = λ/T` **4.9×**, que es el escalar del solver de QSM. Cierra el punto 3 de la cola de la fase 1 |
| **Nuevo (10-09): `rebobina()` arreglado — el respaldo se copia y el lr no vuelve** | dos defectos en una línea, los dos reproducidos como test que falla antes del cambio. El **alias**: `opt.load_state_dict` adoptaba los tensores del respaldo, así que el 2º rebobinado al mismo no devolvía Adam —es la causa del lazo de `t1-guard-2delta-5k`, 45 rebobinados y 41 al mismo respaldo—. El **lr**: el `state_dict` del optimizador traía el lr del respaldo y un rebobinado que cruzaba un hito deshacía ese halving hasta el final (riesgo publicado 6–13 %, ahora cero por construcción). Inerte en `t8delta-100k-s1`: sus 4 rebobinados son primeros a su respaldo y ninguno cruza un hito, así que el control de 31.547 dB sigue pareable |
| **Nuevo (10-08): fase 2 arrancada — 32.766 dB sobre cortes de T1** | `t1-rapido-30k`, 30 000 it en `rapido`, semilla 0, `T_init = 8·δ`, `--channels 1` (387 394 parámetros), 68 cortes de `ldm100k/val`, 3.83 h. Los dos hitos del lr aterrizaron sin que un rebobinado los deshiciera. **No se compara con los 31.547 dB de BSDS**: los cortes traen ~38 % de fondo casi plano y el único anclaje entre dominios es el 20.17 dB de la entrada ruidosa. Disyuntor 0.517 % contra 0.163 % en BSDS, **causa sin medir**. n = 1 |
| Mejor resultado | **31.547 dB** (`t8delta-100k-s1`, iteración 100 000, `referencia`) contra 31.53 publicado. **Objetivo de la fase 1 cumplido el 2026-10-07.** El 31.478 de v4 (bf16, it 68 000) queda tercero, detrás del 31.5257 de `-s0` |
| Corridas completas | 4 con la receta literal; 3 murieron |
| **Nuevo (09-20): la corrida de 10⁵ en `referencia`** | `runs/referencia-100k`: **21.735 dB** de pico y **−16.147 dB** al final. Evento en la it 1 958, **97.3 %** de las iteraciones con `‖g‖ > 10⁴`, cero `NaN`, `T` y `λ` nunca en el clamp. Es `lambdainit/B-seed0` reproducida **bit a bit** y extendida |
| **Nuevo (09-20): evento ≠ muerte ≠ corrida perdida** | la corrida no murió y quedó a 9.8 dB. El endpoint de la fase 1 no puede ser «no muere»: tiene que ser **PSNR a iteración fija** |
| Mortalidad a 4 000 iteraciones | **4/6** con `T_init = 0.0253`, **5/6** con `T_init = 0.1` (Fisher p = 1.0) |
| Modo de muerte corregido | `T → 0` por un pico amplificado por Adam. `T_init = 0.0253` lo elimina; con `T_init = 0.1` la `T` mínima de las seis semillas es 0.0643 |
| Verificado | el gradiente es **exacto** (float64, arquitectura real): el estallido no es un bug |
| **Nuevo (09-06): no es atribuible** | un ULP de fp32, 284 iteraciones antes, borra el evento. Sólo la **tasa** es propiedad de la configuración |
| **Nuevo (09-06): `T` no es identificable** | la predicción registrada falla: `T` no baja hacia `T*`. El invariante es `T·‖∇R‖/‖n‖ ≈ 1.1–1.3` |
| **Nuevo (09-06): `‖g‖` es casi `\|∂L/∂T\|`, pero SOLO a `T` baja** | 0.70–0.86 de la norma en régimen sano **con `T_init = 0.0253`**, 1.0000 durante el estallido de `T → 0`. **Acotado el 10-05:** a `T_init ≈ 0.1` la cuota cae a **0.27–0.36**, que es la *misma* ley —escalar 0.364 por el 4× de `T` da 0.839, dentro de la banda—. Consecuencia práctica: «subir `T_init` baja `max‖g‖` gratis» **se satura** y de 0.1 a 0.1265 no compra nada (razón medida 1.00 en lotes sanos) |
| **Nuevo (09-06): el hazard sube** | 0 eventos en las primeras 1 000 iteraciones, 9.6·10⁻⁴ entre la 3 000 y la 4 000 (Weibull `k` = 3.22, p = 0.0014). **No se extrapola**: el `P(10⁵) = 3.5·10⁻¹¹` queda retirado como número |
| **Nuevo (09-07): la precisión NO es la respuesta** | medido limpio, 12+12 semillas: TF32 contra bf16 da **1.31×** de hazard (IC95 % [0.33, 5.41], log-rank exacto p = 0.61). El `~15×` queda **retirado**: era el confundido. Y bf16 no es el modo que no muere — mortalidad 5/12 contra 6/12 |
| **Nuevo (09-20): `σ_max(A)` medido, y es un parámetro de orden** | el operador adjunto compuesto vale **1.5–5.1 en los nueve checkpoints sanos**, `2.32·10³` en `ref100k best` (21.7 dB) y `2.62·10⁵` en `ref100k last` (−16.1 dB). El amplificador **existe** pero **no está en los estados sanos**: lo construye el entrenamiento |
| **Nuevo (09-20): dos regímenes, mecanismos opuestos** | sano gobernado por la **alineación** (`r(log‖g‖, log cos)` = 0.40–0.95), estallado por la **amplificación** (`r(log‖g‖, log amp)` = 0.52–0.71). En 3 600 lotes sanos la amplificación máxima es **2.05** |
| **Nuevo (09-20): `σ_max` separa vivas de muertas** | AUC **0.968** contra 0.912 de `τ·L`, sobre 65 corridas (excluido el modo `T → 0`, que es otra muerte). **Ninguna de las 38 vivas pasa de `σ_max` = 16.5.** Es el primero que pasa la prueba que fallaron `T*`, `λ/T` y `max‖g‖` — con la advertencia de que es diagnóstico sobre el estado final, no predicción |
| **Nuevo (10-07): el objetivo de la fase 1 está cumplido — 31.547 dB** | `runs/t8delta-100k-s1`, 100 000 iteraciones en `referencia`, contra 31.53 publicado. 163 rechazos, **4 rebobinados** sobrevividos, 37.70 h. Pero el tramo 50 000→100 000 compró sólo **+0.0301 dB** por 18.9 h de GPU —5.4× menos que un sd entre semillas— y el último cuarto **+0.0047 dB** a 19.3 h por centésima. En la it 100 000 los dos escalares **seguían derivando**: `T` 0.0797→0.0826 y `λ` 0.1130→0.1291 |
| **Nuevo (10-07): el disyuntor contiene `‖g‖` hasta 1.3·10¹⁰ — 65 700× el récord anterior** | dos casos independientes. `t8delta-100k-s1`: **`max‖g‖` = 1.295·10¹⁰** con **4 rebobinados** y 163 rechazos de 100 000, terminando en 31.547 dB. `tinit-2delta-50k`: **1.0416·10¹⁰** en la it 3 507, con dos excursiones sostenidas de **25 rechazos seguidos** cada una (it 2 171–2 195 y 3 498–3 522), las dos rebobinadas, saliendo en 31.1548 dB @4 000. Hasta hoy el peor evento sobrevivido escrito era 1.971e5. Y la separación es perfecta: el `‖g‖` máximo de los lotes **aceptados** por el 2·δ es **1 678** contra 3.57e5 el menor rechazado —hueco de **213×**— y 1 678 queda **debajo** del 6 113 sano, o sea que no rechazó nada que no hiciera falta. `t8delta-100k-s0`, en cambio, nunca pasó de **4 006**: la exposición varía 3.2·10⁶× entre dos semillas de la misma configuración |
| **Nuevo (10-07): arrancar `T` en `T*` cuesta 12–72× más intervención** | tasa de rechazo **1.98·10⁻²/it** con `T_init = 2·δ` contra 1.63·10⁻³ de `t8delta-100k-s1` y 2.73·10⁻⁴ de `-s0`, y 3 rebobinados en 4 000 iteraciones contra 4 en 100 000. La dispersión entre las dos semillas del control es 6×, así que el número honesto es el rango. **`2·δ` es un régimen crónicamente inestable que el disyuntor sostiene**, no un arranque equivalente. **ACOTADO el 10-11: ese 12–72× es la tasa de las primeras 4 000 iteraciones, no el costo de la corrida.** A 10⁵, `t2delta-100k-s1` cierra en **1.99×** los rechazos del control, porque el **96 %** de ellos cae en la primera fase de lr y ahí se termina: `δ_Adam` se parte a la mitad en cada hito, así que la reserva se duplica sola |
| **Nuevo (10-07): `T_init` no mueve el PSNR — lo de «31.10–31.19 contra 31.5» era largo de corrida** | a iteración pareada, `T_init` de 2·δ a 10·δ (factor 5) abre **0.026 dB** @2 000 y **0.015 dB** @10 000, contra un piso de detectabilidad de 0.33 dB. El 2·δ va **+0.027/+0.005** arriba @2 000 y **−0.036/−0.017** abajo @4 000: el signo se da vuelta, o sea ruido. Los 31.10–31.19 son corridas de 10 000 iteraciones y los 31.5 de 100 000. **`T_init` cambia la exposición al estallido, no el endpoint** |
| **Nuevo (10-07): el halving del lr hace casi todo el trabajo del tramo final** | el salto de la it 50 000 vale **+0.0098** y **+0.0077** dB contra una mediana de vecinos de +0.0035 y +0.0014 (2.8× y 5.6×), y el mínimo de `T` sobre la traza entera cae en las **it 49 997 y 49 991** —a 9 iteraciones del halving— en las dos semillas, con valores 0.078482 y 0.079627. El salto se encogió **3.4×/3.8×** entre el primer halving y el segundo |
| **Nuevo (10-07): no hay overfitting a las 100 000** | pendiente del PSNR de test **+7.96·10⁻⁷ dB/it** (t = +4.53) contra pendiente de la `loss` de train **−3.30·10⁻⁶** (t = −0.89, no significativa), y correlación **−0.555/−0.375**, que es el signo sano. El overfitting pediría correlación positiva |
| **Nuevo (10-07): `‖Δθ‖` NO sirve de criterio de corte** | las dos semillas están a **68.89** una de otra = **131.8 % de `‖θ‖`** —más lejos que la norma de los pesos— y dan el mismo PSNR. Casi todo el movimiento de pesos es sobre una variedad plana. El criterio que sí sirve es en **espacio de funciones**: `rms(f_ahora(z) − f_vieja(z))` sobre el set de eval, con piso medido entre semillas de **0.95 niveles de gris** |
| **Nuevo (10-05): el disyuntor completo sobrevive un evento real** | `runs/tinit-7p9delta-10k`: pico de `‖g‖` **1.971e5** —peor que el 9.74e4 que mató a `disyuntor-100k`—, centro móvil congelado igual en 29.53, y **salió viva**: 93 rechazos de 10 000, racha máxima 25 (= el umbral), **2 rebobinados**, PSNR subiendo hasta la última evaluación |
| **Nuevo (10-05): lo que previene NO es la razón, es el rebobinado** | los dos eventos entraron por un lote **aceptado** a razón **3.58×** y **3.67×**, debajo de cualquier umbral que no rechace lotes sanos. La razón 5 **contiene** la secuela de 10–180×; el rebobinado es el único que recupera, y lo hizo en **un paso** las dos veces (`loss` 30.95→21.01 y 26.18→20.47) |
| **Nuevo (10-05): la cota absoluta no hizo trabajo propio, pero su valor está bien** | **0 lotes agarrados sólo por ella en 226 000 iteraciones** y 372 rechazos: los 141 que cruzaron 1e4 los agarró también la razón 5 (re-medido el 10-07 sobre siete corridas, `docs/mediciones/cota-absoluta-226k.txt`; antes decía 56 000 iteraciones y 112 rechazos). Sigue **seguro no probado**. Pero su valor no sale de «arriba del máximo sano del corpus (9 397)»: separa las excursiones que **se resuelven solas** —5 312 y 6 113, y la de 6 113 terminó en **30.937 dB sin disyuntor**— de las que **necesitan intervención** —1.971e5 y 6.9e12—. Hueco **[6 113, 1.971e5]**, 32× sin solapar. **Bajarla a 5 000 era mi plan y está mal** |
| **Nuevo (10-05): la ventana tiene que viajar con el respaldo (disyuntor v2)** | `rebobina` **borraba** la mediana móvil, y eso dejaba el relativo desarmado 100 iteraciones aceptadas justo al salir del evento: **200 de las 300 desarmadas** de `tinit-7p9delta-10k` las creaba el rebobinado mismo. Y la ventana que se tiraba era buena: la primera `‖g‖` post-rebobinado vale **48.5** y **21.3** contra centros congelados de 46.71 y 29.53, razones **1.04** y **0.72**. Ahora el respaldo se la lleva y el rebobinado la devuelve |
| **Nuevo (10-05): el respaldo exige 3 limpias seguidas (disyuntor v2)** | dentro de una excursión letal hay lotes **aceptados**: 4 en `tinit-7p9delta-10k` y 2 en `disyuntor-100k`, todos de hueco **1**, y cada uno reinicia `seguidos` —por eso el evento 2 rebobinó en la it 5 901 y no en la 5 864—. Medido, no hipotético: `t8delta-100k-s1` tomó un respaldo en la **it 6003** con 3 rechazos en las 30 previas y **una** limpia detrás |
| **Descartado (10-05): la densidad como disparador del rebobinado** | «N rechazos en las últimas W» en vez de N seguidos. Contraejemplo: `precision/rapido-seed2` tiene **33 rechazos en 50 iteraciones** en su tramo sano, con `‖g‖` a 6 113 y **sin disyuntor**, y terminó en 30.937 dB. A 25/50 habría rebobinado una corrida que no lo necesitaba. **La densidad no separa «se resuelve» de «mata»; la magnitud sí** |
| **Nuevo (10-05): el mejor PSNR del repo en `referencia` es 31.419 dB** | `runs/t8delta-100k-corte-s0` y `-s1` en la **it 18 000**, contra los 31.182 de `disyuntor-100k`. Las dos semillas coinciden dentro de **0.026 dB** a igual iteración, y sus dos mejores valores dentro de **1.9·10⁻⁶ dB** (31.4191394 y 31.4191413) — se anota como bandera, no como conclusión: las predicciones 3 y 4 de la enmienda lo resuelven. Las dos **seguían subiendo** cuando las cortó el corte de luz |
| **Retirado (10-05): `T` @100k = 0.02432 = 0.98·`T*`** | era mi extrapolación del 10-05 y los parciales de 18 000 iteraciones la falsifican. La deriva real a esa altura es **−8.87·10⁻⁷/it**, y con el halving del `lr` el desplazamiento restante da `T` ≈ **0.0585 = 2.36·`T*` = 4.6·`δ`**, con la deriva desacelerando, así que es un piso. **`8·δ` a 100k no pone a prueba la pared de muerte.** Sin el halving sí: `T` terminaría en 0.0114 = 0.90·`δ`, debajo de la pared |
| **Nuevo (10-05): `T_init` 0.1 contra 10·δ no separa** | mejor PSNR **31.34323** contra **31.35234**, diferencia **0.00911 dB** contra un piso de detectabilidad de 0.33 dB: **36× por debajo** de lo medible. El offset de `T` se conserva al **99.2 %** sobre 10 000 iteraciones, confirmando el replay |
| **Nuevo (10-05): a 10 000 iteraciones la PSNR todavía sube** | el mejor PSNR **es** el final en los dos brazos, con incrementos de +0.033 y +0.052 dB en las últimas 2 000 iteraciones. 10 000 it **no** son suficientes para el objetivo de la fase 1, y por eso la tanda siguiente es de 100 000 |
| **Nuevo (09-20): los diez pasos son expansivos** | `‖J_s‖` = 1.18–3.83 en el oráculo, producto **166**, operador compuesto **2.08**. `τ·L` acota el producto, no el operador: por eso nunca separó |
| **Retirado (09-20)** | el «750× por encima de la cota» del pre-registro: comparaba dos estados distintos. Medido en el mismo estado, sobre 4 400 lotes, `\|∂L/∂T\|/cota` **no pasa de 0.50** |
| **Nuevo (09-20): la transición no avisa por la trayectoria, sí por la cola sobre lotes** | dos eventos cazados con la traza puesta (`seed7` it 2 699, `seed8` it 4 721). `amp` salta de **0.70 a 217 en una sola iteración**, con la `loss` en 25.2 y `‖∇R‖` sin moverse (1 555 → 1 560); en 10 011 iteraciones sanas ninguna pasa de 100. Pero sobre **400 lotes** en el checkpoint anterior al salto el máximo vale **203.7 y 120.9**, contra 0.76–13.3 en las ~200 iteraciones previas: **hay aviso, con horizonte de 4 a 9 iteraciones** |
| **Nuevo (09-20): el estado amplificado es metaestable** | las dos corridas **vuelven** al régimen sano una o dos veces antes de decidirse (`seed7` a 0.999, `seed8` a 0.591), y excursiones aisladas a `amp` = 7 y 67 se apagaron solas. Eso le da mecanismo a la no-atribución por un ULP: no cambia la tasa de fugas, cambia un volado que se juega en tres iteraciones |
| **(10-03, SUPERADA el 10-04): el disyuntor con una sola prueba** | rechazaba el lote cuyo `‖g‖` superaba `1/√(1−β₂) = 31.62` veces su mediana móvil de 100: **49 rechazos en 205 829 iteraciones sanas (0.0238 %)** y 48/48 primeros picos interceptados. **Corrió y murió** (`runs/disyuntor-100k`). La razón está en `## Descartado` |
| **(10-04, ACOTADA el 10-05, AMPLIADA a v2 el 10-05): las piezas del disyuntor** | **razón 5** sobre la mediana móvil —costo 0.1545 % de los lotes sanos—; **cota absoluta 10⁴**, que no se puede congelar; **rebobinado** de pesos y `m`/`v` a los 25 rechazos seguidos; y desde el v2, la **ventana viaja con el respaldo** y el respaldo exige **3 limpias seguidas**. Decía que la razón «es la que previene»: **falso**, medido el 10-05 —los dos eventos entraron por lotes aceptados a 3.58× y 3.67×—. La razón contiene, la cota es seguro no probado, y el que previene es el rebobinado |
| **Nuevo (10-03): no toca la aritmética** | con 120 iteraciones en `referencia`, el `train.py` nuevo con `--guard inf` da una corrida **bit a bit** idéntica a la de HEAD. Por eso el control del 9.2 es **pareado**, que es lo que le dio potencia al experimento del clip |
| **Nuevo (10-03): `‖g‖` relativa le gana a `amp`, y es gratis** | la razón sobre la mediana dispara **igual o antes** que `amp > 100` en los dos eventos instrumentados (2 694 contra 2 694; **4 720 contra 4 721**), sin instrumentar el forward. `--guard-amp` queda apagada: cuesta un vjp más y no compra anticipación |
| **Nuevo (10-03): `T_init` no se sube** | el default vigente es 0.1 = 7.9·δ con **39 trayectorias** detrás; 5·δ tiene **una sola** (las cinco de `repeat5` son copias bit a bit) y estalló en la it 1 390. `T` no se mueve de donde arranca —18 corridas sin evento a 0.1 terminan en [0.0763, 0.1016], **0 de 18** bajan de 0.05— así que `T_init` **es el punto de operación**. El rango de 4× ya está medido y no separa (4/6 contra 5/6, Fisher p = 1.0) |
| **Nuevo (10-04): el mejor PSNR del repo en `referencia`** | `runs/disyuntor-100k`: **31.182 dB** en la it 4 000, contra **−30.190 dB** de su control pareado en la misma iteración. Tercero del corpus de 16, y el primero en `referencia`. La corrida igual murió: ese pico es un techo alcanzado, no una corrida sana |
| **Nuevo (10-04): el disyuntor se traba, y trabarse cuesta 15 h** | con todo rechazado el centro deja de actualizarse y los pesos no se mueven: `runs/disyuntor-100k` pasó **45 585 iteraciones sin dar un paso**, `T` en un único valor y `max|dif| = 0`. «No explota» no es «sobrevive» |
| **Nuevo (10-04): una prueba por lote llega siempre un paso tarde** | `‖g‖` se mide en los pesos de ahora, así que dice dónde está la corrida, no a dónde la lleva el paso. El lote que rompió `disyuntor-100k` tenía razón **7.70**, un valor sano ordinario, y `‖g‖` saltó **400×** en ese solo paso. Por eso hace falta deshacer, no sólo rechazar |
| **Nuevo (10-04): `T` es una traslación pura bajo Adam** | replay de `dL/dT` grabado sobre 7 trayectorias de 4 000 iteraciones: la separación inicial entre 0.1 y 0.1265 se conserva al **100.0 % exacto**. Adam no tira a `T` hacia ningún valor preferido, así que `T_init` **es** el punto de operación — ahora con un replay exacto y no sólo con el rango empírico de 18 corridas |
| **Nuevo (10-04): el PSNR a iteración fija tardía no es un endpoint** | 6 corridas de config idéntica (`runs/tinit01`) tienen sd **15.427 dB** en la it 4 000 y rango 38.9 dB, porque lo que domina es si ya estalló. El **mejor** PSNR tiene sd **0.164 dB** y el de la it 1 000, **0.016 dB**. Los contrastes se leen sobre ésos |
| Pendiente principal | **qué selecciona el lote que dispara.** El salto ocurre en una iteración con una `loss` indistinguible de la de sus vecinas y con el operador adjunto sano hasta la iteración anterior. Todo lo **medido** hasta hoy —`T_init`, `λ_init`, clipping, precisión— juega entre 0.9× y 2.5×, y la fase 1 necesita ~15× |
| Irreproducibilidad | resuelta: era `cudnn.benchmark`. `referencia` lo apaga y repite bit a bit |
| Infraestructura | **una corrida = una GPU**, batch 32 acumulado en 2×16. DDP sigue clausurado. **Nuevo (09-20): dos corridas independientes, una por tarjeta, escalan 1.99×** sin cambiar una línea; la afinidad de núcleos y el número de workers no cambian nada |

---

## Cómo retomar

*Reescrita el 2026-10-08, al cerrar la fase 1. La versión anterior está en la historia de git.*

**3D, en `main` desde el 2026-10-15: `train3d.py` está listo para la corrida larga, que el
autor lanza en otra máquina.** Qué copiar, cómo comprobar y cómo lanzar: README, «Llevar el
entrenamiento 3D a otra máquina». La validación mide PSNR, NRMSE, la reconstrucción QSM de COSMOS y
el número de estabilidad PnP, y sigue en W&B. El checkpoint para QSM-TDV es `best_qsm.pth`, no
`best.pth`. Lo último que corrió es la tanda `ventana-3d-12h`, con su RESULTADO al final de este
archivo. La configuración fijada el 2026-10-15 (entrada «Decisiones para la corrida larga») es:
- Blackwell de 96 GB con micro-lote 16;
- 8·δ;
- 2 000 cerebros;
- `--mascara-cerebro`;
- sin `--aug-ganancia`.

Decisiones del autor abiertas:
- N, con el s/it medido en esa tarjeta, para 72–96 h; el calendario es
  `--iterations N --hitos N−15000 N−10000 N−5000` (con N = 40 000, `--hitos 25000 30000 35000`);
- la opción (a) de compatibilidad con QSM-TDV (`docs/mediciones/compat-qsm-tdv.txt`);
- ~~integrar `main` en la rama~~: hecho el 2026-10-15, al integrar la rama en `main` (entrada
  «Integración de `3D-Training` en `main`», al final);
- si `wandb` queda como dependencia obligatoria de `main` o pasa a un extra opcional (misma
  entrada).

**La fase 1 está cerrada** (qué corre ahora: `systemctl --user list-units 'tdv-*'`).
`runs/t8delta-100k-s1` llegó a **31.547 dB** en CBSD68 (σ = 25) en la iteración 100 000, modo `referencia`, contra los 31.53 publicados, y el
autor da por terminada la construcción del entrenador. El veredicto es la entrada del 2026-10-07
(`VEREDICTO de t8delta-100k`); el cierre, con todo lo que quedó abierto, es la del 2026-10-08, al
final de este archivo.

**Dónde está cada cosa:**

- **Los modelos entrenados**, en `checkpoints/Entrenados/`: copias sin el estado del optimizador de
  `t8delta-100k-s1` (it 100 000, 31.547 dB), `t8delta-100k-s0` (it 64 000, 31.526 dB) y
  `replica-v4` (it 68 000, 31.478 dB). Los dos oficiales de VLOGroup/tdv, en
  `checkpoints/Oficiales/`.
- **Las corridas completas** —logs, `traza.npy` y, en las recientes, checkpoints con el estado
  del optimizador—, en `runs/<corrida>/`, que no se versiona. Una ficha legible por corrida, en
  [`resultados/`](resultados/README.md).
- **Qué es cada carpeta de `docs/`**, en [`docs/README.md`](README.md). **Qué significa cada
  número**, en [`metricas.md`](metricas.md), cuya sección 0 explica sin fórmulas qué es una
  semilla, un brazo y el hazard.
- **`scripts/` y `.claude/` son locales** desde el 2026-10-08: las entradas que citan
  `scripts/<algo>.py` siguen valiendo como registro, pero un clon nuevo no tiene esos archivos.

**Qué sigue:**

1. **La tanda `t1-ldm100k`** —dos brazos de `T_init`, 8·δ y 2·δ, sobre cortes 2D de T1 sintéticas
   de LDM_100k, 50 000 iteraciones— está **pre-registrada** en la entrada del 2026-10-07 y **no se
   lanzó**. Su `plan.sh` vive en `runs/t1-ldm100k/`, que no se versiona; los comandos están también
   en el pre-registro.
2. **`rebobina()` está arreglado** (copia del respaldo y lr vigente): el 2026-10-08 en la rama 3D
   y el 10-09 en la 2D, hoy las dos en `main`, con el mismo código (entradas «El lazo era Adam,
   no la ventana», «`rebobina()` arreglado» y «ARREGLO del rebobinado en `main`»).
3. **La fase 2**: el prior en QSM y, antes, el entrenamiento ciego al nivel de ruido
   (`σ ~ U(0, 55)`). Ver «Mejoras candidatas» en `AGENTS.md`.

**Antes de proponer cualquier experimento nuevo:** leer [`## Descartado`](#descartado). Varias
hipótesis se re-abrieron dos veces en este repo por no estar escritas ahí.

---

## Inventario de corridas

Corridas largas, 2 GPUs con DDP, batch global 32, receta literal salvo lo indicado.

| corrida | precisión | mecánica | desviaciones | desenlace | mejor PSNR |
|---|---|---|---|---|---|
| **v1** `replica-bf16` | bf16 | checkpointing | — | dip @5 000 (recuperó), muerte **@19 750** | 31.155 @4 000 |
| v2 `replica-v2` | bf16 | checkpointing | `scalar-lr 0.01`, `t_min 1e-4` | explotó @2 400 | — |
| v3 `replica-v3` | bf16 | checkpointing | + `clip 2000` | explotó @500 | — |
| **v4** `replica-v4` | bf16 | `benchmark` | `clip 35 000` (disyuntor) | evento @69 250, terminó | **31.478 @68 000** |
| **v5** `replica-v5` | fp32 + TF32 | determinista | — | muerte **@2 500** | 30.941 @2 000 |
| **v6** `v6-fp32-ieee` | fp32 IEEE | determinista | — | muerte **@1 750** | — |
| **v7** `v7-tmin` | bf16 | `benchmark` | `t_min 1e-3` | piso @4 250, Modo 1 | 30.935 @2 000 |
| **`referencia-100k`** (09-14) | fp32 IEEE | 1 GPU, sin `benchmark` | `T_init = λ_init = 0.1` | evento @**1 958**, **terminó las 100 000 sin `NaN` y sin clamp** | **21.735 @38 000**, −16.147 al final |

Corridas cortas de control (5 000 iteraciones o menos).

| corrida | precisión | mecánica | desenlace |
|---|---|---|---|
| `x-bf16-det` | bf16 | determinista | 5 000 limpias, **31.194 dB** |
| `x-fp32-bench` | fp32 IEEE | `benchmark` | muerte **@500** |
| `b2-A-control` | fp32 IEEE | `benchmark` | 600 limpias (β₂=0.999) |
| `b2-C-poison` | fp32 IEEE | `benchmark` | 600 limpias (β₂=0.9999) |

Ablaciones de una GPU del 1-sep (`ab-*`, `s1-*`): **todas salvo `s1-literal` llevan
`scalar-lr 0.01`**, la desviación que causó el Modo 1, así que no sirven como evidencia sobre
ninguna otra variable. `s1-literal` corrió 1 500 iteraciones limpias con `T` mín 0.00557 y
`|g|` máx 2·10³; el resto tiene `T` clavada en ~0.00106 y `|g|` máx de 10⁴ a 10⁸.

---

## 2026-08-31 — El port

Se portan `conv.py`, `tdv.py` y `model.py` desde `../tdv/ddr/` (2D, PyTorch nativo, sin `optox`), y
se escriben los tests antes del entrenamiento.

El checkpoint oficial carga con `strict=True` y reproduce los cuatro números publicados al
centésimo de dB, con 387 970 parámetros — exacto, Tabla 3 del paper. Desde acá, cualquier fallo
posterior no puede atribuirse al port.

Detalle que explica lo que vino después: **`VLOGroup/tdv` nunca publicó su código de
entrenamiento.** El repo original trae inferencia y checkpoints. `training/` se escribió desde la
sección 5.1 del paper, sin implementación de referencia contra la cual comparar.

---

## 2026-08-31 — v1: primera corrida completa

Receta literal, bf16, gradient checkpointing.

| iteración | PSNR | `T` | |
|---|---|---|---|
| 4 000 | **31.155** | 0.00988 | pico |
| 6 000 | 27.287 | 0.00202 | primer colapso, `T` cae 4.9× |
| 12 000 | 30.034 | 0.00212 | recuperando |
| 19 000 | 30.698 | 0.00450 | `T` subiendo hace 13 000 iteraciones |
| 20 000 | **20.173** | **0.00000** | segundo colapso, letal |
| 60 000 | 27.730 | 0.00019 | `T` despegó del cero, recuperación parcial |
| 66 000 | 20.173 | 0.00000 | tercera muerte, definitiva |

Diagnóstico del estado muerto, por aritmética:

```
loss  248.9   contra  93·93·3·(25/255)² = 249.4
PSNR   20.173 contra  −10·log₁₀((25/255)²) = 20.172
‖x_S − z‖∞ = 6·10⁻⁷
```

La red devuelve exactamente su entrada. Con `τ = T/S = 0`:

```
x₁ = prox(x₀ − 0·∇R, z, λ/S) = (z + (λ/S)z)/(1 + λ/S) = z
```

y como `∂/∂τ[(z+τz)/(1+τ)] ≡ 0`, se anulan a la vez `∂L/∂θ` (los 387 968 pesos, porque toda la
rama del regularizador va multiplicada por `T`) y `∂L/∂λ`.

`T = 0` **cierra la compuerta del gradiente de los pesos**. Verificado sobre los pesos finales de
v5: `‖grad‖` de los 387 968 pesos = **0.0 exacto**, entradas no nulas: 0.

Que `T` despegue del cero en las iteraciones 40 000–60 000 muestra que el estado no es absorbente
para `T` misma: `∂L/∂T` en `T=0` vale `−2⟨n, ∇R(z)⟩` con los pesos congelados, y su signo decide.
En v5 salió **+3.24·10⁴** (el clamp la sostiene en 0); en v1 salió negativo y `T` subió.

---

## 2026-09-01 — v2 y v3: la desviación que creó un modo de fallo nuevo

Del diagnóstico de v1 salió una corrección: bajar el lr sólo de los escalares
(`--scalar-lr-scale 0.01`) y ponerle piso a `T` (`--t-min 1e-4`).

**Ocho corridas explotaron.** Firma: la pérdida baja limpio (45 → 34 → 32 → 23) y de golpe salta
cuatro órdenes de magnitud, con `|g|` pasando de ~10³ a 10⁶–10⁸.

Causa, con separación perfecta en 13 corridas: `scalar-lr 0.01` deja `T` clavada en 0.001, o sea
`τ = 10⁻⁴`. Para que la imagen cambie algo, `∇R` tiene que valer ~10⁴, y la red sólo puede bajar la
pérdida inflando su propia salida hasta que los pesos explotan.

```
T libre (literal):   0.0075 → 0.0084 → 0.0099     nunca explotó
T clavada (0.01):    0.00108 → 0.00108 → 0.00109  explotó siempre
```

Se revirtió. **`scalar_lr_scale` no debe tocarse.**

`--t-min` entró en el mismo commit y las once corridas de esa tanda lo llevan, así que quedó
confundido con la causa. Además el piso (1e-4) estaba 10× por debajo del `t_init` (1e-3) con `T`
clavada en 0.001: nunca llegó a activarse. Quedó como hipótesis sin probar, hasta v7.

---

## 2026-09-01 — El gradient checkpointing no era la causa

Con las explosiones sobre la mesa, la sospecha cayó en el checkpointing: recomputa el forward
durante el backward, y una recomputación no reproducible daría un gradiente incorrecto.

Refutado por dos vías independientes:

1. **El experimento estaba confundido.** Todas las corridas con checkpointing llevaban también
   `scalar-lr 0.01`. Y v1 —receta literal con checkpointing— llegó a la iteración 19 500.
2. **Medición directa.** Gradiente respecto de los 388 k pesos con y sin `--efficient`, en 6
   configuraciones (fp32/bf16 × compile/sin × determinista/`benchmark`):

   ```
   algoritmos deterministas:  diferencia 0.0 EXACTA, en las 6
   cudnn.benchmark:           7.2·10⁻⁶, contra ruido corrida-a-corrida de 2.9·10⁻⁶
   torch.compile (Triton):    0.0 con determinismo, en ambas precisiones
   ```

En el camino se publicaron dos mediciones falsas propias — "el checkpointing desvía el gradiente un
1.5 %" y "el piso de ruido es 3·10⁻²" — las dos artefactos de medir con el entrenamiento saturando
ambas GPUs. Con `cudnn.benchmark`, la contención cambia la selección de algoritmos.

> **Regla:** no medir con las GPUs ocupadas.

---

## 2026-09-01 — v4: la corrida buena

Receta literal, bf16, `--compile`, sin checkpointing, DDP, `cudnn.benchmark`, clip en 35 000 como
disyuntor no efectivo. 0.249 s/it.

68 000 iteraciones de subida monótona:

```
 2000  30.959      36000  31.424      56000  31.469
 4000  31.170      40000  31.434      60000  31.471
 6000  31.258      44000  31.446      64000  31.476
 8000  30.862 ←    48000  31.448      68000  31.478  ← mejor
10000  31.201      52000  31.464
```

El bache de @8 000 es un colapso de `T` (0.01138 → 0.00725) del que se recuperó.

Evento en la iteración 69 250:

```
[68000]  loss  19.06   T=0.02211  λ=0.03118  |g|=28.3        eval 31.478 dB
[69250]  loss 605.60   T=0.02257  λ=0.03894  |g|=221237.2
[70000]  loss 139.30   T=0.01577  λ=0.05161  |g|=8107        eval 22.740 dB
```

`T` bajó a ~0.0078 y no volvió. El PSNR se recuperó hasta 30.955 pero nunca al pico, y con el lr ya
dividido 8× no le quedaba recorrido.

El clip recortó ~1 200 pasos entre 69 250 y 72 000 (40 % de esa ventana) y cero desde entonces.
**No salvó el entrenamiento:** contuvo la magnitud, pero recortar el 40 % de los pasos corrompe el
segundo momento de Adam. El colapso de `λ` a cero en 76 000–80 000 es plausiblemente consecuencia
del clip. Lo que compró fueron 30 000 iteraciones de trayectoria post-crash, que muestran que el
paisaje tiene otra cuenca usable (`λ`≈0.033, `T`≈0.008, ~31.0 dB).

**Se guardó `runs/replica-v4/checkpoint.pth`**: 31.478 dB, `T`=0.02211, `λ`=0.03118. A 0.05 dB del
paper, desde inicialización aleatoria.

---

## 2026-09-01/02 — v5: la corrida de referencia muere en 2 500

fp32, `--deterministic`, sin clip. Cero desviaciones. La idea era que si volvía a explotar, la
explosión fuera reproducible.

```
[  2250/100000] loss=23.1710  T=0.00906  λ=0.00819  |g|=440.5    growth=0.265
[  2500/100000] loss=248.9643 T=0.00000  λ=0.02761  |g|=43912.9  growth=0.975
...
[ 99250/100000] loss=249.6898 T=0.00000  λ=0.09491  |g|=34139.7
```

97 500 iteraciones —12 horas de las dos GPUs— con el modelo muerto. El PSNR quedó en 20.173 dB
desde la iteración 4 000, dígito por dígito, en las 40 evaluaciones.

El mismo tramo en v4, que sí funcionó:

| iteración | v4 (bf16 + benchmark) | v5 (fp32 + determinista) |
|---|---|---|
| 2000 | `T`=0.00939 `λ`=0.00177 — 30.959 dB | `T`=0.00887 `λ`=0.00563 — 30.941 dB |
| 2250 | `T`=0.00955 `λ`=0.00294 | `T`=0.00906 `λ`=0.00819 |
| 2500 | `T`=0.00967 `λ`=0.00460 | **`T`=0 · loss ×10 · \|g\| ×100** |

Hasta 2 250 las dos trayectorias son la misma corrida con distinto ruido numérico.

**`T = 0` es un punto KKT, no un atasco.** Sobre los pesos finales de v5:

```
T = 0.00000000        λ = 0.09637365
‖grad‖ de los 387 968 pesos = 0.000e+00   (entradas no nulas: 0)
∂L/∂T      = +3.24e+04
∂L/∂λ      = −6.10e-05
```

El modelo identidad satisface las condiciones de primer orden del problema restringido sobre
`[0, Tmax] × Θ`. El optimizador convergió ahí.

Y la teoría del paper no lo excluye. El Teorema 3.1 afirma que el mínimo de `J_S` se alcanza sobre
el compacto `[0, Tmax] × Θ`, que **incluye `T = 0`**. La condición de optimalidad del Teorema 3.3
se deriva dividiendo por `T*` (textual: *"the optimality condition of L w.r.t. T\* reads as
E[−(1/T\*)Σ⟨...⟩] = 0"*), así que presupone `T* ≠ 0` y en cero no dice nada. El modelo identidad
cae en una esquina del conjunto admisible que la teoría no caracteriza. Consecuencia práctica:
evaluar la condición del Teorema 3.3 sólo tiene sentido mientras `T > 0`.

---

## 2026-09-02 — TF32 descartado, y `τ·L > 2` medida

**v6.** Hipótesis: todas las corridas fp32 habían corrido con TF32 activo (encendido por defecto en
cuDNN), mientras que bajo bf16 TF32 es inerte — las convoluciones corren en bf16, que con 7 bits de
mantisa es más grueso que los 10 de TF32. O sea que fp32 IEEE era la celda vacía de la tabla.

v6 la llenó: fp32 sin TF32, murió **@1 750**, antes que v5 (@2 500) con TF32 activo. **TF32
descartado.**

**Medición de `L = ‖∇²R‖`** por iteración de potencia sobre productos Hessiano-vector
(`scripts/measure_lipschitz.py`), evaluada en los estados que el esquema visita:

| modelo | `T` | `τ = T/S` | `τ·L` máx | `growth` realizado |
|---|---|---|---|---|
| inicialización aleatoria | 0.001 | 1.0·10⁻⁴ | 0.00 | — |
| **oráculo oficial (31.53 dB)** | 0.024727 | 2.47·10⁻³ | **4.78** | **0.537** |
| v4 mejor (31.478 dB) | 0.022112 | 2.21·10⁻³ | 2.10 | 0.067 |
| v4 final (post-explosión) | 0.007637 | 7.64·10⁻⁴ | 3.40 | — |

El checkpoint oficial opera a `τ·L = 4.78`. La cota clásica `τL < 2` es la condición para que
descenso de gradiente **converja** sobre una función convexa; acá el esquema corre `S = 10` pasos
fijos sobre una energía no convexa, y la red se entrena *con* esa amplificación adentro del bucle.
Además `L` es el peor autovalor del Hessiano mientras `growth` es la amplificación **realizada**
sobre la trayectoria: el oráculo tiene `τ·L = 4.78` y `growth = 0.537` a la vez.

**Consecuencia: acotar la norma de las convoluciones de los macro-bloques queda descartado.**
Excluiría el régimen donde vive el modelo que funciona (`L ≈ 1930` con `T = 0.0247`).

`τ·L > 2` queda refutada **como explicación de una inestabilidad del forward**. Sobre el backward,
ver más abajo.

---

## 2026-09-02 — v7: el piso para `T` convierte el Modo 2 en Modo 1

`t_min = 1e-3`, bf16, 20 000 iteraciones. El argumento era que el piso sólo actúa dentro de la
espiral de muerte: el oráculo tiene `T = 0.0247` y v4 en su mejor punto 0.0221, o sea 20× por
encima del piso.

```
[3750] loss=16.24    T=0.01018  λ=0.01299  |g|=295    max|g|=789      max_growth=0.266
[4000] loss=55.75    T=0.00291  λ=0.02051  |g|=10072  max|g|=94803    max_growth=1.772
[4250] loss=3791.52  T=0.00100  λ=0.02054  |g|=5.5e6  max|g|=1.35e9   max_growth=1.489
[4750] loss=434.96   T=0.00100  λ=0.03660  |g|=1.9e6  max|g|=1.25e7   max_growth=0.688
```

El piso evitó el cero exacto. A cambio: `T` clavada en el piso, `λ` congelada en 0.03660 al dígito,
y **loss 435 contra 249 de la identidad** — el modelo daña la imagen. Con `τ = 10⁻⁴` la red entra
en el régimen del Modo 1. PSNR 30.935 → 26.569.

**`t_min` queda refutado como corrección.**

Dos observaciones del mismo log, ambas importantes:

- **`max|g| = 2.9·10¹³` con `max_growth = 1.048`.** El forward contrae mientras el gradiente
  explota trece órdenes de magnitud. El fallo vive en el **backward**, no en el forward.
- **`growth` está normalizado mal.** Se define `‖x_S−x_{S−1}‖ / ‖x_1−x_0‖`, o sea que divide por el
  primer incremento. Si el primer paso ya es gigante, el cociente da < 1 aunque `x_S` esté lejísimos
  — exactamente lo que pasó acá. El diagnóstico correcto es `‖x_S−x_0‖/‖n‖` y la amplificación
  adjunta `‖∂L/∂x_0‖/‖∂L/∂x_S‖`.

---

## 2026-09-02 — La precisión separa, el determinismo no

Cruzando precisión contra determinismo, mismo seed 0 y receta literal:

| | `cudnn.benchmark` | determinista |
|---|---|---|
| **bf16** | v4 (@69 250), v7 (@4 250) | `x-bf16-det`: **5 000 limpias, 31.194 dB** |
| **fp32** | `x-fp32-bench`: **muerte @500** | v5 (@2 500), v6 (@1 750) |

Las tres corridas bf16 llegan sanas a ≥5 000 iteraciones; las tres fp32 mueren antes de 2 500.
El determinismo no separa nada.

El contraste más limpio, en la iteración 250 de dos corridas con la misma semilla:

| | `x-bf16-det` | `x-fp32-bench` |
|---|---|---|
| loss | 23.99 | **1447.48** |
| `max_growth` | 1.530 | 1.536 |
| `max|g|` | 470 | **1.11·10⁸** |

El forward se comporta igual en las dos; el backward las separa por cinco órdenes de magnitud.

**Sin explicación.** La hipótesis del redondeo de bf16 como ruido de gradiente no cierra: es
~4·10⁻³ relativo contra una estocasticidad intrínseca de ~100 % por iteración (cada paso sortea 32
parches nuevos y re-muestrea el ruido).

Matiz que hay que tener presente: **bf16 no es inmune** — v1 es bf16 y murió en 19 750. Lo que la
tabla muestra es que bf16 tarda mucho más en morir. Y `x-bf16-det` corrió sólo 5 000 iteraciones:
terminó, no sobrevivió a nada.

**Consecuencia incómoda:** el modo `referencia` —fp32 IEEE, el que existe para producir un número
citable— es el que no entrena. Los tres mejores resultados del repo (31.478, 31.194, 31.155 dB) son
todos bf16.

> **RETIRADO el 2026-10-07.** Se da vuelta: los dos mejores resultados del repo son **31.547** y
> **31.5257**, los dos en **`referencia`**, y el 31.478 de bf16 queda tercero. Esto medía un sesgo
> de **largo de corrida**, no de modo — cuando se escribió, ninguna corrida en `referencia` había
> pasado de 18 000 iteraciones. La comparación de modos a iteración pareada y larga sigue sin
> hacerse.

---

## 2026-09-02 — La respuesta de Adam a un pico, medida

`scripts/measure_adam_response.py`. Entrenamiento real desde `x-bf16-det/checkpoint.pth`; en la
iteración 40 se multiplica **sólo `∂L/∂T`** por `G`, de modo que los pesos siguen la trayectoria
del control y `T` es lo único que diverge. Se lee `T` 34 iteraciones después.

Resultado, en pasos de `lr` (`docs/mediciones/adam-respuesta-al-pico.txt`):

| | `G` = 10⁴ | `G` = 10⁹ | predicho |
|---|---|---|---|
| β₂ = 0.99 | 5.64 | 5.64 | 5.81 |
| β₂ = 0.999 | 5.96 | 5.97 | 6.34 |
| β₂ = 0.9999 | 6.02 | 6.02 | 6.40 |

Tres resultados:

1. **El desplazamiento no depende del tamaño del pico.** Cinco órdenes de magnitud de diferencia en
   `G`, coincidencia en tres cifras significativas, en los tres `β₂`.
2. **Es acotado y satura**, hacia la iteración 25–34 después del pico.
3. **Su magnitud es** `ΔT = lr·√((1−β₂ᵗ)/(1−β₂))`, con 3–6 % de error en las tres condiciones.

La derivación: ante un pico `G` mucho mayor que el fondo, Adam da `m_k = (1−β₁)β₁ᵏG` y
`v_k = (1−β₂)G²`, o sea

```
paso_k = lr·(1−β₁)β₁ᵏ·G / (√(1−β₂)·G)        ← G se cancela
Σ_k    = lr/√(1−β₂)                          ← límite asintótico
```

El factor `√(1−β₂ᵗ)` es la corrección de sesgo de Adam. Domina en `t` chico: en `t`=41 los tres
`β₂` dan ~6 pasos y son indistinguibles, mientras que la forma asintótica predeciría 10, 31.6 y
100. **En el régimen donde ocurren las muertes reales (`t` ≥ 2 000) el término de sesgo ya está
saturado**, y el desplazamiento vale `lr/√(1−β₂) = 0.01265`.

Esa medición hizo nulos por diseño tres experimentos de entrenamiento lanzados el mismo día
(`b2-A/B/C`): variaban `β₂` esperando ver el efecto en 600 iteraciones, donde por construcción no
puede aparecer.

---

## Mecanismo

**Por qué `∂L/∂T` se da vuelta.** Con `r = x_S − z` la corrección y `n` el ruido, a primer orden
`r ≈ T·∂x_S/∂T`, así que

```
∂L/∂T ≈ (2/T)·( ‖r‖² + ⟨n, r⟩ )
```

- **Sub-corrigiendo** (`‖r‖ < ‖n‖`): domina `⟨n,r⟩ < 0` → `∂L/∂T < 0` → `T` sube. Es el régimen
  sano, y por eso `T` trepa monótona durante miles de iteraciones.
- **Sobre-corrigiendo** (`‖r‖ ≫ ‖n‖`): domina `‖r‖²` → `∂L/∂T ≫ 0` → `T` baja.

El cruce ocurre cuando la corrección supera al ruido que debía quitar. No hay nada que empuje a `T`
hacia abajo de forma gradual: un solo lote que sobre-corrige lo hace de un golpe.

**Por qué el golpe es letal.** El freno es correcto en dirección y desproporcionado en magnitud:
Adam entrega hasta `lr/√(1−β₂) = 0.01265`, y `T` vive en 0.008–0.011 durante todo el bootstrap.

| momento | `T` | margen contra un pico |
|---|---|---|
| arranque (`T_init` = 0.001) | 0.001 | −0.0116 |
| meseta del bootstrap | 0.009 | −0.0036 |
| v4 en la iteración 20 000 | 0.0128 | +0.0001 |
| oráculo | 0.0247 | +0.0121 |

**Las cinco muertes del repo tuvieron `T` por debajo de 0.01265, y ningún evento con `T` por encima
mató una corrida** (8 de 8 eventos; `docs/mediciones/eventos-historicos.txt`). v4 es la única
corrida que cruzó el umbral, en la iteración 20 000 con `T`=0.01278 — y después del primer halving
del lr el umbral baja a 0.0063, así que quedó permanentemente a salvo.

**Un segundo motor, sin caracterizar.** Dos de los ocho eventos mueven `T` más de lo que un pico
único puede: `x-fp32-bench` @500 (29.2 pasos de `lr` contra un tope de 19.8) y v4 @69 500 (60.5
contra 31.6, con el clip activo). Los dos tienen `|g|` en 10⁴–10⁹ durante *cientos* de iteraciones.
Ahí `T` camina a `lr` por paso sin tope, y `T_init` no protege.

**Escala de `T_init`.** `T` es el tiempo total del flujo; su valor sólo significa algo relativo a
`‖∇R‖`, que depende de la inicialización de los pesos
(`scripts/measure_t_init_scale.py`, `‖ruido‖` = 27.13):

```
inicialización aleatoria, T=0.001    ‖∇R‖ =    3.82    ‖x_S − z‖/‖n‖ = 0.0001
los MISMOS pesos con T = 0.0247      ‖∇R‖ =    3.82    ‖x_S − z‖/‖n‖ = 0.0035
oráculo entrenado,       T = 0.0247  ‖∇R‖ = 3372.23    ‖x_S − z‖/‖n‖ = 0.9693
```

Con `T = 0.001` la red modifica la imagen en una diezmilésima del ruido: es la identidad numérica.
Y como el gradiente que llega a los pesos va multiplicado por `T`, el entrenamiento arranca con la
compuerta ~7 000× cerrada. Los pesos sólo aprenden si `T > 0`, `T` sólo sube si los pesos sirven, y
mientras se destraban `T` está a 2.5 pasos de Adam del cero.

**`T_init = 0.001` no está en el paper.** Verificado sobre ambos PDF y el suplemento: el texto sólo
da el conjunto admisible `T ∈ [0, Tmax]` para un `Tmax > 0` sin valor numérico, y el valor aprendido
`T* = 0.0247` (color) / `0.0172` (gris). El `0.001` sale de un campo del `config` serializado en el
`.pth`. El paper anterior (CVPR) reporta `T = 0.0297` para TDV3₂₅ con S=10: `T*` depende del modelo
y de la receta, no es una constante.

**Degeneración exacta `T ↔ ‖KN‖`.** `∇R = K1ᵀ ∘ MB^T ∘ KNᵀ(1/m)`, y como el potencial es la
identidad el valor de la salida nunca se usa: `KN` entra linealmente y sólo ahí. Verificado en
float64 sobre el oráculo, `(T, KN) → (cT, KN/c)` deja el esquema idéntico a 10⁻¹⁵ para
`c` = 0.1, 3 y 25 (`docs/mediciones/degeneracion-T-KN.txt`). El mismo oráculo se puede escribir
con `T = 0.0025` o con `T = 0.618`: **`T` no es identificable por sí sola**, sólo el producto
`T·ganancia`. Adam reparte los ajustes de esa ganancia según el paso relativo de cada coordenada
—4.4 %/paso para `T`, ~0.2 %/paso para `‖KN‖`— así que `T` absorbe casi todo, y es la única con una
pared dura.

---

## Corrección adoptada

```
T_init = 2·lr/√(1−β₂) = 2·4·10⁻⁴/√0.001 = 0.0253
```

Dos justificaciones independientes:

1. Queda por encima del desplazamiento máximo de Adam, así que ningún pico aislado puede llevar `T`
   a cero. Después de un pico del peor caso, `T` aterriza en el umbral, no en el clamp.
2. Abre la compuerta del gradiente de los 388 k pesos desde el paso 1, en vez de exigir que `T`
   haga el bootstrap por la zona donde muere.

Las dos entradas de la fórmula —`lr = 4·10⁻⁴` y `β₂ = 0.999`— están publicadas textualmente en la
sec. 5.1, así que no introduce ningún parámetro libre. `T_init` no lo está.

**Advertencia sobre la interpretación:** 0.0253 queda a 2.3 % de `T* = 0.024727`. La derivación no
mira `T*`, pero la coincidencia no puede usarse como evidencia a favor, y si una corrida arranca en
0.0253 y converge a 0.0247 el resultado es discutible. La variante que cierra esa objeción es
arrancar en `T_init = 0.1` (4× por encima de `T*`) y verificar que converge **bajando**. Es seguro:
con pesos aleatorios `‖∇R‖ = 3.82`, así que a `T = 0.1` la corrección es ~1.4 % del ruido y
`τ·L ≈ 0`.

`T_init = 0.0253` **no cubre el segundo motor** (régimen sostenido). Es una corrección parcial.

## 2026-09-02 — Los modos de ejecución, medidos

Tres decisiones que estaban tomadas por intuición y ahora tienen número. Diferencia **relativa** del
gradiente con y sin gradient checkpointing, sobre los 387 970 pesos:

| `benchmark` | determinista | TF32 | diferencia | |
|---|---|---|---|---|
| no | **sí** | no | **0.000e+00** | idéntico |
| no | **sí** | sí | **0.000e+00** | idéntico |
| **sí** | **sí** | no | **0.000e+00** | idéntico |
| **sí** | **sí** | sí | **0.000e+00** | idéntico |
| no | no | no | 9.7e-07 | |
| no | no | sí | 3.2e-06 | |
| sí | no | no | 4.9e-07 | |
| sí | no | sí | 3.5e-06 | |

**1. El checkpointing va atado al determinismo, no a la precisión.** Lo que rompe la recomputación
no es la selección de algoritmo de `cudnn.benchmark` —que se cachea por forma de tensor y es estable
dentro de una corrida— sino los kernels que acumulan con atómicos.
`use_deterministic_algorithms` los prohíbe, y ahí el gradiente es idéntico *con `benchmark`
encendido*. En bf16 con determinismo también sería exacto: no hay nada peligroso en la combinación
bf16 + checkpointing más allá de que `rapido` no corre determinista.

**2. `cudnn.benchmark` es inerte bajo determinismo.** Medido: 0.848 contra 0.847 s/it, y gradiente
idéntico. Con determinismo cuDNN sólo puede elegir entre algoritmos deterministas, así que no queda
nada que cronometrar. Deja de ser una perilla del modo y queda siempre encendido.

**3. TF32 no es reproducible corrida a corrida.** Corrige lo que este repo afirmaba antes
(*"da reproducibilidad intacta pero menos precisión"*). Dos evaluaciones idénticas, `benchmark`
encendido y sin determinismo:

```
TF32 apagado:  |g| = 6.3284163475   y   6.3284163475     idéntico
TF32 encendido:|g| = 6.3278198242   y   6.3278203011     difiere
```

Probablemente indirecto —TF32 habilita otros kernels de cuDNN, y ésos usan atómicos— pero la
consecuencia práctica es la misma. Además duplica largamente el error del checkpointing sin
determinismo (3.2e-6 contra 9.7e-7).

**Consecuencia sobre los modos.** TF32 no puede convivir con bf16 —bajo `autocast` las
convoluciones ya corren en bf16, más grueso, así que cuDNN nunca llega a aplicarlo— ni con
`referencia`, donde destruiría la propiedad por la que ese modo existe. Pero **sí tiene lugar como
modo propio**: 10 bits de mantisa contra los 7 de bf16, a 0.390 s/it contra 0.497 de fp32 IEEE con
`benchmark`. Quedan tres modos, no dos:

| modo | precisión | mantisa | s/it | VRAM/GPU | para qué |
|---|---|---|---|---|---|
| `rapido` | bf16 | 7 bits | 0.248 | 5.97 GB | explorar |
| `intermedio` | fp32 + TF32 | 10 bits | 0.390 | 10.95 GB | testear |
| `referencia` | fp32 IEEE | 24 bits | 0.868 | 3.30 GB | el número que se cita |

`intermedio` es de testeo, no de resultados citables: TF32 no es reproducible corrida a corrida.
Y el checkpointing sigue exigiendo determinismo —no IEEE—, así que sólo `referencia` lo enciende.

---

## 2026-09-03 — La corrección de `T_init`, y lo que destapó

Siete corridas con `T_init = 0.0253`: seis semillas × 4 000 iteraciones y una de 35 000, todas en
`referencia` con DDP.

### La corrección funciona para lo que se diseñó, y no basta

| semilla | desenlace | PSNR @4 000 | `T` final |
|---|---|---|---|
| 0 | viva | 31.190 | 0.0264 |
| 1 | amplificador @1 000 | −3.4 | 0.0324 |
| 2 | `T → 0` @2 400 | 20.804 | 0.0020 |
| 3 | viva | 31.102 | 0.0236 |
| 4 | `T → 0` @4 000 | 20.173 | 0.0000 |
| 5 | viva | 31.141 | 0.0258 |

**Ninguna murió por el mecanismo corregido.** En las tres, `T` valía ≥0.025 la iteración anterior y
cayó en un solo evento; el desangre lento por desplazamiento de Adam no aparece. Y las
supervivientes empatan con el histórico a 4 000 iteraciones —31.190 / 31.141 / 31.102 contra
31.170 (v4), 31.158 y 31.155— así que arrancar con `T` alto no cuesta convergencia. El déficit de
0.42 dB observado a las 2 000 iteraciones es transitorio y se cierra.

La corrida de 35 000 llegó a 31.074 @2 500, estalló @~2 900, quedó 20 000 iteraciones en 15.2 dB,
**se recuperó sola hasta 23.294** —la recuperación se acelera al partirse el `lr` en 25 000— y
volvió a morir @~33 000 con `T → 0`. Que la red vuelva de un estallido es nuevo; que no recupere el
nivel previo, también.

### `τ·L > 2` queda refutado como criterio de muerte

`scripts/measure_lipschitz.py` mide `L = ‖∇²R‖` por iteración de potencia sobre diez checkpoints:

| | `T` | `τ·L` | |
|---|---|---|---|
| oráculo publicado (31.53 dB) | 0.024727 | **4.80** | vivo |
| semilla 2, 400 it antes de estallar | 0.030076 | 2.59 | murió |
| semilla 5 viva | 0.025801 | 2.15 | viva |
| semilla 4, antes de estallar | 0.027456 | 2.03 | murió |
| semilla 0 viva | 0.026410 | 1.02 | viva |
| B, 400 it antes de estallar | 0.027345 | **0.70** | murió |
| inicialización aleatoria | 0.025298 | 0.0007 | — |

Vivas y muertas quedan intercaladas, y los dos extremos matan la hipótesis por separado: el
checkpoint de los autores viola el criterio por un factor 2.4 y funciona; la corrida con el `τ·L`
más bajo de todas estalló 400 iteraciones después. `L` en inicialización aleatoria vale 0.279, no
~800: la energía arranca plana y `L` crece tres órdenes mientras la red aprende.

**Consecuencia práctica:** acotar la norma de las convoluciones de los macro-bloques para mantener
`L < 790` —la corrección que la hipótesis indicaba— habría empujado la red lejos de la solución de
los autores, que vive en `L = 1 943`.

### `referencia` no es reproducible corrida a corrida

Dos corridas con todo idéntico —mismo seed, mismos flags, 300 iteraciones— divergen:

| | primera divergencia | dónde | a las 300 iteraciones |
|---|---|---|---|
| 2 GPU, DDP | iteración 1 | `∂L/∂λ`, 3.7e-5 relativo | `∂L/∂T` difiere 183× |
| 1 GPU, sin DDP | iteración 2 | `∂L/∂T`, ~6e-8 (ULP de fp32) | `loss` difiere 5.5 % |

El forward es idéntico —la `loss` de la primera iteración coincide bit a bit— así que la
divergencia nace en el backward y se amplifica caóticamente. **No es DDP**: sin él también ocurre, y
con dos ranks el allreduce suma exactamente dos flotantes, operación conmutativa. **No es
`warn_only`**: PyTorch no emitió ninguna advertencia de operación no determinista.

Queda como candidato principal `cudnn.benchmark`, y eso **corrige una afirmación de este repo**
(«benchmark es inerte bajo determinismo»). Con determinismo cuDNN sólo elige entre algoritmos
deterministas, lo que garantiza que cada algoritmo repita — no que dos algoritmos distintos den lo
mismo. La medición original se hizo dentro de un proceso, con la caché ya fijada, donde por
construcción no puede variar.

Consecuencias: sigue siendo válido estimar la **tasa** de mortalidad sobre varias semillas; deja de
ser válido atribuir la muerte de una corrida concreta re-corriéndola, y comparar dos
configuraciones con una semilla cada una.

### DDP no es la causa

Seis semillas en **una sola GPU**, batch 32 completo, sin NCCL ni allreduce ni buckets:

| semilla | desenlace | `‖g‖` mediana | `‖g‖` p99 | `max‖g‖` | estallido |
|---|---|---|---|---|---|
| 0 | `T → 0` | 104.1 | 964.9 | 1.29e+08 | 2 195 |
| 1 | `T → 0` | 133.2 | 1 161.4 | 7.0e+08 | 1 451 |
| 2 | **viva** (31.175) | 80.0 | 837.3 | **2.56e+03** | — |
| 3 | `T → 0` | 146.2 | 1 167.3 | 1.71e+11 | 1 577 |
| 4 | amplificador | 97.3 | 824.0 | 2.11e+13 | 2 719 |
| 5 | **viva** (31.163) | 78.7 | 768.7 | **2.20e+03** | — |

4 de 6 muertas sin DDP contra 3 de 6 con DDP. La tasa no baja: **el fallo ocurre en una sola
tarjeta**. La comparación no es pareada —con `world=1` el muestreo de parches es otro— así que lo
que se compara es la tasa, no la trayectoria.

### El estallido, a resolución de una iteración

`traza.npy` (nuevo) guarda por iteración `loss, T, ∂L/∂T, λ, ∂L/∂λ, ‖g‖`. La semilla 4 estalló con
la traza puesta:

```
  it 2716   loss  22.79   T 0.027061   dL/dT     −252.8   |g| 1 988
  it 2717   loss  31.38   T 0.027124   dL/dT   −3.84e+06  |g| 5.05e+06     <- estallido
  it 2718   loss 234.1    T 0.028347   dL/dT   −2.89e+05  |g| 3.02e+05
```

**En la iteración del estallido la `loss` todavía es normal (31.38, dentro del rango habitual
18–40) y el gradiente ya vale 5·10⁶.** La pérdida sólo se dispara en la siguiente, después de que
el optimizador diera un paso sobre ese gradiente. El estallido es un fenómeno del **backward**, no
del forward: la red produce una salida sana y la retropropagación a través de los diez jacobianos
compuestos explota.

El aviso previo es de 2 a 6 iteraciones. No hay `nan` ni `inf` en ninguna de las 4 000 filas.

**No hay congelación de `T`.** Después del estallido `∂L/∂T` no vale 0 en ninguna de las 1 261
iteraciones restantes, y `T` toma 1 252 valores distintos: oscila en la banda 0.0283–0.0381 porque
`∂L/∂T` alterna de signo con magnitud enorme y Adam normaliza por su RMS. La lectura de «`T`
congelada» en la semilla 1 venía de un log a cinco decimales cada 100 iteraciones.

### El predictor

`max‖g‖` separa sin solapamiento: **2.2e+03 y 2.56e+03 en las vivas, 1.29e+08 a 2.11e+13 en las
muertas** — cinco a diez órdenes de magnitud. Y las que murieron corrían más ruidosas desde el
principio: mediana de `‖g‖` 97–146 contra 79–80, p99 de 824–1 167 contra 769–837. Es una
diferencia medible **antes** del estallido, sobre miles de iteraciones sanas.

---

## 2026-09-04 — Las dos ablaciones: `cudnn.benchmark` y el checkpointing

Una variable a la vez, contra la fase A (`referencia`, DDP, benchmark y checkpointing encendidos).
Flags nuevos `--no-benchmark` y `--no-checkpoint`, sólo para ablación.

### Reproducibilidad: es `cudnn.benchmark`, y sólo él

A/B de 300 iteraciones, semilla 7, todo idéntico, comparando `traza.npy` bit a bit en las seis
columnas:

| `benchmark` | `checkpointing` | ¿idénticas? | primera divergencia |
|---|---|---|---|
| sí | sí | **no** (línea base, 2 GPU) | iteración 1 |
| sí | sí | **no** (línea base, 1 GPU) | iteración 2 |
| **no** | sí | **sí, bit a bit las 300** | — |
| sí | **no** | **no** | iteración 1 |

`cudnn.benchmark` es necesario y suficiente para la irreproducibilidad; el checkpointing no
interviene. Apagándolo, `referencia` recupera la propiedad por la que existe.

Esto cierra la corrección abierta ayer sobre la afirmación «benchmark es inerte bajo determinismo».
El determinismo garantiza que cada algoritmo repita, no que dos algoritmos deterministas distintos
den el mismo resultado; y la selección por cronometraje cambia entre procesos. La medición original
se hizo dentro de un proceso, con la caché ya fijada.

**Pendiente de decisión:** `benchmark` está encendido en los tres modos. Para `referencia` el costo
de apagarlo es velocidad —hay que medirlo— contra reproducibilidad exacta, que es justamente lo que
ese modo promete.

### El código no es la causa

`tests/test_gradient.py` verificaba `grad()` —contra autograd y contra una derivada direccional
numérica— pero ninguno de sus ocho tests derivaba `grad()` otra vez, que es lo que hace el backward
del entrenamiento. Dos tests nuevos cubren el hueco, y una comprobación sobre la arquitectura real
(no la miniatura del fixture) en float64:

```
d(gradR)/dx       desviación máx = 1.11e-16
d(gradR)/dtheta   desviación máx = 2.22e-15   (44 tensores)
gradiente completo (10 pasos + prox), manual vs autograd puro:
                  desviación máx = 2.22e-16   relativa = 8.81e-18   (46 tensores)
```

La sospecha concreta —`symmetric_pad2d_transpose` opera con slices, `flip`, `clone` y `+=` in-place,
así que una adjunta correcta podía tener derivada incorrecta— queda refutada. Y el conjunto de
restricciones coincide línea por línea con `../tdv/ddr/tdv.py`: `bound_norm=True` sólo en K₁,
`False` en los tres macro-bloques.

**Consecuencia.** El gradiente de 5·10⁶ que se mide en el estallido es el gradiente verdadero de la
pérdida en ese punto, no un artefacto. La pregunta deja de ser «qué está roto» y pasa a ser «cómo lo
evitaron los autores»: el objetivo tiene cola pesada y la receta publicada no menciona ningún
estabilizador.

---

## 2026-09-04 — Clip 1e4 pareado: descartado con una semilla

`runs/clip/plan.sh`: seis semillas × dos brazos, `referencia` con `benchmark` apagado, 4 000
iteraciones. Única variable: `--clip 1e4`. Al estar apagado `benchmark`, la semilla N de los dos
brazos es la MISMA corrida bit a bit hasta que el clip recorta por primera vez, así que la
comparación es pareada de verdad. Análisis: `scripts/comparar_pareado.py runs/clip base clip`.

### La semilla 0: el clip disparó, recortó el 13 % de la corrida, y murió igual

`traza.npy` de los dos brazos coincide **bit a bit hasta la iteración 3 428**. En la 3 429 `‖g‖`
vale 22 119 —primer recorte— con la `loss` todavía en 43.9, o sea el estallido del backward de
siempre. Desde ahí divergen:

| | base | clip 1e4 |
|---|---|---|
| primera divergencia | — | it 3 429 |
| pasos con `‖g‖ > 10⁴` | 568 / 4 000 | 535 / 4 000 (recortados) |
| `‖g‖` mediana it 3 901–4 000 | — | 1.16e+06 |
| `T` cae por debajo de 0.0126 | it 3 454 | it 3 456 |
| `max‖g‖` | 2.48e+10 | 1.47e+09 |
| `T` final | 0.00017 | 0.00000 |
| PSNR final | 22.46 | **20.17** |
| mejor PSNR | 31.099 | 31.099 (idéntico: es de it 3 000, antes de divergir) |

**El clip compró dos iteraciones y terminó 2.3 dB peor.** Entre 3 501 y 4 000 recorta el 90–99 %
de los pasos: está actuando al máximo y la corrida muere igual.

### Por qué era predecible, con una medición que ya estaba en el repo

`scripts/measure_adam_response.py` midió que el desplazamiento que Adam le imprime a un parámetro
vale `lr/√(1−β₂)` y **no depende del tamaño del pico**, sobre cinco órdenes de magnitud. El clip de
norma global multiplica todos los gradientes por un escalar común, y `m̂/√v̂` es invariante a eso
salvo por el transitorio de las dos EMA. O sea: **el clip no puede acotar el desplazamiento por
paso, porque Adam ya normalizó la magnitud.** Contra el régimen sostenido —cientos de iteraciones
con `‖g‖` en 10⁴–10⁹, el «segundo motor» sin caracterizar— lo que mata es el *signo* sostenido de
`∂L/∂T`, no su tamaño, y el clip no toca el signo.

### Cuánta reserva le hace falta a `T` para aguantar este evento

Replay de Adam sobre la coordenada `T` sola, con la secuencia de `∂L/∂T` registrada en la traza
(validado contra la traza a 1e-7, que es el redondeo fp32↔fp64), quitando el clamp en 0:

```
T sin clamp: mínimo −0.0037 en it 3 540, y ahí se aplana
excursión hacia abajo desde 0.0253:  0.0290
```

Con `T_init > ~0.054` la excursión no habría tocado el clamp en esta secuencia. **Es un
contrafactual sobre una secuencia de gradientes fija** —con otra `T` la trayectoria entera es
otra— así que vale como orden de magnitud de la reserva necesaria, no como predicción.

### Potencia: el experimento ya no puede llegar a 0.05

En una corrida que sobrevive, `max‖g‖` histórico es 2.2e3–2.6e3, o sea **por debajo del umbral de
10⁴: el clip nunca dispara y los dos brazos son literalmente la misma corrida**, un par concordante
con información cero. Los pares que pueden ser discordantes son sólo las semillas en que el brazo
base muere. Con McNemar exacto hacen falta b ≥ 5 discordantes para p ≤ 0.05 a una cola (0.5⁵).

Con la tasa histórica 7/12 = 0.583 y rescate perfecto (el clip salva el 100 % de las que morirían),
que es el techo:

| | P(≥5 muertes en base) | P(≥6) |
|---|---|---|
| al lanzar, 6 semillas | 0.208 | 0.039 |
| tras la semilla 0 (concordante-muerta), 5 restantes | 0.068 | 0 |

Harían falta **10 semillas por brazo** para 80 % de poder a una cola, y eso sigue asumiendo rescate
perfecto. Es la regla de la potencia aplicada tarde: la potencia no se calculó antes de lanzar.

### Veredicto: el brazo se cortó en la semilla 1

**El experimento se detuvo a propósito, no se cayó.** Tres razones, en orden de peso:

1. **El resultado ya está.** Con `max‖g‖` de 2.2·10³–2.6·10³ en las corridas que sobreviven, el
   umbral de 10⁴ **nunca dispara en una corrida sana**: los dos brazos son literalmente la misma
   corrida. La única semilla en que el clip llegó a actuar terminó 2.3 dB peor que el control.
2. **El diseño no puede alcanzar significancia.** Ver el cálculo de potencia de arriba: harían falta
   10 semillas por brazo, y aun así asumiendo rescate perfecto.
3. **Hay un mecanismo que lo explica** —Adam ya normalizó la magnitud, y el clip global no toca el
   signo— así que no es un nulo sin interpretación: es un nulo predicho.

Segunda vez que el clipping se descarta en este repo, ahora con un diseño pareado y un mecanismo.
**No volver a probarlo**, salvo con una variante que actúe sobre el *signo* sostenido de `∂L/∂T` y
no sobre la magnitud.

El brazo `base` (sin clip) siguió corriendo: es un control de mortalidad de 6 semillas en
`referencia` con `benchmark` apagado, y **no existe otro** — las 12 corridas históricas corrieron
todas con `benchmark` encendido. Es el control del experimento siguiente.

---

## 2026-09-04 — `T_init = 0.1`: el control que cierra la objeción de la coincidencia

Lanzado 15:18 en GPU 1, seis semillas × 4 000 iteraciones, `referencia`, `--t-init 0.1`
(`runs/tinit01/plan.sh`). Control: `runs/clip/base-seed*` en GPU 0, misma configuración con
`T_init = 0.0253`.

**Qué prueba, que no es la mortalidad.** La bitácora del 2026-09-03 ya dejó anotada la objeción:
`T_init = 0.0253` cae a 2.3 % de `T* = 0.024727`, así que una corrida que arranca ahí y "converge"
al valor publicado no distingue aprendizaje de coincidencia aritmética. `0.1` está **4× por encima
de `T*`**: si `T` converge *bajando* hacia ~0.0247, el número deja de ser discutible. Esa era la
variante que la propia bitácora proponía, y ésta es la corrida que la ejecuta.

**Por qué es seguro arrancar tan arriba.** Con pesos aleatorios `‖∇R‖ = 3.82`
(`scripts/measure_t_init_scale.py`), así que a `T = 0.1` la corrección vale ~1.8 % del ruido y
`τ·L ≈ 0`: la red sigue arrancando cerca de la identidad, pero con la compuerta del gradiente
abierta ~100× más que con el `0.001` original. Y el replay de Adam de la semilla 0 mide una
excursión hacia abajo de 0.0290 en el evento de muerte: 0.1 deja margen de sobra sobre eso.

**No es pareado.** Cambiar `T_init` cambia la iteración 1, así que la comparación con el control es
de *tasas*, no par a par. Potencia calculada antes de lanzar (`scripts/poder_experimento.py`,
Fisher exacto de dos colas, α=0.05, base 7/12 = 0.583):

| n/brazo | q=0.00 | q=0.10 | q=0.20 | q=0.30 |
|---|---|---|---|---|
| **6** | 0.21 | 0.12 | 0.07 | 0.04 |
| 10 | 0.80 | 0.44 | 0.23 | 0.11 |
| 12 | 0.93 | 0.60 | 0.36 | 0.19 |

Con seis semillas, **este experimento no puede decidir la mortalidad** ni en el caso de rescate
perfecto (21 % de poder). Se lanza igual porque su pregunta principal —¿converge `T` bajando hacia
`T*`?— **se responde con una sola corrida que sobreviva**, y la mortalidad es un subproducto que se
acumula al registro. Anotado por adelantado para no leerlo después como si hubiera sido diseñado
para lo que no puede medir.

**Predicción registrada:** `T` baja monótona desde 0.1 y se estabiliza en 0.02–0.03 antes de la
iteración 4 000. Si en cambio se queda cerca de 0.1, o sube, entonces `T` no está siendo determinada
por el objetivo y toda la lectura de `T*` como "valor aprendido" queda en duda.

**Consistente con el descarte previo** del clipping como estabilizador (umbral 20 000, murió @300
contra @550).

---

---

## Descartado

| hipótesis | veredicto | evidencia |
|---|---|---|
| que lo mínimo detectable en PSNR con una semilla por brazo sea **~0.33 dB** | ❌ | retirada el 2026-10-16 en la regla 5 de `AGENTS.md`. 0.33 es 2·0.164, el ancho de **una** corrida; la diferencia entre dos lleva √2: al 95 %, 1.96·√2·0.164 ≈ **0.45 dB** con σ conocida, ≈ 0.60 con σ estimada en las 6 semillas de `runs/tinit01` (t₅). Las entradas y las salidas crudas que dicen 0.33 quedan como están |
| que `checkpoint_qsm_rapido` se haya reentrenado sobre QSM o a la amplitud de uso, y que su 1.14 **demuestre la receta** | ❌ **afirmación propia retirada** | retirada el 2026-10-15 en la revisión de los `.md`. Es byte a byte `runs/t1-rapido-30k/best.pth` (`cmp`): cortes 2D de T1 en [0, 1], 1 canal, σ = 25, `rapido`, 30 000 iteraciones. El número sigue medido (1.1376 en el punto de operación); la causa, no: contra `8δ` cambian a la vez dominio, canales, modo, iteraciones y calendario, y el oficial gris, entrenado en BSDS, también queda en 1.05. Notas en la entrada «POR QUÉ los checkpoints propios rinden peor…» y en el pre-registro de `ventana-3d-12h`, cuya `--aug-ganancia` se motivó con ese rótulo |
| que **subir `target_std`** rescate a un checkpoint inestable en el ADMM de QSM | ❌ | predicción mía, registrada el 2026-10-12 **antes** de leer el barrido y **fallada**: pedía que `8δ` a `target_std = 0.25` bajara de 28.00 % y dio **31.2121 %**. El knob no mueve sólo la estabilidad: subir la amplitud mete la red en saturación y el prior se debilita (movimiento relativo 0.51 → 0.34 entre std 0.06 y 0.25). **`0.06` es el óptimo de todos los checkpoints**, no sólo del oficial — lo que de paso retira la idea de que el grid-search del solver estuviera atado a él |
| que **`T·‖K_N‖`** prediga el comportamiento como prior PnP | ❌ | es invariante de gauge y **ordena** bien (0.0260 / 0.0260 / 0.0316 / 0.0659), pero `1.1δ` tiene **exactamente** el del oficial y su `λ_max(∇²R)` sale **5.1×** mayor (2044 contra 402): el Hessiano depende de toda la cadena, no de `‖K_N‖`. Lo que sí predice es `(T/S)·λ_max(∇²R)` medido a la amplitud de uso (`scripts/diag_qsm_estabilidad.py`) |
| comparar dos modelos por sus **eigenfunciones** de `R` | ❌ | descartada el 2026-10-10 por el control que trae el propio script: a 1 000 iteraciones de `docs/avance/eigen.py` cada eigenfunción convergida correlaciona con **su propio recorte de partida** tanto como las dos entre sí —0.98716 y 0.99053 contra 0.99015 en img1, 0.93736 y 0.94581 contra 0.95126 en img2—, así que el 0.99 mide el arranque común y no el modelo. Lo que **sí** es propiedad del modelo es `∇R` evaluado en el **mismo** `x`, sin optimización ni arranque, y comparado por **coseno** porque el gauge `(T, K_N) → (cT, K_N/c)` deja el mismo modelo con otro `∇R`: 0.982 en imagen con ruido, 0.95 en limpia (`docs/mediciones/eigen-rapido-vs-referencia.txt`) |
| la **«regla 2-δ»**: que `T_init ≥ 2·δ_Adam` sea un umbral de supervivencia | ❌ | retirada el 2026-10-12, cuestionada por el autor. `δ_Adam` es real (derivado y medido, 3–6 % de error, independiente del tamaño del pico sobre cinco órdenes); **el factor 2 era una elección** —«después de un pico el `T` aterriza en el umbral, no en el clamp», o sea 1·δ de sobra, sin nada que diga que 1·δ sea seguro—. Nunca predijo: `tinit/B-larga` arrancó **en** 2·δ y murió con `T = 0`, y `t1p1delta-rapido-100k` arrancó en **1.1·δ** y terminó las 100 000. El mismo razonamiento ya estaba refutado para `λ` el 09-06 (hacía falta 2.2·δ, 10/12 seguían tocando). Y `δ_Adam` se parte a la mitad en cada hito, así que «2·δ del lr inicial» no es una cantidad fija. Queda un **costo continuo** medido y el default de 8·δ, que es el de la réplica |
| que el solver ADMM de QSM use `w = λ/T`, y que `w` compare dos corridas | ❌ | las dos mías y las dos retiradas el 2026-10-12. `../tdv/wTDV_QSM_torch.py:184-197` usa la red **entera** como denoiser PnP —`vn(z·α, z·α)/α`, con `α = target_std/qsm_std`— y no lee `T` ni `λ`; lo mismo el 3D. Y `w` es **gauge**: `(T, K_N) → (cT, K_N/c)` deja el modelo idéntico y manda `w → w/c`. Medido en las cuatro corridas de 10⁵: `w` abre **12.2×** mientras `T·‖∇R‖` queda dentro de **1.13×** y los cuatro `∇R` tienen coseno **0.978–0.987**. Lo invariante es `ρ = λ‖x−z‖ / T‖∇R‖`, y ahí lo real separa por **modo**: 0.15–0.18 (`referencia`) contra 0.031–0.039 (`rapido`) (`docs/mediciones/gauge-4-corridas.txt`) |
| que `T* = 0.0247` sea un **atractor** de `T` | ❌ | cerrada el 2026-10-11 con las cuatro corridas de 10⁵ sobre la mesa. La rama alternativa de la predicción 6 decía «si queda en [0.024, 0.030], `T*` **es** un atractor». `T_init/δ` → `T` final`/δ`: 1.1 → **2.23**, 2 → **2.76**, 8 → **6.53**, 8 → **6.83**. Se mueven hacia una banda del medio pero el **orden por `T_init` se conserva intacto**, y con el lr en 2.5·10⁻⁵ ya no se mueven más: `T_init` manda sobre `T` final. Es lo que pide el gauge `(T, K_N) → (cT, K_N/c)`, que impide que el modelo prefiera una `T` (`docs/mediciones/modo-y-tinit-100k-s1.txt`) |
| estimar la **reserva exigida** de `T` con un replay de Adam sobre el `∂L/∂T` de otra corrida | ❌ | cerrada el 2026-10-11 por la corrida que predijo. El replay daba 1.638–1.795·δ y de ahí salía «1.1·δ toca 0»: `t1p1delta-rapido-100k` **nunca bajó de 1.437·δ** y terminó en 2.228·δ. El método es una **traslación**, no un contrafactual —el paso de Adam sobre `T` no depende de `T`—, así que usa gradientes de `T ≈ 0.08` para decidir sobre `T ≈ 0.014`. Y le falta lo que sí explicó la supervivencia: **`δ_Adam` se parte a la mitad en cada hito del lr**, o sea que la reserva se duplica sola y la ventana de peligro es finita (`docs/mediciones/t1p1delta-100k.txt`) |
| que el lazo de `t1-guard-2delta-5k` lo arme la ventana restaurada (v2) | ❌ | cerrada el 2026-10-08, el mismo día. Lo arma `opt.load_state_dict`, que **adopta** los tensores del respaldo: desde el 2º rebobinado al mismo respaldo vuelven pesos y ventana pero no Adam. Los 19 rebobinados que fueron **primeros** a su respaldo dan **0 %** de rechazo en las 25 siguientes; los 38 **repetidos** (de 40), **96 %**. Reconstruir la ventana con los rechazados o ensanchar la razón ante un rebobinado repetido trataban el síntoma (regla 1). Arreglado en la rama `3D-Training` el 2026-10-08 y en `main` el 2026-10-09 |
| que `T_init` mueva el endpoint (el PSNR) | ❌ | cerrada el 2026-10-07. A **iteración pareada**, `T_init` de 2·δ a 10·δ —factor 5— abre **0.026 dB** @2 000 y **0.015 dB** @10 000, contra un piso de detectabilidad de 0.33 dB. Y el signo **se da vuelta**: el 2·δ va +0.027/+0.005 arriba @2 000 y −0.036/−0.017 abajo @4 000. Los «31.10–31.19 contra 31.5» que sostenían lo contrario son corridas de **10 000** iteraciones contra una de **100 000**: era largo de corrida, no `T_init`. Lo que `T_init` **sí** cambia es la exposición al estallido, 12–72× en tasa de rechazo. **Confirmada y extendida el 2026-10-11 a 10⁵ iteraciones**: 2·δ cuesta **0.011 dB** y 1.1·δ **0.022 dB** contra el control de 8·δ —o sea que el rango ahora va de 1.1·δ a 10·δ, un factor 9—, mientras la exposición va a 1.99× y 4.7×. El 12–72× es la **tasa de las primeras 4 000 iteraciones**, no el costo de la corrida (`docs/mediciones/modo-y-tinit-100k-s1.txt`) |
| extrapolar `T` a una iteración futura | ❌ | cerrada el 2026-10-07 con **3 de 3 fallas**: 0.02432 (retirada el 10-05), **0.0585** (predicción 1 enmendada, real **0.082642**) y la de PSNR 66k→100k (+0.0245 predicho, **+0.0137** real, 1.8× optimista). El modo de falla es siempre el mismo: la deriva de `T` **se da vuelta en los halvings del lr** —el mínimo cae en las it 49 991 y 49 997— y un ajuste lineal no lo ve. No se extrapola más |
| `‖Δθ‖` como criterio para cortar el entrenamiento | ❌ | cerrada el 2026-10-07 por medición propia: las dos semillas de `t8delta-100k` están a **68.89** una de otra = **131.8 % de `‖θ‖`** —más lejos entre sí que la norma de los pesos— y dan **el mismo PSNR**. Casi todo el movimiento de pesos es sobre una variedad plana, así que la distancia en espacio de parámetros no mide aprendizaje. El criterio que sí funciona es en **espacio de funciones** |
| que una cota ABSOLUTA sobre `‖g‖` prevenga el estallido | ❌ | la afirmé el 2026-10-04 y es falsa el mismo día. A 1e4 **acepta el lote letal** de `disyuntor-100k` (`‖g‖` 510.5) y no toca ninguno de los tres que meten a la cuenca (máximo 280.6). Separa «ya está muerta» de «viva» —hueco de 86× entre 962 y 8.29e4— y eso es diagnóstico, no prevención. Queda implementada por otro motivo: es la única pieza que funciona con la ventana vacía |
| que rechazar el lote alcance, sin deshacer el paso | ❌ | `runs/disyuntor-100k` rechazó **todos** los lotes sobre 1e4 —0 de 4 075 aceptados los superan— y murió igual, trabada 45 585 iteraciones. `‖g‖` se mide en los pesos de ahora: el detector llega un paso tarde por construcción, y el lote que la rompió tenía razón 7.70 |
| la razón `1/√(1−β₂) = 31.62` como umbral del disyuntor | ❌ | queda **por encima** de la banda de entrada medida, 7.70-8.97×: aceptó los tres lotes que metieron `disyuntor-100k` en la cuenca mala, y también el letal de 15.66×. La derivación del pico que duplica `v` es correcta y resuelve el problema equivocado |
| atar `λ = T` fija el gauge gratis | ❌ | no cuesta expresividad, pero la trayectoria de Adam no es invariante: `\|∂L/∂T\|` es 24–99× mayor que `\|∂L/∂λ\|` y los signos se oponen en el 93 % de las iteraciones, así que el escalar atado sería `T` y acoplaría dos modos de fallo hoy independientes. Y los autores no lo hicieron: su `model.py` sólo tiene `fixed` y `learned` por escalar, y los dos `.pth` oficiales traen `learned` en los dos |
| el port está mal | ❌ | reproduce los 4 números publicados y el conteo de parámetros exacto |
| gradiente mal calculado | ❌ | doble backward == autograd a 5·10⁻¹⁶ (float64), en los 18 tensores |
| inicialización mal portada | ❌ | idéntica al original: `normal(0, √(1/(in_ch·k²)))` |
| precisión mixta bf16 desestabiliza | ❌ | invertido: las 3 corridas bf16 sobreviven, las 3 fp32 mueren |
| `torch.compile` / Triton | ❌ | gradiente idéntico bit a bit con determinismo, en ambas precisiones |
| el checkpointing corrompe el gradiente | ❌ | 6 configs: 0.0 exacto con determinismo |
| el checkpointing causa las explosiones | ❌ | confundida con `scalar-lr 0.01`; v1 (literal + ckpt) llegó a 19 500 |
| clipping como estabilizador | ❌ | empeora: umbral 20 000 (31 % de pasos) murió @300 contra @550 sin clip |
| subir `T_init` a 10·`DELTA_ADAM` = 0.1265 | ❌ | está a 1.26× del default 0.1, y el rango de 4× (0.0253 contra 0.1) ya se midió sin separar: 4/6 contra 5/6 eventos, Fisher p = 1.0, con razón de hazard esperada 1.35× que necesita ~300 semillas por brazo. `T` no viaja —0 de 18 corridas sin evento bajan de 0.05 desde 0.1— así que subirlo mueve el punto de operación un 26 % y **aleja** de `T* = 0.0247` (de 4.0× a 5.1×). Y `max‖g‖` escala como `1/T`, así que el diagnóstico mejoraría 26 % gratis sin que la corrida esté más lejos de morir |
| winsorizar `‖g‖` a ~5× la mediana corrida | ❌ | el rango dinámico sano llega a **130× la mediana**: a 5× se rechaza el 0.1545 % del tramo sano con rachas de hasta 18 rechazos seguidos, contra 0.0238 % y 3 a la razón derivada de `β₂`. La recomendación era un orden de magnitud demasiado apretada y se retira |
| abortar la corrida tras N rechazos seguidos | ❌ | `veredicto_precision.py` separa terminada de en vuelo **contando filas** y `hazard_fase1.py` usa `len(traza)` como exposición: una corrida cortada se contaría como censurada viva y el análisis de supervivencia se **invierte** sin que nada se ponga rojo. Retirado antes de commitear; `train.py` no corta nunca |
| clipping como estabilizador (2ª vez, pareado) | ❌ | `--clip 1e4`, semilla 0: recorta el 90–99 % de los pasos finales, compra 2 iteraciones y termina 2.3 dB peor. Y hay mecanismo: Adam ya normalizó la magnitud, el clip global no toca el signo |
| warmup de lr | ❌ | no está en el paper; no se probó |
| `scalar-lr-scale 0.01` como fix | ❌ | causó el Modo 1 |
| `t_min` como fix | ❌ | v7: convierte el Modo 2 en Modo 1, loss 435 > 249 de la identidad |
| degeneración de escala `T ↔ pesos` | ❌ parcial | medida sobre la norma global; `‖KN‖` se mantiene 0.73–1.05 en todos los checkpoints |
| inestabilidad del forward (`τ·L > 2`) | ❌ | el oráculo opera a `τ·L = 4.78` con `growth = 0.537` |
| TF32 | ❌ | v6 (fp32 IEEE) murió @1 750, antes que v5 (@2 500) con TF32 activo |
| `growth` mide `τ·L` | ❌ | miden cosas distintas; y `growth` está mal normalizado (v7) |
| el paralelismo (DDP) causa las muertes | ❌ | 4/6 muertas **sin** DDP, 1 GPU, batch 32: la tasa no baja |
| el paralelismo causa la irreproducibilidad | ❌ | el A/B en 1 GPU también diverge; y con 2 ranks el allreduce suma dos flotantes, que es conmutativo |
| desborde del segundo momento de Adam congela `T` | ❌ | sintético: ni con `\|g\|=1e20` desborda (`v=1e37`); y en la traza `∂L/∂T` nunca vale 0 y `T` toma 1 252 valores distintos en 1 261 iteraciones |
| inestabilidad del paso (`τ·L > 2`) como criterio de muerte | ❌ | medido sobre 10 checkpoints: el oráculo opera a 4.80 y vive; B operaba a 0.70 y estalló 400 it después. Vivas y muertas quedan intercaladas |
| `cudnn.benchmark` causa la irreproducibilidad | ✅ | 2×2 de A/B: apagándolo, dos corridas dan `traza.npy` idéntica bit a bit; el checkpointing no interviene |
| el checkpointing causa las muertes | ❌ parcial | semilla 0 sin checkpointing: 31.127 dB y murió igual (1 semilla, no una tasa) |
| la segunda derivada de la cadena adjunta está mal | ❌ | arquitectura real, float64: `d(gradR)/dx` a 1.1e-16, `d(gradR)/dtheta` a 2.2e-15, gradiente completo del entrenamiento a 8.8e-18 relativo |
| la arquitectura se desvía del original | ❌ | `bound_norm`/`zero_mean`/`bias` idénticos a `../tdv/ddr/tdv.py` línea por línea |
| el ruido de `cudnn.benchmark` desestabiliza | ❌ | v5 corrió determinista y murió antes que v4 con `benchmark` |
| el potencial Student-t redescendente causa el estallido | ❌ | 2026-09-06, dos vías: bf16/fp32 sobre los MISMOS pesos da cociente 0.70–1.85 en 9 checkpoints, nunca 10⁵; y la fracción de pre-activaciones con `\|a\|>1` es **mayor** en las dos supervivientes (0.23) que en las que estallan (0.18–0.22) |
| la estructura residual amplifica la cadena adjunta | ❌ | confunde dos jacobianos. El interno de `∇R` sí crece (3.82 → 2 812), y es necesario. El del paso desenrollado, `(I − τ∇²R)/(1+λ/S)`, **contrae**: 0.83–1.10 por paso y **0.67–0.72 en total, uniforme en los 9 checkpoints** |
| el estallido es un lote raro sobre un estado fijo | ❌ | 64 lotes por checkpoint: 75 iteraciones antes de un estallido de 2·10⁹, `max‖g‖` sobre los 64 vale 191. El estado tiene que moverse primero |
| `amp₀ = ‖p₀‖/‖p₁‖` predice la muerte | ❌ | el oráculo tiene el valor más alto (1.097) y es el objetivo; vivas y muertas quedan intercaladas en 1.01–1.06, sobre 64 lotes y sobre lote único |
| el paso 0 de la cadena adjunta amplifica 8× con `T_init = 0.1` | ❌ **medición propia retirada** | era acumulación de `.grad` sobre un tensor de entrada compartido entre checkpoints; con la entrada clonada `amp₀` vale 1.012–1.097 en todos. Detectado el mismo día |
| subir `λ_init` cambia algo del forward | ❌ | barrido de `λ` sobre CBSD68: todo el rango 0–0.1045 es una meseta de **0.033 dB** en el oráculo y **0.003 dB** en nuestros checkpoints; y llevar `λ` a `λ*` los **empeora** 0.012–0.019 dB. Lo que hace daño es `λ` grande (0.5 → −0.6/−1.8 dB) |
| `λ_init = T_init` baja el hazard | ❌ | 6+6 semillas por brazo. Con `T_init = 0.1` el clamp desaparece del todo y el hazard queda en **1.03×** (log-rank exacto p = 0.97); con `T_init = 0.0253` el clamp **no** desaparece (5/6 lo siguen tocando) y el hazard sube 2.48× **sin mecanismo** — a la it 1 000 `w = λ/T` vale 0.022 contra 0.024 del control, o sea que `λ_init` ya está borrado |
| `λ` importa para el resultado, aunque sea poco | ❌ | el brazo B entrenó las 4 000 iteraciones con `w` **163×** más grande que su control (0.537 contra 0.0033) y dio 31.165/31.167 dB contra 31.188. La meseta de `λ` no es sólo del forward: también es del entrenamiento |
| el replay de Adam predice el conteo de clamp bajo otro `init` | ❌ parcial | acierta **quién** cruza (predijo que `seed4` sería la única de seis que no toca, y acertó la semilla; y 0/6 para `λ_init = 0.1`, exacto) y falla **cuánto**: predijo bajar de ~440 a ~167 iteraciones y la mediana medida es 377. Sirve para tamizar un `init`, no para predecir un conteo |
| la regla 2-δ aplicada a `λ` despeja el clamp | ❌ | replay de Adam sobre la `∂L/∂λ` registrada (valida a 5.6e-7, reproduce el conteo exacto en las 12): con `λ_init = 0.0253` `λ` **sigue tocando el clamp en 10/12**. La reserva que hace falta es 0.021–0.057, o sea 2.2·δ. Con 0.0379 toca 1/12; con 0.1, 0/12 |
| `λ` llega al clamp por un pico, como `T` | ❌ | `δ_Adam = 0.01265` es la asíntota `t→∞` y sólo vale para `t ≫ 1/(1−β₂) = 1000`. `λ` toca el clamp en la iteración **60–91**, donde la corrección de sesgo deja el desplazamiento en ~6·lr. `λ` no la tira un pico: **baja sola**, sostenido, a ~1.3e-4/it, con `∂L/∂λ > 0` en el 62–77 % de las primeras 60 iteraciones |
| una muerte concreta es atribuible a su estado | ❌ | replay de `base-seed0` desde la it 3 000: bit-idéntico 140 iteraciones, 1 ULP de diferencia en la 3 145, y el evento de la 3 429 **no ocurre**. `max‖g‖` 273 contra 2.48·10¹⁰ |
| la red bias-free es escala-equivariante | ❌ | Mohan et al. exige activación 1-homogénea; la Student-t no lo es. Medido: el esquema completo se desvía 15–28 % de `VNet(cz)=c·VNet(z)`. El reescalado entre σ es aproximación de dominio, no simetría |
| el halving del `lr` es el estabilizador de la receta | ⚠️ sin decidir | censurada: ninguna corrida con `traza.npy` llega a 25 000 y 29 de 32 terminan en 4 000. Sólo 3 corridas cruzaron: v1 murió antes del halving, v4 sobrevivió después (a favor), **B-larga murió @32 750 después** (en contra). n=1 contra n=1 |
| `λ` está en la banda letal de Adam como lo estuvo `T` | ✅ | `λ_init = 0.01 < lr/√(1−β₂) = 0.01265`, sale del mismo `config` del `.pth` que el `T_init` letal, `test_recipe.py` no la cubre, y `λ == 0` exacto en 359–971 de 4 000 iteraciones en **las 12** corridas. Falta el experimento |
| `T_init = 2·δ_Adam = 0.0253` como valor de trabajo | ❌ | pedido por el autor el 10-04 y retirado por él mismo el 10-05 con la evidencia sobre la mesa, **sin gastar GPU**. Ya estaba descartado como valor el 09-07 (queda como **piso**). La única corrida larga, `runs/tinit/B-larga` (35 000 it, `guard: None`), murió con `T = 0` y 20.173 dB @32 750: `T` había decaído a **0.0075 = 0.59·δ**, y debajo de `δ` un paso ordinario de Adam toca el clamp en 0, porque `δ` es el desplazamiento que Adam imprime **sin importar la magnitud del pico**. Extrapolando la deriva medida, 2·δ cruza `δ` en la it **7 714**. Y el disyuntor **no** lo cubre: `T → 0` no tiene pico y cada lote se ve sano |
| que el disyuntor de lotes cubra la muerte por `T → 0` | ❌ | es de otra clase. `T → 0` no es un pico: es la deriva de un escalar hacia una pared absorbente, con `‖g‖` normal en cada paso. `ddr/model.py:75` clampea `T` en `[0, 1000]` y `τ = T/S`, así que `T = 0` apaga el regularizador y la salida **es** la entrada: 20.173 dB = `20·log10(255/25)`. `replica-v5` se quedó ahí 92 000 iteraciones y ninguna corrida del repo se recuperó nunca |
| el escalamiento `1/T` de `max‖g‖` se lee sobre la norma global | ❌ | medido el 10-05 entre 0.1 y 10·δ: sobre `max\|∂L/∂T\|` la razón es **0.7655**, dentro de la banda predicha [0.74, 0.85]; sobre `max‖g‖` de lotes sanos es **0.9988** y sobre las primeras 100 iteraciones **1.0375**. La ley vale sobre `\|∂L/∂T\|`, no sobre `‖g‖`, porque a `T ≈ 0.1` esa componente es sólo el 27–36 % de la norma |
| la **densidad** de rechazos como disparador del rebobinado | ❌ | «N rechazos en las últimas W» en vez de N seguidos, para que un aceptado intercalado no reinicie el contador. La idea tiene base —el evento 2 de `tinit-7p9delta-10k` rebobinó en la it 5 901 y la densidad 25/50 se cumplía en la 5 864—, pero el corpus da el contraejemplo: `precision/rapido-seed2` tiene **33 rechazos en 50 iteraciones** en su tramo sano, con `‖g‖` llegando a 6 113 y **sin disyuntor**, y terminó en **30.937 dB**. A 25/50 el disparador habría rebobinado una corrida que no lo necesitaba. La densidad no separa «se resuelve sola» de «mata»; la magnitud sí, y de eso se ocupa la cota |
| bajar la cota absoluta a 5 000, «mejor calibrada» contra la ventana desarmada | ❌ | era mi plan del 10-05 y el argumento parecía bueno: la cota **sólo** actúa con el relativo desarmado, y ahí el techo sano medido sobre **86 corridas** es 2 556, con 0 de 86 arriba de 3 000, así que 10⁴ queda 3.9× por encima de lo que pasa. El error es de población por segunda vez: ese corpus **excluye las excursiones**, y las que se resuelven solas llegan a **6 113**. A 5 000 la cota rechazaría lotes de corridas que no hacía falta tocar. El valor se queda en 10⁴, que cae en el hueco medido [6 113, 1.971e5] |
| que `T_init = 8·δ` aterrice «justo por encima de la pared de muerte» a las 100 000 | ❌ | mi extrapolación del 10-05 daba `T` @100k = 0.02432 = 1.92·`δ`. Los parciales de 18 000 iteraciones la falsifican: la deriva real a esa altura es **−8.87·10⁻⁷/it**, no −1.4·10⁻⁶, y con el halving del `lr` en 25k/50k/75k el desplazamiento restante es 0.0256, así que `T` aterriza en **≈0.0585 = 4.6·`δ` = 2.36·`T*`** — y como la deriva se desacelera sola, eso es un piso. `8·δ` a 100k **no pone a prueba la pared**. El halving sí es el mecanismo que frena la caída, como estaba previsto: sin él `T` terminaría en 0.0114 = 0.90·`δ`, debajo de la pared |
| que `--checkpointing` baje la memoria ~9× en `train3d` | ❌ **medición propia retirada** | cerrada el 2026-10-09: el ~9× es del borrador `scripts/tdv3d_borrador/`, que checkpointea los 10 pasos. `ddr/model.py` checkpointea un paso sólo si su `x` lleva gradiente, y sin `--guard-amp` el `x_0` no lo lleva: el paso 1 guarda sus activaciones, 1.04 pasos medios en CPU. Son ~5× (derivado), y `vram_predicha` lo suma. Con 12.64 GiB predichos, `auto` elegía 32 en la 5070 Ti con `--checkpointing`, y el pico sería de ~23.9. Caen con él, de dos entradas del 10-08, que los números del borrador valgan para este modelo con `--checkpointing` y sin `--guard-amp`, y los ~15 M vóxeles por micro-lote en 64 GB: son ~9 M (derivado, 7.1 kB por vóxel) |
| los `L_init` prueban otro optimizador | ⚠️ sin respaldo | `weight.L_init=1e4`, `T.L_init=1e3` se portan de `../tdv/ddr/` y nunca se leen; el paper dice ADAM con un solo lr |
| que la GPU 1 entrene más lento por su enlace PCIe más angosto | ❌ | la afirmé el 2026-10-08 («~1.3× más lenta en `rapido`») y es un artefacto del mismo día: el ETA del entrenador es un promedio **acumulado**, y el cociente GPU1/GPU0 cambia de **1.44× a la it 250 a 1.21× a la it 500**, que es lo que no puede pasar si fuera de la tarjeta. Despejando arranque y régimen, la GPU 1 arrancó 54 s más lenta y su régimen queda **2.3 % más rápido**, debajo del cuanto del ETA. La medición hecha a propósito concuerda: **0.9 %, y la GPU 1 es la rápida** (`docs/mediciones/dos-gpu.txt`, seis celdas). El enlace es real —gen 4 ×4 contra gen 5 ×16— pero con la CPU 92.8 % ociosa tiene holgura de sobra. Crudo en `docs/mediciones/velocidad-dos-tarjetas.txt` |
| el cociente de Rayleigh con signo como `λ_max` del número de estabilidad (`ddr/qsm3d.estabilidad`) | ❌ | cerrada el 2026-10-12 en la revisión, el mismo día en que entró. Con dos autovalores dominantes de signo opuesto y módulo parecido **no converge a ninguno**: red de 4 features al azar, semilla 3, 6³ a amplitud 0.1 (−2.6731 y +2.6705): 0.97, 0.57 y −1.32 con 40, 200 y 1000 iteraciones, con un `cambio_relativo` de 1.5e-3 a 4e-3 que no lo avisa. Y con un dominante negativo el `paso` sale negativo y nunca marca `*INESTABLE*`. Vuelve `‖Hv‖`, el de `main`: max\|μ\|, que no es menor que el mayor positivo (`docs/mediciones/convergencia-estabilidad.txt`) |

---

## Abierto

*Actualizado 2026-09-04. Lo que se cierra se tacha acá y se explica en la entrada del día.*

1. **El estallido del backward.** `‖g‖` pasa de 10² a 10⁸–10¹³ en **una** iteración, con la `loss`
   todavía en rango normal. Es el pendiente principal y no es un bug: el gradiente está verificado
   exacto en float64 sobre la arquitectura real.
   - **Acotado el 2026-09-06.** No es un lote raro (64 lotes por checkpoint: `max‖g‖` = 191 a 75
     iteraciones de un estallido de 2·10⁹) y **no es atribuible a un estado**: un ULP de fp32
     284 iteraciones antes borra el evento. La cadena adjunta **contrae de punta a punta**
     (0.67–0.72 total, uniforme en los 9 checkpoints). Y en el modo `T → 0` la explosión
     está **entera** en `∂L/∂T` (mediana de `\|∂L/∂T\|/‖g‖` = 1.0000 sobre 568 iteraciones).
   - Lo que queda: la pregunta ya no es «qué tenía ese estado» —no tiene respuesta— sino qué mueve
     la **tasa**. El candidato estructural sigue siendo que sólo `K₁` lleva `bound_norm=True`,
     pero acotar los macro-bloques está descartado por separado (excluye el punto del oráculo).
2. **El «segundo motor»: el régimen sostenido.** Cientos de iteraciones seguidas con `‖g‖` en
   10⁴–10⁹, en las que `T` camina a `lr` por paso sin tope. `T_init` no lo cubre y el clipping
   tampoco (2026-09-04): lo que mata es el *signo* sostenido de `∂L/∂T`, no su magnitud. Nada
   medido lo acota todavía.
3. **Por qué fp32 entra en ese régimen y bf16 no.** Separación 3–3 sin excepciones en las corridas
   largas; el determinismo no separa. Ninguna explicación cierra numéricamente.
4. ~~**Una corrida que sobreviva 10⁵ iteraciones.** Sigue sin existir.~~ **Cerrado como pregunta y
   reformulado el 2026-09-20.** `runs/referencia-100k` corrió las 100 000 en `referencia` y
   **terminó sin `NaN`, sin clamp y sin morir** — y con 21.735 dB de pico. O sea que «sobrevivir» no
   era la pregunta. La reformulada es: **una corrida que llegue a 10⁵ y mantenga el PSNR**. El mejor
   resultado propio (31.478) sigue siendo de bf16 con clip 35 000 como disyuntor, o sea no es la
   receta literal.
7. ~~**El amplificador adjunto.** La cota `|∂L/∂T| ≤ ‖p_S‖·‖∇R‖ ≈ 5.2·10³` se viola por 750× en el
   evento más suave.~~ **Medido y reformulado el 2026-09-20, el mismo día.** La cota no se puede
   violar —es Cauchy-Schwarz sobre una identidad exacta, verificada a `10⁻⁸`— y el «750×» comparaba
   dos estados distintos: en el mismo estado, sobre 4 400 lotes, `|∂L/∂T|/cota` **no pasa de 0.50**.
   Lo que sobrevive es el mecanismo: la premisa falsa era «la cadena adjunta contrae», y en
   `referencia-100k` la cadena **amplifica por 1.5·10⁵**. Lo que la hipótesis erraba es *dónde*:
   `σ_max(A)` vale **1.5–5.1 en los nueve checkpoints sanos**, y en 3 600 lotes la amplificación
   máxima es 2.05. **El amplificador no está latente: lo construye el entrenamiento.** La pregunta
   que queda abierta es la siguiente, y es el punto 8 de `pendientes.md`: **qué dispara la
   transición**. No hay precursor medido — 2 600 iteraciones sanas con `amp` clavada en 0.70.
5. **`T` no es identificable por sí sola** — sólo el producto `T·ganancia`. **Confirmado
   dinámicamente el 2026-09-06:** una corrida que arranca en `T_init = 0.1` termina en 0.0984 y da
   31.188 dB con `‖∇R‖` 3.7× menor; el invariante medido es `T·‖∇R‖/‖n‖ ≈ 1.1–1.3` en el oráculo y
   en las tres supervivientes. `T* = 0.0247` es la coordenada de *ese* checkpoint sobre la órbita.
   Falta decidir si conviene fijar la escala explícitamente en la fase 2.
6. **`λ` no se analizó nunca, y está donde estuvo `T`.** `λ_init = 0.01` bajo el desplazamiento de
   Adam, del mismo `config` del `.pth`, sin guardia en `test_recipe.py`, tocando el clamp en
   359–971 de 4 000 iteraciones en las 12 corridas, con `λ*` del oráculo 10× más arriba. Es el
   experimento siguiente.

**Cerrado recientemente, no re-abrir:** `T_init = 0.0253` verificado en fp32 (2026-09-03: ninguna de
las 6 semillas murió por el mecanismo corregido); la causa de la irreproducibilidad es
`cudnn.benchmark` (2026-09-04, A/B 2×2); el gradiente es exacto (2026-09-04, float64 sobre la
arquitectura real); DDP no causa nada y está clausurado; el clipping no rescata (2026-09-04,
pareado, con mecanismo); `T_init = 0.1` responde su pregunta principal y **falsifica la predicción registrada** (2026-09-06: `T` no baja hacia `T*`); el estallido no es un lote raro ni es atribuible a un estado (2026-09-06, replay con perturbación de 1 ULP); la red **no** es escala-equivariante (2026-09-06, medido sobre el oráculo).

---

## Reglas de método

1. **No medir con las GPUs ocupadas.** Con `cudnn.benchmark`, la contención cambia la selección de
   algoritmos e invalida cualquier comparación numérica. Dos afirmaciones de esta bitácora nacieron
   de ese error.
2. **No cambiar dos cosas a la vez.** La atribución al checkpointing se sostuvo por eso.
3. **Calcular la potencia del experimento antes de lanzarlo.** Los tres brazos de `β₂` dependían de
   un evento que ocurre una vez cada varios miles de iteraciones, dentro de ventanas de 600.
4. **No parchear un síntoma.** Clip, warmup, `scalar-lr-scale` y `t_min` eran los cuatro parches
   disponibles. Los dos que se aplicaron crearon modos de fallo nuevos.
5. **No re-correr una semilla para entender una muerte** (2026-09-06). El replay con perturbación de
   1 ULP lo demuestra: bit-idéntico 140 iteraciones, un ULP en la 3 145, y 284 iteraciones después
   una trayectoria muere y la otra no. Una muerte concreta no es atribuible al estado, al lote ni a
   la trayectoria; sólo la **tasa** es propiedad de la configuración. Corolario: tampoco sirve
   comparar dos configuraciones con una semilla cada una, y `max‖g‖` no se compara entre brazos con
   `T_init` distinto porque escala como `1/T`.

---

---

## 2026-09-04 — Un tercer modo de muerte, en el control (`base-seed1`, en vuelo)

Leído de `traza.npy` a la iteración 2 000, con la corrida todavía andando. No es un veredicto de
tasa: es la caracterización del modo, que **no es `T → 0` ni el amplificador clásico** y que
`T_init` no puede cubrir.

```
evento (primer |g| > 1e4) ......... it 1 726
max|g| ............................ 2.18e+16   <- el mayor del repo (previo: 2.11e+13)
T ................................. 0.0253 -> sube a 0.0471, se asienta en 0.0356
lambda ............................ 0.0100 -> 0.0472
corr (log, it 2 100-2 400) ........ 6.8-7.3   <- la red AMPLIFICA 7x el ruido
loss .............................. ~11 000, estable
```

**`T` no bajó: subió.** El mecanismo corregido por `T_init = 0.0253` no interviene acá.

**Por qué los escalares quedan casi quietos, sin clamp.** `∂L/∂T` cambia de signo en el **43 %** de
los pasos consecutivos posteriores al evento, y su `|media|/rms = 0.060`. Adam avanza
`m̂/√v̂ ≈ 0.06` de su paso nominal: el desplazamiento mediano de `T` es **0.001·lr**, con p90 de
1.79·lr. O sea que `T` toma 213 valores distintos en 300 pasos —no está congelada— pero no va a
ningún lado. Es la **relación señal/ruido de Adam colapsada**, no el desborde de `v` (refutado, y
sigue refutado: `v` no desborda) ni el clamp (`T == 0` exacto en 0 iteraciones).

**Consecuencia para el registro:** el control lleva 2 de 2 muertas a esta altura, por **dos
mecanismos distintos** (semilla 0 `T → 0` @3 429, semilla 1 amplificador @1 726). La corrección de
`T_init` cubre uno de los tres modos observados.

---

## 2026-09-06 — Terminan los dos brazos, y el evento resulta no ser atribuible

Las doce corridas lanzadas el 2026-09-04 terminaron (control `runs/clip/base-seed*` con
`T_init = 0.0253`, brazo `runs/tinit01/seed*` con `T_init = 0.1`; las dos tandas en `referencia`,
`benchmark` apagado, 6 semillas × 4 000). Números crudos en
`docs/mediciones/tinit01-vs-control.txt`.

| | control | `T_init = 0.1` |
|---|---|---|
| corridas con evento (`\|g\| > 10⁴`) | **4 / 6** (semillas 0, 1, 4, 5) | **5 / 6** (todas salvo la 0) |
| `T` mínima sobre las 4 000 iteraciones | 0.0 en dos semillas | **≥ 0.0643 en las seis** |
| mejor PSNR @4 000 de una superviviente | 31.177 (`T` = 0.0287) | **31.188** (`T` = 0.0984) |
| `max‖g‖` de la superviviente más limpia | 2.04·10³ | **835** |

Fisher exacto sobre 4/6 contra 5/6 da p = 1.0, que es lo que la potencia calculada de antemano
anticipaba (21 % aun con rescate perfecto). La mortalidad no se decidió y no podía decidirse.

### La predicción registrada falla, y es el resultado más útil de la tanda

El plan dejó escrito: *«`T` baja monótona desde 0.1 y se estabiliza en 0.02–0.03 antes de la
iteración 4 000. Si en cambio se queda cerca de 0.1, o sube, entonces `T` no está siendo determinada
por el objetivo.»*

`T` **no bajó**. Las seis terminan en 0.064–0.098; la superviviente (semilla 0) va de 0.1 a 0.0984
en 4 000 iteraciones —un 1.6 %— y da el **mejor PSNR a 4 000 iteraciones de todo el repositorio**.
Se aplica la consecuencia registrada: `T` no está determinada por el objetivo.

Lo que sí está determinado es el producto. Sobre un lote común de 16 parches con ruido fijo
(`‖n‖` = 63.23, `scripts/measure_adjoint_chain.py`):

| checkpoint | `T` | `‖∇R‖` | `T·‖∇R‖/‖n‖` | `corr` |
|---|---|---|---|---|
| oráculo publicado | 0.02473 | 2 812 | **1.10** | 0.954 |
| control semilla 2 (viva) | 0.02872 | 2 804 | **1.27** | 0.956 |
| control semilla 3 (viva) | 0.02764 | 2 908 | **1.27** | 0.956 |
| `tinit01` semilla 0 (viva) | 0.09843 | **761** | **1.18** | 0.954 |

(La columna `amplif adj` de la salida cruda de ese script traía el bug descrito más abajo; las
columnas `T`, `‖∇R‖`, `corr` y `|∂L/∂T|/‖g‖` no lo tocan y siguen válidas.)

La red con `T` 4× más grande aprendió una energía 3.7× más débil y llegó al mismo punto de
operación. Las normas de los pesos no cambian (`‖K_N‖` 0.73–0.95 en todos), o sea que la
compensación está en la alineación y no en la escala — consistente con el «❌ parcial» ya anotado.
**`T* = 0.0247` no es un objetivo: es la coordenada que le tocó a ese checkpoint sobre la órbita.**
El invariante es `T·ganancia ≈ 1.2·‖n‖`.

Consecuencia para la fase 2: fijar la escala explícitamente (por ejemplo normalizando `‖K_N‖`) saca
un grado de libertad degenerado del optimizador y le quita a `T` la pared dura del clamp. Cierra el
punto 5 de `## Abierto` como medición; queda abierta la decisión.

### `max‖g‖` es, en su mayor parte, `|∂L/∂T|` — y eso descuenta al predictor

Descomposición de la norma global por tensor, en régimen sano (`‖g‖ < 10³`):

| | mediana de `\|∂L/∂T\| / ‖g‖` |
|---|---|
| control (`T` ≈ 0.028) | **0.70 – 0.78** |
| `tinit01` (`T` ≈ 0.10) | **0.26 – 0.32** |
| oráculo, sobre un lote real | **0.857** |

Un solo escalar carga tres cuartos de la norma de un gradiente de 387 970 dimensiones, y lo hace
también en el checkpoint publicado: **no es una patología, es la estructura del objetivo.**
`∂L/∂T = −(1/(S(1+λ/S)))·Σ_s⟨p_{s+1}, ∇R(x_s)⟩` es un producto interno de dos vectores grandes sobre
25 947 píxeles, sin promediar; cada peso recibe una contribución individual mucho menor.

En el estallido de `base-seed0` la mediana de `|∂L/∂T|/‖g‖` sobre las 568 iteraciones con
`‖g‖ > 10⁴` vale **1.0000** (p10 = 0.998): la explosión está **entera** en el gradiente de `T` y los
388 k pesos no aportan nada. En `tinit01` la misma fracción baja a 0.12–0.76, o sea que ahí sí
estalla el peso.

Dos consecuencias:

1. **Parte del poder de `max‖g‖` como predictor es un artefacto de la parametrización.**
   `|∂L/∂T| ~ 1/T`, así que subir `T_init` 4× baja `‖g‖` mecánicamente. Es por eso que
   `tinit01/seed0` marca `max‖g‖ = 835`, tres veces más silenciosa que cualquier superviviente
   registrada, sin que eso signifique que esté más lejos de morir: 5 de sus 6 hermanas estallaron.
2. **Tercera razón por la que el clip global no puede funcionar**, además de las dos ya anotadas:
   `clip_grad_norm_` reescala por una norma dominada por un escalar, así que encoge los gradientes
   de los pesos en el factor que pide la excursión de `T`.

### `λ` vive pegada al clamp, en las doce corridas

`λ == 0.0` exacto en **359 a 971 de las 4 000 iteraciones**, en las doce, incluidas las tres que
sobreviven. Primer toque en la iteración 60–91, rachas de hasta 293 iteraciones seguidas, último
toque entre la 509 y la 1 459. Durante esas iteraciones `prox(u, z, 0) = u`: el término de datos
está apagado y el esquema es flujo puro sobre `∇R`.

`λ_init = 0.01` está **por debajo** del desplazamiento de Adam `lr/√(1−β₂) = 0.01265` — exactamente
la condición que motivó corregir `T_init`. Y sale del **mismo campo del `config` serializado en el
`.pth` oficial** que dio el `T_init = 0.001` letal (verificado: el checkpoint de color guarda
`T:{init:0.001}` y `lambda:{init:0.01}`). `test_recipe.py::test_t_init_fuera_de_la_banda_letal`
guarda `T` y **no guarda `λ`**.

El destino está mucho más lejos que el de `T`: el oráculo tiene `λ* = 0.1045` (color) y `0.2169`
(gris), o sea 10× y 22× el init, contra 1.04× de `T*`. A 4 000 iteraciones nuestras corridas van por
0.021–0.079 y suben monótonas.

**Lo que está medido** es que `λ` toca el clamp en el 9–25 % del bootstrap en el 100 % de las
corridas. **Lo que no está medido** es que eso haga daño: las tres supervivientes también lo tocan.
Es la hipótesis con el argumento más apretado y sin experimento todavía.

### El evento de `base-seed0` no sobrevive a una perturbación de un ULP

`scripts/replay_evento.py` reanuda `base-seed0/best.pth` (iteración 3 000, con su estado de Adam),
con el mismo flujo de parches indexado y el generador de ruido adelantado 6 000 sorteos, e
instrumenta cada iteración. Salida en `docs/mediciones/replay-ulp.txt`.

```
loss bit-identica a la corrida original hasta la iteracion 3140
T    bit-identica hasta la 3144; en la 3145 difiere 1.9e-09 absoluto = 6.9e-08 relativo (1 ULP fp32)

                        3001-3480      max|g|        #(|g|>1e4)   T final
original                              2.48e+10           48       0.00000   <- muere
replay (perturbado 1 ULP)                273.7            0       0.02668   <- no muere
```

El original estalla en la iteración 3 429 y termina en `T = 0`; el replay atraviesa esa misma
iteración con `‖g‖ = 89.5` y sigue sano hasta la 3 480. **Un ULP de float32, 284 iteraciones antes,
borra el evento.**

Esto convierte en dato medido lo que la bitácora ya sospechaba de forma abstracta (el A/B del
2026-09-03: ULP → 183× en 300 iteraciones). La consecuencia es de método y es dura:

> **Una muerte concreta no es atribuible a nada.** No al checkpoint, no al lote, no a la
> configuración de la trayectoria. Cualquier post-mortem que pregunte «qué tenía este estado» está
> preguntando por un objeto que no existe a nivel de estados. Lo único que es propiedad de la
> configuración es la **tasa**, y para medirla hacen falta las semillas que dice
> `scripts/poder_experimento.py`.

Corolario: el «segundo motor» (punto 2 de `## Abierto`) tampoco se va a caracterizar mirando la
traza de una corrida.

### El estado previo al evento es benigno sobre 64 lotes

`scripts/measure_batch_tail.py` evalúa cada checkpoint sobre 64 lotes distintos de 16 parches con
ruido independiente, y mide la amplificación adjunta del primer paso `‖p₀‖/‖p₁‖` y `‖g‖`.

| checkpoint | amp₀ mediana | amp₀ máx | `‖g‖` mediana | `‖g‖` **máx sobre 64 lotes** |
|---|---|---|---|---|
| oráculo | **1.095** | 1.101 | 28.97 | 83.5 |
| control s2 (viva) | 1.031 | 1.034 | 49.5 | 119 |
| control s0 (estalla 429 it después) | 1.033 | 1.037 | 125.3 | 183 |
| control s5 (estalla **75 it** después) | 1.011 | 1.014 | 99.4 | **191** |
| `tinit01` s0 (viva) | 1.018 | 1.020 | 28.3 | 42.4 |

Setenta y cinco iteraciones antes de un estallido de 2·10⁹, el estado no produce `‖g‖ > 191` en
ninguno de 64 lotes. **El estallido no es un lote raro sobre un estado fijo: el estado tiene que
moverse primero.** Y `amp₀` no separa nada — el oráculo, que es el objetivo, tiene el valor más alto
de la tabla.

### La cadena adjunta contrae, y contrae igual en todos los checkpoints

> **Corregido el mismo día, antes de que nadie lo usara.** La primera versión de esta sección decía
> que «todo lo que amplifica está en el paso 0», con `amp₀` subiendo de 1.10 (oráculo) a 8.23
> (`tinit01`), y concluía que subir `T_init` «paga amplificación en el primer paso».
> **Era un bug de la medición, no un resultado.** `scripts/measure_adjoint_chain.py` compartía el
> tensor de entrada `noisy` entre checkpoints, y `.grad` **acumula** entre backwards
> (`m.zero_grad()` sólo toca los parámetros del modelo), así que `p₀` salía como la suma acumulada
> sobre todos los checkpoints corridos antes y la columna crecía monótona **en el orden de la
> lista** — que es justo el patrón que se leyó como señal. Se detectó auditando el diseño del
> experimento de `λ`. Corregido clonando la entrada en cada corrida.

Con la entrada clonada, `‖p_s‖/‖p_{s+1}‖` para los diez pasos:

```
oraculo                     1.097  0.981  0.982  0.981  0.979  0.972  0.960  0.942  0.912  0.834
base-s2 VIVA                1.033  1.022  0.991  0.980  0.976  0.970  0.960  0.944  0.914  0.836
base-s0 @3000 pre-evento    1.038  0.998  0.989  0.985  0.982  0.976  0.966  0.950  0.920  0.849
base-s5 pre-evento          1.012  0.998  1.001  0.991  0.979  0.976  0.974  0.955  0.922  0.842
```

`amp₀` vale **1.012–1.097** en los nueve checkpoints y la amplificación total `‖p₀‖/‖p_S‖` vale
**0.67–0.72**, sin distinguir vivas de muertas ni un `T_init` del otro. Coincide con lo que
`scripts/measure_batch_tail.py` —que nunca tuvo el bug, porque construye la entrada por lote— venía
midiendo sobre 64 lotes (1.011–1.095). Esa coincidencia es la validación de la corrección.

**Lo que sobrevive, y sale reforzado:** la cadena adjunta **contrae de punta a punta**,
uniformemente, en el oráculo y en las que estallan. La hipótesis (b) queda refutada más limpiamente
todavía: no hay **ningún** paso que amplifique.

**Lo que se retira:** que el paso 0 sea especial, y que subir `T_init` pague amplificación allí. Ese
intercambio **no está medido**, y la frase «el intercambio está medido, no supuesto» era falsa. Sigue
siendo cierto —y no depende de gradientes— que `‖∇R(x₀)‖` es 5–10× mayor que en los pasos
siguientes, porque `x₀` es la imagen ruidosa.

**Lección de método, tercera vez en este repo.** Las dos anteriores fueron medir con las GPUs
ocupadas. Ésta es la misma familia: un artefacto del *instrumento* que produce un patrón limpio y
ordenado, y por lo tanto convincente. La señal de alarma estaba a la vista —la columna crecía
monótona en el orden en que yo había escrito la lista— y se descartó como coincidencia.


---

## 2026-09-06 — Las cuatro hipótesis nuevas, contra la evidencia

Cuatro candidatos propuestos, ninguno en las 24 descartadas. Se probaron los tres que se podían
probar sin gastar GPU en entrenamiento.

### (a) El potencial Student-t redescendente — ❌ en sus dos mitades

**Mitad numérica: refutada en minutos.** El test propuesto era correr el mismo backward sobre los
mismos pesos en fp32 y en bf16, esperando 10⁸ contra 10³ si la cancelación fuera el mecanismo.
Medido sobre nueve checkpoints (`scripts/measure_escala.py` comparte el andamiaje):

```
cociente ||g||_bf16 / ||g||_fp32 :  0.70  0.90  1.00  1.03  1.14  1.15  1.31  1.46  1.85
```

Nunca sale del orden de magnitud, ni siquiera sobre el checkpoint ya muerto (1.85). La separación
3–3 fp32/bf16 de las corridas largas (punto 3 de `## Abierto`) **no es una cancelación por paso**;
es divergencia de trayectoria, que es lo que el replay de arriba demuestra que basta.

**Mitad estructural: no separa.** Fracción de pre-activaciones con `|a| > 1` —donde `φ''` cambia de
signo— sobre un lote común:

| oráculo | control s2 VIVA | control s3 VIVA | s0 pre-ev | s5 pre-ev | `tinit01` s0 VIVA | `tinit01` s1 pre-ev |
|---|---|---|---|---|---|---|
| 0.147 | **0.230** | **0.227** | 0.182 | 0.218 | 0.117 | 0.107 |

Las dos supervivientes del control son las que **más** cruzan `|a| = 1`. La hipótesis predice lo
contrario. Lo que sí aparece, y es real, es la cola: `max|a|` vale 281–2 310 contra una media de
0.7–3.7, o sea 100–600×. Está en las vivas igual que en las muertas.

Un dato lateral del mismo cuadro: la escala interna del oráculo (`|a|` medio 1.17) es la de
`tinit01` (0.93), no la del control (3.1–3.8), pese a que su `T` es la del control. Otra vista de la
degeneración.

### (b) La estructura residual amplifica por construcción — ❌ confunde dos jacobianos

La afirmación es correcta sobre el jacobiano **interno** de `∇R`: cada micro-bloque aporta
`J = I + K₂ diag(φ') K₁` y hay 15 de ellos sin `bound_norm`, así que `‖∇R‖` crece sin cota — de 3.82
en inicialización aleatoria a 2 812 en el oráculo. Eso es cierto, y **es necesario**: el oráculo vive
ahí.

Pero el gradiente del entrenamiento no compone ese jacobiano: compone el del paso desenrollado,
`J_paso = (I − τ∇²R)/(1+λ/S)`, que lleva un **signo menos** y contrae cuando `∇²R` es
semidefinida positiva en la dirección adjunta. Medido arriba: 0.83–1.02 por paso, 0.68–0.85 en total,
en los nueve checkpoints. El piso de 1 no existe en la cadena que importa.

Y acotar la norma de los macro-bloques ya está descartado por separado (2026-09-02): excluiría el
punto donde vive el modelo publicado (`L = 1 943`, `τ·L = 4.8`).

### (c) `λ` nunca se analizó — ✅ el argumento se sostiene, el experimento falta

Es la única de las cuatro que sobrevive. Confirmado con dato: `λ_init = 0.01` está por debajo del
desplazamiento de Adam, sale del mismo `config` serializado que el `T_init` letal, el test de receta
no la cubre, `λ` toca el clamp en 359–971 de 4 000 iteraciones en **las doce** corridas, y el
oráculo está 10× más arriba. Detalle completo en la entrada de arriba.

Lo que **no** se sostiene es la parte de la propuesta que dice que `λ` «se mueve en todas las
muertes»: en el control `λ` termina en 0.028–0.031 en las dos vivas y en 0.043–0.079 en las cuatro
muertas, pero también sube monótona en las vivas, y el 43 % de la separación se explica por las
iteraciones post-mortem. No es un predictor.

`λ_init` no está en el paper, igual que `T_init`, así que subirlo no agrega desviación sobre lo ya
aceptado. **Es el experimento siguiente.**

### (d) El halving del `lr` como estabilizador — ⚠️ no es testeable con lo que hay

La evidencia está censurada por diseño: de las 32 corridas con `traza.npy`, **ninguna llega a
25 000 iteraciones**; 29 de 32 terminan en 4 000 o antes, así que por construcción no pueden tener
un evento posterior al primer halving. Sólo tres corridas del repositorio cruzaron las 25 000:

- **v1** murió @19 750, antes del halving — no informa.
- **v4** tuvo su evento @69 250 y sobrevivió — el único punto a favor.
- **B-larga** murió @32 750, después del halving, con `lr = 2·10⁻⁴` — **contraejemplo**.

Y B-larga da además el mecanismo por el que el halving no protege: bajar el `lr` a la mitad baja el
desplazamiento de Adam a 0.0063, pero para entonces `T` había decaído a 0.0075, así que el margen
no mejoró. El halving sólo ayuda si `T` no decae en proporción.

Queda como no refutada y no respaldada, con n = 1 a favor y n = 1 en contra, y con un costo de
prueba de ≥25 000 iteraciones por semilla. No es el experimento siguiente.

---

## 2026-09-06 — La red **no** es escala-equivariante: se corrige el README

Afirmación vigente en `README.md` y en `AGENTS.md`: *«Todas las convoluciones son bias-free (…) Es
lo que hace a la red escala-equivariante y lo que valida el reescalado entre σ»*, citando a
Mohan et al. (ICLR 2020).

El teorema de Mohan et al. necesita una activación **positivamente homogénea de grado 1** (ReLU).
La Student-t no lo es: `φ(a) = ½log(1+a²)` tiene una escala propia, el `1` de `1+a²`, y
`φ'(ca) = ca/(1+c²a²) ≠ c·φ'(a)`. Ser bias-free no alcanza.

Medido sobre el oráculo (`scripts/measure_escala.py`, `docs/mediciones/escala-equivariancia.txt`):

```
||gradR(cx)|| / (c ||gradR(x)||)   c=0.25: 1.82   c=0.5: 2.47   c=2: 0.40   c=4: 0.12   <- no es grado 1
||gradR(cx)|| /    ||gradR(x)||    c=0.25: 0.46   c=0.5: 1.24   c=2: 0.80   c=4: 0.48   <- ni grado 0
esquema completo, ||VNet(cz,cz) - c*VNet(z,z)|| / ||c*VNet(z,z)||    c=0.5: 0.284   c=2: 0.152
```

El esquema completo se desvía un 15–28 % de la equivariancia exacta. **No hay simetría.**

**Por qué el reescalado entre σ funciona igual**, y reproduce 34.12 / 31.53 / 28.26 al centésimo: no
es una identidad, es llevar la entrada al nivel de ruido con el que se entrenó. Vale como
operación —sin él, σ=50 da 18.02 dB contra 28.26— pero es una aproximación de dominio, no una
consecuencia de la arquitectura.

**Qué cambia.** Nada del port ni de las reproducciones, que nunca dependieron de esto. Cambian dos
justificaciones escritas:

1. El punto 2 de «Restricciones matemáticas» del README dice que bias-free *«es lo que hace a la red
   escala-equivariante»*. Se corrige: bias-free es lo que hace que `R` no penalice offsets, junto con
   la media cero de `K₁`. La equivariancia no se sigue. *[Nota del 2026-10-16: el README
   reescrito ya no tiene esa lista: las restricciones están en `AGENTS.md`, «Mejoras
   candidatas», «Qué NO tocar», y en el README, «El esquema».]*
2. La mejora candidata de `AGENTS.md` —entrenar con σ ~ U(0,55)— tiene su justificación mal
   escrita, pero **la recomendación se refuerza**: como no hay equivariancia de la que colgarse, el
   reescalado es una aproximación real y entrenar ciego al nivel de ruido es la única vía limpia. Es
   además el mismo problema que en QSM obliga a calibrar la escala de entrada, ahora con la causa
   identificada: no hay simetría que explotar, sólo un dominio de entrenamiento.


---

## 2026-09-06 — La auditoría del experimento de `λ`, antes de lanzarlo

Con `λ_init` implementado y `runs/lambdainit/plan.sh` escrito, se auditó el diseño **antes** de
gastar las 9 h de GPU, con cinco lentes independientes (código, diseño estadístico, consistencia con
la bitácora, teoría de `λ`, y «qué falta») y verificación adversarial de cada objeción. 53
objeciones, 52 verificadas. Cuatro de las que sobrevivieron **matan el experimento tal como estaba
diseñado**, y las tres primeras se comprobaron de nuevo a mano antes de escribir esto.

### 1. `λ` es una dirección casi plana: el clamp no es un modo de fallo

Barrido de `λ` sobre CBSD68 completo, σ=25, variando **sólo** `model.lmbda`
(`scripts/measure_lambda_sensibilidad.py`, congelado en `docs/mediciones/lambda-sensibilidad.txt`):

| checkpoint | `λ` propia | `λ`=0 | `λ`=0.01 | `λ`=0.0253 | `λ`=0.1045 | `λ`=0.2 | `λ`=0.5 |
|---|---|---|---|---|---|---|---|
| oráculo | 0.1045 · 31.537 | **−0.033** | −0.027 | −0.019 | 0.000 | −0.059 | −1.787 |
| base-s2 (viva) | 0.0275 · 31.177 | **−0.003** | −0.001 | −0.000 | −0.012 | −0.073 | −0.630 |
| base-s3 (viva) | 0.0305 · 31.070 | **−0.003** | −0.002 | −0.000 | −0.012 | −0.072 | −0.669 |
| tinit01-s0 (viva) | 0.0212 · 31.188 | **−0.001** | −0.000 | −0.000 | −0.019 | −0.087 | −0.624 |

**Apagar el término de datos entero cuesta 0.033 dB en el oráculo y 0.003 dB en nuestros
checkpoints.** Todo el rango donde vive el clamp es una meseta. La dirección que hace daño es `λ`
**demasiado grande**, no demasiado chica.

Y lo más importante: **llevar `λ` a `λ* = 0.1045` empeora nuestros checkpoints** en 0.012–0.019 dB.
`λ*` es la coordenada donde terminó *ese* checkpoint dados *sus* pesos — exactamente la conclusión
que este repo acaba de establecer para `T` el mismo día. El argumento de `λ` estaba re-cometiendo el
error que se acababa de retirar para `T`.

**Consecuencia: «`λ` toca el clamp» queda descartado como modo de fallo.** Sigue siendo cierto que
`λ_init` está mal elegido y mal justificado; deja de ser cierto que arreglarlo pueda mover la
mortalidad por la vía del forward.

### 2. La pregunta principal ya estaba contestada, sin GPU

El paso de Adam sobre un escalar depende **sólo de su propia secuencia de gradientes**, y `traza.npy`
ya guarda `∂L/∂λ` por iteración en las 12 corridas. Re-integrando Adam sobre esa secuencia con otro
`λ_init` (`scripts/replay_lambda_adam.py`, `docs/mediciones/lambda-replay-adam.txt`):

```
validacion: la reconstruccion desde dL/dlambda reproduce la lambda grabada a 5.6e-7
            y el conteo de clamp EXACTO en las 12 corridas (457, 398, 453, 971, ...)

iteraciones con lambda == 0 exacto, sobre 4000:
  lambda_init      0.01   0.0253   0.0379   0.0506    0.1
  toca el clamp   12/12    10/12     1/12     1/12   0/12
  mediana            455      165        0        0      0
reserva necesaria (caida de la trayectoria sin clamp): 0.0208-0.0565, mediana 0.028 = 2.2*delta
```

**El brazo A (`λ_init = 0.0253`) falla su propia pregunta principal**: `λ` sigue tocando el clamp en
10 de 12. La regla 2-δ da 2.00·δ y hace falta 2.2·δ. Con 3·δ = 0.0379 toca 1/12; con 0.1, 0/12.
Esto era calculable en dos minutos y no se calculó — la regla de la potencia aplicada al contenido, no sólo a la
potencia.

### 3. La potencia estaba mal aplicada: el brazo A tiene poder exactamente 0

`scripts/poder_experimento.py` integra sobre los desenlaces de **los dos** brazos, o sea que asume
que el control también se va a sortear. Acá el control ya está corrido y congelado (4/6 y 5/6), así
que lo que corresponde es el poder **condicional**. Con la propia `fisher_p` del script:

| | q=0.00 | q=0.10 | q=0.20 | plan.sh decía |
|---|---|---|---|---|
| brazo A (control fijo 4/6) | **0.000** | 0.000 | 0.000 | 0.35 / 0.22 |
| brazo B (control fijo 5/6) | **1.000** | 0.531 | 0.262 | 0.74 / 0.51 |

Con el control fijo en 4/6, el mejor desenlace concebible del brazo A (0/6 muertas) da p = 0.0606,
por encima de α. **Ningún resultado del brazo A puede alcanzar significancia.** El brazo B está
mejor de lo que el plan afirmaba: 0/6 da p = 0.0152.

El error va en las dos direcciones y es de aplicación, no de aritmética. Pendiente: agregarle a
`poder_experimento.py` un modo de control fijo, para que el próximo experimento contra un control
archivado no lo repita.

### 4. El mecanismo atribuido a `λ` era el de `T`, y no es el suyo

`δ_Adam = lr/√(1−β₂) = 0.01265` es el límite `t→∞` de `lr·√((1−β₂ᵗ)/(1−β₂))`, y sólo se alcanza
cuando `t ≫ 1/(1−β₂) = 1000`. La medición congelada del repo lo dice sin ambigüedad: el pico se
inyecta en `t = 40` y el efecto medido son ~6 pasos de `lr`, no 31.6.

`λ` toca el clamp en la **iteración 60–91**, o sea justo donde la corrección de sesgo domina. Ahí no
hay ningún pico: `max‖g‖` en las primeras 100 iteraciones vale 740–2 328, régimen silencioso, y el
primer evento `‖g‖ > 10⁴` llega entre 1 000 y 3 800 iteraciones después. **`λ` no la tira un pico:
baja sola**, sostenido, a ~1.3·10⁻⁴ por iteración, con `∂L/∂λ > 0` en el 62–77 % de las primeras 60
iteraciones — porque al principio el regularizador no sirve y cualquier tirón hacia `z` mete ruido.

O sea: el argumento de `δ_Adam` —cuya fuerza es que el desplazamiento **no depende del tamaño del
pico**— no aplica a `λ`, porque no hay pico. Contra una caminata la unidad no es `δ` sino
tasa × duración, y da los 0.021–0.057 medidos arriba.

### Qué queda en pie de la hipótesis (c)

| afirmación | estado |
|---|---|
| `λ_init = 0.01` sale del mismo `config` del `.pth` que el `T_init` letal, y no es del paper | ✅ verificado |
| `λ` toca el clamp en 359–971 de 4 000 iteraciones en las 12 corridas | ✅ verificado |
| `test_recipe.py` guardaba `T` y no `λ` | ✅ verificado |
| el mecanismo es el mismo que el de `T` (pico bajo Adam) | ❌ **refutado**: es deriva, no pico |
| `2·δ_Adam` despeja el clamp para `λ` | ❌ **refutado**: hace falta 2.2·δ; 10/12 siguen tocando |
| que `λ` toque el clamp es un modo de fallo | ❌ **refutado**: cuesta 0.003 dB |
| `λ*` = 0.1045 es un destino al que converger | ❌ **refutado**: llegar ahí empeora nuestros checkpoints |

**`λ_init` sigue siendo una elección mal fundada que conviene corregir por procedencia** —viene de un
campo de configuración serializado, no de la receta— **pero deja de ser candidata a explicar las
muertes.** El punto 6 de `## Abierto` se degrada de «el experimento siguiente» a «higiene».

---

## 2026-09-06 — El hazard de la fase 1, y el número que ordena todo lo demás

Nadie había convertido los tiempos de evento en una tasa por iteración y extrapolado. Es el número
que decide qué experimento vale la pena (`scripts/hazard_fase1.py`, `docs/mediciones/hazard-fase1.txt`):

| grupo | eventos | exposición | hazard/it | vida mediana |
|---|---|---|---|---|
| control (`T_init` 0.0253) | 4 | 19 445 | 2.06·10⁻⁴ | 3 370 |
| `tinit01` (`T_init` 0.1) | 5 | 17 939 | 2.79·10⁻⁴ | 2 487 |
| **agrupado** | **9** | **37 384** | **2.41·10⁻⁴** | **2 879** |

```
P(llegar a  4 000 sin evento) = 0.38
P(llegar a 25 000 sin evento) = 2.4e-03
P(llegar a 68 000 sin evento) = 7.8e-08     <- donde v4 marcó 31.478 dB
P(llegar a 100 000 sin evento) = 3.5e-11    <- el objetivo de la fase 1
```

**Para que una corrida de 10⁵ sobreviva con probabilidad 0.5 hay que bajar el hazard 35×; para 0.9,
228×.** Con la receta literal en `referencia`, el objetivo de la fase 1 **no es alcanzable por
muestreo**: haría falta del orden de 10¹⁰ corridas.

Esto reordena las prioridades de golpe:

- La diferencia entre las dos configuraciones ya corridas es **1.35×, y en la dirección equivocada**
  (`tinit01` peor). Cualquier ajuste de `init` juega en ese orden de magnitud. Está a 26× de lo que
  hace falta.
- **El único lever del repo que alguna vez pasó las 20 000 iteraciones es bf16.** Del inventario: v1
  murió @19 750, v4 tuvo su evento @69 250 y **terminó**, `x-bf16-det` corrió 5 000 limpias. Son
  ~2 eventos en ~125 000 iteraciones de exposición bf16, o sea `h ≈ 1.6·10⁻⁵`, **~15× por debajo del
  fp32** — y encima con 3.5× más exposición por hora de GPU. *Caveat*: v1 y v4 llevaban también
  checkpointing, `benchmark` y clip 35 000, así que la comparación limpia **no existe todavía**.
- O sea que el punto 3 de `## Abierto` («por qué fp32 entra en ese régimen y bf16 no») deja de ser
  una curiosidad: **en unidades de hazard es el efecto más grande medido en este repositorio, y el
  único del orden de magnitud que hace falta.**

> **RETIRADO el 2026-09-07 por la tanda `runs/precision/`.** El `h ≈ 1.6·10⁻⁵` de bf16 y el `~15×`
> de estos dos puntos salían de v1, v4 y `x-bf16-det`, que llevaban checkpointing, `benchmark` y
> clip 35 000 encima — el propio *caveat* lo decía. Medido limpio con 12 semillas por brazo, bf16
> tiene mortalidad **5/12** y TF32 **6/12**, razón de hazard **1.31×** (IC95 % [0.33, 5.41], p =
> 0.61). Con la supervivencia medida, que tres corridas bf16 llegaran a ≥5 000 tiene probabilidad
> **0.20** por azar. La afirmación se retira entera; el veredicto está al final de este archivo.

**De ahora en adelante, la unidad en la que se evalúa un experimento de fase 1 es el factor por el
que mueve el hazard**, no la diferencia de mortalidad a 4 000 iteraciones. Y `scripts/poder_experimento.py`
mide lo segundo: para lo primero corresponde un análisis de tiempos (log-rank), que usa los tiempos
de evento en vez de tirarlos a un binario.

---

## 2026-09-06 — `λ/T` no es invariante de gauge: el checkpoint oficial **es** la ec. (7)

Este repo afirma en tres lugares —README, `AGENTS.md` y la nota del paper— que *«los checkpoints
oficiales tienen `λ ≠ T`, o sea que el código generaliza el paper»*. **No se sostiene.**

La degeneración `(T, K_N) → (cT, K_N/c)` deja el esquema idéntico y **no toca `λ`**, así que el
cociente `λ/T` se puede llevar a cualquier valor eligiendo `c`. Transportando el checkpoint oficial
(`scripts/measure_gauge_lambda_T.py`, `docs/mediciones/gauge-lambda-T.txt`):

| `c` | `T` nueva | `λ/T` | `‖K_N‖` | `‖Δx_S‖/‖x_S‖` |
|---|---|---|---|---|
| 0.25 | 0.006182 | 16.91 | 4.2003 | 5.3·10⁻⁸ |
| 1 (original) | 0.024727 | 4.227 | 1.0501 | 5.3·10⁻⁸ |
| **4.22707 = λ\*/T\*** | **0.104522** | **1.0000** | **0.2484** | **5.4·10⁻⁸** |
| 25 | 0.618170 | 0.169 | 0.0420 | 5.4·10⁻⁸ |

Los cuatro desvíos coinciden en 5.3·10⁻⁸, que es el piso de fp32 del esquema y no una dependencia
de `c`. **Con `c = λ*/T*` el checkpoint publicado se escribe exactamente con `λ = T`**, o sea que
*es* la ec. (7) en otras unidades.

Tres consecuencias:

1. **Se retira la afirmación** de que los checkpoints oficiales prueban que el código generaliza el
   paper. El código sí *permite* `λ ≠ T`; los pesos publicados no son evidencia de que haga falta.
2. **Atar `λ = T` no cuesta capacidad expresiva.** Para toda solución `(T, λ, Θ)` existe la
   equivalente `(λ, λ, Θ_{λ/T})` con el mismo mapa. Y de paso **fija el gauge**, que es justo lo que
   pide el punto 5 de `## Abierto` — sin tener que normalizar `‖K_N‖` a mano.
   > ⚠️ **Corregido más abajo (entrada «Atar `λ = T`… y descartado»).** «No cuesta expresividad» es
   > cierto y «es gratis» no: la trayectoria de Adam no es invariante bajo esa reparametrización,
   > `|∂L/∂T|` es 24–99× mayor que `|∂L/∂λ|` y los dos gradientes se oponen en el 93 % de las
   > iteraciones. Y los autores no los ataron. **Descartado.**
3. **Inicializar `λ = T` no es «partir de la ec. (7)» en ningún sentido con contenido**: es una
   elección de unidades que el optimizador deshace en el primer paso. El `LAMBDA_INIT = T_INIT`
   adoptado hoy se justifica por procedencia (0.01 no es del paper), no por la ec. (7).

**Y esto responde mejor la pregunta de normalizar `‖K_N‖`.** Medido: `‖K_N‖` arranca en 0.98 ± 0.24
sobre 20 semillas y termina en 0.63–1.21 tras 4 000 iteraciones, con un cambio de **+0.06 %, +0.49 %
y −1.12 % en las tres corridas que sobreviven** — mientras `‖∇R‖` crece 736× (3.82 → 2 812). O sea
que **`K_N` está prácticamente congelado en su inicialización aleatoria: la ganancia viaja por los
macro-bloques, no por él.** Normalizar `‖K_N‖` fija una dirección plana que Adam **ya no está
recorriendo**: es barato y seguro, y casi un no-op. La fijación de gauge con contenido es atar
`λ = T`, o —más agresiva y sólo para la fase 2— congelar `T` y dejar que los pesos absorban la
escala, que haría estructuralmente imposible el modo `T → 0` y sacaría del optimizador la coordenada
que carga el 70–86 % de la norma del gradiente.

---

## 2026-09-06 — PRE-REGISTRO: la tanda de `λ_init` se lanza igual, y por qué

Escrito **antes de lanzar** y versionado acá a propósito: `runs/` está gitignoreado, así que un
pre-registro que viva sólo en `runs/lambdainit/plan.sh` no tiene sello temporal. Éste sí.

**Decisión del autor, tomada con la auditoría sobre la mesa.** La entrada anterior deja escrito que
`λ` vale 0.003 dB en nuestros checkpoints, que el brazo A falla su propia pregunta según el replay,
y que su poder condicional de mortalidad es exactamente 0. Se lanza igual. Queda anotado que la
objeción se planteó, se cuantificó, y se decidió seguir — no que pasó desapercibida.

**Configuración.** `runs/lambdainit/plan.sh`, 6 semillas × 4 000 iteraciones, modo `referencia`,
una variable por brazo:

| brazo | `t_init` | `λ_init` | control ya corrido |
|---|---|---|---|
| A | 0.025298221281347025 | 0.0253 | `runs/clip/base-seed*` (mismo `t_init`, `λ_init` 0.01) |
| B | 0.1 | 0.1 | `runs/tinit01/seed*` (mismo `t_init`, `λ_init` 0.01) |

**Pregunta principal registrada:** ¿cambia `λ_init` el **hazard**? Es lo único que el replay no
puede contestar, porque el replay congela la secuencia de gradientes y el experimento no. Endpoint:
eventos / iteraciones de exposición, analizado por **tiempos** de evento. Línea base: control
2.06·10⁻⁴, `tinit01` 2.79·10⁻⁴, agrupado 2.41·10⁻⁴.

**Predicciones registradas** (si fallan, *eso* es el hallazgo):

1. **Conteo de clamp:** A sigue tocando en ~10/12 con mediana ~165 iteraciones y primer toque
   corrido de la it 60–91 a la ~300–440; B da 0/12. Si A da cero, el replay está mal.
2. **Hazard:** no cambia más de 1.5× en ninguno de los dos brazos. Fundamento: `λ` vale 0.003 dB a
   pesos fijos, así que sólo puede actuar por la trayectoria, y la trayectoria es caótica.
3. **PSNR @4 000:** no cambia más de 0.033 dB, que es la cota medida a pesos fijos. Una diferencia
   mayor **no es `λ`**, es la trayectoria.

**Potencia condicional al control congelado** (que es la que corresponde, y no la que da
`poder_experimento.py` tal cual):

| | q=0.00 | q=0.10 | q=0.20 |
|---|---|---|---|
| brazo A (control fijo 4/6) | **0.000** | 0.000 | 0.000 |
| brazo B (control fijo 5/6) | **1.000** | 0.531 | 0.262 |

**Cómo no leer los resultados:** `max‖g‖` no se compara entre A y B (escala como `1/T`); ninguna
muerte individual es atribuible (regla de la no-atribución); que `λ` se quede cerca de 0.1 en el brazo B no es
convergencia, porque 0.1 está a 4.3 % de `λ*`; y la pregunta «¿sube `λ` hacia `λ*`?» queda
**retirada**, porque llegar a `λ*` empeora nuestros checkpoints.

**Costo:** ~4.6 h por brazo, los dos en paralelo, una GPU cada uno.

### Decisión sobre el gauge: no se toca

Consultado explícitamente y decidido: **se deja que la red aprenda la escala, como hasta ahora.**
No se normaliza `‖K_N‖`, no se ata `λ = T`, no se congela `T`. Queda anotado para no re-abrirlo:
el análisis está en la entrada de arriba —`K_N` está congelado en su init, normalizarlo es casi un
no-op, y la fijación de gauge con contenido sería atar `λ = T`— y la decisión es no aplicar ninguna
en la fase 1. Se puede reconsiderar en la fase 2; la evidencia ya está escrita.


### Corrección al costo, y la bandera que no queda

El costo real es **~8.4 h por brazo**, no 4.6: los dos arranques comparten un i9 y cada uno levanta
16 workers de `DataLoader`, así que la CPU es el cuello y no la GPU.

Y `--lambda-init` **se retira del código**. Existía sólo para poder pasar el valor histórico 0.01,
y no hace falta: `λ` arranca en `t_init` dentro de `vnet_config`, de modo que los dos brazos de esta
tanda se reproducen con `--t-init 0.0253` y `--t-init 0.1` y nada más. El brazo de control con
`λ_init = 0.01` ya está corrido y archivado en `runs/clip/`; recuperarlo no requiere una bandera
permanente en la receta.

## 2026-09-06 — El brazo B diverge en la it 1900–2000, y no muere

Primera semilla del brazo B (`T_init = λ_init = 0.1`), en curso:

| it | `loss` | `T` | `λ` | `‖g‖` | `max‖g‖` |
|---|---|---|---|---|---|
| 1 900 | 20.86 | 0.10209 | 0.03029 | 46.9 | 141 |
| 2 000 | 4 334 | 0.10817 | 0.04545 | 3.4·10⁷ | 4.5·10⁸ |
| 2 100 | 3.5·10⁷ | 0.07738 | 0.10818 | 1.3·10⁹ | 2.5·10¹⁶ |
| 2 400 | 3.8·10⁷ | 0.07718 | 0.10849 | 1.6·10⁹ | 3.5·10¹¹ |

Dos cosas nuevas respecto de los nueve eventos anteriores:

1. **No produce `NaN`.** La corrida sigue viva 500 iteraciones después, con la `loss` estacionada en
   3.8·10⁷ y la correlación en 394 (sana vale ~0.95). O sea que **«muerte» y «evento» no son lo
   mismo**, y el conteo de mortalidad a 4 000 los mezcla. El hazard, que cuenta eventos por
   iteración de exposición, no depende de si el `NaN` llega.
2. **`λ` sube, no baja.** El evento la empuja de 0.0303 a 0.1085 y a `T` la baja de 0.1021 a 0.0774.
   Es lo contrario del mecanismo de deriva del bootstrap, y es coherente con que el evento sea un
   pico —el régimen donde `δ_Adam` sí vale— y no una deriva.

Por la regla de la no-atribución no se pregunta qué tenía ese estado. Lo que se anota es que el evento ocurrió con
`T ≈ 0.10`, o sea que **`T_init = 0.1` tampoco es inmune**, que era la lectura optimista pendiente
desde la tanda `tinit01`. El veredicto de la tanda espera a las 12 semillas.

## 2026-09-06 — El hazard NO es constante: sube con la corrida

Preguntado explícitamente si el hazard estaba bien planteado. **No del todo.** El número
`h = 2.41·10⁻⁴/it` es correcto como *promedio* sobre las 37 384 iteraciones corridas, pero la
extrapolación a 10⁵ supone además que la tasa es **constante en el tiempo**, y ese supuesto se
rechaza. Tiempos de evento de las nueve corridas que murieron:

```
control (T_init 0.0253):  1726  3075  3215  3429   + 2 censuradas a 4000
tinit01 (T_init 0.1):     1106  2349  2978  3619  3887   + 1 censurada a 4000
```

**Ninguna muere antes de la iteración 1 106**, y cinco de nueve mueren en el último cuarto. Por
ventanas, sin suponer ningún modelo (`scripts/hazard_fase1.py`):

| ventana | eventos | exposición | hazard/it | esperados si fuera constante |
|---|---|---|---|---|
| 0–1 000 | **0** | 12 000 | 0 | 2.9 (p = 0.056) |
| 1 000–2 000 | 2 | 10 832 | 1.85·10⁻⁴ | 2.6 |
| 2 000–3 000 | 2 | 9 327 | 2.14·10⁻⁴ | 2.2 |
| 3 000–4 000 | **5** | 5 225 | **9.57·10⁻⁴** | 1.3 (p = **0.0094**) |

Una Weibull con censura a la derecha da forma **k = 3.22** contra el `k = 1` del hazard constante
(razón de verosimilitudes χ² = 10.2, **p = 0.0014**). El hazard sube aproximadamente como `t²`.
El patrón está en los dos brazos por separado, así que no lo produce mezclarlos.

**Qué se cae y qué se sostiene.**

- **Se cae el número `3.5·10⁻¹¹`** y con él el «hace falta bajar el hazard 35×». No hay un factor
  constante que arregle un hazard creciente. La misma Weibull extrapolada da `S(10 000) = 10⁻¹¹`,
  que es absurdo —v4 llegó a 69 250 en bf16—, o sea que **ninguna de las dos extrapolaciones vale
  fuera de la ventana medida**. No tenemos una sola corrida `referencia` más allá de 4 000.
- **Se sostiene la conclusión cualitativa**, y se refuerza: con hazard constante la fase 1 es
  inalcanzable por muestreo, y con hazard creciente es peor. Lo que cambia es que el objetivo deja
  de ser «dividir una tasa» y pasa a ser **«que la tasa deje de crecer»**.
- **Se sostiene el hazard como unidad de comparación** entre configuraciones, siempre que las dos se
  midan sobre la misma ventana de iteraciones. Comparar un brazo de 4 000 contra uno de 20 000 con
  este hazard mezcla la configuración con la posición en la curva.

**La pista que abre.** La tasa crece con la corrida, y lo único del entrenamiento que crece de forma
comparable es la ganancia aprendida: `‖∇R‖` va de 3.82 a 2 812 (**736×**) mientras `‖K_N‖` queda
congelado. Si el riesgo sigue a la ganancia y no al reloj, entonces el peligro está en **cuánto se
endurece el regularizador**, no en cuánto tiempo lleva entrenando — y eso es medible sobre las
trazas que ya existen, sin GPU: regresar el hazard por ventanas contra `‖∇R‖` en vez de contra `t`.
Queda anotado en `docs/pendientes.md` como el primer experimento, porque es gratis.

**Corolario para el diseño de experimentos.** Una configuración puede verse mejor a 4 000 y no serlo
a 20 000, porque a 4 000 se está midiendo la parte empinada de la curva. Las comparaciones de la
fase 1 siguen valiendo entre sí —todas se corren a 4 000— pero **no se proyectan**.

## 2026-09-06 — Error de método: editar el código con corridas en vuelo

Al sacar la bandera `--lambda-init` de `train.py` **mientras `runs/lambdainit/plan.sh` seguía
corriendo**, las semillas 1–5 de los dos brazos murieron con `unrecognized arguments` y el bucle
las recorrió en dos segundos. Las semillas 0 ya estaban en vuelo y terminaron bien; se perdieron
diez corridas y ~7 h de GPU.

`plan.sh` invoca `train.py` **una vez por semilla**, así que el proceso en vuelo no protege a los
que faltan: cualquier cambio en la interfaz de `train.py` los alcanza. Lo mismo vale para editar el
propio `plan.sh`, que `bash` lee de a pedazos.

**Regla que sale de acá, ya en `AGENTS.md`:** con una tanda en vuelo no se toca ni `train.py` ni el
`plan.sh` que la lanzó. Si el cambio no puede esperar, primero se para la tanda.

Relanzado a las 19:10 con la interfaz nueva —`--t-init` solo, `λ` lo sigue— y el loop en `1 2 3 4 5`.
Las semillas 0 son idénticas: se corrieron con `--lambda-init` explícito y con el mismo valor.

**Resultado de las dos semillas 0**, contra las predicciones registradas:

| | primer toque de `λ = 0` | evento (`‖g‖ > 10⁴`) |
|---|---|---|
| A (`T = λ = 0.0253`) | it ~300, contra la 60–91 del control | 2 660 |
| B (`T = λ = 0.1`) | nunca; `λ` baja de 0.1 a 0.030 en 1 900 iteraciones | 1 958 |

Predicción 1 confirmada en el primer punto de control. Los eventos, con una semilla por brazo, no
dicen nada (regla de la no-atribución); el veredicto espera a las doce.

## 2026-09-06 — Atar `λ = T`: verificado contra el código de los autores, y descartado

La entrada del gauge dejaba abierto que atar `λ = T` «no cuesta expresividad y fija el gauge
gratis». **Gratis no es.** Se verificó por dos vías independientes.

### 1. Qué hicieron los autores — evidencia directa, no del port

Contra `https://github.com/VLOGroup/tdv` (rama `master`), no contra los repos hermanos, porque el
entrenamiento de `../tdv` **no es de los autores**. Lo que publicaron es sólo inferencia:
`model.py`, `denoise.py`, `eigenfunctions.py`, `ddr/{conv,regularizer,tdv}.py` y los dos
checkpoints. **No hay código de entrenamiento, ni config, ni optimizador.**

En su `model.py`, textual:

```python
if config['T_mode'] == 'fixed':      ...
elif config['T_mode'] == 'learned':
    self.T = torch.nn.Parameter(torch.Tensor(1));  self.T.L_init = 1e+3
if config['lambda_mode'] == 'fixed': ...
elif config['lambda_mode'] == 'learned':
    self.lmbda = torch.nn.Parameter(torch.Tensor(1));  self.lmbda.L_init = 1e+3
```

Tres cosas:

- **Dos parámetros independientes**, cada uno con su `_mode`, su `init/min/max` y su `proj`.
  Los únicos modos son `fixed` y `learned`: **no existe un modo atado.**
- El `config` serializado **dentro de los `.pth` oficiales** —que lo escribió el entrenamiento de
  ellos, no el nuestro— dice `T_mode: learned` y `lambda_mode: learned` en los dos checkpoints. En
  las corridas que produjeron los pesos publicados, los dos se aprendieron por separado.
- Aparece `L_init` (`1e+3` en los dos escalares, `1e+4` en `Conv2d.weight`, verificado también
  contra el original). Es una estimación de constante de Lipschitz por parámetro, vestigio de la
  familia IIPG de las redes variacionales anteriores del grupo. **Bajo Adam no hace nada** —Adam
  normaliza por coordenada— y el paper dice Adam explícitamente en la sec. 5.1, así que no cambia la
  receta. Pero deja registrado que en su linaje los dos escalares iban **juntos entre sí y
  distintos de los pesos** (paso 10× más grande), nunca uno atado al otro.

**Veredicto de procedencia: los autores no ataron `λ` a `T`.**

### 2. Por qué atarlos sería malo acá — medido

Atar significa que un solo parámetro recibe `∂L/∂T + ∂L/∂λ`. Sobre las 12 corridas, hasta el evento
(`scripts/measure_grad_T_lambda.py`, `docs/mediciones/grad-T-vs-lambda.txt`):

| | `T_init = 0.0253` | `T_init = 0.1` |
|---|---|---|
| `\|∂L/∂T\| / \|∂L/∂λ\|` (medianas) | **90–99×** | **24–26×** |
| iteraciones con signos **opuestos** | 90–94 % | 93–95 % |

- **El escalar atado sería `T` con ruido.** Con la derivada de `T` entre 24 y 99 veces más grande,
  `λ` dejaría de aprenderse en la práctica: la arrastraría `T`.
- **Los dos gradientes se oponen en el 93 % de las iteraciones**, y eso **sí** es invariante de
  gauge (la degeneración usa `c > 0` y no cambia ningún signo; la razón de magnitudes, en cambio, no
  lo es — escala como `1/T`, que es por qué el bloque de 0.1 da 25× y el de 0.0253 da 95×). Tiene
  sentido mecánico: subir `T` endurece el regularizador y subir `λ` tira hacia la imagen ruidosa.
  **Son dos presiones opuestas, y atarlas es darles una sola perilla.**
- **Acopla dos modos de fallo que hoy son independientes.** Hoy `T → 0` mata (cierra la compuerta
  del gradiente) y `λ → 0` es inocuo (0.003 dB). Atados, un pico que lleve `T` al clamp **apagaría
  también el término de datos**. Y el evento del brazo B del 2026-09-06 subió `λ` de 0.030 a 0.108
  mientras bajaba `T`: atados, ese movimiento no puede ocurrir — la perilla tendría que ir para los
  dos lados a la vez.

### 3. ¿Tiene sentido para TDV, entonces?

Sí, como **normalización**, y por eso la aritmética del gauge da bien: el paso proximal implícito
corresponde a la energía `w·½‖x − z‖² + R(x)` con `w = λ/T` y paso `T/S`. O sea que **`λ/T` es el
peso de fidelidad de la energía**, y `λ = T` es simplemente `w = 1`, con la escala absorbida en `R`.
Como la energía está definida salvo escala global, no se pierde nada — que es exactamente lo que
midió el transporte del checkpoint oficial.

Pero **la trayectoria de Adam no es invariante bajo esa reparametrización.** Adam normaliza por
coordenada, así que el camino en `(T, K_N)` y el camino en `(cT, K_N/c)` son distintos, y lo que
mata las corridas de este repo es la dinámica, no la expresividad. «No cuesta expresividad» y «es
gratis» no son lo mismo, y la entrada anterior las confundía.

Y para fijar el gauge, si algún día hiciera falta, hay una vía **más barata y sin acoplar nada**:
normalizar `‖K_N‖`. Está medido que es casi un no-op —`‖K_N‖` cambia ±1 % mientras `‖∇R‖` crece
736×—, o sea que Adam **ya no recorre esa dirección plana**. No hay problema de gauge que resolver.

### Decisión

**No se atan.** Se mantiene lo que hay: los dos arrancan en el mismo valor —por procedencia, porque
`λ_init = 0.01` sale del `.pth` y no del paper— y después cada uno se aprende libre, como en el
código de los autores. La pregunta queda cerrada; la evidencia está acá para no re-abrirla.

---

## 2026-09-07 — Terminan los dos brazos de `λ_init`: el brazo que cambia `λ` no cambia nada, y el que no la cambia sí

Las doce corridas de `runs/lambdainit/` terminaron. Los `seed0` de los dos brazos —los que alcanzaron
a correr **antes** de que la edición en vuelo rompiera la bandera— son corridas completas y válidas,
así que cada brazo tiene **6 semillas**, no 5, y calza una a una con su control. Verificado que el
único cambio funcional respecto del `train.py` con el que corrieron los controles es `λ_init`: mismo
`t_init`, `referencia`, 4 000 iteraciones, `clip = inf`, sin checkpointing.

Medición congelada en [`docs/mediciones/lambdainit.txt`](mediciones/lambdainit.txt), script en
`scripts/veredicto_lambdainit.py`.

| predicción registrada | resultado | veredicto |
|---|---|---|
| 1. el conteo de `λ == 0` sigue al replay: A ~10/12, B 0/12 | A 5/6, B **0/6** | ✅ en el *cuál*, ❌ en el *cuánto* |
| 2. el hazard no cambia más de 1.5× | B **1.03×**, A **2.48×** | ✅ en B, ❌ en A |
| 3. la PSNR no cambia más de 0.033 dB | −0.001 a +0.043 dB en todos los cortes | ✅ |

### 1. El replay acierta *quién* toca el clamp y falla *cuánto*

El contrafactual de `scripts/replay_lambda_adam.py` predijo, para `λ_init = 0.0253`, que
`base-seed4` sería la única de las seis que no tocaría el clamp. **Acertó la semilla exacta:** en el
brazo A la única que no lo toca es `A-seed4`. Y para `λ_init = 0.1` predijo 0/12; medido, **0/6 en el
brazo B**, exacto.

Pero la magnitud está mal, y en la dirección que importa. El replay predijo que subir `λ_init` a
0.0253 bajaría el conteo de ~440 a ~167 iteraciones. Medido, la mediana del brazo A es **377 contra
445 del control**: prácticamente no bajó. Por semilla, contra su propio control, va **para los dos
lados** — seed1 sube 398 → 1 790, seed2 sube 453 → 763, seed3 baja 971 → 422, seed5 baja 437 → 332.

Es el límite del replay, y ahora está medido: congela la secuencia de gradientes, así que contesta
bien **si** una coordenada cruza un umbral, y no contesta **cuánto tiempo se queda ahí**, porque eso
ya depende de la trayectoria completa. Sigue valiendo para tamizar un `init` antes de gastar GPU;
no vale para predecir un conteo.

### 2. El hazard: el brazo B es un cero limpio, el brazo A se pasa del límite

| grupo | eventos | exposición | hazard/it | iteraciones de los eventos |
|---|---|---|---|---|
| A `λ_init = 0.0253` | 6/6 | 11 749 | 5.11·10⁻⁴ | 1 246, 1 729, 1 769, 2 007, 2 338, 2 660 |
| control A `λ_init = 0.01` | 4/6 | 19 445 | 2.06·10⁻⁴ | 1 726, 3 075, 3 215, 3 429 |
| B `λ_init = 0.1` | 4/6 | 14 000 | 2.86·10⁻⁴ | 289, 1 403, 1 958, 2 350 |
| control B `λ_init = 0.01` | 5/6 | 17 939 | 2.79·10⁻⁴ | 1 106, 2 349, 2 978, 3 619, 3 887 |

**Brazo B: 1.03×.** Log-rank exacto por permutación (las 924 particiones de 6+6) da p = 0.974. Es
el cero más limpio que produjo la fase 1.

**Brazo A: 2.48×**, y el pre-registro decía «no más de 1.5×». Las seis muertes caen antes de la
iteración 2 700 y toman 6 de los primeros 7 lugares del orden conjunto. Log-rank exacto **p =
0.0216**; el test de Poisson sobre los conteos, que ignora los tiempos, da p = 0.19 con IC95%
[0.59, 11.96]. La mortalidad binaria (6/6 contra 4/6, Fisher p = 0.45) no ve nada — exactamente lo
que anticipaba el cálculo de potencia, que daba poder **0** para ese endpoint. El endpoint de tiempos,
que era el registrado como principal, sí tenía poder, y el cálculo de potencia no lo cubría.

### 3. Y sin embargo el brazo A no tiene mecanismo, porque el brazo B lo desmiente

`λ/T` es el peso de fidelidad `w` de la energía. Si `λ_init` actuara por un mecanismo y no por la
trayectoria, tendría que verse ahí. A la iteración 1 000, ya pasado el bootstrap:

| grupo | `λ` | `T` | `w = λ/T` |
|---|---|---|---|
| A `λ_init = 0.0253` | 0.00064 ± 0.00040 | 0.02886 | **0.0220** |
| control A `λ_init = 0.01` | 0.00068 ± 0.00057 | 0.02838 | **0.0236** |
| B `λ_init = 0.1` | 0.05593 ± 0.00213 | 0.10407 | **0.5374** |
| control B `λ_init = 0.01` | 0.00034 ± 0.00048 | 0.10435 | **0.0033** |

**En el brazo A, `λ_init` está borrado a la iteración 1 000**: `λ` colapsa igual que en el control y
`w` queda en 0.022 contra 0.024. El efecto de subir el init dura ~300 iteraciones y después no queda
nada. **En el brazo B, `λ` es 163× la del control durante las 4 000 iteraciones enteras** — y el
hazard es 1.03× y la PSNR se mueve 0.02 dB.

O sea: el brazo donde `λ` **de verdad** cambia no cambia nada, y el brazo donde `λ` **no** termina
cambiando es el que muestra la señal. Cualquier mecanismo que pase por `λ` predice lo contrario. Lo
que queda es la vía que el propio pre-registro anticipó: `λ_init` perturba la trayectoria, la
trayectoria es caótica (un ULP borra un evento, medido), y con 6 semillas eso es lo que se ve. La
señal del brazo A **no es descartable, pero no es atribuible a `λ`**, y no se construye nada encima
sin replicarla.

### 4. La PSNR confirma la meseta, ahora en entrenamiento y no a pesos fijos

A iteración fija y sólo entre corridas vivas —la PSNR de una corrida muerta mide cuándo murió, no el
efecto de `λ_init`—: +0.012 y +0.043 dB en A, −0.007 / −0.001 / −0.019 / −0.022 dB en B. Todo dentro
de la cota de 0.033 dB medida a pesos fijos, salvo el +0.043 de A a la it 2 000, que compara n=3
contra n=5.

Y es una confirmación más fuerte que la de `docs/mediciones/lambda-sensibilidad.txt`: ahí `λ` se movía
sobre pesos congelados; acá el brazo B **entrenó las 4 000 iteraciones con `w` 163× más grande** y
las dos supervivientes dan 31.165 y 31.167 dB, contra 31.188 de la única superviviente del control.
La meseta de `λ` no es sólo del forward: también es del entrenamiento.

### 5. Un modo de fallo que no se había visto

`A-seed3` es la **única de las 24 corridas** que se va a NaN/inf (a la it 2 688, 681 iteraciones
después de su evento). Las otras 23 estallan y se quedan en un régimen numérico finito. Se anota;
con n=1 no es nada más que eso.

### Decisión

**Atar `λ_init = T_init` no compra nada y queda sin justificación empírica.** Lo que se sabe ahora:

- con `T_init = 0.1` el clamp desaparece del todo (0/6) y **eso vale 0.00 dB y 1.03× de hazard**;
- con `T_init = 0.0253` el clamp **ni siquiera desaparece** (5/6 lo siguen tocando, mediana 377
  contra 445) y el hazard sube 2.48× sin mecanismo que lo explique;
- el clamp nunca fue un modo de fallo, así que quitarlo nunca iba a valer nada — y ahora está medido
  que efectivamente no vale nada.

La única razón que quedaba para `λ_init = T_init` era la **procedencia** (0.01 sale del `config`
serializado del `.pth`, no del paper, igual que el `T_init = 0.001` letal), y ésa sigue en pie: el
0.01 no se puede defender. Pero el reemplazo tampoco se puede defender por resultado, y con
`T_init = 0.0253` —que es el default— cae justo en el brazo con la señal fea. **Queda como decisión
abierta para el autor**, con las dos opciones y su costo escritos en
`docs/pendientes.md`, eliminado el 2026-10-08.

Lo que **no** cambia: la cola de corridas. El efecto más grande medido de la fase 1 sigue siendo la
precisión (~15×), y `λ_init` juega en 1.0–2.5× sin mecanismo. El experimento 1 de `pendientes.md`
—`rapido` contra `intermedio`— sigue siendo el siguiente.

> **Corregido el mismo día, a la noche.** El experimento 1 se corrió y **el ~15× no existe**: la
> comparación limpia da 1.31×. La frase de arriba queda como el estado de conocimiento de esa tarde.

---

## 2026-09-07 — Decisión sobre los init, y PRE-REGISTRO de la tanda de precisión

### Los dos escalares arrancan en 0.1, desatados

Decisión del autor con el resultado de `λ_init` sobre la mesa. `train.py` queda con **dos constantes
independientes**, `T_INIT = 0.1` y `LAMBDA_INIT = 0.1`; `vnet_config` ya no deriva `λ` de `T`.

Fundamento de cada una, y lo que **no** lo es:

- **`T_INIT = 0.1`.** Es la corrida con la mejor PSNR del repo a 4 000 iteraciones (31.188 dB) y
  termina en 0.0984, o sea que apenas se mueve. **Honesto sobre el costo:** su hazard medido es
  2.79·10⁻⁴ contra 2.06·10⁻⁴ del `T_init = 0.0253`, un 1.35× peor que no es distinguible del ruido
  con las semillas que hay. Se elige por PSNR, no por hazard.
- **`LAMBDA_INIT = 0.1`.** Es el único valor medido que despeja la pared en 0 del todo (0 de 6
  corridas la tocan) y está medido inocuo: 1.03× de hazard y −0.02 dB contra su control, con `w`
  163× mayor durante las 4 000 iteraciones.
- **La regla 2-δ queda como piso, no como valor.** `0.1 > 2·δ_Adam = 0.0253` con margen, y
  `test_regla_2delta` ahora lo verifica sobre **los dos** escalares. El valor 0.0253 se descarta: es
  el límite `t → ∞`, y el brazo que arrancó los dos ahí murió 6 de 6 contra 4 de 6 del de 0.1.
- **No se atan.** Ya estaba descartado por el gradiente (`|∂L/∂T|` 24–99× mayor, signos opuestos en
  el 93 %) y por el `model.py` de los autores, que no tiene modo atado. Que los dos valgan 0.1 es
  una coincidencia de valor, no un acople: cambiar uno no mueve al otro, y el test lo fija.

### PRE-REGISTRO: `rapido` contra `intermedio`

Lanzado el 2026-09-07 03:20, `runs/precision/`, 12 semillas por brazo, 4 000 iteraciones. Escrito
acá **antes** de mirar ningún resultado, porque `runs/` está gitignoreado y no tiene sello temporal.

**Una sola variable:** la precisión de las convoluciones, 7 bits de mantisa (bf16) contra 10 (TF32).
Los dos modos llevan `cudnn.benchmark` encendido y determinismo apagado, así que no hace falta tocar
el código para aislarla. Es el primer experimento del repo **sin control archivado**: los dos brazos
corren ahora, con el mismo `train.py`.

**Primera tanda con `T_init = λ_init = 0.1`.** Los dos brazos lo comparten, así que no confunde la
comparación; pero el hazard **absoluto** no es comparable con las tandas viejas, que corrían con
`λ_init = 0.01`.

**Predicciones registradas:**

1. `hazard(intermedio) / hazard(rapido) ≥ 3` — bf16 es el modo bueno. Si sale ~1, la evidencia vieja
   era el confundido (checkpointing + `benchmark` + clip 35 000 a la vez) y la hipótesis de la
   precisión se cae entera.
2. La PSNR de bf16 a iteración fija, entre las vivas, **no** es peor que la de TF32 por más de
   0.05 dB. bf16 tiene 3 bits menos de mantisa; si además entrena peor, no sirve igual.
3. Ningún brazo repite bit a bit. Si alguno repite, `benchmark` no hace lo que creemos.

**Endpoint:** log-rank exacto sobre los tiempos de evento. La potencia de 0.89 de
`docs/pendientes.md` es sobre mortalidad y es una **cota inferior**.

**Cómo NO se leen los resultados:** `max‖g‖` no se compara entre brazos si `T` difiere — acá no
difiere, así que esta vez sí vale dentro de la tanda. Y ninguna muerte individual es atribuible:
con `benchmark` encendido ninguna de las 24 corridas se puede replicar.

---

## 2026-09-07 — Auditoría de la teoría y la implementación: qué no cerraba

Con la tanda de precisión en vuelo y sin tocar la GPU, se auditó el repositorio entero por cuatro
frentes independientes —la teoría, `ddr/`, `train.py` y los tests— y después **cada hallazgo pasó por
un revisor cuyo único trabajo era tumbarlo**. De 28 hallazgos, **13 cayeron** en la verificación y
15 quedaron en pie; varios de esos 15 eran el mismo defecto visto desde tres lentes distintas.

Lo que cayó, para no re-abrirlo: el bias de `Conv2d.backward` (no puede activarse), el redondeo del
binomial en bf16, `best.pth` elegido sobre CBSD68 (es el protocolo del paper), las columnas de
`traza.npy` bajo `--clip` (el clip está descartado), `config.json` con `Infinity`, y siete más.

### Lo que sí estaba mal, y ya está corregido

**1. La equivariancia de escala seguía viva en seis sitios.** Se retiró el 2026-09-06, pero aquella
entrada se tituló «se corrige el README» y sólo alcanzó al README y a `AGENTS.md`. Seguían
afirmándola `evaluate.py:8`, `tests/test_gradient.py:108` y cuatro lugares de
`docs/pipeline-explicado.txt` —incluido el glosario, la bibliografía, y la sección que **vende el
entrenamiento ciego al ruido**, que es la mejora candidata más valiosa para QSM—. El peor sitio era
`evaluate.py`, porque es el archivo donde vive el reescalado entre σ, o sea donde alguien va a
buscar su justificación. Corregidos todos, `evaluate.py` incluido (el parche 2 de más abajo).

**Lección de método:** una corrección se busca con `grep`, no con memoria. La entrada del 2026-09-06
enumeró dos sitios y había ocho.

**2. `docs/pipeline-explicado.txt` le atribuía a la norma acotada de `K₁` dos cosas que no se
siguen.** (a) Que elimina la degeneración `ρ ↔ 2ρ`: no la elimina —la degeneración exacta pasa por
`K_N`, que **no** lleva norma acotada, y escalar `K₁` ni siquiera escala `ρ` porque los macro-bloques
son no lineales; medido, `‖∇R‖` crece 736× con `K₁` proyectado en cada paso—. (b) Que acota la
constante del teorema de estabilidad: no alcanza, la cota del paper es un producto sobre **todas**
las convoluciones y en el checkpoint oficial las de los macro-bloques llegan a 3.79 y `K_N` a 1.05,
ninguna proyectada. La justificación correcta de la cota de `K₁` es **la procedencia**: es lo que
hacen los pesos publicados (32 normas por filtro entre 0.9920 y 1.0000) y el código de los autores.
Y no: acotar también los macro-bloques sigue descartado desde el 2026-09-02.

**3. Las notas del paper atribuían a Kobler et al. el `1/m` y la cota `‖·‖₂ ≤ 1`.** Ninguno de los
dos está en el paper: la sec. 2 escribe `r(x,θ) = Ψ(w N(Kx))` sin `1/m`, y de `K` sólo enuncia la
media cero. Los dos salen del **código** de los autores. Las dos son correctas —y el `1/m` la órbita
del gauge lo absorbe exactamente, así que no cambia nada— pero es la misma clase de confusión que
produjo el `T_init = 0.001` letal.

**4. `energy()` no devuelve un escalar por muestra**, devuelve la densidad `ρ` con forma `[B,1,H,W]`;
`R(x)` es su suma sobre píxeles. Estaba mal en `README.md` y en `pipeline-explicado.txt`, bien en el
docstring de `ddr/tdv.py`. Corregido, y el test pasa a llamarse por lo que verifica.

**5. La vida mediana no es `ln 2 / hazard`** — eso supone hazard constante, que este mismo
repositorio refuta seis líneas más abajo. Sobre las 24 corridas: la fórmula exponencial da 2 303 y
**Kaplan-Meier, que no supone nada, da 2 660**; un 13 % de subestimación. Corregido en
`docs/metricas.md`, `AGENTS.md` y `scripts/hazard_fase1.py`, que ahora calcula Kaplan-Meier y marca
explícitamente el bloque de extrapolación como retirado.

**6. Tres tests no verificaban lo que prometían.**
- `test_binomial_blur_is_normalized` sólo miraba que el kernel sumara 1 y su forma. Comprobado: una
  caja `ones(5,5)/25` pasaba ese test **y los ocho de adjunta**, porque `⟨Kx,y⟩ = ⟨x,Kᵀy⟩` es
  indiferente a los valores. La restricción 7 del README quedaba sin guardia. Ahora compara contra
  `[1,4,6,4,1]ᵀ[1,4,6,4,1]/256`.
- `test_constantes_publicadas` no cubría el batch **efectivo** (`ACUMULACION` es división entera:
  con `MICRO_BATCH = 12` el batch real pasa a 24 y el test seguía pasando), ni `CHANNELS`, ni el
  `gamma=0.5` del scheduler — «dividido a la mitad cada 25 000» son dos cosas y sólo se verificaba
  el cada-cuánto. Las tres agregadas.
- `test_grad_is_repeatable` decía ser la razón del refactor a cachés locales. No lo es: comprobado
  que pasa igual con la versión de estado mutable, porque cada `grad()` hace forward y backward de
  forma atómica. El docstring ahora dice qué prueba y qué no.

32 tests pasan (eran 31).

### Lo que está bien y quedó verificado

Las adjuntas de cada operador, el gradiente manual contra autograd, la arquitectura contra el
original, el `state_dict` contra los `.pth` oficiales, la separación train/val, que la pérdida se
evalúe sólo en `x_S`, que la acumulación de micro-lotes promedie bien (no hay factor 2 perdido), y
que la semilla fije las tres fuentes de azar.

### Los tres parches: escritos el 2026-09-07, aplicados en este commit

`train.py` y `evaluate.py` se cargan una vez por semilla, así que **no se tocan con una tanda en
vuelo**. Los tres parches se escribieron con `runs/precision/` corriendo y se aplicaron en una rama
aparte, sin tocar el checkout donde corría la tanda; el merge al checkout vivo espera a que termine
`intermedio`. Los tres, con lo que se verificó de cada uno:

1. **BUG LATENTE: `train.py:265` pasa `SIGMA` (la constante, 25.0) como `sigma_ref`, no `args.sigma`.**
   El modelo se entrena a `args.sigma` y se evalúa reescalando hacia 25, o sea **al revés** si
   alguien entrena con `--sigma ≠ 25`. Con el default 25 el factor es exactamente 1 y **ninguna
   corrida hecha hasta hoy está afectada**. Pero la fase 2 es literalmente entrenar variando σ.
   Parche: pasar `args.sigma` en los dos lugares, que a σ=25 es un no-op bit a bit.
2. `evaluate.py:8` — el docstring todavía dice «es válido porque la red es bias-free y por lo tanto
   escala-equivariante». Cambiar a: el reescalado es una aproximación de dominio, no una identidad.
3. `train.py:226-227` — el ruido hace GPU→CPU→GPU por micro-lote: `.to(device)`, `.cpu()`,
   `add_noise`, `.to(device)`. Son tres transferencias y una sincronización donde alcanza una.
   **Cuidado:** generar el ruido en GPU cambiaría la secuencia y rompería la comparabilidad con
   todas las corridas archivadas. La única versión segura es sumarlo en CPU **antes** de subir.

**Verificación de que los tres son no-ops numéricos.** El 1 lo es por construcción: `args.sigma`
tiene `default=SIGMA`, así que con el default los dos argumentos son el mismo objeto y la llamada es
idéntica. El 2 es un docstring. El 3 **se midió**, que es lo que importa: sobre `cuda:0` —la tarjeta
que no tenía entrenamiento encima— la versión vieja y la nueva dan `clean` y `noisy` **idénticos bit
a bit** (`max|dif| = 0`) en σ = 25, 15 y 50, sobre seis micro-lotes con el mismo `noise_gen`. La
razón: `add_noise` sortea con `torch.randn(..., device=y.device)` y `noise_gen` es un generador de
CPU, así que el sorteo sale igual sobre el tensor original que sobre el que volvió de la GPU —el
viaje de ida y vuelta de un float32 es exacto—, y la suma se hacía en CPU en las dos versiones.
Lo único que cambia es cuántas veces cruza el bus: tres transferencias y una sincronización pasan a
dos transferencias asíncronas.

### ENMIENDA AL PRE-REGISTRO de `runs/precision/` (escrita antes de leer ningún resultado)

**Corregida el mismo día, y hacia abajo.** La primera redacción de esta enmienda presentaba la
contención entre los dos brazos como una amenaza al experimento. **No lo es**, y este repositorio ya
tenía la evidencia:

- `cudnn.benchmark` elige entre algoritmos **matemáticamente equivalentes**. Lo único que cambia son
  los últimos bits. Está medido: es necesario y suficiente para que dos corridas idénticas dejen de
  repetir, y divergen en la iteración 1–2 (A/B de 300 iteraciones, 2026-09-04).
- Esa diferencia de bits es **caótica**, no direccional: un ULP borra un evento (medido). O sea que
  cambia **qué semilla** muere, no **cuántas**. Y lo que este experimento mide es la tasa sobre 12
  semillas, que es exactamente el promedio sobre ese azar.
- La hipótesis «el ruido de `cudnn.benchmark` desestabiliza» **ya está en `## Descartado`**: v5 corrió
  determinista y murió antes que v4 con `benchmark`.
- Y la regla de no medir con las GPUs ocupadas habla de **comparaciones numéricas** (dos trazas, dos
  gradientes) y de **tiempo**. Una tasa sobre 12 semillas no es ninguna de las dos.

Lo que sí corresponde anotar, y por eso la enmienda se queda:

`runs/precision/` es **la primera tanda multi-semilla del repo con `benchmark` encendido** —todas
las anteriores corren en `referencia`, que lo apaga— y los dos brazos corren en paralelo, uno por
tarjeta. La contención no es simétrica en el tiempo: medido en los logs en vuelo, `rapido` va a
~0.45 s/it contra los 0.28 que mide solo (1.6× de penalidad) y `intermedio` a ~0.75 contra 0.70
(1.1×). `rapido` termina ~4 h antes, así que las últimas ~4 semillas de `intermedio` cronometran sus
algoritmos con la máquina ociosa y las primeras ~8 no. La selección se cachea por forma de tensor y
es estable dentro de una corrida, así que la exposición son las primeras iteraciones de cada semilla.

**Decisión del autor: la tanda no se toca.** Se agrega al análisis un control **dentro** del brazo
`intermedio`, que cuesta cero GPU: comparar las semillas que corrieron con la otra tarjeta ocupada
contra las que corrieron solas. No porque se espere una diferencia —no se espera— sino porque es
gratis y deja demostrado lo que hoy es un argumento. El corte sale de los sellos de tiempo del
`plan.sh` contra el `=== rapido FIN ===`.


---

## 2026-09-07 (noche) — VEREDICTO de `runs/precision/`: la precisión no es la respuesta

La tanda cerró a las **22:18:30**, 12 semillas terminadas por brazo. Leída contra las tres
predicciones registradas antes de lanzarla. Salida completa en
[`mediciones/precision.txt`](mediciones/precision.txt), script en `scripts/veredicto_precision.py`.

| | `rapido` (bf16, 7 bits) | `intermedio` (TF32, 10 bits) |
|---|---|---|
| eventos | 5 | 6 |
| exposición | 39 507 it | 36 298 it |
| hazard | 1.266·10⁻⁴/it | 1.653·10⁻⁴/it |
| mortalidad a 4 000 | 5/12 | 6/12 |
| tiempos de evento | 1634, 2149, 2349, 2356, 3019 | 781, 1376, 1857, 2185, 2379, 3720 |
| h por semilla | 0.52–0.55 h | 0.85 h |

### Predicción 1 — FALLA. `hazard(intermedio)/hazard(rapido) ≥ 3`

**Medido: 1.31×**, IC95 % [0.33, 5.41], **log-rank exacto p = 0.61** sobre las 2 704 156
particiones. La mortalidad, que no es el endpoint, da 6/12 contra 5/12 (Fisher p = 1.0).

El pre-registro fijó por adelantado cómo leer este resultado: «si sale ~1, la evidencia vieja era el
confundido (checkpointing + `benchmark` + clip 35 000 a la vez) y la hipótesis de la precisión se
cae entera». 1.31× con p = 0.61 es ~1.

**Qué queda establecido y qué no.** El experimento tenía poder 0.70 contra el 3× registrado
(`AGENTS.md`, regla 5) y no lo encontró: eso es evidencia contra ≥3×, no una refutación. El IC95 %
todavía **contiene** el 3. Lo que el IC sí excluye, y con margen, es el **~15×**: el techo es 5.41.
Y 15× era el número que hacía a la precisión relevante — «el único efecto medido del orden de
magnitud que hace falta». Ese papel se termina acá.

**El supuesto que se rompió, y es el más informativo.** El cálculo de poder del `plan.sh` suponía
mortalidad **0.06 en bf16** y 0.62 en TF32. Medido: **0.42 y 0.50**. O sea que bf16 no es el modo
que casi no muere; muere casi igual que TF32. La evidencia vieja —«tres corridas bf16 llegaron sanas
a ≥5 000 iteraciones»— tiene, con la supervivencia medida de 7/12 a 4 000, una probabilidad de
**0.20** de ocurrir por azar. No hacía falta un mecanismo para explicarla.

**Lo que este experimento NO midió.** Compara 7 bits de mantisa contra 10. El `~15×` retirado
comparaba 7 contra los 23 de `referencia`, y esa comparación además **nunca fue limpia**:
`referencia` apaga `cudnn.benchmark` y las corridas viejas llevaban checkpointing y clip 35 000
encima. Que la precisión no haga nada entre 7 y 10 bits no prueba que no haga nada entre 10 y 23,
pero sí deja sin fundamento la afirmación de que la precisión es el efecto grande de la fase 1.

### Predicción 2 — se sostiene donde hay poder; una violación aislada donde no lo hay

Registrado: la PSNR de bf16 no es peor que la de TF32 por más de 0.05 dB, a iteración fija y entre
las vivas.

| iteración | bf16 | TF32 | diferencia |
|---|---|---|---|
| 1 000 | 30.655 (n=12) | 30.692 (n=11) | −0.037 dB |
| 2 000 | 30.967 (n=11) | 30.968 (n=9) | −0.001 dB |
| 3 000 | 31.006 (n=8) | 31.084 (n=7) | **−0.078 dB** |
| 4 000 | 31.113 (n=7) | 31.156 (n=6) | −0.044 dB |

El único cruce del margen es la iteración 3 000, y **sale de una sola corrida**: `rapido-seed2` pasa
por un bajón a 30.639 en esa evaluación y se recupera a 30.937. Sin ella la media queda en 31.058, o
sea −0.026 dB, adentro del margen. Con 8 contra 7 supervivientes y selección diferencial —comparar
las vivas de dos brazos que no murieron igual— eso es ruido, no una penalización de bf16.

**Análisis post-hoc, no registrado, que el pre-registro dejó pasar:** las semillas están
**apareadas**. `rapido-seedN` e `intermedio-seedN` comparten `--seed N`, o sea el mismo orden de
lotes y la misma inicialización; lo único que difiere es la aritmética. Sobre los pares donde las
dos siguen vivas: **−0.041 dB a la it 1 000** (n=11, Wilcoxon p = 0.90) y **+0.000 dB a la it 2 000**
(n=9, p = 0.50). A 3 000 y 4 000 sobreviven 4 y 3 pares, y ahí no hay nada que concluir. Donde hay
muestra, la diferencia entre 7 y 10 bits de mantisa sobre la PSNR es **cero medible**.

### Predicción 3 — CONFIRMADA

No se podía leer de la tanda: `plan.sh` corre 12 semillas distintas por brazo, así que no hay dos
corridas con la misma semilla que comparar. Se contestó con una réplica corta
(`scripts/replica_bit.sh`, ~2 min de GPU por brazo, corrida a las 22:20 con las tarjetas ya libres),
que re-corre la semilla 0 de cada brazo 100 iteraciones contra su traza archivada:

- `rapido` **diverge en la iteración 5**; a la 100, `∂L/∂T` difiere 31 % y `∂L/∂λ` 60 %.
- `intermedio` **diverge en la iteración 1**; a la 100, `∂L/∂T` difiere 45 % y `∂L/∂λ` 115 %.

Consistente con el A/B de 300 iteraciones del 2026-09-04. `cudnn.benchmark` hace lo que creemos, y
sigue valiendo que ninguna de las 24 corridas es replicable.

### Control de contención (enmienda al pre-registro) — sin diferencia

`rapido` terminó 18:28:57, así que las semillas 0–7 de `intermedio` cronometraron sus algoritmos con
la otra tarjeta ocupada y las 8–11 solas. Hazard 1.135·10⁻⁴ contra 3.037·10⁻⁴, razón **0.37×**,
log-rank exacto **p = 0.23** sobre las 495 particiones; PSNR pareada dentro de ±0.03 dB en las cuatro
evaluaciones. **No hay efecto detectable**, y el signo del punto estimado va al revés de la
preocupación —las que corrieron solas murieron antes—, que es lo que da un grupo de 4 semillas por
azar. La decisión de no tocar la tanda queda demostrada, que era exactamente el motivo de agregar
el control.

### El hazard sigue subiendo

| ventana | eventos | exposición | hazard/it |
|---|---|---|---|
| 0–1 000 | 1 | 23 781 | 4.21·10⁻⁵ |
| 1 000–2 000 | 3 | 21 867 | 1.37·10⁻⁴ |
| 2 000–3 000 | 5 | 16 418 | 3.05·10⁻⁴ |
| 3 000–4 000 | 2 | 13 739 | 1.46·10⁻⁴ |

Sube 7× entre la primera ventana y la tercera. Es la primera tanda con `T_init = λ_init = 0.1`, así
que el hazard **absoluto** no se compara con las tandas viejas (`λ_init = 0.01`); la forma, sí.

### Consecuencias

1. **Se retira «la precisión es el efecto más grande medido, y bf16 es el modo BUENO».** Corregido
   en `AGENTS.md`, `README.md`, `docs/metricas.md` y la tabla de Estado de este archivo, en el mismo
   commit. Las entradas históricas quedan marcadas, no borradas.
2. **La fase 1 se queda sin candidato del tamaño que necesita.** Todo lo medido en este repositorio
   —`T_init`, `λ_init`, el clipping, ahora la precisión— juega entre 0.9× y 2.5×, y lo que hace
   falta es ~15×. No es que falte medir mejor: es que ninguno de los levers probados tiene el tamaño.
3. **El experimento 3 de `pendientes.md` no se dispara.** Su condición era «sólo si el experimento 1
   da un efecto grande», y no lo dio. Pero la pregunta que queda abierta cambió de forma: ya no es
   «¿precisión o `benchmark`?» sino **«¿`referencia` es de verdad peor?»**, que nunca se midió
   limpio. Con 12 semillas por brazo el poder contra 3× es 0.70, así que es contestable si el efecto
   existe y es grande; contra 1.35× no lo es y no se lanza.
4. **Para el experimento 4 —la corrida larga— no hay «modo que gane».** Los dos son
   indistinguibles en hazard, así que el criterio pasa a ser el costo: `rapido` corre a 0.52–0.55 h
   por 4 000 iteraciones contra 0.85 h de `intermedio`, o sea **1.6× más exposición por hora de
   GPU** (más todavía sin contención). Se elige bf16 por throughput, no por seguridad — y esa es la
   única razón que queda para elegirlo.

---

## 2026-09-20 — La primera corrida de 10⁵ en `referencia`: el evento no mata, y tampoco suelta

`runs/referencia-100k`, lanzada el 2026-09-14 13:44 y cerrada el 2026-09-16 03:00. `referencia`,
`T_init = λ_init = 0.1`, σ=25, semilla 0, `clip = inf`, sin checkpointing, una GPU, batch 32 = 2×16.
**37.25 h, VRAM pico 9.94 GB.** Es la corrida que el punto 4 de `## Abierto` pedía desde el
2026-09-02 y la primera `referencia` más allá de 4 000 iteraciones que existe en el repositorio.

| | |
|---|---|
| mejor PSNR | **21.735 dB** (it 38 000) |
| PSNR final (it 100 000) | **−16.147 dB** |
| primer evento (`\|g\| > 10⁴`) | **it 1 958** |
| iteraciones con `\|g\| > 10⁴` | **97 340 / 100 000 (97.3 %)** |
| `max\|g\|` | **1.03·10¹⁸** (it 74 440) — el mayor del repositorio, previo 2.2·10¹⁶ |
| `T` | 0.1 → **0.02138**, mínimo en la última iteración, **nunca toca el clamp** |
| `λ` | 0.1 → **0.32579**, monótona, **nunca toca el clamp** |
| `corr` final | **78.6** — la red amplifica el ruido 79× |
| `nan` / `inf` en las 100 000 filas de `traza.npy` | **0** |

Contra los 31.478 dB de v4 y los 31.534 del oráculo: **9.8 dB corto**, y ni siquiera estable.

### Es `lambdainit/B-seed0` reproducida bit a bit y extendida

`traza.npy` de esta corrida y el de `runs/lambdainit/B-seed0` son **idénticos en las seis columnas
durante las 4 000 iteraciones**, entre dos lanzamientos separados por ocho días. Es la primera
reproducción bit-exacta de una corrida **completa** del repositorio: lo que había eran A/B de 300
iteraciones (2026-09-04) y la réplica de 100 (2026-09-07). `referencia` con `benchmark` apagado
repite de verdad, y lo hace entre procesos y entre días.

Consecuencia directa: todo lo anotado el 2026-09-06 sobre el brazo B —evento en la 1 958, `λ` que
sube en vez de bajar, `loss` estacionada en 3.8·10⁷ sin `NaN`— es de **esta** corrida. Lo que sigue
es cómo termina.

### La corrida no vuelve nunca, pero el PSNR sí — y después lo pierde

Después de la iteración 1 957 **no hay ni una sola racha de 200 iteraciones con `‖g‖ ≤ 10⁴`** en las
98 000 restantes. Y sin embargo:

| ventana | frac `\|g\|>10⁴` | mediana `\|g\|` | `T` med | `λ` med | `loss` med | PSNR |
|---|---|---|---|---|---|---|
| 0–10 k | 0.80 | 2.3·10⁷ | 0.0667 | 0.121 | 9.3·10⁵ | 9.1 → −14.5 |
| 10–20 k | 1.00 | 8.6·10⁶ | 0.0661 | 0.125 | 1.8·10⁵ | −13.6 → 14.3 |
| 20–30 k | 1.00 | 5.1·10⁵ | 0.0655 | 0.128 | 1 577 | 6.0 → 11.9 |
| **30–40 k** | **0.93** | **1.8·10⁵** | 0.0453 | 0.176 | 1 138 | 5.7 → **21.7** → −21.7 |
| 40–50 k | 1.00 | 3.6·10⁸ | 0.0396 | 0.189 | 6.6·10⁶ | −21.7 → −20.0 |
| 50–70 k | 1.00 | ~5·10⁸ | 0.0395 | 0.188 | ~7·10⁵ | −18.6 → −4.8 |
| 70–80 k | 1.00 | 5.7·10⁸ | 0.0426 | 0.186 | 1.0·10⁷ | −2.5 → −0.6 → **−24.2** |
| 80–100 k | 1.00 | ~9·10⁸ | 0.0407 | 0.205 | ~9·10⁶ | −24.2 → −16.1 |

El patrón es un **ciclo**: estallido, 20 000–30 000 iteraciones de recuperación lenta y monótona,
segundo estallido, y de vuelta. Se cumple tres veces (it 1 958, it ~38 000 con `max|g|` = 1.71·10¹⁷,
it ~74 000 con 1.03·10¹⁸) y la recuperación se va frenando con cada halving del `lr`.

**La recuperación del PSNR ocurre dentro del régimen estallado.** En la ventana 36 000–38 000, donde
el modelo marca sus 21.7 dB, el 73–82 % de las iteraciones siguen por encima de 10⁴ y la mediana de
`‖g‖` es 1.7·10⁴ — cien veces la de una corrida sana. O sea que **la calidad de la salida y la salud
del gradiente están desacopladas**: se puede denoiser a 21.7 dB con un gradiente de entrenamiento
cien veces demasiado grande. Es coherente con que el fallo viva en el backward.

Después del segundo estallido el estado se congela: entre las iteraciones 38 000 y 42 000 la `loss`
mediana vale 6.885, 6.871, 6.900 y 6.887 ·10⁶, `T` = 0.03963 y `λ` = 0.18851 **a cinco decimales en
las cuatro ventanas de mil**, con `‖g‖` mediana en 3.7·10⁸. No es el clamp —`T` nunca vale 0— ni el
desborde de `v` (sigue refutado). Es la **relación señal/ruido de Adam colapsada**, el mismo modo
caracterizado en `base-seed1` el 2026-09-04, sostenido ahora durante decenas de miles de iteraciones.

### La deriva de `T` es aritmética de Adam, no convergencia

Sobre las 98 042 iteraciones posteriores al evento, `∂L/∂T` es **positiva en el 86.8 %** de los pasos
—empuja `T` hacia abajo— pero su `|media|/rms` vale **0.0032**. Adam avanza `m̂/√v̂ ≈ 0.0032` de su
paso nominal. Con el schedule real (4, 2, 1 y 0.5 ·10⁻⁴ en los cuatro cuartos) la suma de `lr` sobre
la corrida vale 18.75, así que el desplazamiento predicho es `0.0032 · 18.75 = 0.060`. **Medido:
0.079.** Mismo orden, 25 % de error, sin ningún parámetro ajustado.

**Y de ahí sale la advertencia de esta entrada.** `T` termina en **0.02138**, a un 13 % de
`T* = 0.024727`. Leer eso como «`T` convergió al valor del paper» sería falso de punta a punta: es
una deriva de ruido sobre una dirección exactamente plana (la degeneración `(T, K_N) → (cT, K_N/c)`),
que se detuvo donde la detuvo el schedule del `lr`, en un modelo que **amplifica el ruido 79×**. El
punto 5 de `## Abierto` —`T` no es identificable por sí sola— queda confirmado ahora también de
forma dinámica y sobre 10⁵ iteraciones, y con el caso adversarial: el valor «correcto» de `T`
convive con el peor modelo del repositorio.

### Qué se puede y qué no se puede leer sobre el hazard

**No** se puede leer una tasa. Es una corrida, y la regla de la no-atribución (2026-09-06) vale igual
acá: esta trayectoria concreta no es propiedad de la configuración. Y después del primer evento la
exposición deja de ser comparable, porque el régimen es otro.

**Sí** queda establecido lo que faltaba: la extrapolación del hazard **no se puede seguir usando ni
para arriba ni para abajo**, porque el objeto que predecía —«la corrida muere»— no es lo que pasa.
Con 100 000 iteraciones corridas: cero `NaN`, `T` nunca en el clamp, `λ` nunca en el clamp, y el
modelo inservible de todas formas. **«Evento», «muerte» y «corrida perdida» son tres cosas
distintas**, y el conteo de mortalidad a 4 000 iteraciones mide la primera creyendo medir la tercera.

La consecuencia de diseño es incómoda: el endpoint de la fase 1 no puede ser «no muere», porque esta
corrida no murió. Tiene que ser **PSNR a iteración fija**, que es lo único que distingue los 31.19 dB
de una superviviente de 4 000 de los 21.7 dB de ésta.

### Y el número que ordena lo que sigue

`|∂L/∂T| / ‖g‖` vale **0.28 de mediana en el tramo sano** (las 1 955 iteraciones con `‖g‖ < 10³`) y
0.36 en las estalladas, contra 0.70–0.78 del control con `T ≈ 0.028`. Coincide con los 0.26–0.32 de
`tinit01` y confirma el escalado `|∂L/∂T| ~ 1/T` sobre una corrida entera. A `T_init = 0.1` el
estallido ya **no** está entero en `∂L/∂T`: se lo reparten los 388 k pesos. Lo que quiere decir que
el modo `T → 0` y el estallido del backward son dos cosas separables, y que ésta es una corrida del
segundo en estado puro, 98 000 iteraciones de él.

Medición: `runs/referencia-100k/{train.log, traza.npy, history.json}`.

---

## 2026-09-20 — HIPÓTESIS (pre-registro): el adjunto es un amplificador de 10⁶ al que la trayectoria está casi siempre ortogonal

Ninguna de las 41 hipótesis de `## Descartado` explica el conjunto, y la lista de levers medidos
—`T_init`, `λ_init`, clipping, precisión— juega entre 0.9× y 2.5×. Lo que sigue **reabre una
descartada** («la estructura residual amplifica la cadena adjunta»), con el argumento por el que su
refutación no la alcanza. Se escribe antes de medir nada.

### El argumento: los números de esta bitácora no cierran

`∂L/∂T = −(1/(S(1+λ/S)))·Σ_s⟨p_{s+1}, ∇R(x_s)⟩`. Los dos factores están **acotados por mediciones
que ya están en este archivo**:

- `‖p_S‖ = ‖∂L/∂x_S‖ = 2√(loss/B)`. Con la `loss` en rango normal es **O(1)**. En el estallido de
  la semilla 4 (it 2 717, `loss` = 31.38, B = 32): `‖p_S‖ ≈ 2.0`.
- `‖∇R‖` está fijado por el invariante de gauge medido, `T·‖∇R‖/‖n‖ ≈ 1.1–1.3`. Con `T` = 0.027124
  y `‖n‖` = 63.2: **`‖∇R‖ ≈ 2 600`**. Y no puede ser mucho mayor: si lo fuera, el paso explícito
  `τ∇R` movería la imagen mucho más que el ruido y el forward lo mostraría — no lo muestra
  (`max_growth` = 1.048 con `max|g|` = 2.9·10¹³, v7).
- **Si la cadena adjunta contrae** —`‖p_s‖ ≤ ‖p_S‖`, que es lo medido: 0.67–0.72 de punta a punta en
  los nueve checkpoints— entonces cada uno de los `S` términos está acotado por `‖p_S‖·‖∇R‖`.

```
cota:      |dL/dT|  <=  ||p_S|| * ||grad R||  ~=  2.0 * 2600  =  5.2e3
sano:      |dL/dT|  =   252.8                     (0.05 de la cota: p casi ortogonal a grad R)
estallido: |dL/dT|  =   3.84e6                    <- 750x POR ENCIMA DE LA COTA
```

Y sobre los eventos extremos la violación es de ocho a once órdenes de magnitud (`max|g|` = 2.2·10¹⁶,
1.03·10¹⁸). **Las tres premisas no pueden ser verdaderas a la vez.** Las dos primeras son mediciones
robustas de este repositorio. La tercera es la que hay que soltar.

### Por qué la medición existente no puede ver esto

`scripts/measure_adjoint_chain.py` mide `‖p_s‖/‖p_{s+1}‖` **sobre la dirección realizada**, o sea
sobre `p_S = 2(x_S − y)/B`: **un vector, en 25 947 dimensiones**. Eso no es la norma del operador
`Π_s J_sᵀ` con `J_s = (I − τ∇²R(x_s))/(1+λ/S)`. Un operador con `σ_max = 10⁶` da cociente realizado
0.7 siempre que la dirección realizada esté fuera del cono expansivo, que es el caso genérico cuando
ese cono es de dimensión baja. **La medición y la hipótesis son compatibles**, y la frase
«no hay **ningún** paso que amplifique» no se sigue de lo medido.

Y hay sitio de sobra en el espectro para el amplificador: `τ·L` vale **4.78 medido en el oráculo**
(`measure_lipschitz.py`), así que `I − τ∇²R` tiene autovalores de módulo hasta `1 + τL = 5.78` —el
script hace iteración de potencia sobre HVPs, que **devuelve el mayor autovalor en módulo y no su
signo**, y `∇²R` tiene autovalores negativos porque la Student-t es cóncava para `|a| > 1` y
`max|a|` medido vale 281–2 310—. Sobre los `S = 10` pasos:

```
5.78^10 = 4.2e7        con tau*|lambda| = 4.78  (el punto de operacion del oraculo)
1.95^10 = 750          con tau*|lambda| = 0.95  (lo que hace falta para el evento de la s4)
12.6^10 = 1e11         con tau*|lambda| = 11.6  (lo que hace falta para max|g| ~ 1e13-1e18)
```

**El rango dinámico del mecanismo es exactamente el rango dinámico de los eventos observados.**

### Y `τ·L` **sí** es invariante de gauge — es la única coordenada de riesgo que lo es

Bajo la degeneración exacta `(T, K_N) → (cT, K_N/c)`: `∇R → ∇R/c`, y por lo tanto `∇²R → ∇²R/c` y
`L → L/c`, mientras `τ = T/S → cτ`. **El producto `τ·L` queda igual.** Lo mismo la amplificación
`(1 + τ|λ_min|)^S`.

Eso importa más de lo que parece. Este repositorio lleva tres correcciones sucesivas por confundir
una coordenada con una cantidad física: `T*` no es un objetivo, `λ/T` no es invariante, `max‖g‖`
escala como `1/T`. **`τ·L` sobrevive a la prueba que las tres fallaron**, así que una hipótesis
escrita en `τ·L` es una afirmación sobre el modelo y no sobre la parametrización — que es
exactamente lo que le faltó a todo lo probado hasta hoy.

Y da un control inmediato y gratis sobre la objeción que ya está anotada en `pendientes.md` §5
(«`tinit01` tiene `‖∇R‖` 3.7× menor y hazard 1.35× mayor», o sea que la ganancia cruda no explica el
riesgo entre configuraciones): **medir `τ·L` en la superviviente de `tinit01` contra la del control.**
`T` difiere 4× entre las dos y `‖∇R‖` 3.7× en sentido contrario. Si `τ·L` sale parecido, la
discrepancia de `pendientes.md` §5 se disuelve —la ganancia cruda no era la coordenada— y el riesgo
queda escrito en la única cantidad que el gauge no mueve. Si sale distinto, hay que explicar por qué
dos configuraciones con el mismo hazard operan a curvaturas distintas.

### Qué explica, sin parámetros libres

| observación | lo que dice la hipótesis |
|---|---|
| el forward está sano y el backward estalla | el forward está acotado por `τ‖∇R‖ ≈ ‖n‖`; el adjunto multiplica por el **espectro** de `I − τ∇²R`, que incluye `1+τ\|λ\|` en las direcciones cóncavas |
| estalla en **una** iteración, con la `loss` normal | la alineación de `p_S` con el cono expansivo es una variable aleatoria por (estado, lote); no se acumula |
| un ULP 284 iteraciones antes borra el evento | la alineación depende del estado exacto y la trayectoria es caótica (6 %/it). Sólo la **tasa** es medible — la hipótesis predice eso, no lo contradice |
| 64 lotes sobre el estado pre-evento no ven nada | con hazard 2.4·10⁻⁴, `P(ningún evento en 64 lotes) = 0.985`. **El nulo de `measure_batch_tail.py` está sub-potenciado ~200×**: harían falta ~12 000 lotes. La conclusión «el estado tiene que moverse primero» **no se sigue de esa medición** |
| **la primera ventana de 1 000 casi no tiene eventos** | 0 eventos en 12 000 iteraciones de exposición (control + `tinit01`) y 4.2·10⁻⁵ en 23 781 (tanda de precisión, un evento @781), contra 3.05·10⁻⁴ en la tercera ventana. En inicialización aleatoria `L = 0.279` y `τL ≈ 0`: `I − τ∇²R ≈ I`, **no hay nada que pueda amplificar**. El amplificador tiene que construirse primero |
| el hazard sube como `t²` | `L` crece 7 000× durante el entrenamiento (0.279 → 1 943) mientras `T‖∇R‖` queda fijo, así que `τL` crece y el cono expansivo se abre. Es la «pista» anotada el 2026-09-06 —regresar el hazard contra `‖∇R‖` en vez de contra `t`— con un mecanismo detrás |
| la precisión no separa (1.31×, p = 0.61) | **predicho**: no es una cancelación catastrófica, es una propiedad espectral del objetivo exacto. El `~15×` viejo era el confundido, y la hipótesis no lo necesita |
| el clip no rescata | correcto por las tres razones ya anotadas, y la hipótesis agrega la cuarta: el problema no es la magnitud del gradiente sino la **dirección** en la que se amplifica |
| el «segundo motor» (régimen sostenido) | cuando el estado entra en una región de curvatura negativa grande el cono expansivo deja de ser raro y la amplificación pasa de cola a **típica**. Es la misma hipótesis en su fase saturada, y explica las 98 000 iteraciones de `referencia-100k` |
| `τ·L > 2` no separa vivas de muertas | correcto y esperado: `τL` es la **ganancia** del amplificador, no el disparador. El disparador es la alineación, que es azar |

### La identidad que explica el `1/T`, y que es gratis de verificar

Como el potencial es la identidad, `K_N` entra **linealmente y sólo una vez** (verificado en float64,
`degeneracion-T-KN.txt`), así que `L` depende de `(T, K_N)` sólo a través del producto. Derivando:

```
T · dL/dT  ==  <K_N, dL/dK_N>        (exacta, para todo estado y todo lote)
```

O sea que **`∂L/∂T` es exactamente `1/T` por la componente radial del gradiente de `K_N`**. Eso
convierte en teorema lo que la bitácora tenía como observación empírica («`|∂L/∂T| ~ 1/T`, así que
subir `T_init` 4× baja `‖g‖` mecánicamente») y explica por qué un solo escalar carga el 70–86 % de
la norma: es la única coordenada por la que pasa el modo de escala global de la red, dividido por
`T ≈ 0.025`. Es también una **ley de conservación exacta y gratis** para vigilar en `traza.npy`.

### Las tres mediciones que la deciden, y ninguna entrena

1. **`σ_max(Π_s J_sᵀ)` por iteración de potencia**, sobre los checkpoints que ya existen (oráculo,
   vivas, pre-evento). Misma maquinaria que `measure_lipschitz.py`, minutos de GPU.
   **Predicción registrada: `σ_max` entre 10³ y 10⁸ en todos, incluido el oráculo, mientras el
   cociente realizado `‖p_0‖/‖p_S‖` sigue dando 0.67–0.72.** Si `σ_max ≈ 0.7`, la hipótesis muere
   acá y no hay que hacer nada más.
2. **El signo del autovalor extremo de `∇²R`.** Iteración de potencia sobre `cI − ∇²R` para obtener
   `λ_min`, en los mismos checkpoints. **Predicción: es negativo, y `τ|λ_min|` crece con el
   entrenamiento.** Decide cuál es el lever: si el extremo es negativo se puede acotar la curvatura
   **por un solo lado** — y eso **no** está descartado, porque lo descartado (2026-09-02) es acotar
   `‖∇²R‖`, que sí excluiría el punto del oráculo (`L = 1 943`).
3. **`‖p_s‖` por paso, registrado en cada iteración**, en una corrida corta de `rapido` (0.5 h) hasta
   cazar un evento. **Predicción: en la iteración del estallido `‖p_s‖` crece geométricamente hacia
   atrás desde `p_S`, con razón por paso de 2 a 13, mientras la `loss` y `‖p_S‖` siguen normales.**
   Es la observación directa del mecanismo.

Las tres son falsables por separado y las dos primeras cuestan cero GPU de entrenamiento. Se corren
en ese orden: si la 1 falla, las otras dos no se hacen.

### Lo que la hipótesis NO explica, anotado antes de medir

- **No da un número para la tasa.** Predice que la tasa es `P(alineación)` y que crece con `τL`, pero
  no la calcula. La comparación cuantitativa con el hazard medido queda pendiente.
- **No explica la corrida B con `τL = 0.70` que estalló 400 iteraciones después.** A esa ganancia el
  techo de amplificación es `1.7¹⁰ = 200`, insuficiente para 10¹⁰. O `L` se movió muchísimo en esas
  400 iteraciones, o la medición local de `L` en un estado no representa el máximo sobre los diez
  estados visitados y sobre la cola de los lotes. **Es el punto más débil, y la medición 1 lo toca de
  frente** porque `σ_max` del operador compuesto es justamente lo que `L` local no captura.
- **No dice qué hicieron los autores.** Si el amplificador está también en el oráculo, entonces la
  receta publicada convive con él y la pregunta pasa a ser qué mantiene la trayectoria fuera del
  cono — no cómo eliminar el cono.

---

## 2026-09-20 — Las tres mediciones del amplificador adjunto: el mecanismo es el correcto, el estado no

Las tres mediciones que quedaron pre-registradas esta misma mañana en `pendientes.md`
§7 se corrieron el mismo día, en las dos GPUs. **La predicción central es falsa y la hipótesis
sobrevive**, porque lo que falla es la mitad que decía *dónde* está el amplificador, no la que decía
*qué* hace. Conviene leer esta entrada junto con la anterior: es su corrección.

### 7.1 — `σ_max` del operador adjunto compuesto

`scripts/measure_adjoint_sigma.py`. Iteración de potencia sobre `AᵀA` con `A = J_0ᵀJ_1ᵀ⋯J_{S-1}ᵀ`,
normalizando **por muestra** —el Hessiano es diagonal por bloques en la dimensión de batch, así que
cada parche converge a su propio `σ_max`—. Como `J_s = (I − τ∇²R(x_s))/(1+λ/S)` y `∇²R` es
simétrico, aplicar `J_s` y `J_sᵀ` es el mismo producto Hessiano-vector y sólo cambia el orden de la
cadena: la medición reusa la maquinaria de `measure_lipschitz.py`. Control: la columna `realizado`
del mismo script reproduce el 0.67–0.72 de `adjunta-por-paso.txt`, así que la cadena es la misma y
lo único que cambia es que ahora se mide la **norma del operador**.

```
checkpoint                  T        tau*L    sigma_max(A) min/med/max    realizado   prod ||J_s||
-------------------------------------------------------------------------------------------------
oraculo 31.53 dB         0.02473      4.87     1.60    1.81    2.08         0.621          166
base-s2 VIVA             0.02872      3.92     1.58    1.95    4.04         0.663          152
base-s3 VIVA             0.02764      5.50     1.49    1.81    3.18         0.659          269
base-s0 pre-evento       0.02692      4.57     1.72    1.91    2.34         0.686           86
base-s4 pre-evento       0.02678      5.46     1.54    1.91    3.12         0.678         97.6
base-s5 pre-evento       0.02589      5.41     1.58    1.96    5.11         0.673          707
tinit01-s0 VIVA          0.09843      3.82     1.77    2.09    3.23         0.674          149
tinit01-s1 pre-evento    0.10185      5.58     1.61    1.78    2.45         0.666         93.3
tinit01-s3 pre-evento    0.10277      9.87     1.59    1.87    61.1         0.697          973
-------------------------------------------------------------------------------------------------
ref100k best  21.735 dB  0.04405     25.69     34.3     176   2.32e3        4.186       5.82e9
ref100k last -16.147 dB  0.02138   65891.3   6.84e3  3.30e4   2.62e5     1741.918       6.51e14
```

**La predicción registrada era `σ_max` entre 10³ y 10⁸ *en todos, incluido el oráculo*. Es falsa.**
En los nueve checkpoints sanos `σ_max` vale **1.5–5.1**. La condición de muerte escrita —`σ_max ≈
0.7`— **no se cumplió**: `σ_max` es de 2 a 7 veces el cociente realizado, así que el punto
metodológico de la hipótesis (la dirección realizada `p_S` es un vector en 25 947 dimensiones y no
mide la norma del operador) **es correcto**. Lo que no es correcto es el tamaño.

**Y en los estados de `referencia-100k` sí aparece la banda predicha**: `2.32·10³` en el checkpoint
de 21.735 dB y `2.62·10⁵` en el de −16.147 dB. El amplificador existe y llega a cinco órdenes. No
es una propiedad de la arquitectura: **es algo que el entrenamiento construye.**

### El producto por pasos colapsa, y eso es lo que mantiene sano al oráculo

Los autovalores extremos por paso del oráculo, con `τ = 2.473·10⁻³`:

```
  s    lambda_min     lambda_max    tau*|lmin|   tau*|lmax|   ||J_s||
  0       -348.98         733.01       0.863        1.813      1.844
  1        -77.68         254.51       0.192        0.629      1.180
  2        -96.55         422.53       0.239        1.045      1.226
  3       -119.57         630.96       0.296        1.560      1.282
  4       -128.10         980.06       0.317        2.423      1.409
  5       -195.89        1795.16       0.484        4.439      3.403
  6       -343.85        1967.95       0.850        4.866      3.826
  7       -128.86        1061.27       0.319        2.624      1.607
  8        -90.95         867.14       0.225        2.144      1.212
  9        -49.35         959.78       0.122        2.373      1.359
                                             producto = 166.1     sigma_max(A) = 2.081
```

**Los diez pasos son expansivos** —`‖J_s‖` va de 1.18 a 3.83, ninguno contrae— y su producto vale
166. El operador compuesto mide **2.08**. Las direcciones expansivas de pasos consecutivos son casi
mutuamente ortogonales y el producto **colapsa por 80×**.

Eso cierra, por fin, por qué `τ·L` nunca funcionó como criterio y por qué el oráculo tolera
`τ·L = 4.87`: **`τ·L` acota el producto, no el operador.** La cota clásica `τ·L < 2` es una
condición sobre un paso; acá hay diez pasos cuyos conos expansivos rotan, y el objeto que entra en
el gradiente es la composición.

### 7.2 — El signo del autovalor extremo: confirmado

`λ_min(∇²R) < 0` **en los diez estados** del oráculo, entre −49.35 y −348.98. El Hessiano de la
energía es indefinido en todo el recorrido, que es lo que predice la Student-t: `φ''` cambia de
signo en `|a| = 1` y el `max|a|` medido vale 281–2 310. `τ|λ_min|` llega a 0.86, así que la rama
cóncava aporta expansión modesta; la grande la pone `τλ_max = 4.87`.

Queda abierto, como estaba escrito, si acotar **sólo la parte cóncava** excluye o no el punto del
oráculo. Lo que esta medición agrega es que la pregunta tiene sentido: la parte cóncava existe y es
grande en todos los estados visitados.

### 7.3 — La identidad, y cuál de los tres factores crece

`scripts/measure_evento_cota.py` y `scripts/measure_cota_lotes.py`. Se mide, término a término,

```
dL/dT  =  -(1/(S(1+sig))) * sum_s < p_{s+1}, grad_R(x_s) >
```

y los tres factores que la componen. **La identidad se verifica numéricamente**: reconstruida desde
la suma contra el valor de autograd, el error relativo es de `1·10⁻⁸` en 460 iteraciones seguidas.
No hay término olvidado.

Sobre **400 lotes distintos por checkpoint** (6 400 parches, contra los 64 lotes del nulo
sub-potenciado de `measure_batch_tail.py`):

```
checkpoint                |g| med    |g| MAX   |dT| med   |dT| MAX  cota med   cota MAX  |dT|/cota  amp MAX
-----------------------------------------------------------------------------------------------------------
oraculo 31.53 dB            30.33      94.18      23.30      91.45      4505      5472     0.0200     0.738
base-s2 VIVA                56.45      129.9      43.68      121.0      4603      5544     0.0264     0.744
base-s3 VIVA                31.71      99.28      20.14      88.34      4951      6544     0.0200     0.741
base-s0 pre-evento          117.8      202.3      109.1      188.2      4924      5902     0.0401     0.752
base-s4 pre-evento          39.89      142.7      26.00      129.2      5156      6837     0.0226     1.075
base-s5 pre-evento          93.69      197.1      81.91      183.5      4987      5938     0.0365     0.756
tinit01-s0 VIVA             27.28      107.8      5.976      24.04      1285      2927     0.0191     2.049
tinit01-s1 pre-evento       36.72      84.82      5.740      31.19      1398      1782     0.0212     0.915
tinit01-s3 pre-evento       69.22      138.2      29.87      57.52      1331      1701     0.0461     0.980
-----------------------------------------------------------------------------------------------------------
ref100k best 21.735 dB    2.25e+04   7.76e+05       4415   1.67e+05  6.18e+05  9.32e+06     0.0291      4476
ref100k last -16.147 dB   8.77e+07   6.92e+08   8.12e+07   1.11e+08  4.90e+08  7.35e+10     0.5023    147209
```

**En los nueve estados sanos, en 3 600 lotes, la amplificación máxima es 0.74–2.05.** La cadena
contrae y no tiene cola. El amplificador no está latente esperando una dirección: no está.

### La corrección al argumento de la «violación de 750×»

Hay que decirlo con todas las letras: **la cota no se puede violar.** Es Cauchy-Schwarz término a
término sobre una identidad exacta. Lo que se hizo esta mañana fue comparar un `|∂L/∂T|` medido
**en** un evento contra una cota calculada en un checkpoint de hasta 400 iteraciones **antes**.
Medidos en el mismo estado, `|∂L/∂T|/cota` no pasa de **0.50** en ninguno de los 4 400 lotes.

El argumento servía igual, y por la razón que decía: de las tres premisas, la que cae es **«la
cadena adjunta contrae»**. En `referencia-100k` la cadena amplifica por `1.5·10⁵`. Lo que la
hipótesis no podía saber es que eso **no ocurre en los estados sanos**, y que por lo tanto el
amplificador no es la explicación del *primer* evento sino la descripción de lo que pasa *después*.

El número queda cerrado del otro lado: el estallido de la semilla 4 (it 2 717, `loss` 31.38,
`∂L/∂T = −3.84·10⁶`) exige, con `‖p_S‖ = 2√(31.38/32) = 1.981` y `‖∇R‖` en rango normal, una
amplificación de **≈ 6.9·10²**. La distribución medida en `ref100k best` —un estado cuyo `|g|`
mediano es cien veces menor que el del evento— llega a **4 476**. El número que hacía falta está
dentro de lo medido.

### Los dos regímenes tienen mecanismos distintos, y se separan por correlación

Dentro de cada checkpoint, sobre sus 400 lotes:

```
checkpoint              r(log|g|, log cos)   r(log|g|, log amp)
------------------------------------------------------------------
oraculo 31.53 dB                 0.891                0.326
base-s2 VIVA                     0.903               -0.331
base-s3 VIVA                     0.733               -0.024
base-s0 pre-evento               0.946                0.173
base-s4 pre-evento               0.840               -0.121
base-s5 pre-evento               0.939                0.369
tinit01-s0 VIVA                  0.436                0.334
tinit01-s1 pre-evento            0.404               -0.015
tinit01-s3 pre-evento            0.676                0.142
------------------------------------------------------------------
ref100k best 21.735 dB           0.049                0.516
ref100k last -16.147 dB          0.159                0.705
```

**En régimen sano el gradiente lo mueve la alineación; en régimen estallado lo mueve la
amplificación.** La correlación se da vuelta entera. Y el tercer factor no participa: `‖∇R(x_0)‖`
vale 2 011–15 700 en los nueve sanos y **6 202 en `ref100k best`** —normal, comparable al 8 128 del
oráculo— mientras su `|g|` mediano es 2.2·10⁴. Ahí no hay una energía más inclinada; hay
amplificación pura. Recién en `ref100k last` sube a `1.11·10⁶`.

### La trayectoria: en entrenamiento sano la amplificación está congelada

`scripts/measure_traza_adjunta.py`, 2 600 iteraciones en `referencia` desde inicialización
aleatoria, registrando `‖p_s‖` por paso en **cada** iteración. Por bloques de 200:

```
    1- 200  amp 0.7143 +- 0.1067      1401-1600  amp 0.6971 +- 0.0164
  201- 400  amp 0.7273 +- 0.0707      1601-1800  amp 0.6989 +- 0.0262
  401- 600  amp 0.7129 +- 0.1088      1801-2000  amp 0.7367 +- 0.4582
  601- 800  amp 0.7039 +- 0.1268      2001-2200  amp 0.6984 +- 0.0233
  801-1000  amp 0.6950 +- 0.0245      2201-2400  amp 0.7033 +- 0.0754
 1001-1200  amp 0.7040 +- 0.0810      2401-2600  amp 0.6989 +- 0.0178
 1201-1400  amp 0.6964 +- 0.0229

  r(log|g|, log cos)  = +0.749        max |dT|/cota = 0.469
  r(log|g|, log cota) = +0.474        max |g| = 870.2, cero eventos en 2 600
  r(log|g|, log amp)  = +0.070
```

`amp` no se mueve de 0.70 en 2 600 iteraciones y la variación de `|g|` la explica el coseno. El
perfil `‖p_s‖/‖p_S‖` es suave y monótono hacia atrás: `1.000 0.849 0.782 0.748 0.734 0.720 0.699
0.685 0.673 0.676 0.701`.

**La corrida no reprodujo el evento de la it 1 958 de `referencia-100k`.** Era el riesgo anotado:
el `retain_grad` mueve un ULP y está medido que un ULP borra un evento. Lo mismo pasó con el replay
de `base-seed0` entre las 3 001 y las 3 460, que corrió las 460 iteraciones con `max|g| = 172`. Así
que **no hay precursor medido**: es el hueco que queda abierto.

### El barrido: `σ_max` es el mejor discriminador que tiene el repositorio

`measure_adjoint_sigma.py --scan --quick` sobre los **156 checkpoints** del repositorio, las dos
tarjetas, ~75 min. Es una pasada gruesa —25 iteraciones de potencia, lote de 4— y los órdenes de
magnitud coinciden con la fina, pero los extremos se subestiman: sirve para tener `n`, no para
citar valores. Se usa sólo el `last.pth`, que es el desenlace. Quedan **76 corridas etiquetadas
con PSNR: 38 vivas (≥30 dB) y 38 muertas**, 11 de ellas por el modo `T → 0`.

Ese modo se separa aparte porque es otra muerte, ya caracterizada, y ahí el diagnóstico es `T`
misma: con `τ = 0` el paso es la identidad y `σ_max ≈ 1` **por construcción**, no por sanidad.

```
                          AUC tau*L    AUC sigma_max
  todas          (n=76)      0.701         0.780
  sin `T -> 0`   (n=65)      0.912         0.968

  al mejor umbral de cada una, sobre las 65 sin `T -> 0`:
    tau*L     > 13.10   ->  61/65,  0 falsos positivos,  4 muertas no detectadas
    sigma_max > 16.46   ->  64/65,  0 falsos positivos,  1 muerta no detectada
```

**Ninguna de las 38 corridas vivas tiene `σ_max > 16.5.**` Las 38 vivas se agrupan en 1.79–16.46 y
las muertas llegan a `6.97·10¹⁰`. La única que se escapa es `lambdainit/B-seed5` (21.91 dB,
`τ·L = 3.52`, `σ_max = 2.14`): sana por las dos medidas y terminó a 21.91 dB.

`r(log τ·L, log σ_max) = +0.580` sobre los 72 puntos con `τ·L > 0`: correlacionados, pero lejos de
determinarse. Es lo que cabía esperar de que `τ·L` acote el producto y `σ_max` mida el operador.

**Advertencia, y no es menor: esto es un diagnóstico, no una predicción.** Se mide el estado final,
así que para las destruidas la separación es en parte tautológica. El caso que sí informa es
`ablacion/seed0` (17.39 dB): `τ·L = 4.40`, **indistinguible del 4.87 del oráculo**, y
`σ_max = 63.67`. Ahí `τ·L` dice sana, `σ_max` dice muerta, y el PSNR le da la razón a `σ_max`. La
versión predictiva —`σ_max` en un checkpoint *pre-evento*— no está medida.

### Qué queda en pie, y qué cambia

| | |
|---|---|
| Confirmado | `σ_max(A)` es un parámetro de orden real: **1.5–5.1 sano, 2.3·10³ a 21.7 dB, 2.6·10⁵ a −16.1 dB** |
| Confirmado | la dirección realizada **no** mide la norma del operador: hay un factor 2–7× en los sanos y 150× en `ref100k best` |
| Confirmado | `λ_min(∇²R) < 0` en los diez pasos |
| Confirmado | la identidad de `∂L/∂T`, a `10⁻⁸` |
| **Refutado** | que el amplificador esté ya en los estados sanos. No está: 3 600 lotes, amp máxima 2.05 |
| **Retirado** | el «750× por encima de la cota». Era comparar dos estados distintos; medido en el mismo estado el máximo es 0.50 |
| Nuevo | los diez pasos son **todos** expansivos y el producto colapsa por 80×. `τ·L` acota el producto, no el operador |
| Nuevo | dos regímenes con mecanismos opuestos: sano gobernado por la alineación (`r` = 0.40–0.95), estallado por la amplificación (`r` = 0.52–0.71) |
| Nuevo | **`σ_max` separa vivas de muertas**: AUC 0.968 contra 0.912 de `τ·L` sobre 65 corridas, y **ninguna de las 38 vivas pasa de 16.5**. Es el primero que pasa la prueba que fallaron `T*`, `λ/T` y `max‖g‖` — pero es diagnóstico sobre el estado final, no predicción |
| Abierto | **qué dispara la transición.** Sin precursor medido: 2 600 iteraciones sanas con `amp` clavada en 0.70 |
| Abierto | la corrida B con `τ·L = 0.70` sigue sin explicación |

La consecuencia práctica para la fase 1 es que **el objetivo a vigilar deja de ser `max‖g‖` y pasa a
ser `σ_max(A)`**, que cuesta minutos de GPU sobre un checkpoint y no depende de la parametrización
—mientras `T*`, `λ/T` y `max‖g‖` fallaron esa prueba—. Y la consecuencia teórica es que el lever
que la fase 1 necesita, si existe, está en el **backward**: la red sana ya opera con diez pasos
expansivos y lo único que la salva es que sus conos no se alineen.

---

## 2026-09-20 — Las dos tarjetas escalan 2× sin tocar nada, y la afinidad de núcleos no aporta

Medido porque la cola de experimentos se duplicó y valía la pena saber si dos corridas simultáneas
se estorban. `scripts/bench_dos_gpu.sh`, 250 iteraciones de `referencia` cronometradas de punta a
punta —arranque de `torch.compile` y evaluación final incluidos, que son idénticos en las cinco
configuraciones—.

**Esto NO reabre DDP.** Lo que se mide acá son **dos corridas independientes, una por tarjeta**, no
una corrida repartida. DDP sigue retirado por las razones del 2026-09-04, que no eran de
rendimiento.

```
config                                          GPU0        GPU1      total
-----------------------------------------------------------------------------------
A  una sola GPU (linea base)                  366.1 s         --    0.683 it/s
B  dos GPUs, sin afinidad          pasada 1   427.7 s    366.8 s    1.267 it/s  1.85x
                                   pasada 2   370.2 s    367.2 s    1.356 it/s  1.99x
C  dos GPUs, taskset 0-7,16-23 /   pasada 1   370.1 s    367.0 s    1.357 it/s  1.99x
                 8-15,24-31        pasada 2   370.3 s    367.1 s    1.356 it/s  1.99x
D  como C, 8 workers por proceso              370.3 s    366.7 s    1.357 it/s  1.99x
E  como C, 4 workers por proceso              370.0 s    366.9 s    1.357 it/s  1.99x
```

**Las dos tarjetas ya escalan 1.99× sin cambiar una línea.** Basta `CUDA_VISIBLE_DEVICES=0` y
`CUDA_VISIBLE_DEVICES=1` en dos terminales. Todas las celdas de dos GPUs caen entre 366.7 y 370.3 s,
o sea dentro del 1 % de la corrida sola.

**La afinidad de núcleos no aporta nada medible**, y el número de workers tampoco: 16, 8 y 4 dan el
mismo tiempo dentro del 1 %. Era previsible una vez mirado el código: `PatchDataset2D` carga las 400
imágenes de BSDS en RAM en `__init__` (~740 MB, compartidos por *copy-on-write* con los workers), y
`__getitem__` sólo recorta 93×93, hace flips y una rotación. Con la GPU al 95 % y la CPU al 93 %
ociosa, el pipeline de datos no es el cuello de botella; **no hay nada que recuperar.**

El único dato en contra fue el `427.7 s` de la primera pasada de B, un 16 % peor. **No se
reprodujo**: repetida la misma configuración dio 370.2 s. Queda registrado como transitorio, no
como efecto. Un solo valor atípico en catorce celdas.

La diferencia sistemática que sí aparece es `GPU0` contra `GPU1`: 370.2 contra 367.1 s, **0.9 %**,
constante en las seis celdas. La GPU 0 maneja el display (278 MiB residentes contra 15 MiB). Es real
y es despreciable.

**Qué hacer con esto, en una línea:** lanzar las tandas como dos corridas independientes, una por
tarjeta, y no gastar esfuerzo en repartir núcleos. Las fases D y E del script necesitan un
`TDV_WORKERS` en `train.py` que **no se dejó aplicado**, justamente porque midió cero; la línea
exacta está en el encabezado de `bench_dos_gpu.sh` por si alguna vez hace falta re-correrlas.

---

## 2026-09-20 — La transición no tiene precursor: `amp` salta de 0.70 a 10² en una iteración, con la `loss` normal

El punto 8, registrado esta misma tarde, se corrió esta misma noche. Dos corridas de 6 000
iteraciones en `referencia`, semillas 7 y 8, **una por tarjeta** — 134 min las dos juntas, que es lo
que habría tardado una sola, y es la primera vez que el 1.99× medido se usa en producción.

**Las dos cazaron un evento.** `seed7` en la iteración 2 699, `seed8` en la 4 721.

### El salto

```
seed7
   2692  loss  22.453  |g|    43.42  amp   0.7113  |cos| 0.0383
   2693  loss  23.217  |g|    71.71  amp   0.7015  |cos| 0.0222   <- sana
   2694  loss  25.237  |g|     4592  amp    217.1  |cos| 0.0596   <- SALTO, loss normal
   2695  loss  36.879  |g|    785.3  amp   0.9992  |cos| 0.3851   <- vuelve
   2696  loss  42.167  |g|     1816  amp    19.62
   2697  loss 223.701  |g|     4127  amp    1.483                 <- vuelve otra vez
   2698  loss 126.542  |g|     7294  amp    17.43
   2699  loss  54.203  |g| 1.015e04  amp    310.5   <- primer |g| > 1e4; ya no vuelve
   2700  loss  97.650  |g| 1.147e05  amp    514.6
   2703  loss 2751.35  |g| 2.599e06  amp     4790

seed8
   4718  loss  22.999  |g|    60.69  amp   0.7180                 <- sana
   4719  loss  22.353  |g|    298.6  amp     10.9   <- SALTO, loss normal
   4720  loss  27.891  |g|     1460  amp    21.13
   4721  loss  56.528  |g| 4.422e05  amp     8513   <- primer |g| > 1e4
   4722  loss  52.436  |g|     2005  amp    74.93
   4725  loss 372.726  |g|     4273  amp    1.046                 <- vuelve
   4726  loss 292.125  |g|     3541  amp   0.5909                 <- vuelve DEL TODO
   ...   y muere en NaN en la iteracion 4 792
```

**No hay precursor en la trayectoria realizada.** En las 2 693 iteraciones previas de `seed7` y las
4 718 de `seed8`, más las 2 600 de la corrida de control de la tarde, o sea **10 011 iteraciones
sanas**:

```
                      amp mediana   p99    max    >2      >5    >20   >100
  seed7 (1..2693)        0.6966    1.077   9.70    6       1      0     0
  seed8 (1..4718)        0.6979    0.896  67.42   10       4      3     0
  control (1..2600)      0.6963    0.910   6.97    4       1      0     0
```

**Cero iteraciones por encima de `amp` = 100 en las 10 011.** Y la transición cruza 10² y 10³ en dos
iteraciones. No es una cola pesada que crece: es una fase estable con excursiones chicas, más una
fuga.

### Sí hay precursor en la COLA SOBRE LOTES, y dura entre 4 y 9 iteraciones

Las dos corridas se relanzaron con checkpoint **cada 10 iteraciones** alrededor del salto
(`seed7-fino`, `seed8-fino`). Reprodujeron bit a bit —la it 2 600 imprime `loss = 21.920`,
`|g| = 95.56`, `amp = 0.7043`, idéntica— así que los checkpoints caen sobre la misma trayectoria.
Sobre cada uno se corrieron **400 lotes distintos**:

```
  seed7   faltan   amp max   lote   amp p99   amp mediana   #amp>2  #amp>10
    2500     194     0.985     78     0.730      0.6855        0       0
    2510     184     4.108    374     0.740      0.6871        2       0
    2550     144     8.921    139     0.785      0.7064        2       0
    2580     114     2.319     70     0.753      0.6976        2       0
    2600      94     3.708    139     0.734      0.7015        1       0
    2620      74     3.331    139     0.756      0.7164        1       0
    2640      54     1.023     78     0.735      0.6944        0       0
    2650      44    13.315    374     0.734      0.6875        1       1
    2660      34     9.296    374     0.784      0.6963        1       0
    2670      24     1.120    166     0.747      0.7047        0       0
    2680      14     2.615    139     0.775      0.6856        1       0
    2690       4   203.725    374     1.046      0.7104        2       1

  seed8   faltan   amp max   lote   amp p99   amp mediana   #amp>2  #amp>10
    4550     169     0.761    361     0.733      0.6983        0       0
    4600     119     1.100     12     0.747      0.7128        0       0
    4650      69     3.235    275     0.734      0.6913        1       0
    4680      39     0.825     41     0.730      0.6923        0       0
    4700      19     0.832    397     0.732      0.6931        0       0
    4710       9   120.874    235    21.257      0.7360       48      11
```

**La mediana no se mueve nunca** —0.685 a 0.736 en las dieciocho mediciones— y el `p99` tampoco,
salvo en la última de `seed8`. Lo que se mueve es **el máximo sobre 400 lotes**, y la lectura
honesta es que **fluctúa entre 0.76 y 13.3 durante las ~200 iteraciones previas y sólo alcanza 10²
en las últimas 4 a 9**. No es una subida suave: en `seed7` vale 1.12 a 24 iteraciones del salto y
2.62 a 14.

**Pero el estado de la última medición antes del evento es inconfundible.** `seed7` a 4 iteraciones
tiene un lote en 203.7 con `‖g‖ = 1.19·10⁴`, o sea **un gradiente de magnitud de evento que la
trayectoria no llegó a sortear**. `seed8` a 9 iteraciones es más claro todavía: no es un lote, es la
cola entera —**48 de 400 por encima de 2, 11 por encima de 10, `p99` = 21.3**— contra 0–2 y 0–1 en
todos los demás estados medidos.

Eso corrige la lectura de `σ_max`: medido sobre un **lote fijo** vale 1.90 en la it 2 690, cuatro
iteraciones antes del salto. Pero `σ_max` **depende del lote**, porque los estados `x_s` dependen de
la entrada, y sobre el lote 374 ese mismo estado da 203.7. La magnitud que sirve no es `σ_max` en un
lote, es **su cola sobre lotes**.

Y hay un detalle que vale la pena mirar: en `seed7` el lote 374 es el primero o el segundo de los
400 en siete de las doce mediciones, y su `amp` va 0.80 → 4.11 → 0.71 → ... → 13.3 → 9.30 → 0.81 →
1.14 → **203.7**. Su `loss` vale 17.9, **más baja que la mediana de 24**: no es un lote difícil, es
uno particular. Pero cae al puesto 100–142 en cuatro de las doce, así que no es identificable de
antemano con esta resolución.

### La firma es exactamente la predicción registrada en 7.3

`‖p_s‖/‖p_S‖`, de `s = 0` a `s = S`:

```
seed7  2693 (sana)    0.7015 0.6538 0.661  0.6701 0.6906 0.7239 0.7268 0.747  0.7844 0.8509 1.000
seed7  2694 (salto)   217.1  173.1  121.6  68.68  31.88  5.071  1.674  1.061  0.7964 0.8603 1.000
seed8  4718 (sana)    0.7180 0.7047 0.7065 0.7034 0.7119 0.7236 0.7403 0.7663 0.8152 0.8684 1.000
seed8  4721 (evento)  8513   9506   5534   798.7  299.2  168.9  42.56  11.9   2.741  1.428  1.000
```

La predicción escrita en `pendientes.md` §7.3 decía: «en la iteración del estallido `‖p_s‖` crece
geométricamente **hacia atrás** desde `p_S`, con razón por paso entre 2 y 13, mientras la `loss` y
`‖p_S‖` siguen en rango normal». **Se cumple en las dos semillas, y los números caen dentro del
rango predicho:** las razones por paso van de 1.8 a 6.9, la `loss` en la iteración del salto vale
25.237 y 22.353 (mediana previa 25.5 y 22.5) y `‖p_S‖` vale 1.776 y 1.672.

Y los otros dos factores **no se mueven**. En `seed8`, entre la 4 718 y la 4 719, `‖∇R(x_0)‖` pasa de
**1 555 a 1 560** y `‖p_S‖` de 1.696 a 1.672. La energía no se inclinó, la pérdida no subió, la
salida es sana. **Lo único que cambió es el operador adjunto.**

En `seed7` el crecimiento arranca entre los pasos 4 y 6 —razones 6.3 y 3.0—, que son exactamente los
pasos donde el oráculo tiene sus `‖J_s‖` máximos (3.40 en `s = 5`, 3.83 en `s = 6`). En `seed8`
arranca cerca del final de la cadena y después se propaga plano. O sea que el paso que dispara
varía, pero la estructura no.

### El parpadeo, que es la respuesta a «a veces explota y a veces no»

Las dos corridas **alternan** entre los dos regímenes durante unas pocas iteraciones antes de
decidirse. `seed7` vuelve a `amp = 0.999` en la 2 695 y a 1.483 en la 2 697, y recién en la 2 699 se
engancha. `seed8` vuelve a 1.046 en la 4 725 y a **0.5909** en la 4 726 —o sea contrayendo de nuevo,
más que de costumbre— y sobrevive 66 iteraciones más antes de irse a `NaN` en la 4 792.

Y en la corrida de control, una excursión a `amp = 6.97` en la iteración 1 963 se apagó sola y la
corrida llegó sana a las 2 600. `seed8` tuvo una a **67.42** que también se apagó.

**El estado amplificado es alcanzable desde el sano en un paso, y es metaestable.** Que una corrida
muera o no depende de si la fuga se engancha en las dos o tres iteraciones siguientes. Eso es lo que
estaba detrás de la no-atribución por un ULP, y le da un mecanismo: un ULP no cambia la tasa de
fugas, cambia el resultado de un volado que se juega en tres iteraciones.

### Los dos modos de muerte, en la misma tanda

`seed7` se enganchó y corrió las 3 300 iteraciones restantes en régimen estallado **sin un solo
`NaN`**, con `max‖g‖ = 1.21·10¹⁶` — que es el patrón de `referencia-100k`. `seed8` se soltó, volvió a
contraer, y murió en `NaN` 71 iteraciones después. Dos desenlaces distintos del mismo evento, en la
misma tanda, con la única diferencia de la semilla.

### Qué queda

| | |
|---|---|
| Contestado | **no hay precursor en la trayectoria realizada.** Una iteración, con la `loss` normal. Ni `amp` ni `σ_max` sobre un lote fijo se mueven antes |
| Contestado | **sí lo hay en la cola sobre lotes, con horizonte de 4 a 9 iteraciones.** El máximo sobre 400 lotes llega a 203.7 y 120.9 en la última medición antes de cada evento, contra 0.76–13.3 en las ~200 iteraciones previas. La mediana nunca se mueve |
| Contestado | la firma predicha en 7.3 se cumple en las dos semillas, con las razones por paso dentro del rango escrito |
| Contestado | **sólo cambia el operador adjunto**: `‖∇R‖`, `loss` y `‖p_S‖` quedan donde estaban |
| Nuevo | el estado amplificado es **metaestable**: las dos corridas vuelven al régimen sano una o dos veces antes de decidirse, y excursiones a `amp` = 7 y 67 se apagaron solas |
| Nuevo | `amp` **nunca pasa de 100 en 10 011 iteraciones sanas**, y la transición cruza 10³ en dos |
| Abierto | **qué hace que un lote sea el que dispara.** En `seed7` el 374 es el primero o segundo de 400 en siete de doce mediciones, con `loss` 17.9 —más baja que la mediana de 24— pero cae al puesto 100–142 en cuatro. No es identificable de antemano con esta resolución |
| Abierto | por qué `seed7` se engancha y `seed8` se suelta |
| Contestado esa misma noche | **el horizonte corto no es un artefacto de muestreo.** Con 4 000 lotes en vez de 400, los cuatro estados medidos entre 19 y 69 iteraciones antes del salto llegan a `amp` 3.6–16.5 y **ninguno a 100**, mientras que a 4 y 9 iteraciones bastan 400 lotes para dar 204 y 121. La predicción registrada decía que aparecería un lote sobre 100 a 24 iteraciones: **es falsa**. El estado cambia de verdad en las últimas ~10 |

Lo práctico, en dos piezas. **Vigilar `amp` en la trayectoria no sirve como aviso** —no hay aviso—
pero **sondear la cola sobre 400 lotes sí**, con un horizonte de 4 a 9 iteraciones y un contraste de
dos órdenes. Cuesta ~3 min de GPU por sonda, así que sondear cada 10 iteraciones cuesta un 30 % del
tiempo de entrenamiento: caro para producción, barato para caracterizar. Y como **disyuntor**, `amp`
sí sirve. Es una
columna más en `traza.npy`, cuesta dos normas por paso, y distingue la iteración del salto de sus
vecinas por un factor 300 cuando `‖g‖` todavía sólo se movió 64×. Un corte en `amp > 100`, con el
umbral apoyado en 10 011 iteraciones sanas que nunca lo tocaron, dispararía en la 2 694 de `seed7` y
en la 4 721 de `seed8`. Qué hacer cuando dispara —descartar el lote, retroceder— es otra pregunta, y
el clipping ya está medido y descartado como respuesta.

---

## 2026-10-03 — El disyuntor de lotes: intercepta los 48 primeros picos del corpus rechazando 1 de cada 4 200 lotes sanos

Implementado en `train.py` a partir de `batch_rejection_spec.txt`, revisado contra las trazas del
repo y con dos cosas del spec corregidas. Calibración sin GPU: `scripts/calibrar_disyuntor.py` pasa
la columna `‖g‖` de las 81 `traza.npy` por la función de producción `rechaza_lote`. Datos crudos en
[`mediciones/disyuntor-calibracion.txt`](mediciones/disyuntor-calibracion.txt).

**Qué hace.** Después del backward y antes del paso, compara `‖g‖` contra la mediana de las últimas
100 normas **aceptadas**. Si la supera por más de `GUARD_RATIO`, el lote no da el paso: `opt.step()`
no se llama y el pico no entra en `m` ni en `v`. El lote igual se consumió y el `sched` igual
avanza.

**Por qué no es el clip otra vez**, que está descartado dos veces (2026-09-04). El mecanismo del
descarte es el que obliga a esta forma: el clip multiplica todos los gradientes por un escalar
común y `m̂/√v̂` es invariante a eso; el desplazamiento que Adam imprime vale `lr/√(1−β₂)` y **no
depende del tamaño del pico**. Reescalar la magnitud no puede acotar el desplazamiento. El
disyuntor no reescala: **saca el lote**, y un paso que no se da no desplaza nada.

### La razón sale de `β₂`, y la medición elige la misma

**Derivada.** Adam acumula `v` con `β₂ = 0.999`. Un lote de norma `R` veces la típica aporta
`(1−β₂)R²` veces la `v` de régimen; igualar eso a 1 —el pico que **duplica** el denominador y con él
le parte el paso a la mitad por las ~1 000 iteraciones del horizonte— da `R = 1/√(1−β₂) = 31.62`.
Misma clase de derivación que `DELTA_ADAM`, del mismo `β₂`.

**Medida, 2026-10-03.** 205 829 iteraciones sanas —las anteriores al primer `‖g‖ > 10⁴` de cada
corrida— con mediana 72.07, p99.9 1 448.8 y máximo **9 397, que es 130× la mediana**:

| razón | rechazos en el tramo sano | captan el 1er pico | máx. rechazos seguidos |
|---|---|---|---|
| 5.00 | 318/205829 = 0.1545 % | 48/48 | 18 |
| 10.00 | 132/205829 = 0.0641 % | 48/48 | 7 |
| 20.00 | 70/205829 = 0.0340 % | 48/48 | 5 |
| **31.62** | **49/205829 = 0.0238 %** | **48/48** | **3** |
| 50.00 | 30/205829 = 0.0146 % | 48/48 | 2 |
| 100.00 | 7/205829 = 0.0034 % | **47/48** | 1 |

De 10 a 50 hay una **meseta** donde los 48 primeros picos quedan interceptados; en 100 se escapa
uno. La razón derivada cae dentro de la meseta y rechaza **1 lote cada 4 200 sanos**, dos órdenes
por debajo del 1–2 % que el propio spec se fijaba como techo. No se eligió el valor que mejor mide:
se eligió el que sale de `β₂`, y la medición dice que sirve.

**Y el rango dinámico sano de 130× explica por qué un umbral apretado no servía.** La recomendación
que yo mismo había dado antes —winsorizar a ~5× la mediana— rechaza el **0.15 %** del tramo sano y
llega a 18 rechazos seguidos. Queda retirada: era un orden de magnitud demasiado apretada.

### El centro tiene que ser una mediana, no una EMA

La `EMA(‖g‖)` del spec falla en el arranque, y es medible: `‖g‖` sube de **26 a 828 en las doce
primeras iteraciones** (mediana sobre las 81 corridas). Una EMA arrancada en el primer valor
rechaza ahí.

| centro | tasa de rechazo en sano, razón 10 | con warmup de 50 |
|---|---|---|
| EMA `β = 0.99` | 1.135 % | 0.052 % |
| mediana móvil de 100 | **0.064 %** | 0.064 % |

La mediana es **inmune al arranque** —queda armada en la it 101, y el evento más temprano del
corpus es el de `lambdainit/B-seed5` en la **289**— y además no la mueve el pico, porque sólo entran
los lotes aceptados. Se adopta la mediana y el warmup del spec sobra.

### Dispara en la iteración 2 694, que es la del contrafactual

`‖g‖` no salta de golpe. En `caza-adjunta/seed7`, con la mediana móvil en 61.3:

```
      it      loss         |g|   mediana        amp  rechaza
    2693    23.217       71.71      61.3     0.7015  -
    2694    25.237        4592      61.3      217.1  |g|+amp
    2695    36.879       785.3      61.3     0.9992  -
    2696    42.167        1816      62.0      19.62  -
    2697   223.701        4127      62.5      1.483  |g|
    2698   126.542        7294      62.5      17.43  |g|
    2699    54.203   1.015e+04      62.5      310.5  |g|+amp   <- el evento por ||g||>1e4
```

La it 2 694 es **exactamente** el paso cuya omisión evitó el estallido entero en el contrafactual
del 2026-09-20: 0 de 1 307 iteraciones por encima de 10⁴ contra 1 302 de 1 307 en el control, con la
pérdida siguiendo a la baja. O sea que el disyuntor derivado de `β₂`, sin instrumentar nada,
identifica el mismo paso que la intervención ya medida.

### La prueba de `amp` queda como diagnóstico, no como componente

La entrada del 09-20 dejó escrito que un corte en `amp > 100` dispararía en la 2 694 de `seed7` y en
la 4 721 de `seed8`. Medido ahora contra la prueba de `‖g‖`:

| | `seed7` (evento en 2 699) | `seed8` (evento en 4 721) |
|---|---|---|
| `amp > 100` dispara en | 2 694 | 4 721 |
| razón de `‖g‖` dispara en | **2 694** | **4 720** |

**La prueba gratuita es al menos tan temprana que la instrumentada en los dos eventos**, y en
`seed8` llega una iteración antes. Así que `--guard-amp` queda implementada y **apagada**, para el
9.2 y para diagnóstico, no como parte del mecanismo.

**Y acá se cae una predicción mía.** Había escrito que instrumentar el forward cambiaría la
aritmética, porque está medido que un ULP separa dos trayectorias en ~280 iteraciones. **Es falsa**:
con `--guard-amp 100`, 120 iteraciones en `referencia` dan la traza **idéntica bit a bit** a la de
HEAD. El vjp extra no realimenta el gradiente de los parámetros y el determinismo fija el orden de
las reducciones, así que no hay ULP que mover. Queda apagada por costo —un vjp más por el paso 0— y
porque no compra anticipación. La invariancia está verificada sobre 120 iteraciones y un solo modo:
no se extiende a 10⁵ ni a `rapido` sin medirlo.

**`‖p_S‖` no sale de `retain_grad`, y eso costó una corrida.** `VNet2D.forward` devuelve los estados
**apilados en un solo tensor**, así que `states[-1]` es una vista nueva en cada indexado: el hook se
le pega a un temporal y `.grad` queda en `None`. Sale de la pérdida en forma cerrada,
`p_S = 2(x_S − clean)/(MICRO_BATCH·ACUMULACION)`, verificada contra autograd en los tests. **Y la
validación cruzada sale bien:** la `amp` que produce tiene mediana **0.6795** sobre 120 iteraciones,
contra **0.6975** medida con `retain_grad` sobre otras trazas y otro código.

Sobre `amp` en sí: en las **11 324 iteraciones sanas independientes** de las trazas instrumentadas
—`seed7-fino` y `seed8-fino` son réplicas bit a bit, y los dos brazos del contrafactual comparten
2 693 filas con `seed7`— la mediana vale 0.6975, el p99.9 6.41, y hay **una sola** por encima de
100: la it 2 694 con 217.05. No es un falso positivo: es el salto, cinco iteraciones antes de que el
criterio de `‖g‖` lo llame evento. Consistente con el «ninguna pasa de 100 en 10 011 iteraciones
sanas» del 09-20, cuya ventana sana termina justo antes del salto.

### No toca la aritmética, y por eso el control es pareado

**Medida, 2026-10-03.** 120 iteraciones, modo `referencia`, GPU 0, tres brazos con la misma semilla:
`train.py` de HEAD, el nuevo con `--guard inf`, y el nuevo con el disyuntor armado.

| brazo | `loss` @120 | `T` | `λ` | `max‖g‖` | PSNR | columnas 0–5 contra HEAD |
|---|---|---|---|---|---|---|
| `train.py` de HEAD | 36.068 | 0.10470 | 0.08502 | 784 | 29.507 | — |
| nuevo, `--guard inf` | 36.068 | 0.10470 | 0.08502 | 784 | 29.507 | **idénticas, `max\|dif\|` = 0** |
| nuevo, disyuntor armado | 36.068 | 0.10470 | 0.08502 | 784 | 29.507 | **idénticas, `max\|dif\|` = 0** |

Cero rechazos en las 120 —el disyuntor se arma en la 101— y el mismo pico de VRAM, 9.94 GB.

Ésa es la propiedad que le faltaba a todo lo probado antes: mientras el disyuntor no rechace, la
corrida es **bit a bit** la misma que sin él, así que `--guard inf` contra el valor derivado es un
par concordante de verdad y la tanda del 9.2 tiene el diseño pareado que le dio potencia al
experimento del clip.

### Por qué no aborta la corrida

El primer intento abortaba tras 50 rechazos seguidos. **Se retiró antes de commitear**:
`scripts/veredicto_precision.py` separa corrida terminada de corrida en vuelo **contando filas** de
la traza, y `scripts/hazard_fase1.py` usa `len(traza)` como tiempo de exposición. Una corrida
cortada se contaría como censurada viva y **el análisis de supervivencia se invertiría sin que nada
se ponga rojo**. `train.py` no corta nunca, y el disyuntor no es la excepción.
`tests/test_disyuntor.py` afirma que no hay `break` en el bucle.

### Qué NO está medido

El replay sobre las trazas es contrafactual **válido sólo hasta el primer rechazo**: después, la
trayectoria real se separa de la registrada. Lo medido es (a) la tasa de rechazo en el tramo sano y
(b) que el primer pico queda interceptado en las 48 corridas que estallan. **El hazard no está
medido**, y es lo único que decide. Y la objeción del 9.2 sigue en pie: el estado amplificado es
**metaestable**, así que descartar el lote puede sólo postergar — en `seed8` la corrida volvió sola
a `amp = 0.59` cuatro iteraciones después del evento y murió igual 66 más tarde, mientras el
contrafactual de `seed7` dice lo contrario con n = 1.

**Consecuencia práctica:** el mecanismo está implementado, calibrado y con los tests puestos, y la
tanda pareada de seis semillas por brazo es lo que queda. Pre-registro en
`pendientes.md`, 9.2.

### `T_init`: no hay evidencia para subirlo, y el corpus de 5·δ es **una sola trayectoria**

Preguntado si conviene subir `T_init` de 5·`DELTA_ADAM` a 10·`DELTA_ADAM`. Medido con
`scripts/corpus_tinit.py`, sin GPU. Tres cosas, en orden de peso:

**1. El default vigente no es 5·δ, es 0.1 = 7.9·δ**, y el corpus está repartido así —contando
trayectorias distintas, no directorios, porque hay tandas de copias bit a bit:

| `T_init` | ×δ | corridas | trayectorias distintas | con evento | sin evento | `T` final (sin evento) |
|---|---|---|---|---|---|---|
| 0.025298 | 2.0 | 32 | 29 | 18 | 14 | [0.02615, 0.02949] |
| **0.063246** | **5.0** | **5** | **1** | **5** | **0** | — |
| 0.100000 | 7.9 | 44 | 39 | 25 | 19 | [0.07632, 0.10155] |

Las cinco de `runs/repeat5` son **una** trayectoria replicada bit a bit, y estalló en la it 1 390.
O sea que 5·δ no está medido como configuración: tiene n = 1. 10·δ = 0.1265 está a **1.26×** del
default que tiene 39 trayectorias detrás.

**2. `T` no se mueve de donde arranca.** Sobre las 18 corridas de ≥4 000 iteraciones **sin evento**
a `T_init = 0.1`, la `T` final queda en [0.07632, 0.10155] con mediana 0.09915, y **0 de 18** baja
de 0.05. A `T_init = 0.0253` la `T` final queda en [0.02615, 0.02949], o sea que **sube** un 9–16 %.
Esto confirma y cuantifica lo que el 2026-09-06 ya dejó anotado como «`T` no es identificable»: la
predicción registrada el 2026-09-04 —`T` baja monótona desde 0.1 y se estabiliza en 0.02–0.03 antes
de la it 4 000— **es falsa**, ahora con n = 18. Consecuencia para la pregunta: `T_init` **es el
punto de operación**, no una semilla que el optimizador corrige, así que subirlo 26 % mueve `τ` un
26 % de forma permanente, y aleja de `T* = 0.024727` en vez de acercar (de 4.0× a 5.1×).

**3. El rango de 4× ya está medido y no separa.** `T_init` 0.0253 contra 0.1, seis semillas por
brazo, modo `referencia`: **4/6 contra 5/6 eventos, Fisher p = 1.0**
([`mediciones/tinit01-vs-control.txt`](mediciones/tinit01-vs-control.txt)). Y la razón de hazard
esperada para ese contraste es 1.35×, que según `scripts/poder_logrank.py` necesita ~300 semillas
por brazo. Un contraste de 1.26× es **más chico que el que ya no se puede medir**.

**Y la trampa que lo haría parecer una mejora.** `max‖g‖` escala como `1/T`, así que subir `T_init`
un 26 % baja el diagnóstico un 26 % **gratis**, sin que la corrida esté más lejos de morir. Es una
de las trampas ya anotadas en las reglas de método, y es exactamente el modo en que este cambio se
leería como un éxito.

**Lo que se descarta:** subir `T_init` a 10·δ. El argumento a favor era la reserva contra la
excursión de `T` hacia abajo, medida en **0.0290** en el evento de muerte de `clip/base-seed0`; el
default 0.1 ya la cubre **3.4×**, y ninguna de las 44 corridas a 0.1 tocó el clamp en 0. Se queda en
`T_INIT = 0.1`.

**Lo que queda abierto, y apunta al otro lado:** si `T` no viaja y el default opera a **4× el `T*`
publicado**, la pregunta interesante es si conviene **bajar** `T_init` hacia `T*`, no subirlo. Esa
celda es el 2 de `pendientes.md`, descartado por potencia el 2026-09-07 — pero el descarte fue sobre
mortalidad, y el endpoint de la fase 1 ya se corrigió a **PSNR a iteración fija**, que necesita
muchas menos semillas. Vale re-evaluar ese cálculo antes que tocar `T_init` hacia arriba.

---

## 2026-10-04 — VEREDICTO de `runs/disyuntor-100k`: 2 de 4 predicciones, mejor PSNR del repo en `referencia`, y se trabó

Matada a mano en la it 50 607 de 100 000, el 2026-10-04 a las 12:14. **Trabada desde la
it 4416**, la siguiente al último paso que se dio:
desde ahí no dio **un solo paso** en 45 585 iteraciones, con `T` y `λ` en un único valor
—0.09809705615 y 0.02930751257, `max|dif| = 0`— y 15 h de GPU gastadas sin que nada pudiera
cambiar. `estado.json` lo dice escrito, para que no se lea como censurada viva.

### Las cuatro predicciones, una por una

**1. «El disyuntor dispara en la it 1 958 o antes.» CONFIRMADA.** Disparó en la **it 1 953**, cinco
iteraciones antes de que el control cruce 1e4. Y el pareo salió exacto: hasta la it 1 952 las seis
columnas compartidas con `runs/referencia-100k` son **idénticas bit a bit**, `max|dif| = 0`.

**2. «La fracción de iteraciones con `‖g‖ > 1e4` queda bajo el 5 %.» FALLA.** Salió **91.81 %**
(45 904 de 50 000). Con una salvedad que hay que decir o el número engaña: entre los lotes
**aceptados** es **0 de 4 075 = 0.000 %**. El 91.81 % son gradientes calculados sobre pesos
congelados y descartados — mide el trabamiento, no la trayectoria del optimizador.

**3. «El mejor PSNR supera los 21.735 dB del control; forma fuerte, supera los 30 dB.» CONFIRMADA,
incluso la forma fuerte.** **31.182 dB en la it 4 000**, contra −30.190 dB del control en la misma
iteración: **+61.37 dB** pareado. Es el **mejor PSNR en modo `referencia`** de todo el repo y el
tercero del corpus de 16, detrás de `replica-v4` (31.478 @ 68 000) y `x-bf16-det` (31.194 @ 5 000,
bf16).

**4. «Los rechazos quedan bajo 100 en 100 000 iteraciones.» FALLA.** **45 925** en 50 000.

Dos de cuatro. Por la regla registrada el 10-03 —«si falla cualquiera de las cuatro, el disyuntor se
descarta»— quedaría descartado. Pero las dos que fallan miden **el mismo evento único**, el
trabamiento, no dos cosas independientes; y la 3 es el endpoint de la fase 1. La lectura honesta es
que **la detección funciona y la recuperación no existía**: «no explota» no es «sobrevive».

### Cómo murió, que es lo que cambió el diseño

| | it 4072-4074 (meten a la cuenca) | it 4075 | it 4415 (el paso letal) |
|---|---|---|---|
| `‖g‖` | 280.6 / 248.6 / 245.3 | 9.74e4 | 510.5 |
| razón contra el centro | 8.97× / 7.94× / 7.70× | — | **15.66×** |
| razón 31.62 (la de ayer) | acepta las tres | rechaza | **acepta** |
| cota absoluta 1e4 | no toca ninguna | rechaza | **acepta** |
| razón 5 (la de hoy) | **rechaza las tres** | rechaza | **rechaza** |

Entre la it 4074 y la 4075 `‖g‖` salta **400×** —de 245.3 a 9.74e4— por un paso aceptado cuya razón
era **7.70**, un valor sano ordinario. No es un pico: es una **pérdida de estabilidad**. Con `T` en
0.098 y `S = 10`, `τ = T/S = 0.0098`, que es **4.0× el `T*` publicado** de 0.024727.

**Y acá se retira una afirmación mía de hoy mismo.** Escribí que la cota absoluta era el arreglo,
con el argumento de que separa perfecto: 0 de 4 075 aceptados sanos pasan de 1e4 y 45 904 de 45 925
rechazados sí, con un hueco de **86×** entre la máxima aceptada (962) y la mínima desde que se trabó
(8.29e4). El número es cierto y la conclusión era falsa: **a 1e4 la cota acepta el lote letal de
510.5 y no toca ninguno de los tres de entrada**, cuyo máximo es 280.6. Separa «ya está muerta» de
«viva», que es un **diagnóstico**, no una prevención. Lo que previene es bajar la razón.

> **Acotado el 2026-10-05.** «Lo que previene es bajar la razón» es falso: los dos eventos de
> `runs/tinit-7p9delta-10k` entraron por un lote **aceptado** a razón 3.58× y 3.67×, debajo de
> cualquier umbral que no rechace lotes sanos. La razón **contiene** la secuela; el que previene
> la muerte es el **rebobinado**. Ver el veredicto del 2026-10-05.

### Lo estructural: una prueba por lote llega siempre un paso tarde

`‖g‖` se mide en los pesos de **ahora**, así que informa dónde está la corrida, no a dónde la lleva
el paso que falta dar. El lote que la rompió tenía razón 7.70. Con los pesos ya en la cuenca, las
dos salidas son simétricas y **las dos se observaron en esta corrida, en ese orden**: rechazar todo
→ el centro deja de actualizarse, los pesos se congelan y se traba; aceptar la cola baja → el paso
se da sobre pesos dañados y se hunde más. Por eso el disyuntor pasa a tener tres piezas con
trabajos distintos, y por eso ninguna prueba por lote sola podía salvar **esta** corrida.

**Lo que no está medido:** que el rebobinado salve una divergencia real. Lo medido es el mecanismo,
en 200 iteraciones con `--guard 1.0` a propósito absurdo: 180 rechazos seguidos y PSNR 25.129 dB sin
él, contra 5 seguidos como máximo, 7 rebobinados y 29.015 dB con él. Eso valida que rebobinar
funciona, no que alcance.

---

## 2026-10-04 — PRE-REGISTRO: `T_init` 0.1 contra 10·δ a 10⁴ iteraciones, con la cota nueva

**Reapertura deliberada.** Subir `T_init` a 10·`DELTA_ADAM` = 0.1265 está en `## Descartado` desde
ayer, con tres argumentos. El usuario la reabre a pedido explícito y con su razonamiento anotado:
«tiene sentido que sea un valor alto al principio, ya que corrige de más, pero no tiene mucha
info». Se corre. Queda escrito que es reapertura y no pregunta nueva, porque en este repo ya se
reabrieron hipótesis dos veces por no mirar esa tabla primero.

**1. Qué cambia, y que es una sola cosa.** `--t-init`: 0.1 (= 7.9057·δ) contra 0.1264911064
(= 10·δ). Todo lo demás idéntico y verificado en los dos `config.json`: `referencia`, `seed 0`,
10 000 iteraciones, `clip inf`, sin checkpointing, y el mismo disyuntor —razón 5, cota absoluta
10⁴, ventana 100, rebobinado a los 25 seguidos, respaldo cada 250—. Los dos brazos salen del mismo
`train.py`.

**2. Control: concurrente**, los dos a la vez, GPU 0 y GPU 1. `runs/disyuntor-100k` **no sirve de
control** aunque comparta semilla y `t_init`: corrió con la razón 31.62 y sin cota absoluta ni
rebobinado, así que difiere en cuatro cosas.

**3. Qué contestó el replay sin GPU, y qué no.** `T` bajo Adam, con la secuencia `dL/dT` grabada,
arrancando de 0.1 y de 0.1265 sobre 7 trayectorias de 4 000 iteraciones: **la separación inicial de
0.026491 se conserva al 100.0 % exacto**, hasta el último dígito. Adam no tira a `T` hacia ningún
valor preferido: el offset es una traslación pura que no decae. Eso confirma el mecanismo —`T_init`
**es** el punto de operación y el contraste de 1.26× es permanente, no transitorio— y por eso el
experimento no es vacío. Lo que el replay **no** puede contestar: mantiene fija la secuencia de
gradientes, y un `T` distinto cambia el forward y por lo tanto los gradientes. Da el *si*, no el
*cuánto*. (Validación del replay: las 6 corridas archivadas reconstruyen `T` con error máximo
1e-7; `disyuntor-100k` da 0.0097 porque el lote rechazado de la it 1953 sí omitió un paso, que es
exactamente lo que el replay no modela.)

**4. Endpoint: el MEJOR PSNR, no el PSNR a iteración fija.** Medido hoy sobre las 6 corridas de
`runs/tinit01` —misma config, sólo cambia la semilla—:

| endpoint | n | sd entre semillas | rango |
|---|---|---|---|
| PSNR en la it 1 000 | 6 | **0.016 dB** | 0.041 dB |
| PSNR en la it 4 000 | 6 | **15.427 dB** | 38.939 dB |
| **mejor PSNR** | 6 | **0.164 dB** | 0.472 dB |

A iteración fija tardía el PSNR es inservible: en la it 4 000 la dispersión entre semillas idénticas
es de 38.9 dB porque lo que domina es si la corrida ya estalló, no la configuración. El mejor PSNR
en cambio tiene sd 0.164 dB. Se registran los tres y el veredicto se lee sobre el mejor PSNR y
sobre el de la it 1 000. Mortalidad binaria no se usa, nunca.

**5. Predicciones, con umbral.**

1. El mejor PSNR de 10·δ **no supera** al de 0.1 por más de 0.33 dB (= 2 sd entre semillas). Si lo
   supera por más, la tabla `## Descartado` queda mal y hay que revisarla.
2. El mejor PSNR de 10·δ **no cae** más de 0.33 dB por debajo del de 0.1. Si cae más, el descarte
   queda confirmado con una medición directa y no sólo por argumento.
3. El PSNR en la it 1 000 difiere entre brazos **menos de 0.05 dB** (= 3 sd del endpoint temprano).
   Si difiere más, `T_init` muerde antes de lo que dice el corpus.
4. `max‖g‖` del brazo de 10·δ queda **entre 0.74× y 0.85×** el del brazo de 0.1, porque `‖g‖`
   escala como `1/T` y 0.1/0.1265 = 0.79. Es el control de sanidad: si sale ~1, el escalamiento
   `1/T` que la bitácora da por medido está mal.
5. Ninguno de los dos brazos se **traba**: 0 iteraciones con todos los lotes rechazados hasta el
   final. Si uno se traba, el rebobinado no alcanza y hay que rediseñarlo.

**6. Potencia, y lo que esta tanda NO pregunta.** El contraste es 1.26×. El techo medido el
2026-09-07 con 12 semillas por brazo da poder 0.11 contra 1.35×, y 1.35× necesita ~300 semillas por
brazo. Con **n = 1 por brazo no hay potencia para ningún endpoint de evento o de hazard**, así que
**la pregunta de hazard no se hace**. Lo que n = 1 sí permite es el mejor PSNR, porque su sd entre
semillas es 0.164 dB: un contraste mayor a 0.33 dB es más grande que el ruido de semilla. Un
contraste menor **no es medible con esta tanda** y así queda escrito antes de verlo.

**7. Costo.** 10 000 iteraciones en `referencia` a 1.53 h/4 000 it = **3.83 h por brazo**, los dos en
paralelo en tarjetas separadas ≈ **4 h de reloj**. GPUs verificadas libres antes de lanzar: 277 MiB
y 15 MiB, 0 % de uso las dos.

**8. Cómo NO se leen estos resultados.**
- Si los dos mejores PSNR caen dentro de 0.33 dB, eso **no** es evidencia de que `T_init` no
  importa. Es exactamente lo que predice el ruido de semilla con n = 1, y la celda sigue abierta.
- Si 10·δ gana por menos de 0.33 dB, eso **no** alcanza para cambiar el default. Los tres
  argumentos de `## Descartado` —que `T` no viaja, que el rango de 4× ya se midió sin separar, y que
  subirlo aleja de `T*` de 4.0× a 5.1×— no los tumba una semilla.
- La comparación con `runs/disyuntor-100k` (31.182 dB) **no entra en el veredicto**: difiere en
  cuatro cosas además de `T_init`.
- Y el `max‖g‖` más bajo del brazo de 10·δ **no es que esté más sano**: es la trampa del `1/T`, que
  está medida y es mecánica.

---

## 2026-10-05 — VEREDICTO: `T_init` 0.1 contra 10·δ **no separa**, y el disyuntor sobrevivió un evento real

Las dos terminadas a 10 000 iteraciones, `estado.json` propio en las dos, 3.80 h y 3.74 h.
Salida cruda en `docs/mediciones/tinit-10k.txt`.

| | A — `T_init` 0.1 (7.91·δ) | B — `T_init` 10·δ (0.12649) |
|---|---|---|
| **mejor PSNR** | **31.34323 dB** | **31.35234 dB** |
| PSNR final | 31.34323 (= el mejor) | 31.35234 (= el mejor) |
| `T` @10 000 | 0.08977 | 0.11605 |
| `λ` @10 000 | 0.05189 | 0.04294 |
| lotes rechazados | 93 de 10 000 (0.93 %) | 4 de 10 000 (0.04 %) |
| racha máxima | **25** (= el umbral de rebobinado) | 1 |
| rebobinados | **2** (it 2 298 y 5 901) | 0 |
| `max‖g‖` | 1.971e5 | 4 428 |

### Predicción 1 — el mejor PSNR de 10·δ no supera al de 0.1 por más de 0.33 dB → **CONFIRMADA**

Supera por **+0.00911 dB**, que es **36× menos** que el piso. La tabla `## Descartado` no queda mal.

### Predicción 2 — ni cae más de 0.33 dB por debajo → **CONFIRMADA**

No cae: queda arriba por 0.009 dB. Los dos brazos son indistinguibles, que es el resultado.

### Predicción 3 — el PSNR en la it 1 000 difiere menos de 0.05 dB → **NO MEDIBLE, y es error mío**

`eval_every = 2000`, así que **no existe evaluación en la it 1 000** en ninguno de los dos brazos.
Registré un endpoint que no instrumenté. El pre-registro decía «se registran los tres»: se
registraron dos. Para la tanda de 100 000 no se arrastra, porque ahí el endpoint temprano no entra.

### Predicción 4 — `max‖g‖` de 10·δ entre 0.74× y 0.85× el de 0.1 → **FALLA, y deja una medición**

| cantidad | razón B/A | banda [0.74, 0.85] |
|---|---|---|
| `max‖g‖` todas (con el evento de A) | 0.0225 | fuera |
| `max‖g‖` sólo lotes sanos | **0.9988** | fuera |
| `max‖g‖` it 1–100 (`T ≈ T_init`) | 1.0375 | fuera |
| mediana `‖g‖` sanas | 0.9648 | fuera |
| **`max\|∂L/∂T\|` sanas** | **0.7655** | **dentro** |

El pre-registro decía: *«si sale ~1, el escalamiento `1/T` que la bitácora da por medido está
mal»*. Sale 1.00 y el escalamiento **no** está mal — está mal el objeto sobre el que lo leí.
Vale sobre `|∂L/∂T|` (0.7655 contra 0.790 predicho) y no se propaga a `‖g‖` porque a `T ≈ 0.1`
esa componente es el **27–36 %** de la norma, no el 70–86 % que registra `AGENTS.md`. Y los dos
números son la **misma ley**: escalar 0.364 por el 4× de `T` da **0.839**, y 0.272 da 0.816, las
dos dentro de 0.70–0.86. Lo que hay que acotar es el corolario práctico, no el número.

### Predicción 5 — ninguno de los dos se traba → **CONFIRMADA**

Ninguno. La racha máxima de A es **25**, que es exactamente el umbral: llega, rebobina y el
contador vuelve a cero, así que por construcción no puede pasarlo mientras haya respaldo. Contra
los **45 585** rechazos seguidos de `disyuntor-100k`.

### Lo que la tanda dio además, y no estaba preguntado

**A reprodujo la muerte de `disyuntor-100k` y salió viva.** Misma firma, peor pico:

| | `disyuntor-100k` (cota vieja) | A (cota nueva) |
|---|---|---|
| pico de `‖g‖` | 9.74e4 | **1.971e5** |
| centro móvil | congelado en 32.5974 | congelado en 29.53 |
| rechazos seguidos | **45 585** → trabada | **25** → rebobina |
| `‖g‖` el paso siguiente | seguía sobre 1e4 | **21.3** |
| PSNR | 31.182 → 27.631, nunca volvió | 31.182 → **31.343, subiendo** |

**Lo que previene no es la razón, es el rebobinado.** Los dos eventos de A entraron por un lote
**aceptado**: la it 2 273 con `‖g‖` 165.1 (razón **3.58×**) y la it 5 836 con 105.9 (razón
**3.67×**). Ningún umbral que no rechace lotes sanos los agarra. La razón 5 contiene la secuela
—7 de los 93 rechazos caen en la banda 5–10× que la razón 31.62 aceptaba, y 5 de esos 7 están
dentro del evento grande—, pero recuperar lo hace el rebobinado, en **un paso** las dos veces.

**La cota absoluta no hizo trabajo propio.** De los 97 rechazos de los dos brazos, **0** los agarró
sólo ella; los 18 que cruzaron 1e4 los agarró también la razón. Los tres tramos con la ventana
desarmada (it 1–100, y los 100 que siguen a cada rebobinado) tuvieron máximos de **828 / 117.6 /
54.1** y **cero rechazos**. Sigue implementada porque es la única pieza que no se estima de nada y
no puede congelarse —el modo exacto en que murió `disyuntor-100k`—, pero es **seguro no probado**.

**La PSNR no se había estancado.** El mejor PSNR **es** el de la última evaluación en los dos, con
incrementos de +0.197, +0.074, +0.046, +0.033 (A) y +0.178, +0.081, +0.044, +0.052 (B). A 10 000
iteraciones la curva sigue subiendo ~0.04 dB cada 2 000, y faltan 0.157 dB para los 31.5. **10 000
iteraciones no alcanzan para el objetivo de la fase 1.**

**El offset de `T` se conserva al 99.2 %** sobre 10 000 iteraciones: 0.0264911 inicial contra
0.0262713 medido. El replay sin GPU predijo 100.0 % y acertó con 0.8 % de error, ahora verificado
en entrenamiento y no sólo en replay.

### Un error propio que costó 4 h de GPU

Las dos terminaron a la **01:45** y las de 100 000 se lanzaron a las **05:56**: las tarjetas
estuvieron libres cuatro horas. El esperador en segundo plano usaba
`pgrep -f 'train.py --out runs/tinit-'`, que matchea **también los wrappers de bash** que lanzaron
las corridas, así que la condición no se cumplió hasta que el tope de 5.5 h cortó el bucle.
`AGENTS.md` ya tiene esa trampa escrita, con el patrón anclado y el antecedente del 2026-09-03.
Es la tercera vez. **Lo correcto acá era esperar `estado.json`**, que existe desde el 10-04
justamente para no deducir el estado de un proceso.

---

## 2026-10-05 — PRE-REGISTRO: `T_init = 8·δ_Adam` a 10⁵ iteraciones, dos semillas

Escrito **antes** de ver un solo número de las dos corridas, lanzadas a las 05:56. `runs/` está
gitignoreado y no tiene sello temporal, así que el pre-registro vive acá.

**Decisión del autor**, corrigiendo el plan de la noche anterior: «quiero que inicializes con 8 *
delta adam, así llega justo por encima del límite de muerte a las 100.000 (por el halving). no uses
2*delta_adam, deséchalo. Tampoco uses 0.1 directo, prefiero un poco más alto».

**La aritmética de 8·δ, que es el motivo.** `8·δ = 0.1011928851`. Con la deriva de `T` medida en el
brazo de 10k (−1.6399·10⁻⁶ por iteración entre la it 1 000 y la 6 000, escalada por el halving del
`lr` cada 25 000):

| `T_init` | | `T` @25k | @50k | @75k | **@100k** | contra `δ` | contra `T*` |
|---|---|---|---|---|---|---|---|
| **8·δ** | 8.00× | 0.06020 | 0.03970 | 0.02945 | **0.02432** | **1.92×** | **0.98×** |

> **Falsificada el mismo 10-05 por los parciales de 18 000 iteraciones** (ver la enmienda al final
> de este archivo). La deriva real a la it 18 000 es −8.87·10⁻⁷/it y `T` aterriza en ≈0.0585 =
> 2.36·`T*`, no en 0.02432. La predicción 1 de abajo queda registrada como estaba y se leerá como
> falla; la banda enmendada es [0.050, 0.075].
| 0.1 | 7.91× | 0.05900 | 0.03850 | 0.02825 | 0.02313 | 1.83× | 0.94× |
| 9·δ | 9.00× | 0.07284 | 0.05235 | 0.04210 | 0.03697 | 2.92× | 1.50× |
| 10·δ | 10.00× | 0.08549 | 0.06499 | 0.05475 | 0.04962 | 3.92× | 2.01× |
| 2·δ | 2.00× | — | — | — | cruza `δ` en la it **7 714** | muerta | — |

8·δ es el único de la lista que aterriza **arriba de la pared y sobre `T*`** a la vez.

**1. Qué cambia.** Entre los dos brazos, **sólo la semilla**: 0 y 1. No es una tanda de contraste,
es **n = 2 sobre una configuración**: `T_init = 0.1011928851`, `referencia`, 100 000 iteraciones,
disyuntor completo (razón 5, cota 1e4, ventana 100, rebobinado a los 25, respaldo 250),
`LAMBDA_INIT = 0.1`, `clip inf`.

**2. Control.** No hay brazo de contraste, a propósito. El control del *recorrido* es la tanda de
10k de hoy, que da la deriva de `T` y el nivel de PSNR. Control del *desenlace a 10⁵* no existe:
`runs/disyuntor-100k` murió en la it 4 416 con la cota vieja y `runs/referencia-100k` corrió sin
disyuntor.

**3. La pregunta principal, y qué contestó el replay.** ¿`T` saliendo de 8·δ aterriza en
`T* = 0.024727` **bajando**, sin cruzar nunca la pared en `δ`? Es textualmente la variante que esta
bitácora prescribió el 2026-09-02 para cerrar la circularidad: *«si una corrida arranca en 0.0253 y
converge a 0.0247 el resultado es discutible. La variante que cierra esa objeción es arrancar (…)
por encima de `T*` y verificar que converge bajando»*. El replay ya corrió y contesta otra cosa: el
offset de `T_init` se conserva al 100.0 % exacto sobre la `dL/dT` grabada, así que el punto de
operación es permanente. No puede decir **dónde aterriza**, porque la `dL/dT` registrada viene de
trayectorias que nunca estuvieron ahí.

**4. Endpoint: el mejor PSNR**, más `T` @100 000 y la primera iteración con `T < δ` si la hay. Sobre
6 semillas de config idéntica el mejor PSNR tiene sd **0.164 dB** y el PSNR a iteración fija tardía
**15.427 dB**. Con n = 2 el error estándar del promedio es 0.116 dB y **una diferencia menor a
0.33 dB no es medible**.

**5. Predicciones, con umbral.**

1. `T` @100 000 cae en **[0.018, 0.032]**; el punto es 0.02432 = 0.98·`T*`. Si falla, la deriva no
   extrapola —es 17× la ventana medida— y hay que medir la curvatura en vez de suponerla lineal.
2. `T` **nunca** cruza la pared: ninguna evaluación con `T < 0.0126491`. Si falla, 8·δ no alcanza
   de margen y el piso de la regla 2-δ hay que subirlo.
3. El mejor PSNR llega a **≥ 31.40 dB** en CBSD68. Si falla, 10⁵ iteraciones no alcanzan desde
   inicialización aleatoria, o `T_init` no es la perilla.
4. Ninguna de las dos se traba: rachas de rechazo **< 1 000** seguidos. Si falla, el rebobinado no
   escala de 10⁴ a 10⁵ iteraciones.
5. Las dos semillas concuerdan: |mejor PSNR(s0) − mejor PSNR(s1)| **< 0.33 dB**. Si falla, la
   varianza de endpoint a 10⁵ es mayor que la medida a 4 000 y n = 2 no alcanza.
6. La cota absoluta 1e4 **nunca dispara sola**: 0 lotes con `‖g‖ > 1e4` y razón ≤ 5 en las 200 000
   iteraciones. Si falla, deja de ser seguro no probado y pasa a ser pieza que trabaja.

**Riesgo conocido de la predicción 1, anotado antes del resultado.** La deriva **se desacelera**:
−1.6399·10⁻⁶ (it 1 000–6 000), −1.5401·10⁻⁶ (it 1 000–10 000), **−1.4154·10⁻⁶** (it 6 000–10 000).
Con la más reciente el punto se corre a **0.03485 = 1.41·`T*`**, que cae **fuera** de la banda
registrada. La banda queda como está: lectura honesta es que `T` probablemente aterrice en
0.029–0.035, arriba de la pared pero no sobre `T*`, y que la predicción 1 falle por mi modelo de
deriva y no por el `T_init` elegido.

**6. Potencia.** n = 2: **no se hace ninguna pregunta de hazard** (techo del 2026-09-07: poder 0.11
contra 1.35×). Las predicciones 1–4 son de magnitud grande y no necesitan n alto; la 5 es la que
mide si n = 2 alcanza.

**7. Costo.** 1.347 s/it medido en la tanda de 10k con las dos GPUs ocupadas. 100 000 it = **37.4 h
por brazo**, concurrentes en tarjetas separadas ≈ **37.4 h de reloj**, 74.8 h de GPU. Dos
`referencia` concurrentes no se interfieren: `cudnn.benchmark` es inerte bajo determinismo
(2026-09-02: 0.848 contra 0.847 s/it, gradiente idéntico), y los dos brazos de 10k corrieron los dos
a 1.347 s/it. GPUs verificadas libres antes de lanzar: 277 MiB y 15 MiB, 0 % las dos.

**8. Cómo NO se leen estos resultados.**
- Si `T` @100k cae en la banda, eso **no** prueba que `T*` sea un atractor. Prueba que una
  trayectoria que arranca 8× arriba pasa por ahí a las 100 000 iteraciones. Atractor exigiría
  mostrar que vuelve después de una perturbación, y eso no se mide acá.
- Si la PSNR no llega a 31.40 dB, eso **no** culpa a `T_init`. La receta tiene más cosas sin
  verificar —el `lr` inicial, el tamaño de lote, el crop— y esta tanda no las separa.
- Si una se traba, **no** se promedia con la otra. Se reportan por separado y la trabada cuenta
  como desenlace, no como ruido.
- `max‖g‖` **no** se compara contra ninguna corrida con otro `T_init`: la trampa del `1/T` está
  medida, y hoy quedó acotada a `|∂L/∂T|`, que es sólo el 27–36 % de la norma a este `T`.
- El `estado.json` manda sobre el conteo de filas de `traza.npy`. Una corrida matada a mano **no**
  es una censurada viva.

---

## 2026-10-05 — ENMIENDA AL PRE-REGISTRO de `t8delta-100k`: corte de luz, y el disyuntor v2

Escrita **antes** de relanzar y antes de leer ningún resultado nuevo.

**Qué pasó.** Corte de luz. La máquina rebotó a las **12:58** (`who -b`), último log a las 12:56:57.
Los dos brazos iban por la **it 18 500 de 100 000** (18 750 y 18 500), sin `estado.json`, con
`last.pth` de las 12:40. Quedan archivados en `runs/t8delta-100k-corte-s0` y `-s1`: son el control
pareado del disyuntor v1 sobre las primeras 18 000 iteraciones, con la misma semilla y el mismo
`T_init`, y `referencia` es bit-reproducible, así que sirven de patrón exacto.

**Lo que dejaron medido** (las dos trayectorias coinciden dentro de 0.026 dB a igual iteración, que
es 13x por debajo del piso de 0.33 dB):

| it | PSNR s0 | PSNR s1 | T s0 | T s1 |
|---|---|---|---|---|
| 10 000 | 31.337 | 31.340 | 0.090208 | 0.093545 |
| 14 000 | 31.401 | 31.401 | 0.085775 | 0.089571 |
| 16 000 | **31.419** | 31.411 | 0.083872 | 0.087823 |
| 18 000 | 31.393 | **31.419** | 0.081994 | 0.086154 |

Rechazos 11 y 4, rebobinados 0 y 0. Las dos seguían subiendo: el mejor PSNR es el de la última
evaluación en s1, y en s0 el de la penúltima.

### El disyuntor v2, y los tres valores que NO se tocan

Con las GPUs libres se pagó la deuda de la §9.5 de `docs/pendientes.md`. Dos cambios, y tres
números que se quedan con la razón medida que los fija.

**Cambio 1 — la ventana viaja con el respaldo.** `rebobina` devolvía pesos y momentos y **borraba**
la escala. Es inconsistente: el rebobinado pone los pesos de vuelta en el respaldo, así que la
escala que valía en el respaldo vuelve a valer. Borrarla dejaba el relativo desarmado 100
iteraciones aceptadas justo al salir del evento —**200 de las 300 desarmadas de
`tinit-7p9delta-10k` las creaba el rebobinado mismo**—, y la ventana que se tiraba era buena: la
primera `‖g‖` después de cada rebobinado vale **48.5** y **21.3** contra centros congelados de
46.71 y 29.53, razones **1.04** y **0.72**. Las dos caen dentro de la ventana descartada.

**Cambio 2 — `GUARD_LIMPIAS = 3`.** El respaldo se tomaba en cualquier iteración aceptada, y dentro
de una excursión letal hay lotes aceptados: **4 en el evento de `tinit-7p9delta-10k`** y **2 en
`disyuntor-100k`**, todos de hueco **1**. Medido, no hipotético: `t8delta-100k-s1` tomó un respaldo
en la **it 6003** con 3 rechazos en las 30 previas y **una sola** limpia detrás. El 3 es 3x el
hueco máximo medido, y en `referencia` —4 a 11 rechazos en 18 000— no retrasa nada.

**Se quedan, y por qué:**

- **razón 5.** Único valor de la tabla bajo la banda de entrada medida 7.70–8.97x.
- **cota 10⁴.** Su criterio real no es «arriba del máximo sano del corpus (9397)»: es **separar las
  excursiones que se resuelven solas de las que necesitan intervención**. Las primeras llegan a
  **5312** (`tinit-7p9delta-10k` ev.1) y **6113** (`precision/rapido-seed2`: 33 rechazos en 50
  iteraciones, sin disyuntor, y terminó en **30.937 dB**). Las segundas, a **1.971e5** y **6.9e12**.
  Hueco medido **[6113, 1.971e5]**, 32x sin solapamiento, y 10⁴ cae adentro. **Bajarla a 5000 era
  mi plan y está mal**: queda debajo de 6113. El techo sano de la ventana desarmada es 2556 sobre
  86 corridas (0 de 86 arriba de 3000), pero ese corpus excluye justo las excursiones que importan.
- **25 seguidos.** El corpus sano llega a 18 a razón 5.

**Descartado con contraejemplo: la densidad como disparador.** «N rechazos en las últimas W» en vez
de N seguidos, porque un aceptado intercalado reinicia el contador —y eso retrasó el rescate del
evento 2 de la it 5862 a la 5901—. Pero `precision/rapido-seed2` tiene **33 rechazos en 50
iteraciones** en su tramo sano y terminó en 30.937 dB sin tocar nada: a 25/50 el disparador habría
rebobinado una corrida que no lo necesitaba. **La densidad no separa «se resuelve» de «mata»; la
magnitud sí**, y de eso ya se ocupa la cota.

**Prueba de humo del v2**, `--guard 1.1 --guard-ventana 20 --guard-seguidos 5 --guard-respaldo 30`:
191 rechazos de 300, **31 rebobinados**, exit 0, y **cero `mediana=nan`** en todo el log. Con v1
cada rebobinado dejaba 100 iteraciones de mediana `nan`. 53 tests pasan.

### Predicciones enmendadas

**1 (reemplaza la registrada).** `T` a la it 100 000 cae en **[0.050, 0.075]** = 4.0–5.9·δ. La
banda registrada era [0.018, 0.032] y los parciales la falsifican: la deriva medida a la it 18 000
es **−8.87·10⁻⁷/it** (s0 −9.39e−7, s1 −8.35e−7) y el halving del lr la divide por 2 en 25k, 50k y
75k, así que el desplazamiento restante es `8.87e−7 × (7000 + 12500 + 6250 + 3125)` = **0.0256**, y
`T` aterriza en **0.0585**, que es **2.36·`T*`**. Como la deriva además se desacelera sola, 0.0585
es un **piso**. Si cae fuera de [0.050, 0.075], mi modelo de deriva está mal por segunda vez.

**2.** Ningún brazo llega a `T < δ`. Ahora es casi seguro y por eso **deja de ser informativa**:
`8·δ` a 100k **no pone a prueba la pared de muerte**. Queda escrito para que no se recicle como
éxito. El halving es lo que la salva: sin él el desplazamiento sería `8.87e−7 × 82 000` = 0.0727 y
`T` terminaría en **0.0114 = 0.90·δ**, debajo de la pared.

**3.** **La semilla 1 reproduce su propia trayectoria v1 hasta la it 18 000**, evaluación por
evaluación: 31.003 / 31.172 / 31.005 / 31.310 / 31.340 / 31.377 / 31.401 / 31.411 / 31.419. El v2
no toca el camino sano —s1 tiene 4 rechazos y 0 rebobinados, y la única diferencia es el respaldo de
la it 6003 corriéndose a ~6005, que sólo importa si dispara un rebobinado—. **Si no reproduce, el
cambio no es inerte y hay que sacarlo.**

**4.** Lo mismo para la semilla 0 (31.393 a la it 18 000, 11 rechazos, 0 rebobinados).

**5.** PSNR a la it 100 000 en **[31.45, 31.60]**. Base: 31.419 a la 18 000 con incrementos ya
caídos a ~+0.01 cada 2 000, y tres halvings por delante. Debajo de 31.45, 100 000 iteraciones no
alcanzan a este `T_init`; arriba de 31.60, los halvings pesan más que la tendencia.

**6.** Si algún brazo rebobina, la primera `‖g‖` posterior se lee contra la mediana **restaurada** y
da una razón finita en la banda de v1 (1.04 y 0.72), no `nan`.

### Cómo NO se leen

- Los dos brazos **no** son evidencia independiente sobre `T_init`: comparten el valor. Son una
  prueba de reproducibilidad, y es para lo único que valen a n = 2.
- Si falla la 3 o la 4, **nada más de la tanda es legible**: el v2 cambió el camino sano y la
  comparación contra el corpus archivado se rompe.
- **El disyuntor v2 no queda validado si no dispara ningún rebobinado.** Dos brazos limpios hasta
  100k dejan al v2 exactamente donde está la cota: sin probar. Eso no es un resultado negativo, es
  ausencia de resultado, y se escribe así.
- Un `T` que aterriza cerca de un número publicado es coincidencia hasta que una segunda semilla lo
  reproduzca.

**Costo:** 2 × 37.4 h = **74.8 h de GPU**, 37.4 h de reloj, las dos tarjetas. `referencia` es
inmune a la contención (0.848 contra 0.847 s/it, gradiente idéntico), así que las dos conviven.

---

## 2026-10-06 — PRE-REGISTRO de `tinit-2delta-50k`: `T` arrancando en `T*`, con el disyuntor puesto

Se cortó a mano `t8delta-100k-s0` en la it 66 000 (`estado.json`, mejor 31.5257 dB) y su GPU se
reasignó a esta corrida. El corte está justificado abajo en «Por qué se cortó el control».

**1. Qué variable cambia, y es una sola.** `T_init = 0.025298221281347025` (2·`δ_adam`) contra
`0.1011928851` (8·`δ_adam`) del control. Todo lo demás idéntico: semilla 0, modo `referencia`,
`λ_init = 0.1`, `lr = 4e-4` con halving cada 25 000, y los siete parámetros del disyuntor v2 sin
tocar (`guard=5`, `abs=1e4`, `ventana=100`, `seguidos=25`, `respaldo=250`, `limpias=3`,
`amp=inf`). Las iteraciones pedidas difieren —50 000 contra 100 000— así que **el contraste se lee
a iteración pareada**, nunca endpoint contra endpoint.

**2. Control.** `runs/t8delta-100k-s0`, **archivado** (cortado a mano, no terminado): 66 000
iteraciones, mejor 31.5257 @64 000, `T` final 0.07962, 18 rechazos, 0 rebobinados.
*Contexto archivado que NO es control:* `runs/clip/base-seed*`, seis semillas × 4 000 it con el
mismo `T_init` pero `λ_init = 0.01` y **sin disyuntor** — 3 de 6 muertas (fila 1 de su traza:
`T=0.0252982 lambda=0.0100000`). No es control porque difiere en dos variables más.

**3. La pregunta, y qué contestó ya el replay sin gastar GPU.** ¿Un modelo con `T` arrancando en
`T*` llega a la misma PSNR que uno que arranca 4× arriba y nunca baja de 0.078? El replay de Adam
sobre la coordenada `T` sola ya corrió, **validado a 6.85e−7** contra la `T` grabada de la seed 0
(que no tiene rebobinados), y contesta el *si*: con la secuencia de `∂L/∂T` de la seed 1,
`T_init = 2δ` toca cero **sin** disyuntor —19 iteraciones bajo cero, mínimo −0.007867— y **no** lo
toca con él —mínimo +0.001513—. Reserva exigida medida: **1.88·δ con** el disyuntor contra
**2.62·δ sin** él, o sea que la guarda bajó el requisito 1.4× en esa secuencia y 2.3× contra el
~4.3·δ que había medido el 2026-09-03. No contesta el *cuánto*: la PSNR no sale de un replay de una
coordenada.

**4. Endpoint.** PSNR a iteración fija sobre CBSD68, leída a 4 000 / 16 000 / 50 000 contra los
mismos puntos del control. No mortalidad binaria. El evento (`‖g‖ > 1e4`) se cuenta aparte, como
diagnóstico del disyuntor, no como desenlace.

**5. Predicciones.**

1. **`T` no toca cero en las 50 000.** El replay da mínimo +0.001513, o sea 0.12·δ de margen sobre
   una reserva exigida de 1.88·δ: **margen del 6 %**. Si falla, la lectura es que 1.88·δ está
   subestimado porque el contrafactual corre sobre gradientes de `T ≈ 0.08`.
2. **PSNR @4 000 en [31.05, 31.25].** Las tres supervivientes del 2026-09-03 con el mismo `T_init`
   dieron 31.190 / 31.141 / 31.102, y eso fue con `λ_init = 0.01`. Si cae debajo de 31.05,
   arrancar en `T*` cuesta convergencia temprana.
3. **`T` @4 000 en [0.023, 0.028].** Réplica del histórico (0.0264 / 0.0236 / 0.0258). Si no
   replica, cambió algo entre esa época y ahora —`λ_init` de 0.01 a 0.1, o el disyuntor— y hay que
   encontrar qué.
4. **PSNR @16 000 ≥ 31.09.** El control dio 31.4191 ahí. 31.09 es el control menos 0.33 dB, que es
   el mínimo contraste detectable con n=1. Entre 31.09 y 31.42 **no se concluye nada**.
5. **PSNR @50 000 en [31.48, 31.58].** El control dio 31.5031 @48 000 y 31.5192 @56 000, ≈31.51
   interpolado.
6. **`T` @50 000 > 0.0300**, o sea que `T` **sube** saliendo de `T*`. Base: el 10-06 se midió que
   la deriva de `T` se invierte en los halvings, y la tabla `## Descartado` ya registra que `T` no
   viaja (0 de 18 corridas sin evento bajan de 0.05 desde 0.1). Si queda en [0.024, 0.030],
   entonces `T*` **es** un atractor, y sería el primer resultado positivo del repo sobre `T*`.
7. **Rechazos > 18** en las primeras 50 000 (el control acumuló 18 en 66 000). Base: `max‖g‖`
   escala como `1/T`, y 0.0796/0.0253 = 3.1×, así que la cota absoluta de 1e4 debería cruzarse más
   seguido. Si no dispara más, la ley `1/T` no vale fuera del rango donde se midió.

**6. Potencia, y es mala: esta tanda es descriptiva.** n = 1. Para la pregunta de calidad sólo
detecta diferencias ≥ 0.33 dB (2 sd de mejor-PSNR = 0.164 dB). Para la pregunta de supervivencia
es **peor**: contra la tasa base histórica de 3/6, una sola corrida que sobreviva da p = 0.5, o sea
**nada**. Se lanza por lo que sí da sin poder estadístico: la trayectoria de `T` desde abajo de
donde el control se estaciona, que nunca se observó, y la comprobación de si la predicción del
replay se cumple en una corrida real.

**7. Costo.** 50 000 × ~1.385 s/it medido = **~19 h de GPU**, en la GPU 1. Concurrente con
`t8delta-100k-s1` en la GPU 0; dos `referencia` no se interfieren (medido: 0.848 contra 0.847 s/it,
gradiente idéntico).

**8. Cómo NO se leen los resultados.**

- Si la PSNR sale dentro de ±0.3 dB del control, **no** se lee «arrancar en `T*` es equivalente».
  Se lee «no se separa con n=1», y la pregunta queda abierta.
- Si sobrevive, **no** se lee «el disyuntor salva 2δ»: p = 0.5 contra el 3/6 histórico. Eso lo
  contesta una tanda de semillas, no esta corrida.
- Si `T` converge a 0.0247 desde 0.0253, **no** vale como evidencia de que aprendió `T*`. Es la
  objeción de la coincidencia aritmética que la bitácora registró el 2026-09-03: 0.0253 está a
  2.3 % de `T*`. Sólo una corrida que arranque arriba y baje puede probarlo, y el 10-06 se midió
  que **no baja** —se estaciona en 0.0796 y se da vuelta en los halvings—.
- Si muere, **no** invalida el disyuntor. El contrafactual del replay corre sobre gradientes de
  `T ≈ 0.08`; a 0.0253 la ley `1/T` dice que son ~3× más grandes, así que la reserva exigida real
  puede ser mayor que el 1.88·δ medido.

**Por qué se cortó el control.** En el tramo 50 000→66 000 la seed 0 ganó +0.0159 dB con una
pendiente de +7.96e−7 dB/it, o sea **3.8 h de GPU por 0.01 dB** contra 0.2 h/0.01 dB en el tramo
25 000→50 000: 19× peor. Proyectado a la 100 000 quedaban +0.027 dB, que es 6.6× menos que un sd
entre semillas (0.164 dB) y 13× menos que el mínimo contraste detectable. Y la medición que lo
cierra, en píxeles y no en dB: la salida sobre seis imágenes de CBSD68 cambió **rms 1.135 niveles
de gris sobre 255** entre la it 18 000 y la 66 000, mientras **dos semillas distintas a la misma
iteración difieren rms 0.949**. Entrenar 48 000 iteraciones más mueve el modelo lo mismo que tirar
otra semilla. Lo único que seguía moviéndose era `λ` (+8.5 % en el tramo); `T`, la loss y la
mediana de `‖g‖` estaban planas.

**Lo que el corte cuesta, dicho explícito.** Las predicciones 1 y 5 del pre-registro del 10-05
estaban escritas sobre la it 100 000. Con la seed 0 cortada quedan apoyadas sólo en la seed 1, que
sigue viva y llega a la 100 000. Si la seed 1 también se cortara, esas dos predicciones pasan a
**ausencia de resultado** y se escriben así.

---------------------------------------------------------------------------------------------------

## 2026-10-07 — PRE-REGISTRO de `t1-ldm100k`: dos brazos de `T_init` sobre cortes de T1, con el calendario del lr comprimido

Primera tanda fuera de BSDS. El dominio nuevo son **cortes 2D de T1w sintéticas**: el dataset
`LDM_100k` (Pinaya et al. 2022, 100 000 volúmenes generados por un modelo de difusión latente
entrenado sobre UK Biobank), en `/home/santicien/Public/LDM-data/LDM_100k/LDM_100k/rawdata`.

**Dicho de frente para que no quede un número mal atribuido en el repo: NO son QSM.** Son T1w,
`sub-*/anat/sub-*_T1w.nii.gz`, uint16, 160×224×160 a 1 mm isotrópico. El pedido original decía
QSM y en `/home/santicien/Public` no hay ningún QSM. Lo que esta tanda mide es el cambio de
dominio «imagen natural → corte de cerebro», no «imagen natural → mapa de susceptibilidad». Un
mapa de QSM tiene valores con signo, otro contraste y artefactos de estriado: nada de lo que salga
acá se traslada a QSM sin volver a medirlo.

El cache lo arma `scripts/cache_ldm100k.py`: un corte por plano (sagital, coronal, axial) y por
sujeto, sujetos sorteados con semilla fija entre los 100 000, normalizado a [0, 1] por el percentil
99.5 de los voxels positivos. **8 000 sujetos × 3 planos = 24 000 cortes** de entrenamiento y
**68 cortes axiales de 68 sujetos disjuntos** para evaluar — 68 iguala el cardinal de CBSD68, así
que el costo del eval no cambia. Verificado: solapamiento de sujetos entre train y val = 0.
Fracción de tejido por corte 0.37–0.76 (media 0.62), formas 160×224, 160×160 y 224×160, todas por
encima de `PATCH_SIZE = 93`. 3.1 GB en disco.

### 1. Qué variable cambia, y la prueba de que es una sola

`T_init`, entre los dos brazos: **8·δ_Adam = 0.1011928851** y **2·δ_Adam = 0.0252982212**.

Los dos comandos difieren en exactamente `--t-init` y `--out`. Todo lo demás es idéntico y está
fijado por la misma invocación: `--mode referencia`, `--seed 0`, el mismo cache, `--channels 1`,
`λ_init = 0.1`, `σ = 25`, los siete parámetros del disyuntor v2 en sus defaults, y el mismo
calendario de lr. Van concurrentes, una por GPU, así que la contención es la misma para las dos.

**Lo que sí cambió respecto de todo lo archivado, y por eso no se puede acreditar: dos cosas a la
vez.** El dataset (BSDS400 en color → 24 000 cortes de T1 en gris) y el calendario del lr (un
halving cada 25 000 → cinco halvings de 5 000 a partir de la 25 000). La regla 4 se cumple
*dentro* de la tanda, que es donde hay contraste; contra el archivo no hay contraste de ninguna de
las dos, y el campo 8 lo repite.

`--channels 1` baja el conteo de 387 970 a **387 394** parámetros: son los 32×2×3×3 = 576 pesos de
los dos canales de entrada que sobran en la primera convolución. El resto de la red es idéntico.

### 2. Control: archivado o concurrente

**Concurrente y pareado**, que es la forma más fuerte disponible: misma máquina, dos RTX 5070 Ti
iguales, lanzadas juntas, misma contención. No hay control archivado posible porque no existe
ninguna corrida previa sobre este dataset.

### 3. La pregunta principal, y por qué no la contesta el replay

**¿Arrancar `T` pegado a `T*` sobrevive en un dominio nuevo, y cuánto PSNR da TDV sobre cortes de
cerebro?**

El replay obligatorio **no se puede correr**, y el motivo es honesto y no una excusa: `scripts/
replay_lambda_adam.py` y el `replay_t.py` del 10-06 reproducen la trayectoria de `T` a partir de
una **traza de `∂L/∂T` ya medida**, y no existe ninguna traza de `∂L/∂T` sobre datos de T1. La
primera es la de esta tanda. Replayar con la traza de BSDS contestaría la pregunta de BSDS, que ya
está contestada. Lo único que hay es la prueba de humo del 10-07 (10 iteraciones, `T_init = 8·δ`
sobre T1): `|g|` 106–167 y `T` subiendo de 0.10119 a 0.10171, o sea gradientes del mismo orden que
en BSDS. Es demasiado corto para extrapolar y queda anotado como tal.

### 4. Endpoint

**PSNR sobre los 68 cortes de evaluación a la iteración 50 000**, fija, que es lo que la skill
manda para la fase 1. Secundarios, declarados ahora: el mínimo de `T` sobre las 50 000
iteraciones, la tasa de rechazos del disyuntor, y el salto de PSNR en la primera evaluación
posterior a cada uno de los cinco halvings.

La entrada ruidosa vale **20.17 dB** en los dos dominios por construcción: con `data_range = 1` y
`σ = 25/255`, PSNR = −20·log₁₀(25/255) sin depender de la imagen. Es el único anclaje que cruza
dominios.

### 5. Predicciones numeradas

**Antes de predecir, lo que la evidencia archivada dice de `T_init`, que no es lo que estaba
escrito.** A iteración pareada en BSDS:

| corrida | `T_init` | it 2 000 | it 10 000 |
|---|---|---|---|
| `tinit-2delta-50k` | 2·δ | **31.008** | — |
| `t8delta-100k-s1` | 8·δ | 31.003 | 31.340 |
| `tinit-7p9delta-10k` | 7.9·δ | 30.993 | 31.343 |
| `t8delta-100k-s0` | 8·δ | 30.982 | 31.337 |
| `tinit-10delta-10k` | 10·δ | 30.998 | 31.352 |

`T_init` recorre un **factor 5** y el PSNR se mueve **0.026 dB** a la it 2 000 y **0.015 dB** a la
it 10 000, con el brazo de 2·δ *arriba*. El «los 2·δ quedaban en 31.10–31.19 contra 31.5 del 8·δ»
que circulaba estaba confundido con el **largo de la corrida** —10 000 iteraciones contra
100 000—, no con `T_init`. Las predicciones salen de esta tabla, no de aquella frase.

1. **Ningún brazo baja de `δ_Adam = 0.0126491` en ninguna iteración.** Umbral: `min(traza[:, 1]) >
   0.0126491` en los dos. Si falla en el brazo de 2·δ, el muro de muerte depende del dominio y el
   margen de 1.88·δ medido en BSDS el 10-06 no se puede trasladar.
2. **El contraste de PSNR a la it 50 000 es `|Δ| < 0.10 dB`.** Sale directo de la tabla: el efecto
   de `T_init` sobre el PSNR, a iteración pareada, es de orden 0.02 dB. Si falla con `|Δ| > 0.33 dB`
   —2 sd del PSNR-mejor, el mínimo detectable con n = 1— entonces en T1 `T_init` sí mueve el
   endpoint y es un hallazgo nuevo que pide semillas.
3. **El brazo de 2·δ registra una tasa de rechazos mayor que el de 8·δ.** Umbral:
   `rechazos(2δ) > rechazos(8δ)`. Fundamento medido hoy sobre BSDS: el 2·δ llevaba **54 rechazos y
   2 rebobinados en 2 750 iteraciones** (1.96·10⁻² por iteración) contra **163 rechazos y 4
   rebobinados en 100 000** del 8·δ (1.63·10⁻³) — un factor **12**. Si falla, la inestabilidad de
   arrancar en `T*` es de BSDS y no de `T_init`.
4. **PSNR a la it 50 000 ≥ 32.5 dB en los dos brazos.** Fundamento débil y dicho así: los cortes
   tienen 38 % de fondo casi plano, que es trivial de denoisear, y el único punto medido es
   21.376 dB a la it 10 contra 20.17 de la entrada ruidosa. Banda ancha a propósito. Si cae debajo,
   el problema más probable es la normalización del cache, no el modelo.
5. **El salto de PSNR en la primera evaluación posterior al halving es positivo en al menos 4 de
   los 5 halvings, en los dos brazos.** Fundamento: en BSDS los halvings de la 25 000 y la 50 000
   dieron +0.0333/+0.0295 y +0.0098/+0.0077 dB, 4 de 4 positivos y por encima de la mediana de sus
   vecinos. Si falla, cinco halvings separados por 5 000 iteraciones están demasiado juntos para
   que `v` de Adam se adapte, y el calendario comprimido no sirve.
6. **El salto decrece entre halvings consecutivos: salto(hito 5) < salto(hito 1).** Fundamento: en
   BSDS el salto cayó 3.4× y 3.8× entre el primer halving y el segundo.

### 6. Potencia

**Condicional, y mala en el eje de contraste: n = 1 por brazo.** El mínimo detectable del PSNR con
una semilla por brazo es **2 sd = 0.33 dB** (sd del PSNR-mejor = 0.164 dB), y la predicción 2 dice
que el efecto real es de orden **0.02 dB**, o sea **16× por debajo del piso**. El contraste de
`T_init` sobre el PSNR **no es medible en esta tanda** y queda **descriptivo**. En el eje de
exposición al estallido es peor: el techo medido el 2026-09-07 con 12 semillas por brazo da poder
0.11 contra 1.35×, y con una semilla no hay test de log-rank que correr.

Lo que esta tanda sí mide, y que no es un contraste entre brazos, es el **nivel absoluto** del
PSNR de TDV sobre cortes de cerebro y el **comportamiento del calendario comprimido**. Eso es
n = 1 pero no es una comparación, y por eso vale.

### 7. Costo en horas de GPU

`referencia` mide **1.347–1.385 s/it** corriendo sola. 50 000 × 1.35 s = **18.75 h por brazo**, los
dos en paralelo, uno por GPU: **~19 h de reloj y ~37.5 h de GPU**. VRAM pico medida en la prueba de
humo: **9.88 GB** por corrida, contra 16.3 GB por tarjeta. Se re-mide el s/it en la primera hora y
se anota si se va de esa banda.

### 8. Cómo NO se leen los resultados

- **Un PSNR de 34 dB en T1 no es «mejor que los 31.5 dB de BSDS».** El 38 % de fondo casi plano de
  los cortes infla el PSNR y los dos números no son comparables. La única comparación legítima
  entre dominios es contra la entrada ruidosa, que vale 20.17 dB en los dos por construcción.
- **Si los dos brazos salen a menos de 0.33 dB, no se lee «2·δ es igual de bueno».** Con n = 1 por
  brazo eso es exactamente lo que la predicción 2 anticipa y el test no tiene poder. Se lee «no se
  detectó diferencia», que es otra afirmación.
- **Si el brazo de 2·δ sobrevive, no se lee «2·δ es seguro».** La mortalidad medida el 2026-09-03
  con `T_init = 0.0253` es **4/6**, o sea **2/6 vivas**: una sola corrida que sobreviva da
  p = 2/6 = **0.33** y una que muera da p = 4/6 = **0.67**. Ninguno de los dos mide nada.
- **Si el calendario comprimido rinde, no se lee «el calendario del paper estaba mal».** No hay
  brazo con el calendario de 25 000 sobre T1, así que el contraste del calendario es contra BSDS y
  entre dominios, o sea no es un contraste. Para acreditarlo hace falta un brazo de T1 con
  `--lr-halve-every 25000`, que no está en esta tanda.
- **La tasa de rechazos del disyuntor no se lee como mortalidad.** «Evento», «muerte» y «corrida
  perdida» siguen siendo tres cosas distintas (10-05), y un rechazo es un no-op sobre el
  optimizador: no llama a `opt.step()`, así que el pico no entra en `m` ni en `v`.

### GPUs y comandos

GPU 0 y GPU 1 libres al lanzar (280 MiB de base cada una; el tren de eigen de la presentación se
mató antes y la corrida de BSDS de 2·δ se cortó a mano, ver su propia entrada).

```sh
# brazo 8 delta -> GPU 0
train.py --out runs/t1-8delta-50k --mode referencia --iterations 50000 \
    --t-init 0.1011928851 --seed 0 --channels 1 \
    --data-root datasets/ldm100k --train-splits train --eval-split val \
    --eval-list datasets/ldm100k/lista_eval.txt \
    --lr-halve-start 25000 --lr-halve-every 5000 --lr-halvings 5

# brazo 2 delta -> GPU 1
train.py --out runs/t1-2delta-50k --mode referencia --iterations 50000 \
    --t-init 0.0252982212 --seed 0 --channels 1 \
    --data-root datasets/ldm100k --train-splits train --eval-split val \
    --eval-list datasets/ldm100k/lista_eval.txt \
    --lr-halve-start 25000 --lr-halve-every 5000 --lr-halvings 5
```

Hitos del lr resultantes: **25 000, 30 000, 35 000, 40 000, 45 000**, y el lr termina en
4·10⁻⁴/32 = **1.25·10⁻⁵**. El piso del paso no es libre: Adam promedia sobre
1/(1−β₂) = 1 000 iteraciones, así que 5 000 son 5× esa memoria y el calendario sigue siendo un
calendario; debajo de ~2 500 dejaría de serlo.

---------------------------------------------------------------------------------------------------

## 2026-10-07 — VEREDICTO de `tinit-2delta-50k`: cortada en la 4 000, y el disyuntor contuvo `‖g‖` = 1.04·10¹⁰

Medición cruda en `docs/mediciones/tinit-2delta-50k.txt`. La corrida se cortó a mano en la
iteración **4 000** para reasignar la GPU a la tanda `t1-ldm100k` sobre cortes de T1. Se la dejó
llegar al eval de la 4 000 a propósito: la traza se escribe a disco recién en cada evaluación, y
cortarla antes habría perdido la fila por iteración de las dos excursiones.

**3 CONFIRMADAS, 1 PARCIAL, 3 SIN DATO por el corte. Ninguna FALLA.**

### Predicción 1 — `T` no toca cero en las 50 000 — PARCIAL

`min T = 0.024898 = 1.97·δ` sobre las 4 000 iteraciones corridas, contra la pared en
`δ = 0.0126491`. Nunca se acercó. Pero la predicción hablaba de 50 000 y se corrieron 4 000: está
confirmada en el **8 %** de la ventana y **sin dato** en el resto. El mínimo del replay
(+0.001513) caía más adelante, así que no hay contradicción — hay ausencia de medición.

### Predicción 2 — PSNR @4 000 en [31.05, 31.25] — CONFIRMADA

**31.1548 dB**, casi en el centro de la banda.

### Predicción 3 — `T` @4 000 en [0.023, 0.028] — CONFIRMADA

**0.027129**. `T` subió de 0.0252982 a un máximo de 0.029756, bajó a 0.024898 y terminó en
0.027129: se queda rondando `T*` sin fugarse para ningún lado.

### Predicción 4 — PSNR @16 000 ≥ 31.09 — SIN DATO

### Predicción 5 — PSNR @50 000 en [31.48, 31.58] — SIN DATO

### Predicción 6 — `T` @50 000 > 0.0300 — SIN DATO

La dirección es la predicha —`T` venía subiendo desde su mínimo— pero 4 000 iteraciones son el 8 %
de la ventana y la deriva de `T` se da vuelta en los halvings (medido el 10-07 sobre
`t8delta-100k`), así que extrapolar desde acá sería la cuarta extrapolación de `T` de este repo, y
las tres anteriores fallaron. No se extrapola.

### Predicción 7 — Rechazos > 18 en las primeras 50 000 — CONFIRMADA

**79 rechazos en 4 000 iteraciones**, 4.4× el umbral en el 8 % de la ventana.

### Lo que la corrida midió y nadie había pedido: el disyuntor contra 1.04·10¹⁰

Dos excursiones sostenidas, las dos contenidas, las dos de **exactamente 25 rechazos seguidos** —el
umbral de rebobinado— y las dos rebobinadas:

| evento | iteraciones | `‖g‖` | mediana congelada | salida |
|---|---|---|---|---|
| 1 | 2 171 – 2 195 | 3.57·10⁵ a **3.77·10⁹** | 63.25 | rebobinado a la 2 003 |
| 2 | 3 498 – 3 522 | 2.43·10⁸ a **1.04·10¹⁰** | 103.84 | rebobinado a la 3 ~500 |

**`max‖g‖` = 1.04158·10¹⁰ en la it 3 507: 52 800× el 1.971·10⁵** que era el peor evento sobrevivido
*escrito* en el repo (`tinit-7p9delta-10k`, 10-05).

Corrección al leer el veredicto de `t8delta-100k` el mismo día: el récord absoluto **no** es éste.
`t8delta-100k-s1` llegó a **1.295·10¹⁰** y también sobrevivió, con 4 rebobinados; simplemente su
veredicto no estaba escrito todavía. Son **dos casos independientes** de contención por encima de
10¹⁰, uno con `T_init = 8·δ` y otro con `2·δ`.

Y el número que cierra la calibración: **el `‖g‖` máximo de los lotes ACEPTADOS es 1 678.08**,
contra **3.57·10⁵** el menor de los rechazados dentro de las excursiones. Hueco de **213×** sin
solapamiento, y 1 678 queda **por debajo** del 6 113 que es la mayor excursión que se resuelve
sola. O sea que el disyuntor separó perfecto y no rechazó nada que no hiciera falta.

**El hueco publicado [6 113, 1.971·10⁵] que justifica `GUARD_ABS = 1e4` NO se retira.** Ese hueco
lo definen la mayor excursión que se resuelve sola (6 113) y la menor que necesita intervención
(1.971·10⁵); 1.04·10¹⁰ necesitó intervención y cae muy por arriba, del mismo lado. Ninguno de los
dos extremos se mueve y nada de `README.md`, `AGENTS.md`, `train.py` ni `docs/metricas.md` cambia.

### Lo que sí cambia de criterio: arrancar en `T*` no es un arranque limpio

| corrida | `T_init` | rechazos / it | tasa | rebobinados |
|---|---|---|---|---|
| `tinit-2delta-50k` | 2·δ | 79 / 4 000 | **1.98·10⁻²** | 3 |
| `t8delta-100k-s1` | 8·δ | 163 / 100 000 | 1.63·10⁻³ | 4 |
| `t8delta-100k-s0` | 8·δ | 18 / 66 000 | 2.73·10⁻⁴ | 0 |

**Factor 12 a 72.** La dispersión entre las dos semillas del control (6×) dice que la tasa es
ruidosa, así que el número honesto es el rango, no el punto. Pero el orden de magnitud no es
ambiguo: `T_init = 2·δ` es un **régimen crónicamente inestable que el disyuntor sostiene**, no un
arranque equivalente al de 8·δ. Esto es lo que quedó pre-registrado como predicción 3 de
`t1-ldm100k`.

### Y el PSNR: `T_init` no lo mueve, y lo de «31.10–31.19 contra 31.5» era largo de corrida

| it | 2·δ | `t8delta-s0` | `t8delta-s1` |
|---|---|---|---|
| 2 000 | 31.0084 | 30.9817 (**+0.0268**) | 31.0031 (**+0.0053**) |
| 4 000 | 31.1548 | 31.1908 (**−0.0360**) | 31.1717 (**−0.0168**) |

El signo del contraste **se da vuelta** entre la it 2 000 y la 4 000, y las cuatro diferencias
viven en ±0.036 dB contra un piso de detectabilidad de **0.33 dB** (2 sd del PSNR-mejor): es ruido.
Sumado a los brazos de 7.9·δ y 10·δ del 10-05, `T_init` recorre un **factor 5** y el PSNR se mueve
0.026 dB @2 000 y 0.015 dB @10 000.

**Queda corregido en el repo**: la frase que circulaba —«los `T_init = 2·δ` quedaban en 31.10–31.19
contra 31.5 del 8·δ»— confundía `T_init` con el **largo de la corrida**. Los 31.10–31.19 son
corridas de 10 000 iteraciones y los 31.5 son de 100 000; a iteración pareada no se separan.
`T_init` cambia la **exposición al estallido**, no el endpoint.

### Retirado (10-07): la tasa base de supervivencia de `T_init = 0.0253` no es 3/6

Es **4/6 de mortalidad**, o sea **2/6 vivas**, y está medido desde el 2026-09-03 en la propia tabla
`## Estado` de este archivo («Mortalidad a 4 000 iteraciones: **4/6** con `T_init = 0.0253`, **5/6**
con `T_init = 0.1`, Fisher p = 1.0»). El «3/6» lo escribí yo el 10-06 en el campo 6 del
pre-registro de `tinit-2delta-50k` y lo repetí hoy en el campo 8 del de `t1-ldm100k`, sin mirar la
tabla que estaba doce líneas más arriba.

Lo que cambia: con n = 1, una corrida que **sobreviva** da p = 2/6 = **0.33** y una que **muera**
da p = 4/6 = **0.67**, no 0.5 en los dos casos. **La conclusión no cambia** —ninguno de los dos
números mide nada— pero el número publicado estaba mal y queda retirado.

Cobertura del retiro, `grep -rn "3/6"` sobre `*.md`, `*.py` y `*.txt` fuera de `runs/`: corregidos
`docs/pendientes.md` §9.5 y el campo 8 del pre-registro del 10-07. Las dos apariciones en el
pre-registro del 10-06 (campos 6 y 8) **se dejan como están**: un pre-registro no se reescribe
después de ver los datos, que es justo lo que lo hace valer, y su corrección es esta entrada.

---------------------------------------------------------------------------------------------------

## 2026-10-07 — VEREDICTO de `t8delta-100k`: objetivo de la fase 1 cumplido en 31.547 dB, y `T` subió en vez de bajar

Medición cruda en `docs/mediciones/t8delta-100k.txt`. Predicciones leídas contra las **enmendadas**
del 10-05, no contra las originales. **4 CONFIRMADAS, 1 CONFIRMADA PERO NO INFORMATIVA (así se
registró), 1 FALLA.**

La semilla 1 terminó las 100 000 iteraciones: **31.547008 dB**, 163 rechazos, **4 rebobinados**,
37.70 h, VRAM pico 9.94 GB. La semilla 0 se cortó a mano en la 66 000 con 31.5257 dB.

### Predicción 1 — `T` @100 000 en [0.050, 0.075] — FALLA

Medido **0.082642 = 6.53·δ**, contra una banda de [3.95, 5.93]·δ. El pre-registro decía «si cae
fuera, mi modelo de deriva está mal por segunda vez», y está mal por segunda vez.

Y falla por el lado contrario al que el modelo contemplaba. La forma real de la curva:

| it | 18 000 | 50 000 | 100 000 |
|---|---|---|---|
| `T` (s1) | 0.086154 | 0.079690 | **0.082642** |

El mínimo de `T` sobre la traza entera está en la **it 49 991** (s1) y la **49 997** (s0), o sea a
nueve iteraciones del halving del lr, y **después `T` sube**. El modelo extrapolaba una deriva
negativa que se desacelera hasta aterrizar en 0.0585; lo que pasa es que la deriva **se da vuelta**
en el halving. Ninguna de las tres extrapolaciones de `T` de este repo acertó: 0.02432 (retirada el
10-05), 0.0585 (esta) y la de 66k→100k en PSNR, que salió 1.8× optimista (+0.0245 predicho contra
+0.0137 medido). **`T` deja de extrapolarse.**

### Predicción 2 — ningún brazo llega a `T < δ` — CONFIRMADA, y no informativa

`min T` = **0.078482** (s0) y **0.079627** (s1), o sea **6.2–6.3·δ**, un margen de 6×. Quedó
pre-declarada como no informativa justamente porque era casi segura, y se cumple: **`8·δ` a 100 000
no pone a prueba la pared de muerte.** No se recicla como éxito.

### Predicción 3 — la semilla 1 reproduce su trayectoria v1 hasta la it 18 000 — CONFIRMADA

Las nueve evaluaciones, una por una, con `|dif|` máxima de **0.000492 dB** —el redondeo de los tres
decimales que tenía el registro—:

| eval | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| registrado | 31.003 | 31.172 | 31.005 | 31.310 | 31.340 | 31.377 | 31.401 | 31.411 | 31.419 |
| medido | 31.003129 | 31.171660 | 31.005485 | 31.309534 | 31.340219 | 31.377077 | 31.400508 | 31.411360 | 31.419141 |

**El disyuntor v2 es inerte sobre el camino sano, medido y no argumentado.** Era la condición para
que el resto de la tanda fuera legible («si falla la 3 o la 4, nada más de la tanda es legible»).

### Predicción 4 — lo mismo para la semilla 0 — CONFIRMADA

PSNR @18 000 = **31.393218** contra 31.393 registrado, **11 rechazos** contra 11 registrados y
**0 rebobinados** contra 0. Los tres números, exactos.

### Predicción 5 — PSNR @100 000 en [31.45, 31.60] — CONFIRMADA

**31.547008 dB**, contra los 31.53 publicados por Kobler et al. **El objetivo de la fase 1 está
cumplido.**

### Predicción 6 — tras un rebobinado la razón es finita, no `nan` — CONFIRMADA

Los cuatro rebobinados de la semilla 1, con su primera iteración posterior:

| rebobinado en la it | `‖g‖` siguiente | mediana restaurada | razón |
|---|---|---|---|
| 24 796 | 13.93 | 17.01 | **0.819** |
| 27 123 | 14.54 | 13.44 | **1.082** |
| 29 419 | 10.50 | 14.67 | **0.716** |
| 99 365 | 12.98 | 13.99 | **0.928** |

Las cuatro finitas y las cuatro dentro de la banda de v1 (1.04 y 0.72). **Ésta es la predicción que
valida el arreglo del v2**: la ventana viaja con el respaldo, el rebobinado la devuelve, y el
relativo queda armado inmediatamente en vez de desarmado 100 iteraciones.

### El récord de exposición, que no estaba pedido

**`max‖g‖` de la semilla 1 = 1.295·10¹⁰**, sobrevivido con 4 rebobinados. Contra los **4 006** de
la semilla 0, que nunca cruzó ni la cota absoluta de 10⁴. **La exposición varía 3.2·10⁶× entre dos
semillas de la misma configuración**, lo que vuelve a decir —tercera vez en este repo— que la tasa
de eventos no se estima con n = 2. El peor evento sobrevivido que estaba escrito era 1.971·10⁵: el
nuevo lo supera **65 700×**.

### El costo del tramo final, medido

| tramo | ΔPSNR | h de GPU | h por 0.01 dB |
|---|---|---|---|
| 50 000 → 100 000 | +0.0301 | 18.8 | 6.3 |
| 66 000 → 100 000 | +0.0137 | 12.8 | 9.3 |
| 76 000 → 100 000 | **+0.0047** | 9.0 | **19.3** |

La mitad de la corrida compró **+0.03 dB**, que es **5.4× menos que un sd** entre semillas
(0.164 dB). El corte de la semilla 0 en la 66 000 resignó **+0.0137 dB** —el **8 %** de un sd,
inmedible— y ahorró 12.8 h de GPU: la decisión queda **validada por medición**, no por proyección.

Y en la it 100 000 los dos escalares **seguían derivando monótonamente**: `T` de 0.0797 a 0.0826 y
`λ` de 0.1130 a 0.1291 (+14 %). Los pesos ya no movían el PSNR y los dos escalares nunca se
asentaron. Eso **no** es overfitting: la pendiente del PSNR de test es **+7.96·10⁻⁷ dB/it**
(t = +4.53) contra una pendiente de la `loss` de train de −3.30·10⁻⁶ (t = −0.89, no significativa),
con correlación −0.555/−0.375 — el signo sano. El overfitting pediría correlación positiva.

### Cómo NO se leyó

Las cuatro prohibiciones del pre-registro se respetan. En particular: **los dos brazos no son
evidencia sobre `T_init`** —comparten el valor— y valen sólo como prueba de reproducibilidad, que
es lo que las predicciones 3 y 4 midieron. Y el v2 **sí** quedó validado, porque **sí** disparó
rebobinados: 4 en la semilla 1. Si no hubiera disparado ninguno, el pre-registro obligaba a
escribir «ausencia de resultado», y no hizo falta.

---------------------------------------------------------------------------------------------------

## 2026-10-08 — Cierre de la fase 1 y reordenamiento del repositorio

Entrada de registro, sin medición nueva. Deja escritas tres cosas que de otro modo habría que
reconstruir del `git log`.

### La fase 1 queda cerrada en 31.547 dB

`runs/t8delta-100k-s1`, iteración 100 000, modo `referencia`: **31.547 dB** en CBSD68 (σ = 25),
contra los 31.53 publicados. El veredicto, predicción por predicción, es la entrada del 2026-10-07
`VEREDICTO de t8delta-100k`. **El autor da por terminada la construcción del entrenador**: para la
réplica no se lanza nada más.

### `docs/pendientes.md` se elimina, y esto es lo que tenía abierto

Era la cola de la fase 1, y con la fase cerrada deja de ser una cola. Se borra con `git rm`; su
última versión se lee con `git show 8d71465:docs/pendientes.md`. Lo cerrado ya estaba en esta
bitácora —0 (`λ_init`), 1 (bf16 contra TF32), 7 (el amplificador adjunto), 8 (la transición),
9.1 (`disyuntor-100k`), 9.3 (`T_init` 0.1 contra 10·δ), 9.4 (el horizonte corto) y 9.5
(`t8delta-100k`)—, con una excepción de puntero: **`pendientes.md` era el único sitio que citaba
`docs/mediciones/transicion-adjunta.txt`**, los datos crudos de los puntos 8 y 9 del 2026-09-20.
Queda citado acá.

Lo que quedó abierto o sin correr, con su número y título originales:

- **2 — `T_init = 0.1` contra `0.0254`, con `λ_init = 0.1` en los dos.** No se corre por potencia:
  con 12 semillas por brazo, 0.11 contra la razón de hazard esperada de 1.35×. Lo que queda es de
  teoría y cuesta cero GPU: **cuál es la versión invariante de gauge del balance entre el término
  de datos y el regularizador**. El candidato escrito es `λ·‖x−z‖ / (T·‖∇R‖)`; si se sostiene, ese
  balance lo fija `λ` sola. Importa en la fase 2, que tiene que fijar ese balance en otro dominio.
- **3 — ¿precisión o `cudnn.benchmark`?** No se disparó: su condición era un efecto grande en el 1,
  que dio 1.31×. La pregunta reformulada, **«¿`referencia` es de verdad peor?»**, sigue sin medirse
  a iteración pareada. `t8delta-100k` muestra que `referencia` llega a 31.547, no que los otros dos
  modos lleguen menos. Importa sólo si la fase 2 quiere entrenar en `rapido` por costo.
- **4 — Una corrida larga**, reformulada el 2026-09-20 a PSNR a iteración fija, con su candidata
  **4.1 — extender `runs/repeat5` (seed 7, `T_init = 5·δ`) a 100k iteraciones**: ~37.2 h de GPU,
  no lanzada. Con la fase cerrada pierde el motivo.
- **5 — Registrar `‖∇R‖` por iteración.** No se hizo: `traza.npy` sigue en 10 columnas, ninguna
  de ellas `‖∇R‖` (`train.py`, la definición de `traza`). Es lo que contestaría si el hazard crece
  con el reloj o con la ganancia aprendida, y en la fase 2 es una columna barata de agregar.
- **6 — Fase 2: entrenamiento ciego al nivel de ruido**, `σ ~ U(0, 55)`. Es lo que sigue. Detalle
  en «Mejoras candidatas» de `AGENTS.md`.
- **9 — Qué selecciona el lote que dispara** (parcialmente cerrada), y dentro de ella **9.2 — el
  disyuntor de lotes**: seis semillas por brazo, `--guard inf` contra el disyuntor, pareado bit a
  bit hasta el primer rechazo. **Nunca se corrió con ese diseño.** El disyuntor corrió sólo en
  configuraciones de una o dos semillas (`disyuntor-100k`, `tinit-7p9delta-10k`,
  `tinit-10delta-10k`, `tinit-2delta-50k` y las dos de `t8delta-100k`, con sus parciales `-corte`;
  `guard` en el `config.json` de cada una), así que **sigue sin medirse si cambia la tasa de
  eventos o sólo cuál lote la dispara**, que era la objeción de la metaestabilidad.
- **9.5, «siguiente slot, no lanzada»: `rapido` a 10⁵ iteraciones con el disyuntor.** La pregunta:
  si el rebobinado rescata una corrida larga en bf16 —`replica-bf16` corrió las 10⁵ sin disyuntor
  y terminó en 20.173 dB—, una corrida de 10⁵ baja de 37.4 h a ~13 h, según la estimación que
  hacía `pendientes.md` (no medida). Importa por el costo de la fase 2.

Fuera de la cola, **la tanda `t1-ldm100k`** —cortes de T1 de LDM_100k, dos brazos de `T_init`,
8·δ y 2·δ— está **PRE-REGISTRADA** en la entrada del 2026-10-07 y **no se lanzó**. Su `plan.sh`
vive en `runs/t1-ldm100k/`, que no se versiona; si ese directorio se pierde, los dos comandos están
en el pre-registro, sección «GPUs y comandos».

### El reordenamiento, y las rutas que cambiaron

| antes | ahora |
|---|---|
| `bsds500/` | `datasets/bsds500/`, junto a `datasets/ldm100k/` |
| `scripts/cache_ldm100k.py` | `datasets/cache_ldm100k.py` |
| `checkpoints/tdv3-3-25-f32-{color,gray}.pth` y su `README.md` | `checkpoints/Oficiales/` |
| — | `checkpoints/Entrenados/`: `t8delta-100k-s1.pth` (it 100 000, 31.547 dB), `t8delta-100k-s0.pth` (it 64 000, 31.526 dB) y `replica-v4.pth` (it 68 000, 31.478 dB) |
| `docs/pipeline-explicado.txt` | `docs/presentacion/pipeline-explicado.txt` |
| `docs/pendientes.md` | eliminado; lo abierto, arriba |
| `scripts/`, `.claude/`, `CLAUDE.md` | locales: gitignorados y fuera del índice |

Los tres de `Entrenados/` son copias con las claves `config`, `model`, `iteration` y `psnr`, **sin
el estado del optimizador** (1.6 MB cada uno); los originales siguen en
`runs/t8delta-100k-s1/best.pth`, `runs/t8delta-100k-s0/best.pth` y
`runs/replica-v4/checkpoint.pth`, con sus logs. `tests/` sigue versionado: es el contrato del port.
Y `docs/README.md` es nuevo: dice qué es cada carpeta de `docs/` y qué depende de qué.

**Ojo con las rutas viejas.** Las entradas anteriores de esta bitácora, `docs/mediciones/` y
`docs/resultados/` citan `bsds500/`, `checkpoints/tdv3-…`, `scripts/…`,
`docs/pipeline-explicado.txt` y `docs/pendientes.md` como estaban el día en que se escribieron, y
**no se reescriben**: son registro. Para traducir una ruta, esta tabla. Los `scripts/` citados
existen sólo en la máquina del autor.

### Un defecto encontrado al documentar: rebobinar puede deshacer un halving del lr

**No corregido; queda escrito para que no se pierda.** `rebobina()` restaura `opt.state_dict()`, y
el lr de cada grupo viaja dentro de ese estado. `MultiStepLR` divide a la mitad el lr que encuentra
en el optimizador al llegar a cada hito, así que si un hito cae entre el respaldo y el rebobinado,
el rebobinado devuelve el lr de antes del hito y ese halving se pierde: el lr queda al doble del
calendario hasta el final de la corrida. Comprobado con un optimizador de juguete en CPU (Adam +
`MultiStepLR(milestones=[5, 10])`, respaldo en la 4, rebobinado en la 7): el lr vuelve de 2·10⁻⁴ a
4·10⁻⁴ y el hito 10 sólo lo baja a 2·10⁻⁴. **No** se comprobó en una corrida.

`t8delta-100k-s1` no está afectada: sus cuatro rebobinados (`salida.log`: 24 796 → 24 755,
27 123 → 27 005, 29 419 → 29 255 y 99 365 → 99 255) no cruzan ningún hito (25 000, 50 000,
75 000). El riesgo crece con el calendario comprimido —un hito cada 5 000 iteraciones en vez de
cada 25 000—, y el síntoma se ve en el log: la columna `lr=` de las líneas posteriores a un
`REBOBINADO` no coincide con el calendario.

### Un cambio de código que entra en este commit

El default de `T_INIT` en `train.py` pasó de `10·δ_Adam` (0.1265) a `8·δ_Adam` (0.1012), el valor
de la corrida que replicó el paper. Es un cambio del autor, hecho fuera de la sesión que reordenó el
repositorio; el README y las notas de `train.py` ya describen el default nuevo.

---------------------------------------------------------------------------------------------------

## 2026-10-08 — El calendario del lr, medido: el tramo de lr completo deja de pagar en la it 14 000

Medición sobre archivo, sin GPU, hecha para decidir el calendario de la primera tanda sobre cortes
de T1. Salida cruda en `docs/mediciones/calendario-lr.txt`; la herramienta,
`scripts/measure_calendario_lr.py`, es local.

Sobre tres corridas largas **sanas**: `t8delta-100k-s1`, `t8delta-100k-s0` y `replica-v4`. Sanas =
nunca cayeron al estado degenerado de 20.17 dB, que es la PSNR de la entrada ruidosa sin tocar. Las
otras cuatro que llegaron a 100 000 iteraciones —`replica-v5`, `replica-bf16`, `referencia-100k` y
`disyuntor-100k`— sí cayeron, y su trayectoria no informa sobre el calendario. `replica-v4` sólo
sirve hasta el primer halving: en el tramo de lr 1e-4 pierde 2.77 dB y recupera 0.77.

### Qué es nuevo acá y qué ya estaba

Ya estaba medido el **salto** de cada halving, en el pre-registro de `t1-ldm100k` del 10-07
(predicciones 5 y 6): +0.0333/+0.0295 dB en el hito de la 25 000 y +0.0098/+0.0077 en el de la
50 000, 4 de 4 positivos, con el salto cayendo 3.4× y 3.8× del primero al segundo. Esto no lo
vuelve a contar. Mide otras cuatro cosas: dónde se aplana el tramo de lr completo, cuánto vale el
**tramo entero** de cada lr, en qué parte del tramo se cobra, y la resolución efectiva del eval.

### El tramo de lr completo se aplana en la it 14 000

| | `-s1` | `-s0` | `replica-v4` |
|---|---|---|---|
| it 2 000 → 14 000 | +0.3974 | +0.4191 | +0.3282 |
| it 14 000 → 24 000 | **+0.0526** | **+0.0423** | **+0.1016** |
| ya cobrado a la 14 000 | **88 %** | **91 %** | **76 %** |

La ganancia por ventana de 2 000 iteraciones cae debajo de 0.03 dB desde la it 14 000 en las tres.
Las últimas 10 000 iteraciones del tramo compran 0.042–0.102 dB: por encima de los 0.026 dB en que
coinciden las dos semillas a iteración pareada —así que es una pérdida real, no ruido— y 6× debajo
del piso detectable de 0.33 dB con n = 1.

**25 000 no es un mínimo.** En BSDS, cortar el tramo de lr completo en 14 000 cuesta ~0.05 dB de
mediana. Dónde cae ese codo en otro dominio no está medido.

### La ganancia del tramo entero decae con razón ~0.28

| lr | tramo | ganancia (`-s1`) | razón contra el tramo anterior |
|---|---|---|---|
| 4e-4 | it 2 000 → 24 000 | +0.4499 dB | — |
| 2e-4 | 24 000 → 50 000 | +0.0638 | 0.142 |
| 1e-4 | 50 000 → 74 000 | +0.0179 | 0.281 |
| 5e-5 | 76 000 → 100 000 | +0.0047 | 0.262 |

Los tres primeros halvings compran +0.086 dB en total. Las ventanas de la cola de `-s1` (it 80 000
a 100 000) se dispersan **±0.0018 dB**, que es la resolución efectiva del eval. Extrapolando la
razón 0.28 —y queda dicho que **es una extrapolación**, en un repositorio que tiene «extrapolar `T`
o el hazard» en `## Descartado` y tres extrapolaciones propias falladas— el cuarto halving daría
~+0.0013 dB y el quinto ~+0.0004, los dos debajo de esa resolución.

### Dónde se cobra la ganancia dentro de un tramo

El calendario comprimido quiere tramos de 5 000 iteraciones en vez de 25 000. Sobre los dos tramos
que tienen las dos semillas:

| después del hito | % del tramo cobrado (`-s1` / `-s0`) |
|---|---|
| +2 000 it | 46 % / 51 % |
| +4 000 it | 55 % / 45 % |
| +6 000 it | 53 % / 61 % |
| +10 000 it | 61 % / 72 % |

**2 000 y 5 000 iteraciones por tramo cobran casi lo mismo:** la diferencia son ~7 puntos de un
tramo que vale 0.064 dB, o sea 0.004 dB. El argumento para no bajar de 5 000 no es la ganancia, es
que `1/(1−β₂) = 1 000` iteraciones es la memoria del segundo momento de Adam. Con tramos de 2 000,
cada régimen de lr dura dos memorias y `v` todavía arrastra gradientes del lr anterior cuando llega
el hito siguiente; con 5 000 son cinco. El piso defendible es 2 500.

### Lo que se decidió con esto

El calendario de la tanda de T1 quedó en **20 000 iteraciones de lr completo, hitos en 20 000 y
25 000, cierre en 30 000**, lr final 1e-4. Cede ~0.04 dB contra las 100 000 completas —+0.536 dB de
ganancia acumulada contra ~+0.496 estimados con el 52 % de cobro por tramo— por el 30 % de las
iteraciones. Las 20 000 en vez de 14 000 son el seguro contra un codo más tardío en un dominio
nuevo: esta tanda lo mide, y la siguiente puede arrancar con el número medido.

---------------------------------------------------------------------------------------------------

## 2026-10-08 — Primeras corridas sobre cortes de T1: registro de lanzamiento, no un pre-registro

**Esto no es un pre-registro y no hay que leerlo como uno.** Las dos corridas se lanzaron a pedido
del autor, con un plazo de 4 horas hasta una presentación, y este registro se escribe con ellas ya
en vuelo. Queda escrito porque la regla 8 lo pide y porque `runs/` no se versiona.

**Tampoco es una tanda controlada.** Los dos brazos difieren en **tres** cosas a la vez —modo,
`T_init` y largo—, así que ningún contraste entre ellos se acredita (regla 4). Son dos cosas
distintas: una primera medición de PSNR sobre T1, y un diagnóstico del disyuntor en el dominio
nuevo.

| | `t1-rapido-30k` | `t1-guard-2delta-5k` |
|---|---|---|
| propósito | primer PSNR sobre cortes de T1 | diagnóstico del disyuntor |
| GPU | 0 (PCIe gen 5 ×16) | 1 (PCIe gen 4 ×4) |
| modo | `rapido` | `referencia` |
| `T_init` | 8·δ = 0.1011928851 | 2·δ = 0.0252982212 |
| iteraciones | 30 000 | 5 000 |
| hitos del lr | 20 000 y 25 000 (lr final 1e-4) | ninguno |
| eval | cada 2 000, 68 cortes de `val` | cada 1 000 |

Comunes: `datasets/ldm100k` (24 000 cortes de train, 68 de val), `--channels 1` → **387 394
parámetros**, `--seed 0`, `λ_init = 0.1`, los siete parámetros del disyuntor en sus valores
publicados. El dataset es **T1w sintética** (LDM_100k, Pinaya et al. 2022), **no es QSM**.

### Desvíos contra el pre-registro de `t1-ldm100k` del 10-07

1. **El brazo de `referencia` a 30 000 no se corrió.** Necesita ~11.5 h a 1.5 h/4 000 it y el plazo
   era de 4. A la hora de la presentación habría estado en la it ~10 400, antes del primer hito.
2. **El calendario se comprimió** de 25 000 + 5 hitos cada 5 000 (50 000 it) a 20 000 + 2 hitos
   cada 5 000 (30 000 it). El fundamento está en la entrada de hoy sobre el calendario del lr.
3. **`T_init` dejó de ser la variable.** El brazo de 2·δ pasó a ser diagnóstico, no comparación.

### Dos mediciones instrumentales que salieron del lanzamiento

**Las dos GPUs negociaron enlaces distintos —la 1 PCIe gen 4 ×4 y la 0 gen 5 ×16— pero eso NO se
paga en velocidad de entrenamiento.** ~~La misma configuración en `rapido` estimó 5.6 h restantes a
la it 250 en la GPU 1 contra 3.9 h en la GPU 0, o sea ~1.3× más lenta la 1.~~ **RETIRADO el mismo
día**, más abajo: ese 1.3× es el arranque de `torch.compile`, no el régimen. El primer intento en la
GPU 1 quedó en `runs/t1-rapido-30k-gpu1-descartada` como evidencia, y relanzar costó ~500
iteraciones que no hacía falta pagar.

**El disyuntor interviene mucho más en T1 que en BSDS, en los dos brazos.** La mediana de `‖g‖`
en T1 con `T_init = 8·δ` es **18–22**, contra **~247** en BSDS a la it 250 — 14× más chica. Como la
razón 5 es relativa a esa mediana, el umbral absoluto baja con ella: **68 rechazos y 2 rebobinados
en 2 000 iteraciones**, contra 163 rechazos y 4 rebobinados en **100 000** de `t8delta-100k-s1`.
El brazo de 2·δ también: rebobinó en la it **499** (`‖g‖` = 220, vuelta a la it 253).

Sobre el efecto de `T_init` **no se concluye nada acá**. Lo que se ve es un desfase de arranque: a
la it 250 el brazo de 8·δ llevaba 28 rechazos y un rebobinado y el de 2·δ **cero**, con medianas de
18–22 contra **34.3**; pero a la it 499 el de 2·δ rebobinó igual. O sea que la diferencia medida es
**cuándo empieza**, no si pasa. La dirección es además la opuesta a BSDS, donde 2·δ exigía 12–72×
más intervención, y eso basta para no extrapolar la calibración de un dominio al otro — pero con
una semilla por brazo y tres variables distintas entre ellos (regla 4) no hay atribución posible.
**La causa de la diferencia de tasa NO está medida.** Los dos mecanismos que esta entrada propuso
—el nivel de la mediana y la heterogeneidad de los lotes— se midieron el mismo día y los dos
resultaron falsos; el detalle está en la entrada de retirada más abajo.

### RETIRADAS el 2026-10-08, el mismo día: las tres explicaciones de la tasa de rechazo en T1

Las tres se publicaron y las tres se midieron después, con verificación adversaria. Crudo en
`docs/mediciones/nivel-g-t1-vs-bsds.txt`, `umbral-guard-escala.txt` y `dispersion-g.txt`.

**1. «La mediana móvil se asienta 5.4× más abajo en T1 (~10 contra ~54).» MAL ESCALADA.** Los dos
números son correctos y están pareados (it 1 500–2 000, cociente 5.418), pero **ninguna de las dos
corridas está asentada ahí**: `t8delta-100k-s1` sigue cayendo 4.2× después, hasta **13.14** sobre
it 50 000–100 000, y de T1 no hay traza más allá de la it 12 000. Y el cociente no es un nivel: en
ventanas pareadas de 2 000 baja monótono **6.99× → 3.05×** dentro de los datos que ya había en
disco. Elegir 5.4× fue elegir una fila de esa curva.

**Y el 3.05× que queda es escala de canales, no dominio.** La `loss` suma 93·93·C valores con C = 1
en T1 y C = 3 en BSDS; el cociente de la loss es **plano en 0.317–0.329** en las seis ventanas y
1/3 = 0.3333 (`docs/metricas.md` ya lo registra: no hacer nada vale 83.1 en gris y 249.4 en color,
cociente 0.3332). Corrigiendo T1 por ×3, el cociente de la mediana móvil va **2.331 → 1.016** y a
la it 10 000–12 000 **no queda brecha de dominio medible**. Lo que la escala no explica queda
confinado a las primeras ~4 000 iteraciones.

**2. «El umbral efectivo queda en ~50 en T1 contra ~270 en BSDS, y eso explica la tasa.» FALSA, y
refutada en signo.** **El guard relativo es invariante de escala por construcción**: decide
`gnorm > ratio · centro` con `centro` = mediana de las últimas 100 normas aceptadas de la propia
corrida, o sea `gnorm/centro > 5`, adimensional. Reescalar toda la serie de `‖g‖` por `c` reescala
`centro` por `c` y deja el set de rechazos **idéntico bit a bit** —verificado reimplementando el
guard sobre `‖g‖` de T1 por 0.1, 1, 5.4 y 27—. El «umbral efectivo en `‖g‖`» **es** la escala de la
corrida, y la escala es exactamente lo que el cociente cancela: no puede explicar ninguna
diferencia de tasa.

El contrafactual además corre al revés: a un umbral **absoluto** fijo de 273, en it 0–2 000 T1
tiene **25** lotes por encima y BSDS **172** (s1) y **194** (s0). Y la prueba pareada dentro de s1
lo mata: su umbral efectivo cae 6.6× a lo largo de sus 100 000 iteraciones (404 → 62), pasa por el
valor de T1, y a la it 80 000–90 000 —umbral 63.5, un 10 % por encima del de T1— rechaza **3 de
10 000 = 0.030 %** contra **68 de 2 000 = 3.4 %** de T1: **113× menos al mismo umbral**. Sobre las
50 ventanas de s1, Spearman(umbral, rechazos) = **−0.438**: las de umbral bajo rechazan *menos*.

El «0 rechazos en BSDS» de la tabla retirada era además **una semilla elegida**: `t8delta-100k-s0`
rechaza **6** en esa misma ventana, y lo imprime la propia medición que la entrada citaba.

**3. «Los lotes de T1 son más heterogéneos porque el cache se armó con la mayor variedad posible, y
eso ensancha la dispersión de `‖g‖`.» FALSA, premisa y consecuencia.** La consecuencia: en las
cinco ventanas pareadas asentadas, T1 queda **en o por debajo** de la banda entre las dos semillas
de BSDS en los cuatro estadísticos de dispersión relativa (`p99/p50`, `p95/p50`, `p75/p25`,
`MAD/p50`). No hay ensanchamiento. La premisa: armando 512 lotes de 32 parches con el mismo
muestreo de `PatchDataset2D`, los lotes de T1 son **1.7–3.7× MENOS** heterogéneos que los de BSDS,
entre lotes y dentro del lote. 24 000 cortes de la misma modalidad, mismo normalizado y misma
anatomía son más uniformes por parche que 400 fotos naturales: **la variedad está en el número de
imágenes fuente, no en el contenido del parche**. Era mi decisión de diseño y la leí al revés.

### Lo que queda en pie, y el puntero que sí quedó

- **La diferencia de tasa es real**: 68 rechazos en 2 000 iteraciones en T1 contra 0 y 6 en las dos
  semillas de BSDS, y 5.9 % de iteraciones tiradas por rebobinado contra 0.0–0.4 %.
- **El signo de la diferencia de medianas se sostiene**: T1 queda por debajo en las seis ventanas
  pareadas sin cruzarse nunca, y las dos semillas de BSDS coinciden entre sí al 1–6 %, así que no es
  efecto de semilla. Pero es sobre todo escala de canales.
- **La causa de la tasa no está medida.** El puntero que dejó la verificación: las excursiones
  llegan en ráfagas de **exactamente `GUARD_SEGUIDOS` = 25**, que es imposible bajo lotes raros
  independientes —es el contador del propio disyuntor, no la data—, y en esas filas la `loss` queda
  **normal** mientras `‖g‖` sube hasta ×428. Un lote patológico subiría las dos cosas. El residuo
  de la loss explica **~1 %** de la varianza de `log‖g‖`, en los dos dominios. O sea que lo que hay
  que buscar está en la dinámica, no en el lote.
- **El pareo por índice de iteración es válido**: progreso efectivo 93.9 % en T1 contra 100.0 % en
  BSDS en it 0–12 000, o sea un desfase máximo de ~730 iteraciones ≈ 1.5 % de nivel.

Y **es del dominio, no de `T_init`**: a la it 500 los dos brazos de T1 llevan exactamente **28
rechazos y 1 rebobinado**, con `T_init` distinto por un factor 4.

Dos controles que impiden sobre-leerlo:

- **El solapamiento no es de T1.** En T1 el mayor `‖g‖` aceptado (205.4) está *por encima* del
  menor rechazado (96.81), razón 0.47×. Pero `t8delta-100k-s0` da **0.51×** en la misma ventana:
  es una propiedad de comparar contra una mediana **móvil**, no un defecto del dominio. El «hueco
  de 213×» publicado se midió sobre una corrida entera, no sobre las primeras 2 000 iteraciones.
- **La cota absoluta sigue sin trabajo propio:** 0 lotes por encima de 1e4 en las 2 000 iteraciones
  de T1, consistente con las 226 000 de `docs/mediciones/cota-absoluta-226k.txt`.

> **RETIRADO el 2026-10-08, el mismo día que se escribió.** Acá decía: «el hueco que calibra el
> disyuntor en BSDS es [6 113, 1.971e5], y tanto el `‖g‖` = 1.37e3 que disparó el segundo
> rebobinado como el `max‖g‖` = 5.51e3 caen debajo de 6 113, o sea dentro de la banda que en BSDS
> se resuelve sola» — y de ahí que el disyuntor estuviera interviniendo de más. **Es falso, y por
> el error que `AGENTS.md` ya nombra:** ese hueco está en `‖g‖` **absoluto**, y comparar `max‖g‖`
> entre configuraciones con escalas de gradiente distintas está prohibido en este repo. Normalizado
> por la mediana: el techo que se resuelve solo en BSDS es 6 113/54 = **113× la mediana**, y la
> excursión de T1 es 5 514/10 = **551×**, o sea **4.9× más allá** de ese techo —equivalente a
> ~29 800 en unidades de BSDS, dentro de la banda que pide intervención—. **El disyuntor rebobinó
> bien.**

También queda retirada la lectura de «transitorio de arranque». La tasa de rechazo por ventana de
500 iteraciones es **28, 15, 0, 25**: viene en ráfagas, no decae. Se había dicho que «la tormenta
se cortó sola» mirando sólo las it 250–500.

### RETIRADA el 2026-10-08, el mismo día: el «~1.3× más lenta la GPU 1 en `rapido`»

Crudo en `docs/mediciones/velocidad-dos-tarjetas.txt`.

**El 1.3× es arranque de `torch.compile`, no régimen.** El ETA que imprime el entrenador es
`(time.time() − t0) / it × (iterations − it)`: un promedio **acumulado**, así que a it chica el
arranque lo domina. Los dos logs tienen ETA a la **misma** iteración, lo que parecía hacer la
comparación válida, pero el cociente GPU1/GPU0 **cambia con la iteración** —1.44× a la it 250 y
1.21× a la it 500—, que es justamente lo que no puede pasar si fuera una propiedad de la tarjeta.
Despejando `acum(it) = W/it + s` con los dos puntos de cada corrida: la GPU 0 arrancó en **4.1 s** y
la GPU 1 en **58.1 s**, y el régimen queda en **0.4555 contra 0.4451 s/it**, o sea la GPU 1 un
**2.3 % más rápida** — y esos 0.0105 s/it caen **debajo** del cuanto de redondeo del ETA
(0.0122 s/it), así que ni el signo está medido.

Concuerda con la única medición hecha a propósito, `docs/mediciones/dos-gpu.txt` del 09-20: 250
iteraciones de `referencia`, seis celdas, **370.2 s la GPU 0 contra 367.1 s la GPU 1** —0.9 %, y la
GPU 1 es la rápida porque la 0 maneja el display—. Ahí la CPU quedó **92.8 % ociosa** con las dos
tarjetas al 93–95 %: el pipeline de datos no es el cuello, y por eso el enlace más angosto tiene
holgura de sobra. El hecho de hardware **no** se retira: la 1 negoció PCIe gen 4 ×4 y la 0 corre
gen 5 ×16, 8× de ancho de banda en el papel. Lo que se retira es la inferencia de ahí a s/it.

**Lo que queda sin medir** es si el hueco por tarjeta es mayor en `rapido` que en `referencia`.
`referencia` tiene su 0.9 %; `rapido` no tiene ninguna medición limpia. El argumento de que debería
pesar más —`rapido` mueve ~3.5× más lotes por segundo sobre el mismo enlace— es razonable y **no es
evidencia**: se mide con `scripts/bench_dos_gpu.sh --mode rapido` y las dos tarjetas libres.

**Costo del error:** relanzar en la GPU 0 tiró ~500 iteraciones de `runs/t1-rapido-30k-gpu1-descartada`.

### Cómo NO se leen estos resultados

- **Un PSNR alto en T1 no es «mejor que los 31.547 dB de BSDS».** Los cortes tienen ~38 % de fondo
  casi plano, trivial de denoisear, y el único anclaje entre dominios es que la entrada ruidosa da
  20.17 dB en los dos por construcción.
- **El calendario comprimido no se acredita**: cambió junto con el dataset y el modo.
- **El contraste `rapido` contra `referencia` no existe en esta tanda**: el brazo de `referencia`
  es de 5 000 iteraciones y otro `T_init`.
- **Riesgo conocido, 6–13 % de probabilidad:** ~~46 %, con 2 rebobinados en 1 619 iteraciones —1
  cada 810—~~ **corregido el mismo día: ese 810 se midió entero dentro de la tormenta de arranque.**
  Con los 5 rebobinados reales de la corrida (it 241, 1 619, 6 337, 7 206, 15 284) en 17 500
  iteraciones la tasa es 1 cada 3 500, y 1 cada 5 167 si se saca la tormenta (3 en las it
  2 000–17 500). Con 2 × 250 iteraciones de ventana expuesta tras los hitos de 20 000 y 25 000,
  `1 − exp(−500/3 500)` = **13.3 %** y `1 − exp(−500/5 167)` = **9.2 %**. Esos 250 son además una
  **cota superior**: `GUARD_RESPALDO` es cada cuánto se *intenta* el respaldo, y la edad mediana del
  que de verdad se usa es **118** sobre 15 rebobinados sanos —sólo 2 de 15 pasan de 250—, lo que
  baja la probabilidad central a **6.2 %** (`docs/mediciones/edad-del-respaldo.txt`). El rango
  honesto es **6–13 %** (defecto documentado hoy,
  `rebobina()` restaura el lr con `opt.state_dict()`). Es n = 1. No se mitigó: costaría 14 min de un
  presupuesto de 3.2 h para comprar
  0.01–0.03 dB, que es lo que vale un halving de 5 000 iteraciones. Se detecta en el log, por la
  columna `lr=` posterior a un `REBOBINADO`, y el calendario efectivo se reconstruye exacto.

---------------------------------------------------------------------------------------------------

## 2026-10-08 — El rebobinado descarta el 72 % de las iteraciones en el peor rincón de T1

Medición sobre los logs y las trazas de las dos corridas de hoy contra las archivadas de BSDS.
Crudo en `docs/mediciones/rebobinados-t1.txt` y `docs/mediciones/escalares-t1.txt`.

### La tasa que importa es la de rebobinado, no la de rechazo, y nunca se había contado así

Un lote rechazado cuesta **una** iteración: no se llama a `opt.step()` y se sigue. Un **rebobinado**
cuesta hasta `GUARD_RESPALDO = 250`, porque devuelve los pesos a un respaldo anterior y tira todo
lo aprendido desde él. Contando el par (iteración del rebobinado, iteración del respaldo) de cada
evento:

| corrida | dominio | `T_init` | modo | it | reb | 1 cada | **% tirado** |
|---|---|---|---|---|---|---|---|
| `t8delta-100k-s0` | BSDS | 8·δ | `referencia` | 67 750 | 0 | — | **0.0 %** |
| `t8delta-100k-s1` | BSDS | 8·δ | `referencia` | 100 000 | 4 | 25 000 | **0.4 %** |
| `tinit-7p9delta-10k` | BSDS | 7.9·δ | `referencia` | 10 000 | 2 | 5 000 | **2.0 %** |
| `t1-rapido-30k` | **T1** | 8·δ | `rapido` | 3 500 | 2 | 1 750 | **9.7 %** |
| `tinit-2delta-50k` | BSDS | 2·δ | `referencia` | 4 000 | 3 | 1 333 | **13.2 %** |
| `t1-guard-2delta-5k` | **T1** | 2·δ | `referencia` | 1 000 | 3 | 333 | **72.2 %** |

Los dos factores empujan en la misma dirección y **se componen**: el peor rincón —T1 con `T_init`
en el piso de la regla 2-δ— descarta **72 % de sus iteraciones**. Su primer eval da 30.087 dB a la
iteración nominal 1 000, pero su iteración **efectiva** es ~278.

### Lo único atribuible a una variable sola

`tinit-2delta-50k` y `t1-guard-2delta-5k` difieren en **exactamente** el dominio: las dos en
`referencia`, las dos con `T_init = 2·δ`, las dos con `--seed 0`, los dos con el disyuntor en sus
valores publicados. **13.2 % contra 72.2 %**, o sea 5.5×.

Pero **n = 1 por celda, y la regla 6 dice que eso mide suerte, no configuración.** La prueba está
en esta misma tabla: `t8delta-100k-s0` y `-s1` son la *misma* configuración y dan 0 rebobinados en
67 750 iteraciones contra 4 en 100 000. Un conteo de eventos con una semilla no es una tasa. Lo que
sí queda establecido es que **el peor rincón existe y es severo**, que es información suficiente
para no gastar GPU en T1 con `T_init` bajo.

La fila de `t1-rapido-30k` además está **confundida con el modo**: es la única en `rapido`.

### Anatomía de la excursión de 9.57e4, con el disyuntor conteniéndola

`t1-guard-2delta-5k`, it 718–753, con la mediana móvil **congelada en 49.6–49.7** durante todo el
evento:

| it | `‖g‖` | razón contra la mediana | `seguidos` |
|---|---|---|---|
| 718 | 3 588 | 72× | 1 |
| 719 | 2 715 | 55× | 2 |
| 720 | **95 716** | **1 926×** | 3 |
| 722–724 | 896, 872, 1 199 | 18–24× | 1, 2, 3 |
| 726–727 | 373, 658 | 7.5–13× | 1, 2 |
| 729–731 | 14 931, 16 814, **88 006** | 300–**1 771×** | 1, 2, 3 |
| 753 | 2 920 | → **REBOBINADO a la it 503** | |

Tres cosas que esto mide, ninguna nueva en BSDS pero las tres nuevas en T1:

1. **Los lotes aceptados dentro de la excursión reinician `seguidos`.** Mirá la columna: llega a 3
   y se reinicia, tres veces. Entre las it 720→722, 724→726 y 727→729 hubo lotes **aceptados**. Por
   eso el rebobinado disparó en la 753 y no en la 731. Estaba medido en BSDS (4 lotes aceptados
   dentro de una excursión letal en `tinit-7p9delta-10k`).
2. **La cota absoluta sigue sin trabajo propio.** El `‖g‖` = 95 716 cruza 1e4, pero su razón es
   1 926 ≫ 5: el relativo lo agarra igual.
3. **`T` casi no se movió: 0.02738 → 0.02724, el −0.5 %**, y quedó a **2.15×** de la pared de
   muerte (`δ_Adam` = 0.01265). Un pico desplaza a cualquier parámetro **un `δ` entero** bajo Adam,
   sin importar su tamaño: eso habría dejado `T` en 0.0146, a 1.15× de la pared. No pasó porque los
   lotes rechazados nunca llaman a `opt.step()` y el rebobinado deshizo el resto. **Es el mecanismo
   del disyuntor medido directo sobre el escalar que protege.** La loss tocó 13.365 y volvió a
   9.248.

### Los dos escalares en el dominio nuevo

Primeras 2 000 iteraciones, `T_init = 8·δ` en las tres:

| | T1 | BSDS `-s1` | BSDS `-s0` |
|---|---|---|---|
| deriva de `T` | **+0.00836** | +0.00194 | +0.00191 |
| `λ` final | 0.06093 | 0.03666 | 0.02875 |
| `w = λ/T` | **0.5562** | 0.3555 | 0.2789 |
| mediana `‖g‖` | **12.89** | 83.27 | 84.06 |
| mediana `\|∂L/∂T\|` | 4.63 | 22.68 | 24.01 |
| fracción `\|∂L/∂T\|/‖g‖` | 0.360 | 0.272 | 0.286 |

- **`T` deriva 4.3× más rápido y hacia arriba**, o sea se *aleja* de la pared. Y como `max‖g‖`
  escala como `1/T`, una `T` que sube encoge la escala del gradiente: realimentación
  **estabilizante**. En este eje T1 es más seguro que BSDS, no menos.
- **La fracción `|∂L/∂T|/‖g‖` da 0.360 contra 0.272–0.286.** `AGENTS.md` ya registra que a
  `T ≈ 0.1` esa componente es el 27–36 % de la norma: T1 cae en el techo de la banda y BSDS en el
  piso. **No hay régimen nuevo.**
- **`w = λ/T` es 1.75× más alto en T1.** Toca el punto 2 que quedó abierto al cerrar la fase 1 —la
  versión invariante de gauge del balance entre el término de datos y el regularizador—: el balance
  que Adam elige **depende del dominio**. Sale de trazas, sin GPU.
- Confundido con el modo: la columna de T1 es `rapido` y las dos de BSDS son `referencia`. Juicio,
  no medición: el modo no explica un factor 6.46 en la magnitud del gradiente, porque bf16 contra
  TF32 dio 1.31× de hazard y la precisión cambia los bits de mantisa, no la escala. Desconfundirlo
  pide un `referencia` + 8·δ sobre T1, que no está corrido.

---------------------------------------------------------------------------------------------------

## 2026-10-08 — DEFECTO del disyuntor v2: el rebobinado puede entrar en un lazo del que no sale

> **RETIRADA LA ATRIBUCIÓN el 2026-10-08, el mismo día; el arreglo se aplicó ese día en la rama
> `3D-Training` y en `main` el 2026-10-09.** El lazo existe y sus conteos (período de 26, respaldo
> de la 1 756, 45 rebobinados) son correctos, pero **la causa no es la ventana**: es que
> `rebobina()` le entrega el respaldo a `opt.load_state_dict`, que **adopta** sus tensores en vez de
> copiarlos. Desde el segundo rebobinado al mismo respaldo vuelven los pesos y la ventana, pero Adam
> sigue como estaba. Medido en `docs/mediciones/alias-respaldo-adam.txt`; el detalle, en la entrada
> «El lazo era Adam, no la ventana», y los arreglos en «`rebobina()` arreglado» (10-08, rama
> `3D-Training`) y en la del 2026-10-09 (`main`). **Y dos números de esta entrada no cuadran con la
> traza**: los rechazados de los ocho ciclos van de **94.5 a 2 497.6** (razones 5.20 a 137.5), no de
> 101.1 a 173.3 —ése es el rango del log, que imprime sólo los tres primeros de cada racha—; y `T`
> queda en **0.027360**, **2.16×** su pared, no en 0.0271 y 2.14× (`estado.json`; 2.17× sale de
> redondear `T` antes de dividir).

**El arreglo de v2 es la causa, y el mecanismo está medido entero.** Crudo en
`docs/mediciones/lazo-disyuntor-t1.txt`; la herramienta, `scripts/measure_lazo_disyuntor.py`.

Encontrado en `runs/t1-guard-2delta-5k` —cortes de T1, `T_init = 2·δ`, `referencia`, seed 0, el
disyuntor en sus valores publicados—, que a partir de la it 1 988 rebobina **al mismo respaldo**
cada **26 iteraciones**, que es `GUARD_SEGUIDOS + 1`:

| rebobinado | it | espaciado | → respaldo | `‖g‖` |
|---|---|---|---|---|
| 6º | 1 988 | 166 | 1 756 | 2 310 |
| 7º | 2 014 | **26** | 1 756 | 129 |
| 8º | 2 040 | **26** | 1 756 | 130 |
| 9º | 2 066 | **26** | 1 756 | 111 |
| 10º | 2 092 | **26** | 1 756 | 113 |
| 11º | 2 118 | **26** | 1 756 | 102 |
| 12º | 2 144 | **26** | 1 756 | 94.5 |
| 13º | 2 170 | **26** | 1 756 | 115 |

### El lazo

1. `rebobina()` restaura los pesos **y la ventana de la mediana móvil** del respaldo. Eso es el
   arreglo de v2 del 2026-10-05, que entró porque borrar la ventana dejaba el relativo desarmado
   100 iteraciones justo al salir de un evento.
2. La mediana restaurada vale **18.2** y queda **CONGELADA** ahí los ocho ciclos. La escala de
   gradiente real de la corrida es ~110, así que el umbral del relativo —`5 × 18.2 = 91.0`— lo
   cruza **todo** lote: los rechazados van de **101.1 a 173.3**, razones de **5.55 a 9.52**.
3. **Un lote rechazado no actualiza la mediana**, porque la ventana sólo ve los aceptados. La
   mediana no puede subir.
4. `seguidos` llega a 25 → rebobina → vuelve la misma mediana de 18.2 → al paso 2. **No hay salida
   por dentro del mecanismo.**

> **RETIRADO el 2026-10-08:** «la escala de gradiente real de la corrida es ~110». En los pesos
> restaurados `‖g‖` vale **5.1 a 32.2** —el primer lote tras cada uno de los 38 rebobinados
> repetidos analizados, de 40—, o sea la escala de la mediana de 18.2: la ventana describía bien
> los pesos con los que volvía. Los 94.5–2 497.6 son `‖g‖` **después de un paso** dado con el Adam
> que no se restauró.

El efecto: a la iteración nominal 2 196 los pesos son los de la it **1 756**. La corrida no muere
—`T` queda en 0.0271, a **2.14×** de su pared— y su último eval da 30.477 dB, que es el estado de
la 1 756. Simplemente deja de avanzar.

### Por qué BSDS nunca lo tocó, y qué lo dispara

La validación de v2 (entrada del 2026-10-05) midió que el primer `‖g‖` post-rebobinado valía
**48.5** y **21.3** contra centros congelados de **46.71** y **29.53**: razones **1.04** y **0.72**,
lejos del 5. El arreglo se probó justamente donde la razón post-rebobinado es ~1.

El lazo necesita que la razón post-rebobinado sea **> `GUARD_RATIO`**, o sea una mediana restaurada
muy por debajo de la escala real. Eso es lo que da T1, donde la mediana móvil se asienta en ~10–18
contra ~54 en BSDS (`docs/mediciones/disyuntor-t1.txt`). **`GUARD_VENTANA = 100` y
`GUARD_RESPALDO = 250` también importan**: con el respaldo 250 iteraciones atrás, la ventana que
viaja con él puede describir un régimen que ya no existe.

**No se corrige acá** —hay una tanda en vuelo (regla 2) y el arreglo pide decidir qué hacer con la
ventana, no un parche—. Las dos direcciones obvias, sin medir ninguna: reconstruir la ventana con
los lotes **rechazados** cuando la racha pasa de `GUARD_SEGUIDOS` (o sea, admitir que la escala
cambió), o detectar el rebobinado repetido al mismo respaldo y ensanchar la razón sólo ahí. La
primera contradice el motivo por el que la ventana viaja; la segunda es un caso especial.

> **RETIRADO el 2026-10-08:** «el lazo necesita una mediana restaurada muy por debajo de la escala
> real, y eso es lo que da T1». BSDS nunca rebobinó dos veces al mismo respaldo (0 de 9 rebobinados
> en tres corridas), así que nunca pudo tocar el defecto; en T1 con 2·δ los eventos son lo bastante
> seguidos para que dos caigan sobre el mismo respaldo. El dominio explica la **frecuencia** de los
> rebobinados, no el lazo. Y las dos direcciones de arreglo quedan **descartadas por la regla 1**:
> las dos le harían aceptar al guard lotes medidos en pesos empujados por un Adam viejo.

### Un error de contabilidad propio, corregido

`scripts/measure_rebobinados_t1.py` sumaba `it − respaldo` por rebobinado. Dentro de un lazo el
respaldo no se mueve y esa resta **crece** en cada ciclo (258, 284, 310, 336, …), hasta sumar más
iteraciones que las que tiene la corrida. El costo real no puede pasar de lo ganado desde el
rebobinado anterior: `min(it − respaldo, it − rebobinado_anterior)`. Corregido, y es **no-op** para
las corridas sin lazo —las cuatro de BSDS dan 0.4 %, 0.0 %, 13.2 % y 2.0 % igual que antes—, así
que el **72.2 %** publicado hoy más arriba, calculado a la it 1 000 y antes del lazo, no cambia.
Con el contador bueno: **14 rebobinados y 64.9 % de las iteraciones tiradas** a la it 2 000.

### La prueba de que el lazo no tiene salida: la ventana es cerrada bajo su propio filtro

Añadido el mismo día, con la traza del lazo ya en disco. Sobre it 1 988 → 3 000, 1 012 iteraciones:

| | valor |
|---|---|
| rechazados | 973 = **96.1 %** |
| aceptados | 39, o sea **25.9 iteraciones por aceptado** — exactamente 1 por ciclo de 26 |
| mediana móvil | 18.17 a 18.61: **3 valores distintos** en 1 012 iteraciones |
| umbral `5 × mediana` | 90.8 a 93.1 |
| `‖g‖` de los **aceptados** | **5.1 a 32.2** |
| `‖g‖` de los **rechazados** | 94.5 a 2 497.6 |

El guard acepta un lote si `‖g‖ ≤ GUARD_RATIO × mediana(ventana)`, y la ventana se arma **sólo con
los aceptados**. Entonces todo valor de la ventana es, por construcción, ≤ el umbral que esa misma
ventana define: **la ventana es cerrada bajo su propio filtro**. Cuando la mediana baja lo
suficiente para que el lote típico la cruce, los únicos que entran son los atípicamente chicos —acá
de 5.1 a 32.2 contra rechazados de 94.5 a 2 497.6—, y ésos la mantienen baja. Es un **punto fijo**,
no un transitorio: 39 lotes aceptados movieron la mediana **0.44** en 1 012 iteraciones.

Eso cierra la duda que había quedado abierta más arriba. El mecanismo de renovación de la ventana
—`GUARD_VENTANA = 100` ranuras y 1 aceptado por ciclo— **no puede** subir la mediana, porque los
lotes que lo alimentan son justamente los que están por debajo del umbral.

**Y precisa la atribución a v2.** El punto fijo es una propiedad del guard *relativo*; lo que lo
vuelve permanente es que `rebobina()` **restaura** la ventana baja en vez de reconstruirla.
Razonamiento, no medición —no se corrió el código pre-v2—: antes, la ventana se borraba y el
relativo quedaba desarmado 100 iteraciones, en las que entraba todo, incluidos los lotes grandes,
y la mediana subía. **Las 100 iteraciones desarmadas que v2 vino a corregir eran la salida.** v2
cambió un modo de fallo —quedar desprotegido justo al salir de un evento— por otro: quedar trabado.

> **RETIRADO el 2026-10-08:** que la renovación de la ventana sea lo que traba la corrida, y que
> «las 100 iteraciones desarmadas eran la salida» (nunca se midió). La clausura de la ventana bajo
> su propio filtro es una propiedad cierta del guard relativo, pero en el lazo los lotes que
> rechaza son **legítimamente malos**: los mide en pesos que un Adam sin restaurar sacó del
> respaldo. Con Adam restaurado, los 19 rebobinados del corpus que fueron los primeros a su
> respaldo dan **0 rechazos** en las 25 iteraciones siguientes.

### El costo, medido, y el corte

PSNR **30.4775 dB a la it 2 000 y 30.4712 a la it 3 000**: **−0.0063 dB en mil iteraciones
nominales**. Progreso cero. Cortada a mano en la it 3 000 para liberar la GPU
(`runs/t1-guard-2delta-5k/estado.json`, `"terminada": false`, con su `motivo`): **45 rebobinados,
41 al mismo respaldo**, tasa de rechazo global **39.7 %** (1 192 de 3 000), pesos de la it **1 756**.
El régimen antes del lazo era 11.0 % de rechazo.

**No murió.** `T` terminó en 0.02736 y su mínimo fue 0.02479, **1.96×** su pared, y `max‖g‖` llegó
a 95 716. El disyuntor protegió el escalar que tiene que proteger y a la vez trabó la corrida: las
dos cosas a la vez, y por el mismo mecanismo.

---------------------------------------------------------------------------------------------------

## 2026-10-08 — El lazo era Adam, no la ventana: `opt.load_state_dict` comparte los tensores del respaldo

Medición sin GPU, sobre trazas y logs archivados más un ejemplo mínimo en CPU. Crudo en
`docs/mediciones/alias-respaldo-adam.txt`; la herramienta, `scripts/measure_alias_respaldo.py`
(local). Corrige la atribución de la entrada anterior: el lazo existe, pero no lo arma la ventana.
Encontrado el 10-08 trabajando en `3D-Training`; `main` arrastró el defecto hasta el 2026-10-09.

### El mecanismo

`respalda()` guarda `copy.deepcopy(opt.state_dict())`, y `rebobina()` se lo pasa tal cual a
`opt.load_state_dict`. PyTorch no copia los tensores de estado al cargar: `exp_avg` y `exp_avg_sq`
pasan por `.to(dtype, device)`, que devuelve **el mismo objeto** si ya coinciden, y `step` se
devuelve tal cual. Desde ese momento `exp_avg`, `exp_avg_sq` y `step` del optimizador **son** los
del respaldo, y cada `opt.step()` aceptado los muta en el lugar. Un segundo rebobinado al mismo
respaldo restaura pesos y ventana, pero deja a Adam como estaba en el instante del rebobinado.
Verificado en torch 2.13.0+cu130:

| tensor | mismo objeto tras `load` | respaldo mutado por los pasos | el 2º rebobinado lo devuelve |
|---|---|---|---|
| `step` | sí | sí | **no** |
| `exp_avg` | sí | sí | **no** |
| `exp_avg_sq` | sí | sí | **no** |

Los pesos no tienen el problema: `model.load_state_dict` copia con `param.copy_`. La ventana
tampoco: `escala[:] = respaldo[2]` copia los elementos.

### La firma en las trazas

57 de los 59 rebobinados de las cinco corridas con traza que rebobinaron —quedan afuera los dos
últimos de `t1-guard-2delta-5k`, en la 2 976 y la 3 002, que no tienen 25 iteraciones después—,
separados por si eran el primero a ese respaldo (Adam restaurado de verdad) o uno repetido (vuelven
los pesos y Adam no):

| | n | `‖g‖` en la it siguiente | `‖g‖` una it después | rechazo en las 25 siguientes |
|---|---|---|---|---|
| primero a su respaldo | 19 | 16.4 | **24.4** (0.9× la mediana restaurada) | **0 %** en los 19 |
| repetido | 38 | 13.9 | **210.7** (11.6× la mediana restaurada) | **96 %** en los 38 |

- **El primer `‖g‖` es sano en las dos clases**: en los pesos restaurados el lote típico cae en la
  escala de la ventana que volvió con ellos. Lo que separa las clases es **un paso de Adam**.
- **El lazo de `t1-guard-2delta-5k` arranca exactamente en el primer rebobinado repetido**: el 5º
  (it 1 822, primero al respaldo de la 1 756) siguió sano —1 823 a 1 834 con `‖g‖` de 5.7 a 47.6—
  y el 6º (it 1 988, segundo al mismo) rechazó el 96 % de ahí en adelante.
- **Los 38 repetidos son un solo episodio, en una corrida, y en serie**: no son 38 observaciones
  independientes ni una tasa. Lo que sostiene la atribución es que el mecanismo es determinista
  (tabla de arriba) y que la separación es total.
- **Ninguna otra corrida rebobinó dos veces al mismo respaldo** —tampoco las tres de BSDS, 9
  rebobinados—, así que el defecto nunca había tenido ocasión de mostrarse. Hace falta que dos
  eventos caigan sin un respaldo nuevo en medio: el respaldo pide 3 aceptadas seguidas y 250
  iteraciones desde el anterior. `t1-rapido-30k`, en vuelo, lleva 5 rebobinados a 5 respaldos
  distintos; está expuesta.

### El arreglo, preparado y sin aplicar

*Nota del 2026-10-15, al integrar `3D-Training` en `main`: se aplicó ese mismo día en la rama
(entrada «`rebobina()` arreglado», abajo) y el 10-09 en `main` (entrada «ARREGLO del rebobinado
en `main`»).*

Una línea en `rebobina()`: `opt.load_state_dict(copy.deepcopy(respaldo[1]))`. Es inerte sin
rebobinados y en el primer rebobinado a cada respaldo (los tensores ya eran una copia), así que
sólo cambia los repetidos. **No se aplicó** porque `tdv-t1-rapido` está en vuelo (regla 2). Queda
como contrato `test_el_segundo_rebobinado_al_mismo_respaldo_tambien_devuelve_adam`, marcado
`xfail(strict=True)`: con el arreglo pasa —verificado parcheando `train.rebobina` en memoria, junto
con el test del rebobinado simple—, y la marca estricta obliga a sacarla en el mismo commit.

El defecto del lr (`rebobina()` también restaura el lr del respaldo) vive en la misma línea y se
decide en el mismo cambio: según `AGENTS.md`, la ventana viaja con el respaldo y el lr no.

### Qué cambia de lo publicado

En la entrada anterior quedan retiradas, con nota en el sitio: la escala «real» de ~110, el «por
qué BSDS nunca lo tocó» y las dos direcciones de arreglo, que la regla 1 descarta. La clausura de
la ventana bajo su propio filtro sigue siendo una propiedad cierta del guard relativo, pero en el
lazo rechazaba lotes legítimamente malos. Y `‖g‖` = 95 716 no fue parte del lazo: es la excursión
de la it 720, que el disyuntor contuvo con el 2º rebobinado (it 753, al respaldo de la 503), mucho
antes del lazo.

---------------------------------------------------------------------------------------------------

## 2026-10-08 — RESULTADO de `t1-rapido-30k`: 32.766 dB sobre cortes de T1, y el calendario comprimido se corrió exacto

Crudo en `docs/mediciones/t1-rapido-30k.txt`, ficha en `docs/resultados/t1-rapido-30k.md`.

**No es un veredicto de tanda, porque no hubo pre-registro.** El registro de lanzamiento de hoy lo
dice: las dos corridas salieron a pedido del autor con un plazo de 4 horas, sin los ocho campos y
sin predicciones numeradas. Así que no hay nada que calificar CONFIRMADA/FALLA: esto es el
resultado, y lo que no se acredita queda dicho abajo.

| | |
|---|---|
| PSNR | **32.766 dB** en 68 cortes de `ldm100k/val`, iteración 30 000 (mejor = final) |
| configuración | `rapido`, semilla 0, `T_init = 8·δ = 0.1011928851`, `λ_init = 0.1`, `--channels 1` → **387 394** parámetros |
| calendario | 20 000 a lr completo, hitos en 20 000 y 25 000, lr final 1e-4 |
| costo | **3.827 h** en la GPU 0, VRAM pico **5.02 GB** |
| disyuntor | 155 rechazos de 30 000 (**0.517 %**) en 17 tandas, máximo 25 seguidos, **5 rebobinados** |
| salud | `max‖g‖` aceptado **205.4**, máximo visto 5 514, **0 lotes por encima de 10⁴**, cero no finitas |

### El calendario que se pidió es el que se corrió

Los dos hitos aterrizaron y **ninguno se deshizo**: `4.00e-04 → 2.00e-04` en la it 20 000 y
`→ 1.00e-04` en la 25 000, sin un rebobinado que cruzara ninguno de los dos y sin que el lr volviera
a subir en ninguna línea posterior. El defecto de `rebobina()` **no se disparó**, con una
probabilidad estimada de 8.0 % a la tasa final (5 rebobinados en 30 000 = 1 cada 6 000, ventana
nominal de 2 × 250) y de ~4 % con la edad mediana del respaldo medida hoy. Verificado con el test
exacto `respaldo ≤ hito < rebobinado`, no leyendo la columna a ojo.

### Dónde se cobró la ganancia

| tramo | iteraciones | ganancia | por 1 000 it |
|---|---|---|---|
| lr completo, 14 000 → 20 000 | 6 000 | +0.063 dB | 0.0105 |
| dos halvings, 20 000 → 30 000 | 10 000 | +0.082 dB | 0.0082 |
| últimas 2 000 | 2 000 | +0.005 dB | 0.0025 |

Los saltos del eval caen monótonos hasta +0.013 dB en la it 18 000, y los halvings los devuelven a
+0.030 y +0.031. **Cerrar en 30 000 fue correcto**: los dos últimos tramos dieron +0.002 y +0.005,
o sea que la corrida estaba agotada a ese lr. Lo que **no** se puede decir es que este calendario
sea mejor que el de 25 000 + 5 hitos: ese brazo no se corrió.

### `T` se da la vuelta en el primer halving

`T` baja de 0.10954 (it 2 000) a **0.09365** en la it 20 000 y de ahí **sube** a 0.09561. El mínimo
coincide con el primer hito. Es el mismo fenómeno que tituló el veredicto de `t8delta-100k` en BSDS
—`T` subió en vez de bajar— y aparece ahora en otro dominio, con otro número de canales y con un
calendario distinto. `λ` crece monótona de 0.06095 a 0.12978 y nunca toca el clamp.

### El disyuntor en T1, sin explicación todavía

0.517 % de rechazo global contra **0.163 %** de `t8delta-100k-s1` en BSDS: **3.2× más**, con cero
lotes sobre 10⁴ en los dos casos. La causa **sigue sin medirse**, y hoy se retiraron las tres
explicaciones que se habían publicado. Lo que sí está cerrado: no es el nivel de `‖g‖`, porque el
guard relativo es invariante de escala; y no son las cotas, que no se recalibran por dominio
(`docs/mediciones/cotas-disyuntor-dominio.txt`). Las 17 tandas y el máximo de exactamente 25
seguidos dicen que lo que se ve es el contador del propio disyuntor, no una propiedad del lote.

### Cómo NO se leen estos resultados

- **32.766 dB no es «mejor que los 31.547 dB de BSDS».** Los cortes traen ~38 % de fondo casi plano,
  trivial de denoisear. El único anclaje entre dominios es que la entrada ruidosa da 20.17 dB en los
  dos por construcción.
- **El calendario comprimido no se acredita**: cambió junto con el dataset, el modo y los canales.
- **El modo `rapido` tampoco**: no hubo brazo de `referencia` a 30 000, se descartó por el plazo.
- **Una semilla.** La sd del mejor PSNR entre semillas es 0.164 dB, así que a n = 1 nada por debajo
  de 0.33 dB es distinguible, y esta tanda no mide ninguna diferencia.

---------------------------------------------------------------------------------------------------

## 2026-10-08 — El TDV33 3D en la 5070 Ti: 2.3 µs por vóxel en `rapido`, 21× más en `referencia`

Primera medición de la fase 2 en 3D. Crudo en `docs/mediciones/costo-tdv3d.txt`; las herramientas,
en `scripts/tdv3d_borrador/` (locales). Es una **estimación de costo en la GPU 1** con la GPU 0
ocupada por `t1-rapido-30k`, no una comparación entre tarjetas (regla 3).

### El port 3D

Port mecánico de `ddr/conv.py` + `ddr/tdv.py`: kernels 3³, el binomial `[1,4,6,4,1]/16` en 3D
aplicado al kernel antes del stride 2 (un 3³ se vuelve 7³), bajada y subida independientes,
Student-t. En CPU y float64, las adjuntas cierran a ~10⁻¹⁵ y `grad` contra autograd a 4.8·10⁻¹⁶. El
TDV33 en gris tiene **1 162 112** parámetros sin `T` ni `λ`: 3.0× los 387 392 del TDV33 2D en gris
(387 394 con los dos escalares), que es 27/9.

### Costo y memoria por micro-lote (S = 10, forward + backward)

| modo | parche × micro-lote | s | µs/vóxel | VRAM |
|---|---|---|---|---|
| `rapido` | 32³ × 4 | 0.363 | 2.77 | 5.21 GB |
| `rapido` | 48³ × 1 | 0.334 | 3.02 | 4.14 GB |
| `rapido` | **48³ × 4** | **0.999** | **2.26** | **14.72 GB** |
| `rapido` | 64³ × 1 | 0.619 | 2.36 | 8.78 GB |
| `rapido` | 64³ × 2 | — | — | no entra |
| `rapido`, sin `torch.compile` | 64³ × 1 | 0.952 | 3.63 | 14.16 GB |
| `rapido`, checkpointing | 96³ × 1 | 2.631 | 2.97 | 3.33 GB |
| `rapido`, checkpointing | 64³ × 4 | 3.023 | 2.88 | 4.11 GB |
| `intermedio` | 48³ × 1 | 0.927 | 8.39 | 7.55 GB |
| `referencia` | 48³ × 1 | 5.299 | 47.9 | 7.60 GB |

- **`referencia` cuesta ~21× más que `rapido`** (47.9 contra 2.26 µs/vóxel) e `intermedio` 2.8× (a
  micro-lote 1). En 2D la distancia era ~3×. La regla 3 sigue: estos cocientes son de una tarjeta
  ocupada a medias, y sirven para decidir, no para publicar un factor.
- **`torch.compile` sobre `R.grad` ahorra 35 % de tiempo y 38 % de memoria** a 64³.
- **El checkpointing baja la memoria ~9×** (≈4 kB por vóxel contra ≈36) por **+27 %** de tiempo.

### Horas por 100 000 iteraciones en una 5070 Ti

Derivadas: s por parche del mejor micro-lote que entra × parches por iteración × 10⁵.

| parche | lote 4 | lote 8 | lote 32 (el del paper) |
|---|---|---|---|
| 32³ (`rapido`) | 10 h | 20 h | 81 h |
| 48³ (`rapido`) | **28 h** | **55 h** | 222 h |
| 64³ (`rapido`) | 69 h | 137 h | 550 h |
| 48³ (`referencia`) | 589 h | 1 178 h | 4 710 h |

**Consecuencia práctica:** en 3D `referencia` no es un modo de trabajo; queda para tests cortos de
exactitud (la reanudación bit a bit, el arreglo de `rebobina()`). Las corridas largas van en
`rapido`, así que de ellas se lee la **tasa** sobre semillas, nunca una corrida contra otra. Y en la
tarjeta de destino de la corrida larga el límite sería cómputo, no memoria: **si** tiene 64 GB —el
autor lo dio como aproximado, sin confirmar—, con checkpointing alcanzan para ~15 M vóxeles por
micro-lote (derivado: 64 GB / 4.2 kB), o sea 8 parches de 120³.

---------------------------------------------------------------------------------------------------

## 2026-10-08 — En 3D el arranque sano cruza `GUARD_ABS = 1e4`: la cota tiene que escalar con los vóxeles del parche

Crudo en `docs/mediciones/arranque-g-3d.txt`; la herramienta, `scripts/tdv3d_borrador/arranque_g.py`.
Receta de `train.py` llevada a 3D (σ = 25, S = 10, Adam 4e-4, `T_init` = 8·δ, `λ_init` = 0.1,
`rapido`), 80 iteraciones sobre parches de LDM_100k con al menos 50 % de tejido. Una semilla por
configuración.

| parche × lote | mediana de `‖g‖` | máximo | iteración del máximo | cruces de 1e4 |
|---|---|---|---|---|
| 32³ × 4 | 213 | 10 329 | 7 | 1 |
| 48³ × 1 | 926 | 8 367 | 14 | 0 |
| 48³ × 4 | 747 | 8 201 | 17 | 0 |
| 64³ × 1 | 1 602 | **15 985** | 14 | **2** |

**Por qué importa:** la cota absoluta es la única prueba del disyuntor armada antes de que la ventana
de 100 aceptadas se llene, y está calibrada para no tocar lo sano (el máximo sano de las 81 corridas
de BSDS es 9 397). En 3D lo sano la cruza en el arranque.

**Por qué escala:** la pérdida es una **suma** por parche, así que `‖g‖` crece con los valores
sumados. Ya estaba medido en 2D con los canales: el cociente de medianas entre T1 y BSDS es escala
de canales, y corregido por ×3 no queda brecha a la it 10 000–12 000; en el arranque (it 0–2 000)
queda un cociente de 2.33× que los canales no explican (`docs/mediciones/nivel-g-t1-vs-bsds.txt`).
O sea que la escala por valores sumados vale asentada, y en el arranque es aproximada. Acá la
mediana sube 3.5–4.3× de 32³ a 48³ (3.4× vóxeles) y 1.7–2.1× de 48³ a 64³ (2.4×).

**Lo que se propone (derivado, no medido en corridas largas):** escalar la cota por los valores
sumados por parche, con BSDS como referencia: `GUARD_ABS · P³ / (93²·3)`. Da **4.26·10⁴** a 48³ y
**1.01·10⁵** a 64³: 5.1× y 6.3× sobre el máximo de arriba. Ese margen **no calibra nada**: es de 80
iteraciones con una semilla. En 2D tampoco hay un margen comparable: el máximo sano de 9 397
(`tests/test_disyuntor.py`) es del tramo de cada corrida **antes** de su primer `‖g‖` > 1e4, así que
queda debajo de la cota por construcción. Que el hueco [6 113, 1.97·10⁵] de BSDS escale igual es un
**supuesto**; el máximo sano en 3D lo tiene que medir el tronco del piloto. El guard relativo no
necesita nada: es invariante de escala (`docs/mediciones/umbral-guard-escala.txt`).

---------------------------------------------------------------------------------------------------

## 2026-10-08 — LDM_100k para 3D: sin metadatos, con cráneo, sin fallas groseras, y la variedad se elige por miniaturas

Crudo en `docs/mediciones/variedad-ldm100k.txt`; las herramientas, `scripts/escanea_ldm100k.py` y
`scripts/selecciona_ldm100k.py` (locales; su versión versionada es `datasets/cache_ldm100k3d.py`).

### Lo que hay

- `participants.tsv`, `participants.json` y `dataset_description.json` pesan **0 bytes**: edad,
  sexo y volúmenes —las variables con que se condicionó el modelo generativo— no están. La variedad
  hay que medirla en las imágenes.
- Los 100 000 son `uint16` de 160×224×160 a 1 mm, **alineados entre sí**, **con cráneo** y con el
  defacing de UK Biobank. El tejido (> 0.05 tras normalizar) ocupa ~47 % del volumen entero, no de
  la caja del tejido. El tamaño de los ventrículos varía mucho de un sujeto a otro.

### El escaneo

Miniatura por promedio en bloques de 8³ (20×28×20, comparable vóxel a vóxel porque los volúmenes
están alineados) y siete escalares por sujeto: **100 000 de 100 000 en 1 529 s**, cero fallidos, con
8 procesos a `nice 19`. 32 componentes principales explican **69.3 %** de la varianza de las
miniaturas (la primera, 12.0 %).

### Control de calidad: no hay nada grosero que sacar

A z > 6 desviaciones robustas el control excluía **31**. Mirados los cortes medios de 8 de ellos:
anatomía plausible, casi todos ventrículos muy grandes. **Ningún volumen pasa de z = 8.1** en ningún
descriptor, así que el corte en 6 recortaba la cola legítima, que es justo la variedad buscada. El
umbral queda en 10, que no excluye ninguno: el mecanismo queda como salvaguarda.

### Cobertura: k-center, sorteo y la mezcla

Cobertura = distancia (en el espacio PCA) de cada sujeto del corpus a su elegido más cercano.

| N | regla | mediana | p99 | máximo |
|---|---|---|---|---|
| 2 000 | k-center voraz | 4.618 | 5.359 | 5.522 |
| 2 000 | sorteo | 3.845 | 5.643 | **9.766** |
| 2 000 | **mezcla** | **4.000** | **5.381** | **5.987** |
| 8 000 | sorteo | 3.496 | 5.036 | 7.921 |
| 8 000 | mezcla | 3.606 | 4.730 | 5.083 |

El k-center cubre las colas y el sorteo sigue la densidad; la **mezcla** (mitad de cada uno) queda a
4 % del sorteo en la mediana y a 8 % del k-center en el máximo. **La mezcla de 2 000 cubre las colas
mejor que un sorteo de 8 000** (máximo 5.987 contra 7.921). La evaluación son 12 sujetos, el más
cercano a cada centroide de un k-means: representativos de los modos, no extremos.

### Cómo NO se lee esto

- La cobertura es en miniaturas de 8 mm: dice que los **cerebros** son variados, no que los
  **parches** de 48³ lo sean. En T1 2D los lotes resultaron 1.7–3.7× menos heterogéneos que los de
  BSDS (entrada de retiradas de hoy): la variedad está en el número de sujetos fuente.
- Es **T1 sintética, con cráneo**: no es QSM. Un parche que cae en el cuero cabelludo enseña
  texturas que en QSM no existen (el χ se reconstruye dentro de una máscara de cerebro).

---------------------------------------------------------------------------------------------------

## 2026-10-08 — Fase 2 en 3D: qué cambia respecto del paper, y por qué cada cosa

Propuesta, no veredicto. Cada punto dice si está **medido**, **derivado**, **citado** o **supuesto**.
El código vive en la rama `3D-Training`.

### Lo que cambia sí o sí

- **La dimensión** (citada para el 2D, derivada para el 3D): kernels 3³ y binomial 5³; las 8
  orientaciones de la receta (flips y rotaciones de 90°) pasan a las **48** simetrías del cubo,
  válidas porque LDM es isotrópica a 1 mm. Para la etapa con el dipolo dejan de valer: el eje de B0
  no es intercambiable (README, fase 2).
- **El modo de las corridas largas** (medido arriba): `rapido`. `referencia` cuesta ~21×.
- **El lote** (derivado): el «32» del paper son 32×93²×3 = 830 k valores por iteración en BSDS
  color; 32 parches de 48³ son 3.5 M y 222 h por 100 000 iteraciones en una 5070 Ti. El lote 3D se
  decide midiendo, no copiando el número (abajo).

### Infraestructura para corridas largas: no es receta

Los puntos 1 y 2 son inertes sin rebobinados, y el 3 y el 4 no tocan la aritmética. El 5 sí:
cambia qué lotes rechaza la cota absoluta.

1. **`rebobina()` copia el respaldo** (bug medido en la entrada «El lazo era Adam»).
2. **El lr se fija en cada iteración desde el calendario**: con eso una rama puede reanudar con
   otro calendario, y además un rebobinado no puede deshacer un halving (que `rebobina()` ya no hace
   desde su arreglo).
3. **Reanudación exacta y ramas** (`--guardar-cada`, `--reanudar`): el estado completo de la corrida
   —pesos, Adam, ventana, respaldo, contadores, generadores, posición en los datos— viaja en el
   checkpoint. Sirve para dos cosas: sobrevivir un corte en una máquina remota (ya hubo uno, en
   `t8delta-100k`) y medir dónde poner el lr/2 con ramas desde un tronco, en vez de adivinarlo.
4. **Cuatro columnas más en la traza**, las diez de `train.py` sin moverse: `‖g‖` del primer
   micro-lote, lr vigente, iteración efectiva y σ del lote. La primera, junto con `‖g‖` del lote
   entero, estima la escala de ruido del gradiente `B_simple = tr(Σ)/|G|²` (McCandlish et al., 2018):
   si el lote está muy por debajo de `B_simple`, duplicarlo casi divide por dos las iteraciones; si
   está por encima, la tarjeta de destino conviene usarla para más semillas, no para lotes más
   grandes.
5. **`GUARD_ABS` escalada por los vóxeles del parche** (entrada «En 3D el arranque sano cruza
   `GUARD_ABS = 1e4`»).

### Candidatas, una por vez y apagadas por defecto (regla 4)

- **Ruido ciego**, σ ~ U(σ_min, σ_max) por parche. Es la de más valor para QSM según `AGENTS.md`:
  elimina el `alpha` de calibración del solver ADMM. **Supuesto, no medido:** con un solo `T` para
  todos los σ, la PSNR a σ = 25 baja; y como la pérdida escala con σ², la dispersión de `‖g‖` entre
  lotes sube y el guard relativo lo va a ver.
- **Signo**, x → −x con probabilidad 1/2. QSM tiene susceptibilidades de los dos signos y T1 es
  positiva. **Derivado:** con padding simétrico una imagen constante sigue constante, y K1 tiene
  media cero, así que `K1(x + c) = K1(x)` exacto y la energía TDV ya es invariante a un offset; lo
  que queda por cubrir es el signo y la escala.
- **Ganancia** log-uniforme en [1/g, g]. **Medido** que la red no es escala-equivariante (15–28 %):
  entrenar con ganancias es lo que podría volver menos crítico el `alpha`. **Supuesto**.
- **No implementadas todavía:** EMA de los pesos (su estado tendría que viajar con el respaldo);
  evaluar el denoising sobre los χ de COSMOS y Sim2 que hay en `../tdv/data_mat/` como chequeo de
  transferencia de dominio; y la etapa con el término de datos del dipolo. **Citado:** el propio
  abstract de QSM propone reentrenar en 3D y contra el streaking (Milovic et al., ISMRM 2021,
  Discusión). **Derivado:** que eso pase por el dipolo, porque el streaking nace de su cono nulo. La
  cita: «Retraining may also target
  suppression of QSM streaking artifacts»).

### Lo que no se toca

Lo de `AGENTS.md` («Qué NO tocar»): normalizaciones, ReLU, atención, bilineal, más features o
bloques, otro ψ. En 3D la red ya tiene 3× los parámetros sin agregar nada.

### El piloto en la 5070 Ti, y por qué no una corrida de 100 000 con calendario fijo

1. **Mismo contenido por iteración que en la tarjeta de destino**, armado con acumulación: el
   gradiente es el mismo porque ninguna capa acopla muestras, así que sólo cambia el reloj. Si la
   tarjeta de destino usa otro
   lote, las iteraciones del piloto no se trasladan, salvo que `B_simple` diga que el lote está en el
   régimen lineal; ahí el calendario se traslada en **parches vistos**.
2. **Un tronco a lr constante con estados guardados, y ramas con lr/2**, en vez de una sola corrida
   con los hitos puestos de antemano. En BSDS el tramo de lr completo ya había cobrado 76–91 % de su
   ganancia en la 14 000 y cada halving cobra ~la mitad de su escalón en 2 000 iteraciones
   (entrada del calendario, de hoy): ramas de 5 000 desde la 10 000, 20 000 y 30 000 dicen dónde está
   el codo en 3D. Costo derivado a 48³ con lote 8: tronco de 40 000 = 22 h y tres ramas = 8 h,
   contra 55 h de una corrida de 100 000 que contesta una sola pregunta.
3. **Del tronco sale la calibración del disyuntor en 3D**: tasa sana de rechazo a razón 5 y la racha
   sana más larga, que es lo que fija `GUARD_SEGUIDOS` (en BSDS, 18 contra 25).

---------------------------------------------------------------------------------------------------

## 2026-10-08 — `rebobina()` arreglado: copia el respaldo y conserva el lr, y es inerte sin rebobinados

`opt.load_state_dict(copy.deepcopy(respaldo[1]))` y el lr vigente conservado (`train.py`, y su nota
en el bloque final). Los dos tests que lo exigen pasan; uno nuevo verifica que un rebobinado que
cruza un hito no deshace el halving. **Inerte sin rebobinados**, verificado como pide `AGENTS.md`:
120 iteraciones en `referencia`, semilla 0, BSDS. Las GPUs estaban ocupadas por otra sesión y una
corrida 2D en `referencia` pide 9.94 GB, así que las dos versiones corrieron con `--checkpointing`
en la GPU 1 compartida; y la de antes se corrió además sin `--checkpointing`, con la GPU libre.

| corrida (120 it, `referencia`, semilla 0) | código | `--checkpointing` | PSNR | `traza.npy` |
|---|---|---|---|---|
| `antes` | sin el arreglo | no | 29.502 | — |
| `antes-ckpt` | sin el arreglo | sí | 29.502 | idéntica a `antes` |
| `despues-ckpt` | con el arreglo | sí | 29.502 | **idéntica bit a bit a las dos** |

Cero rechazos y cero rebobinados en las 120, como se esperaba: el arreglo sólo actúa al rebobinar.
De paso queda medido que `--checkpointing` es exacto en `referencia` también con la GPU compartida.
Salidas en `runs/arreglo-rebobina/`.

`main` sigue con el defecto: el arreglo vive en esta rama hasta que se integre o se lleve el commit.

---------------------------------------------------------------------------------------------------

## 2026-10-08 — El entrenador 3D en la rama `3D-Training`

Hecho por agentes con una especificación escrita, revisado por cuatro frentes (matemática, bucle,
datos, estilo) con dos refutadores por hallazgo: **18 confirmados y corregidos en dos rondas, 5
refutados**. Lo que entró:

- `ddr/conv3d.py`, `ddr/tdv3d.py` y `VNet3D` en `ddr/model.py`: espejo de los operadores 2D. El
  `state_dict` 2D no cambia (huella SHA-256 de los 5 checkpoints con `strict=True`, antes y después),
  y el TDV33 3D sale idéntico bit a bit al borrador con que se midió el costo, así que esos números
  valen para este modelo.
- `ddr/data3d.py`: las 48 orientaciones, parches con al menos 35 % de tejido, semilla por índice
  global (reanudar = `offset = it · lote`), recortes fijos de 128³ para evaluar, ruido ciego y los
  aumentos de signo y ganancia, apagados.
- `datasets/cache_ldm100k3d.py`: escaneo, selección y cache. Reproduce bit a bit la salida de los
  scripts con que se midió la variedad. Las listas versionadas salen con `--z 10`.
- `train3d.py`: la receta de `train.py` importada, lr por iteración, reanudación exacta y ramas, cota
  absoluta escalada, 14 columnas de traza.

### Cómo se verificó

- **Suite: 130 pasados, 0 fallados, 0 xfail** (CPU, sin los PSNR publicados).
- **Reanudación bit a bit** (`tests/test_train3d.py`): en CPU y `referencia`, 12 iteraciones seguidas
  contra 6 + 6 y 9 + 3 reanudadas dan traza, `history.json`, log y estado final idénticos, con ruido
  ciego, aumentos y rebobinados forzados a los dos lados del corte.
- **Mutaciones**: de 19 mutaciones de `train3d.py`, 17 ponen algún test en rojo y las otras 2 son
  equivalentes; 8 de 8 en el modelo y 8 de 8 en los datos.
- **En GPU**, `rapido`, 60 iteraciones con lote 8 de 48³ y `--checkpointing`, sobre los 2 000
  volúmenes cacheados: corre entero, PSNR de evaluación **28.759 dB en la 30 y 29.585 en la 60**
  (12 recortes de 128³; la entrada ruidosa da 20.17), 0 rechazos, VRAM pico 3.59 GB, RAM pico 26.8
  GB (log en `docs/mediciones/humo-train3d.txt`). Una rama reanudada en GPU desde
  `estado-000030.pth`, con lr 2e-4 desde la 31, corre entera y sale con `reanudada_de` en su
  `estado.json`; su 29.490 dB contra los 29.585 del tronco en la 60 es de 30 iteraciones en
  `rapido` y **no es un efecto**.

**Lo que no se probó:** una corrida larga. Todo lo de arriba es de decenas de iteraciones; la tasa
de rechazos, el codo del lr y la escala de ruido los contesta el piloto.

---------------------------------------------------------------------------------------------------

## 2026-10-08 — PRE-REGISTRO (borrador, sin lanzar) del piloto 3D: un tronco a lr constante y tres ramas con lr/2

**Borrador.** Lo lanza el autor (`AGENTS.md`, mecánica de la sesión), y antes hay que mirar las
GPUs: al escribir esto las dos estaban ocupadas por un grid search de `../tdv`. El comando va al
final, impreso, no ejecutado. Código: rama `3D-Training`, `train3d.py`.

**1. Qué cambia.** En el tronco, nada: es descriptivo. En cada rama, **una sola cosa**: el lr pasa
de 4e-4 a 2e-4 desde la iteración k + 1, con k = 10 000, 20 000 y 30 000. Rama y tronco salen del
mismo `estado-<k>.pth`, ven los mismos parches (el parche `i` depende sólo de `(semilla, i)`) y el
mismo ruido (el generador viaja en el estado); difieren sólo en el lr. En `rapido` no son bit a bit
iguales, pero el par es lo más apareado que se puede tener sin `referencia`, que en 3D cuesta ~21×.

**2. Control.** El propio tronco, concurrente, entre k y k + 5 000.

**3. Preguntas, escritas antes.** Principal: **dónde se aplana el tramo de lr completo en 3D y
cuánto compra el primer halving según cuándo se aplica**, que es lo que decide el calendario de la
corrida larga en la tarjeta de destino. Secundarias, del tronco: la escala de ruido del gradiente
`B_simple` (columnas 5 y 10 de la traza), que decide el lote de la tarjeta de destino; y la
calibración del disyuntor en 3D (tasa sana de rechazo a razón 5, racha sana más larga, máximo `‖g‖`
aceptado). Ninguna se contesta sin GPU: no hay ninguna traza 3D más larga que 80 iteraciones.

**4. Endpoint.** PSNR medio sobre los 12 recortes de 128³ de `lista_eval.txt`, en todo el recorte y
en tejido. Para las ramas: diferencia rama − tronco a la iteración k + 5 000, **pareada**. Para el
codo: ganancia del tronco por ventana de 2 000 iteraciones. Si hay rebobinados, a iteración
efectiva (columna 12).

**5. Predicciones**, todas **supuestos**; de dónde sale cada una, entre paréntesis:

1. El tronco cobra a la **14 000** al menos el **75 %** de la ganancia que tiene entre la 2 000 y
   la 24 000 (misma definición que en BSDS, donde dio 76–91 %: entrada del calendario del lr, de
   hoy). Si falla, el codo en 3D está más tarde y el tramo de lr completo de la tarjeta de destino
   se alarga.
2. Cada rama queda **entre +0.01 y +0.06 dB** sobre el tronco a k + 5 000 (BSDS, primer halving:
   +0.030 y +0.033, misma entrada). Si sale ≤ 0, el halving no paga en 3D a esa altura; si sale >
   0.1, el tramo de lr completo estaba sobrando.
   *[Nota del 2026-10-15, al integrar `3D-Training` en `main`: corregida en `main` el 2026-10-12
   (entrada «¿El recorte del calendario se traslada al 3D?»): +0.030/+0.033 son ventanas crudas que
   incluyen la pendiente previa; descontada, el primer halving de BSDS vale +0.0065 a +0.0182 en las
   cuatro corridas de 10⁵. La banda honesta es +0.007 a +0.018 dB; el techo de +0.06 queda ~3×
   alto.]*
3. La tasa de rechazo del tronco fuera de rachas de rebobinado queda **≤ 1 %** (BSDS sano: 0.15 %,
   `tests/test_disyuntor.py`; T1 2D llegó a 3.4 % en el arranque, entrada de retiradas de hoy). Si
   falla, la razón 5 está apretada en 3D.
4. Todo lote del tronco con `‖g‖` (columna 5) por encima de `cota_absoluta(48)` = **4.26·10⁴** cae
   dentro de una racha que termina en rebobinado (derivado de la escala de la cota, entrada «En 3D
   el arranque sano cruza `GUARD_ABS = 1e4`»). Si alguno queda fuera, la cota rechaza lotes sanos y
   hay que recalibrarla.
5. `T` **no baja de 2·δ = 0.0253** en las 40 000 (en BSDS subió desde 8·δ: veredicto de
   `t8delta-100k`). Si baja, la deriva hacia la pared también existe en 3D.
   *[Nota del 2026-10-15, al integrar `3D-Training` en `main`: el umbral sale de la «regla 2-δ»,
   retirada en `main` el 2026-10-12 (entrada «RETIRO dos afirmaciones mías…»; `## Descartado`): el
   factor 2 era una elección y nunca predijo (`tinit/B-larga` murió arrancando en 2·δ;
   `t1p1delta-rapido-100k` terminó desde 1.1·δ). Se lee como descriptiva. En el tronco a lr
   constante `δ_vivo` no se parte, así que 0.0253 sigue siendo 2·δ_vivo toda la corrida.]*

`B_simple` no lleva predicción: es una medición con regla de decisión fijada ahora. Si su mediana
sobre la 10 000–40 000 es ≥ 16, la tarjeta de destino usa lote ≥ 16 y el calendario se traslada en
parches vistos; si es < 8, lote 8 y la capacidad sobrante va a semillas.

**6. Potencia.** No aplica: una semilla, preguntas descriptivas (la regla 5 lo admite si la pregunta
principal está escrita antes). En BSDS el salto de un halving fue positivo en 4 de 4 sobre dos
semillas, o sea detectable dentro de una corrida; en 3D eso es un supuesto.

**7. Costo**, de `docs/mediciones/costo-tdv3d.txt` (0.250 s por parche de 48³ en `rapido`): tronco
de 40 000 iteraciones con lote 8, **22 h**; tres ramas de 5 000, **8 h**. ~30 h de una GPU, más las
evaluaciones: 23 s en la única evaluación medida, de la corrida de humo con la GPU compartida (la
primera, 59 s, recompila `R.grad` para 128³; crudo en `docs/mediciones/humo-train3d.txt`), o sea
~15 min sobre el tronco con `--eval-every 1000`. Si se corren en paralelo, el tronco en la GPU 0 y
las ramas en la 1 a medida que aparece cada estado, termina en ~22 h.

**8. Cómo NO se leen los resultados.**

- El PSNR de T1 3D con cráneo no se compara con T1 2D, ni con BSDS, ni con nada de QSM.
- Rama contra tronco en `rapido` no es bit a bit: una diferencia por debajo de la dispersión entre
  evaluaciones vecinas del tronco es ruido, no efecto.
- El codo medido en iteraciones vale para la tarjeta de destino **sólo** con el mismo lote y
  parche, o traducido a parches vistos si `B_simple` lo permite.
- Una semilla no dice nada de la tasa de estallidos: un rebobinado o su ausencia es un caso.

**Comando** (no ejecutado):

```bash
# desde un checkout de la rama 3D-Training con runs/ y datasets/ldm100k3d/images enlazados
REPO=$PWD; PY=/home/santicien/Documents/Training-TDV/.venv/bin/python
# tronco, cuando haya una GPU libre
systemd-run --user --unit tdv-piloto3d-tronco --working-directory=$REPO -E CUDA_VISIBLE_DEVICES=0 \
    $PY train3d.py --out runs/piloto3d/tronco --hitos --iterations 40000 --eval-every 1000
# cada rama, cuando exista su estado (k = 10000, 20000, 30000)
systemd-run --user --unit tdv-piloto3d-rama-10k --working-directory=$REPO -E CUDA_VISIBLE_DEVICES=1 \
    $PY train3d.py --out runs/piloto3d/rama-10k \
    --reanudar runs/piloto3d/tronco/estado-010000.pth --hitos 10000 --iterations 15000 --eval-every 1000
```

`--eval-every 1000` en los dos para tener el par exacto en k + 5 000; la frecuencia de evaluación no
toca la aritmética (es una de las banderas libres al reanudar).

### ENMIENDA del 2026-10-09: lote 32, y el piloto puede correr en la tarjeta de destino (escrita antes de lanzar nada)

Lo de arriba no se reescribe. Cambia el lote, por decisión del autor (entrada del 2026-10-09,
«Lote 32 y micro-lote automático para una tarjeta de 48 GB»), y con él el costo, dónde corre y
cuánto pesan dos predicciones.

- **Lote 32** parches de 48³ en vez de 8: 3.54 M vóxeles por paso, 4.3× los 830 k valores del lote
  de BSDS color (con 8 eran 885 k). El micro-lote lo elige la memoria (`--micro-lote auto`). Rama y
  tronco siguen apareados como dice el punto 1: el lote viaja en el estado y `--reanudar` rechaza
  cambiarlo.
- **Dónde corre.** El piloto puede correr en la tarjeta de destino; el comando de arriba es el de
  esta máquina. Tronco y ramas, en la misma tarjeta y con el mismo micro-lote: con otro, la rama
  sigue la misma receta pero el par deja de compartir el orden de suma (que en `rapido` tampoco era
  bit a bit), y `--reanudar` lo avisa.
- **Costo (punto 7), derivado.** En la 5070 Ti, a 0.250 s por parche (micro-lote 4): tronco =
  0.250 × 32 × 40 000 = **89 h**, y tres ramas de 5 000, **33 h**. Con el micro-lote 2 que `auto`
  elige allá (0.265 s por parche), 94 h y 35 h. En la tarjeta de destino **no está medido**: antes
  de lanzar, correr la sonda y 200 iteraciones (abajo), y anotar acá el s/it. *[Nota del 2026-10-15:
  en la 5070 Ti, la sonda del 2026-10-13 midió 0.280–0.291 s por parche con el micro-lote 2
  (8.96–9.30 s/it), más que la grilla, y `ventana-3d-12h` corrió a 9.17–9.24 s/it. El piloto no se
  lanzó, y la tarjeta de destino sigue sin medir.]*
- **Las predicciones 1 y 2 se calibraron con el lote de BSDS**, 32 parches con 830 k valores en
  total. Con lote 8 el piloto veía del mismo orden por paso; con 32 ve 4.3×. Quedan como
  **supuestos más débiles**: si el codo no cae donde dice la 1, o el halving paga distinto de lo que
  dice la 2, puede ser por el contenido por paso y no por la dimensión. Se leen igual, y un fallo
  dice menos que antes sobre el 3D.
- **La regla de `B_simple` (punto 5) ya no decide el lote**, que lo fijó el autor. Se mide igual,
  con las columnas 5 y 10 (lotes de `--micro-lote` y de 32), y se lee como pregunta: si la mediana
  sobre la 10 000–40 000 queda ≥ 32, el lote no pasa la escala de ruido y los parches de más pagan;
  si queda muy por debajo, un lote menor compraba casi lo mismo por iteración, a menos costo.
  Con un solo micro-lote (`--checkpointing` en 48 GiB, donde `auto` da 32) la columna 10 no estima
  nada: ahí, `--micro-lote 16`.

Antes de lanzar, en la tarjeta de destino:

```bash
# el pico medido de cada micro-lote contra el predicho; no entrena, y sólo escribe sonda-vram.json
$PY train3d.py --out runs/piloto3d/sonda --sonda-vram
# el s/it: con systemd-run, como arriba, el journal fecha cada línea; s/it = (hora de [200] − hora
# de [100]) / 100, porque las primeras pagan la compilación
$PY train3d.py --out runs/piloto3d/medida --hitos --iterations 200 --log-every 50 --guardar-cada 0
```

---------------------------------------------------------------------------------------------------

## 2026-10-09 — ARREGLO del rebobinado en `main`: el respaldo se copia y el lr no vuelve

Pedido por el autor tras ver que el arreglo vivía sólo en `3D-Training`. **Son dos defectos en la
misma línea**, y los dos quedan reproducidos como test que falla antes del cambio:

| defecto | qué hacía | test que lo fija |
|---|---|---|
| **alias del respaldo** | `opt.load_state_dict(respaldo[1])` adopta los tensores: los pasos siguientes mutan el respaldo y el 2º rebobinado al mismo ya no devuelve Adam | `test_el_segundo_rebobinado_al_mismo_respaldo_tambien_devuelve_adam` |
| **lr restaurado** | el `state_dict` del optimizador trae el lr del respaldo, así que un rebobinado que cruza un hito deshace ese halving hasta el final | `test_un_rebobinado_que_cruza_un_hito_no_deshace_el_halving` |

Contra el código de antes los dos fallan; el del lr con `assert 0.0004 == 0.0002`, o sea el lr
volviendo a `4e-4` desde `2e-4`. Con el arreglo pasan los dos, y la suite en CPU queda en **47
pasados y 3 salteados** (`CUDA_VISIBLE_DEVICES="" pytest -k "not published_psnr"`).

### El cambio

`rebobina()` restaura una **copia** del estado de Adam y **conserva el lr vigente**:

```python
lr = [g["lr"] for g in opt.param_groups]
opt.load_state_dict(copy.deepcopy(respaldo[1]))
for g, valor in zip(opt.param_groups, lr):
    g["lr"] = valor
```

El criterio es el que ya estaba escrito: **la ventana viaja con el respaldo y el lr no**. La ventana
describe los pesos que vuelven; el lr es del calendario, que no retrocede.

### Es inerte donde importa, y por eso el pareo sigue valiendo

El arreglo sólo cambia dos situaciones: un rebobinado **repetido** al mismo respaldo, y uno que
**cruza un hito**. En el primer rebobinado a cada respaldo los tensores ya eran una copia, y sin
hito cruzado el lr que se conserva es el mismo que traía el respaldo.

**`t8delta-100k-s1` no está afectada**: sus cuatro rebobinados (24 796→24 755, 27 123→27 005,
29 419→29 255, 99 365→99 255) son **todos primeros a su respaldo** —así los clasifica
`docs/mediciones/alias-respaldo-adam.txt`— y ninguno cruza un hito (25 000, 50 000, 75 000,
100 000). Lo mismo `t1-rapido-30k`: 5 rebobinados a 5 respaldos distintos, ningún hito cruzado.
Así que el control de 31.547 dB sigue siendo comparable con lo que se corra de acá en adelante. **No
se re-corrió para verificarlo bit a bit**: son 37.7 h de GPU, y el argumento es de código más la
clasificación de la medición, no una réplica.

### Lo que el arreglo NO arregla

- **No baja la frecuencia de rebobinados.** Quita la trampa, no los eventos. T1 con `T_init = 2·δ`
  seguirá rebobinando seguido; lo que cambia es que ahora el segundo rebobinado al mismo respaldo
  devuelve Adam de verdad y la corrida puede salir.
- **No se midió una corrida con el arreglo puesto.** El único episodio del defecto en el corpus es
  `t1-guard-2delta-5k`, y está cortada. Que el lazo no vuelva a pasar es predicción, no medición.
- **La ventana de exposición del defecto del lr no era `GUARD_RESPALDO`**: la edad mediana del
  respaldo que se usa es **118**, no 250 (`docs/mediciones/edad-del-respaldo.txt`). El riesgo
  publicado de 6–13 % queda como historia: desde hoy es cero por construcción.

---------------------------------------------------------------------------------------------------

## 2026-10-09 — PRE-REGISTRO de `modo-y-tinit-100k-s1`: dos brazos contra el control de 31.547 dB, LANZADA a las 14:32

> **LANZADA el 2026-10-09**, a pedido explícito del autor y con las dos GPU **libres** (0 % de uso
> y sin procesos de cómputo: el otro proyecto que las tenía al 97–100 % terminó). Brazo B a las
> **14:32:49** en la **GPU 0** —la misma del control—, brazo A a las **14:32:51** en la **GPU 1**.
> Los `config.json` de las dos confirman `seed 1`, 100 000 iteraciones y los hitos del lr en
> [25 000, 50 000, 75 000, 100 000], o sea el calendario del control; el brazo B difiere de él sólo
> en `t_init` y el A sólo en `mode`. Las dos corren con el `rebobina()` **arreglado** (commit
> `81b3701`, pasado a `main` por fast-forward antes de lanzar), y el comando exacto de cada una
> quedó en `runs/<brazo>/plan.sh`. Primer indicio sano: a la it 250 el brazo A da `T` = 0.10537
> contra **0.10535** del control en la misma iteración, o sea misma semilla y aritmética todavía
> sin separarse; y su ETA de arranque da 13.1 h, dentro del rango de la predicción 2 —que **no se
> acredita con eso**, porque a la it 250 ese número es casi todo arranque de `torch.compile`.

Pedido por el autor el 2026-10-09, con la instrucción explícita de **no lanzar y esperar su
confirmación**. Los dos brazos cuelgan del mismo control archivado, `runs/t8delta-100k-s1`, y cada
uno difiere de él en una dimensión.

| | control (archivado) | brazo A | brazo B |
|---|---|---|---|
| nombre | `t8delta-100k-s1` | `rapido-100k-s1` | `t2delta-100k-s1` |
| `--mode` | `referencia` | **`rapido`** | `referencia` |
| `--t-init` | 8·δ = 0.1011928851 | 8·δ = 0.1011928851 | **2·δ = 0.0252982212813** |
| `--seed` | 1 | 1 | 1 |
| `--iterations` | 100 000 | 100 000 | 100 000 |
| hitos del lr | 25k / 50k / 75k / 100k | iguales (defaults) | iguales (defaults) |
| disyuntor | v2, valores publicados | iguales | iguales |
| resultado | **31.547008 dB** @100 000 | — | — |

**1. Qué variable cambia, y lo que hay que decir de frente.**

- **Brazo B cambia una sola cosa**: `--t-init`. Semilla, modo, calendario, dataset, canales y los
  siete parámetros del disyuntor quedan idénticos al control. Es un contraste limpio.
- **Brazo A NO cambia una sola cosa.** `rapido` difiere de `referencia` en **tres**: bf16 en vez de
  fp32 IEEE, `cudnn.benchmark` encendido y determinismo apagado. Es la trampa que la skill `tanda`
  nombra en su campo 1, y acá no se puede evitar porque la pregunta es sobre el modo como paquete.
  Queda registrado como **pregunta de ingeniería** —¿se puede usar `rapido` para una corrida de
  10⁵?— y **no** como experimento de una variable. Los tres ingredientes ya están cerrados por
  separado en `## Descartado` (bf16 ❌, `cudnn.benchmark` ❌, TF32 ❌ como desestabilizadores), así
  que lo que se mide acá es el endpoint del paquete, no la causa de nada.
- **Hay un segundo cambio contra el control, y es de código**: el control corrió con el
  `rebobina()` sin arreglar. Los dos brazos correrán con el arreglo del 2026-10-09. Es **inerte en
  la trayectoria del control** —sus 4 rebobinados son primeros a su respaldo y ninguno cruza un
  hito—, así que el pareo vale. Pero si un brazo **rebobina dos veces al mismo respaldo**, en ese
  tramo deja de ser comparable con una corrida pre-arreglo, y hay que decirlo en vez de promediar.

**2. Control.** `runs/t8delta-100k-s1`, **archivado y terminado**: `"terminada": true`,
100 000/100 000, mejor = final = **31.547008477 dB**, 163 rechazos, `max_seguidos` 25, **4
rebobinados**, 37.697 h, VRAM pico 9.94 GB. Su `config.json` es la referencia literal de las
banderas. El control **no se vuelve a correr**: la potencia de los dos brazos es condicional a él.

**3. La pregunta, y qué contestó el replay sin gastar GPU.**

*Brazo B.* ¿Sobrevive `T_init = 2·δ` a 100 000 iteraciones con el disyuntor puesto? El replay de
Adam sobre la coordenada `T` sola corrió hoy, **validado a 6.85·10⁻⁷** contra la `T` grabada de
`t8delta-100k-s0` (66 000 filas, 0 rebobinados) y a 4.49·10⁻⁷ contra el prefijo limpio de la s1
(24 795 filas) — el mismo error que registró el 10-06, reproducido con una implementación
independiente. Crudo en `docs/mediciones/replay-T-2delta-100k.txt`. Contesta el *si*:

| desde | min `T` | × la pared | ¿toca 0? |
|---|---|---|---|
| secuencia de `-s0`, 2·δ | 0.002588 | **0.205** | no |
| secuencia de `-s1`, 2·δ | 0.004583 | **0.362** | no |

Reserva exigida (caída máxima sin clamp desde un init alto): **1.795·δ** en `-s0` y **1.638·δ** en
`-s1`, o sea que 2·δ aguanta las dos secuencias con un margen de **0.21·δ y 0.36·δ**. Pero eso
contesta mucho menos de lo que parece, y hay que decirlo antes de gastar 38 h:

> **RETIRADA el 2026-10-11 como forma de decidir un `T_init`, por la corrida que predijo.** De esta
> reserva salió «1.1·δ toca 0»: `t1p1delta-rapido-100k` **nunca bajó de 1.437·δ** y terminó en
> 2.228·δ. Falla por las dos cosas que ya están dichas abajo —es una traslación, y usa gradientes de
> `T ≈ 0.08` para decidir sobre `T ≈ 0.014`— y por una tercera que no estaba: **`δ_Adam` se parte a
> la mitad en cada hito del lr**, así que la reserva se duplica sola y la ventana de peligro es
> finita. Fila en `## Descartado`; veredicto en la entrada del 10-11.


> **El replay crudo es una TRASLACIÓN, no un contrafactual.** El paso de Adam sobre `T` vale
> `−lr·m̂/√v̂` y depende **sólo de la secuencia de gradientes**, no del valor de `T`. Rearrancar la
> misma secuencia desde otro `T_init` traslada la trayectoria entera en vertical, y se ve: el
> mínimo del replay desde 2·δ cae en la it **49 997** en `-s0` y en la **24 772** en `-s1`, que son
> **exactamente** las iteraciones del mínimo de la corrida grabada. Lo único con contenido es la
> **reserva exigida**; «2·δ no toca cero» es «2·δ − 1.795·δ > 0», y vale sólo si los gradientes
> fueran los mismos, que es justo lo que otro `T_init` no garantiza.

> **Y la única corrida real de 2·δ dice lo contrario en dirección.** En `tinit-2delta-50k` el
> mínimo de `T` está en la iteración **2** —0.0248982, 1.97× la pared— y de ahí `T` **sube**, hasta
> 0.02710 en la it 4 000 (`estado.json` da 0.027129, que es después del paso). A `T` ≈ 0.025 los
> gradientes no son los del control trasladados: empujan `T` para arriba. El replay **no cierra la
> pregunta**, y por eso este brazo vale la GPU.

> **De paso, revisa una objeción del 2026-10-06.** El pre-registro de `tinit-2delta-50k` leyó que
> «a 0.0253 la ley `1/T` dice que los gradientes son ~3× más grandes, así que la reserva exigida
> real es mayor». Eso confunde magnitud de gradiente con tamaño de paso: **`δ_adam` es el
> desplazamiento que Adam imprime sin importar la magnitud del pico**, y está escrito así en el
> repo. Aplicada paso a paso, la ley `1/T` mueve el mínimo **para arriba** —0.205·δ → 1.036·δ en
> `-s0` y 0.362·δ → 1.138·δ en `-s1`—, no para abajo. Esa columna tampoco es la respuesta: es una
> realimentación que empuja `T` hacia su trayectoria grabada. Las dos se leen como **cota**.

*Brazo A.* Ningún script de CPU la contesta: es una pregunta de aritmética, no de `init`. El mejor
resultado bf16 escrito es **31.478 @68 000** (`replica-v4`, otra config, con `clip 35 000` y DDP),
contra 31.547 @100 000 en `referencia`. No es una comparación: difieren en todo menos la precisión.
Lo que la hace valer la GPU es que **la fase 2 ya corre en `rapido`** (`t1-rapido-30k`) y nada en el
repo licencia eso a largo de 10⁵.

**4. Endpoint.** PSNR a **iteración fija** sobre CBSD68 (σ = 25), leída a 4 000 / 25 000 / 50 000 /
75 000 / 100 000 contra los mismos puntos del control. Nunca mortalidad binaria. El evento
(`‖g‖ > 10⁴`) se cuenta aparte, como diagnóstico del disyuntor, no como desenlace.

**5. Predicciones, con umbral numérico.**

*Brazo A (`rapido`).*

1. **PSNR @100 000 en [31.22, 31.88]**, o sea el control ±0.33 dB, que es el mínimo contraste
   detectable con n = 1 (2 sd del mejor PSNR = 0.164 dB). Si cae debajo de 31.22, `rapido` cuesta
   endpoint a largo de 10⁵ y el uso que la fase 2 ya le da hay que revisarlo.
2. **Termina en 13.0–13.8 h** (0.52–0.55 h/4 000 it medido en `rapido`). Si pasa de 16 h, esa cifra
   no vale para BSDS en 3 canales. Ojo: concurrente, así que esto NO se lee como velocidad (campo 8).
3. **Rechazos en [82, 489]**, o sea entre 0.5× y 3× los 163 del control. Fundamento: las dos
   semillas del control difieren **6×** entre sí (1.63·10⁻³ contra 2.73·10⁻⁴), así que todo lo que
   caiga dentro de esa dispersión es ruido, no efecto del modo.
4. **VRAM pico ≈ 5.97 GB** contra 9.94 del control (medido por modo, tabla del 10-05).

*Brazo B (2·δ).*

5. **`T` no toca 0 en las 100 000.** El replay da margen 0.21·δ y 0.36·δ. Si muere, la reserva
   exigida está subestimada y hay que rehacer el razonamiento de la pared, no culpar al disyuntor.
6. **`T` @100 000 > 0.0300**, o sea `T` **sube** saliendo de `T*`. Fundamento medido:
   `tinit-2delta-50k` toca su mínimo en la **iteración 2** (0.0248982) y de ahí sube a 0.02710 en la
   4 000, y `t1-rapido-30k` mostró a `T` darse vuelta exactamente en el primer halving. Si queda en
   [0.024, 0.030], `T*` **es** un atractor, y sería el primer resultado positivo del repo sobre `T*`.
7. **Rechazos > 163** (los del control). Fundamento: 1.98·10⁻²/it medido para 2·δ a 4 000 contra
   1.63·10⁻³ del control, o sea 12×. Umbral flojo a propósito: el signo es la predicción.
8. **Rebobinados > 4** (los del control). Fundamento: 3 en 4 000 iteraciones en `tinit-2delta-50k`.
9. **PSNR @100 000 en [31.22, 31.88]**, igual que A1. Fundamento: que `T_init` mueva el endpoint
   está **cerrado con ❌** (10-07: 0.026 dB @2 000 y 0.015 @10 000 por un factor 5 de `T_init`,
   contra un piso de 0.33). Si sale afuera, **no se cree**: reabre una pregunta cerrada y pide una
   réplica en otra semilla antes de escribir nada.
10. **Ningún rebobinado repite respaldo.** Si alguno lo hace, es la **primera prueba en vivo del
    arreglo de hoy**: la corrida tiene que seguir avanzando en vez de entrar en el lazo de período
    26. Si se traba igual, el arreglo está incompleto.

**6. Potencia, y es mala: la tanda es descriptiva.** n = 1 por brazo, condicional a un control ya
corrido. Para la PSNR detecta sólo diferencias **≥ 0.33 dB**. Para eventos **no tiene poder**:
`scripts/poder_logrank.py` arranca en n = 6 por brazo y ahí el poder contra 1.35× es **0.10** y
contra 2× es **0.21**; con n = 1 queda debajo de todo eso. El techo conocido no se mueve: contra
1.35× harían falta ~300 semillas por brazo. **Ninguno de los dos brazos puede establecer
equivalencia**, y eso está dicho antes de ver los datos.

**7. Costo.** Brazo A **13.0–13.8 h**; brazo B **~37.7 h**, que es el medido del control con la
misma config salvo `T_init` (un lote rechazado igual hace forward y backward, así que los rechazos
extra del 2·δ no cambian el costo por iteración). Concurrentes en las dos tarjetas: **~38 h de
reloj** y ~51 h de GPU, con una contención medida de ~0.5 % (`docs/mediciones/dos-gpu.txt`: dos
corridas rinden 1.99×).

**8. Cómo NO se leen los resultados.**

- **Si los dos brazos caen a menos de 0.33 dB del control, no se lee «`rapido` es equivalente» ni
  «2·δ es equivalente».** Se lee «no se separa con n = 1», y las dos preguntas quedan abiertas.
- **La velocidad del brazo A no contesta «hay diferencia entre las dos GPU en `rapido`».** Corre
  concurrente con un `referencia`, y bajo contención `cudnn.benchmark` elige otros algoritmos
  (regla 3). Esa medición pide las dos tarjetas libres y `scripts/bench_dos_gpu.sh --mode rapido`.
- **`rapido` no es reproducible bit a bit**, así que el brazo A arrastra una varianza entre
  corridas que **nunca se midió a 10⁵**. Una sola corrida en `rapido` no se puede replicar contra
  sí misma, y eso no se arregla leyendo el número con más decimales.
- **Si 2·δ sobrevive, no se lee «2·δ es seguro».** El replay ya decía que sobrevive estas
  secuencias; lo que agregaría es que sobrevive una corrida real con el disyuntor, **una vez**.
  `2·δ` sigue descartado como valor de trabajo y sigue siendo **piso** de la regla 2-δ.
- **Si el brazo B da más PSNR que el control, no se lee «2·δ es mejor».** El endpoint de `T_init`
  está cerrado con ❌ y 0.33 dB es el piso de detección.
- **Una diferencia del brazo A no se le atribuye a bf16.** Cambian tres cosas a la vez, y las tres
  ya están cerradas por separado como causa de inestabilidad.
- **Si un brazo rebobina dos veces al mismo respaldo**, ese tramo no se compara con el control
  pre-arreglo: se reporta aparte.

**Precondición de lanzamiento, verificada hoy y PENDIENTE.** Al momento de escribir esto las dos
GPU están al 97–100 % con dos procesos de **otro proyecto** (`Documents/tdv`, PID 1103948 y
1111912): no hay tanda de este repo en vuelo —regla 2 verificada con `pgrep`—, pero sí contención,
y con las GPU ocupadas no se mide (regla 3). **Las dos tarjetas tienen que estar libres.**

```sh
# brazo B (una sola variable: T_init) -> GPU 0, la misma del control
train.py --out runs/t2delta-100k-s1 --mode referencia --iterations 100000 \
    --t-init 0.025298221281347025 --seed 1

# brazo A (el paquete `rapido`) -> GPU 1
train.py --out runs/rapido-100k-s1 --mode rapido --iterations 100000 \
    --t-init 0.1011928851 --seed 1
```

Todo lo demás va en default y es lo que corrió el control: BSDS400 en color (3 canales), σ = 25,
`λ_init = 0.1`, `clip = inf`, hitos del lr en 25k/50k/75k/100k, y los siete del disyuntor v2. El
`--t-init` del brazo B es el valor de precisión completa que usó `tinit-2delta-50k`, para que los
dos 2·δ de BSDS sean un par de semillas y no dos valores distintos.

**Confirmado y lanzado el 2026-10-09 a las 14:32.** Desde acá rige la regla 2: con una
tanda en vuelo no se toca `train.py`.

### Observación en vuelo (2026-10-09, it ~3 000 del brazo A): `rapido` le recorta a `λ` la mitad del margen

**No es el veredicto** —las predicciones se califican cuando las dos terminen—, pero es una medición
y va escrita: `docs/mediciones/lambda-rapido-vs-referencia.txt`, herramienta
`scripts/measure_lambda_modo.py` (local).

**El pareo es limpio, y se verificó.** La correlación de la `loss` entre el brazo A y el control
sobre las primeras 2 000 iteraciones es **1.0000**: misma semilla, mismo orden de lotes, mismos
sorteos de ruido. Lo único que difiere es la aritmética.

**`λ` cava un pozo mucho más profundo en `rapido`, y da la vuelta a la misma iteración.**

| | mínimo de `λ` | × `δ_adam` | en it |
|---|---|---|---|
| control, `referencia`, 3 canales | 0.030028 | **2.37** | **2 831** |
| brazo A, `rapido`, 3 canales | **0.013524** | **1.07** | **2 848** |
| `t1-rapido-30k`, `rapido`, **1 canal** | 0.060337 | **4.77** | 1 962 |

**Los dos mínimos caen a 17 iteraciones uno del otro.** bf16 hace el pozo **2.22× más profundo** y
**no mueve** el instante en que se revierte: 0.6 % de diferencia en la iteración. Eso separa las dos
cosas: la profundidad es aritmética, el momento del giro no lo es.

`λ = 0` es absorbente y `λ_init = 0.1` se eligió porque con 0.0253 `λ` tocaba el clamp en 10 de 12
corridas. El brazo A llegó a **poco más de un desplazamiento de Adam** del cero y volvió. No
disparó, pero el margen se partió al medio.

**Dónde está el mecanismo: en el gradiente del escalar, no en la contabilidad de Adam.** Sobre el
prefijo común, `∂L/∂λ` media **0.461** en `rapido` contra **0.385** en `referencia` —cociente
**1.198**, y positiva el 58.1 % de las iteraciones contra 56.5 %—, y `λ` baja cuando ese gradiente
es positivo. Simétricamente `∂L/∂T` es un 8.8 % **más negativa**, que empuja `T` para arriba: por eso
`T` apenas se separa (0.1048 contra 0.1046 de media) y lo hace en la dirección segura. La `loss`
media queda a 0.1 % (27.782 contra 27.813), o sea que bf16 no degrada el forward.

**Lo que esto NO separa:** pasadas las primeras iteraciones los pesos ya difieren, así que el 1.198
mezcla el sesgo aritmético con la deriva acumulada. El mismo-pesos está medido aparte y da
**0.70–1.85** en 9 checkpoints, del mismo orden, así que un sesgo sistemático es compatible — y
compatibilidad no es atribución.

**La fase 2 no está en riesgo.** `t1-rapido-30k` corre en `rapido` y su `λ` nunca bajó de **4.77·δ**.
El margen fino es de **3 canales + `rapido`**, no de `rapido` solo: la `loss` suma 93·93·C, así que
en gris `∂L/∂λ` es un tercio y el pozo es más chico. Los dos efectos empujan en sentidos opuestos.

**El endpoint, por ahora, no se separa.** Dos lecturas pareadas: **30.997** contra 31.003 dB en la
it 2 000 y **31.162** contra 31.172 en la 4 000, o sea −0.006 y −0.010 dB. Las dos un orden de
magnitud por debajo del piso de 0.33 dB, así que la predicción 1 sigue viva y nada está acreditado.

**Cómo NO se lee.** Es n = 1, y lo medido es que el pozo de `λ` es 2.2× más profundo — **no** que la
próxima corrida en `rapido` toque el clamp. Y una nota de lectura del log, para no repetir el error:
**`max‖g‖` en la línea de progreso es el máximo de la ventana de 250, no el acumulado** (el brazo B
va 1.94·10³ → 479 → 362, y un acumulado no baja). El máximo global sale de la traza.

---------------------------------------------------------------------------------------------------

## 2026-10-09 — Lote 32 y micro-lote automático para una tarjeta de 48 GB

El autor entrena en una tarjeta Blackwell de la que puede usar **48 GB de VRAM** (puede ser una
porción de una más grande, compartida), con RAM de sobra, y quiere exprimirla. Decisión suya, no
medición: **lote 32**. Esta entrada dice qué cambió en `train3d.py` y qué de eso está medido,
derivado o supuesto. Reemplaza el punto «El lote (derivado)» de la entrada «Fase 2 en 3D», que lo
dejaba a la medición.

### Qué cambió

- `LOTE = 32` (antes 8) y `MICRO_LOTE = "auto"` (antes 4): `--micro-lote auto` elige el mayor
  divisor de `--lote` cuyo pico predicho entra en `--vram-gb` (default 48, en GiB) con 10 % de
  margen. Un entero sigue valiendo, y si no divide el lote es un error.
- `--vram-gb` se recorta a la memoria de la tarjeta, y si queda por debajo,
  `set_per_process_memory_fraction` no deja a PyTorch pasar de ahí: la tarjeta puede estar
  compartida. En CPU el presupuesto es `--vram-gb`, así que `auto` no depende de la máquina.
- `--sonda-vram`: en GPU, 2 iteraciones de forward + backward con cada divisor del lote, de menor a
  mayor y compilado como para entrenar, hasta el primero que no entra; imprime predicho contra
  medido, deja `sonda-vram.json` y sale sin entrenar.
- `micro_lote` y `vram_gb` pasan a las banderas libres al reanudar (y `sonda_vram`, que no toca
  nada): seguir en otra tarjeta está permitido, con un AVISO en el log.
- `WORKERS = 2` (antes 1). **Supuesto**: con 32 parches por iteración y el costo medido de uno
  (0.31–0.81 ms, `docs/mediciones/tejido-ldm100k3d.txt`) uno alcanza, pero la tarjeta nueva es
  otra máquina con otra CPU, y dos cuestan poco.
- El log de arranque imprime dispositivo, memoria, presupuesto, micro-lote y pico predicho; el
  final y `estado.json`, el pico medido junto al predicho. `config.json` y el estado guardan el
  micro-lote efectivo, no `auto`.

### El lote es receta; el micro-lote, hardware

Con acumulación el gradiente es el mismo para cualquier partición del lote: ninguna capa acopla
muestras (TDV no tiene normalizaciones), la pérdida del lote es la media de las de sus parches y su
gradiente, la media de los gradientes. El micro-lote se elige por la memoria y el lote no depende
de la tarjeta. Lo que cambia con la partición es el orden de suma. **Medido** en CPU y `referencia`
(`tests/test_train3d.py`): seguir con micro-lote 2 un estado de micro-lote 1 queda a lo sumo a
7.8·10⁻⁷ relativo de la corrida seguida en las 6 iteraciones siguientes (pérdida, `T`, `λ`, sus
gradientes y `‖g‖`; `T` y `λ`, idénticos), sin ser bit a bit.

Eso vale porque el ruido no cambia: `torch.randn` en CPU genera de a 16 valores, y con σ fijo y P³
múltiplo de 16 (48³ lo es) da los mismos números partido de cualquier manera (**medido** con torch
2.13: 4 parches de 48³ de una vez, 2 + 2 y 1 + 1 + 1 + 1, idénticos bit a bit; a 15³, no). Con
`--sigma-max` no: cada micro-lote sortea sus σ antes que su ruido. Y con `--guard-amp`, la
amplificación adjunta suma normas micro-lote por micro-lote, así que depende de la partición, y con
ella lo que rechaza el disyuntor. El AVISO al reanudar nombra lo que cambia en cada caso.

### Lote 32: el número del paper, no su contenido

**Citado** (Kobler et al., *A Stable Regularizer*, sec. 5.1): «mini batch size of 32». Es el mismo
**número** de parches por paso, no el mismo contenido: 32 × 48³ = **3.54 M vóxeles**, **4.3×** los
830 k valores de 32 × 93² × 3 en BSDS color. El paper no tiene versión 3D, así que ningún lote 3D
es «el del paper»: el 8 de antes igualaba los valores (885 k), el 32 iguala los parches. **No está
medido** cuál entrena mejor por hora: lo mide el piloto, y la pregunta concreta es `B_simple`
(columnas 5 y 10 de la traza). Si el lote queda muy por encima de la escala de ruido del gradiente,
cada parche de más compra menos de lo que cuesta (McCandlish et al., 2018): la iteración cuesta
proporcionalmente más y las iteraciones que hacen falta casi no bajan.

### En 48 GiB el micro-lote queda en 8, y usa ~60 % de la tarjeta

Derivado del modelo de memoria (abajo), con `--micro-lote auto` y el 90 % del presupuesto útil:

| caso | presupuesto GiB | útiles | `auto` | pico predicho GiB | el divisor siguiente |
|---|---|---|---|---|---|
| **48 GiB, 48³, `rapido`** | 48 | 43.2 | **8** | **28.83** | 16: 57.06 |
| porción de 48 GB decimales | 44.7 | 40.23 | 8 | 28.83 | 16: 57.06 |
| 5070 Ti (15.45 GiB según torch) | 15.45 | 13.90 | 2 | 7.67 | 4: 14.72 |
| 48 GiB, 64³ | 48 | 43.2 | 4 | 34.06 | 8: 67.51 |
| 48 GiB, `--checkpointing` | 48 | 43.2 | 32 | 23.93 | — |
| 48 GiB, `--checkpointing --guard-amp` | 48 | 43.2 | 32 | 12.64 | — |
| 5070 Ti, `--checkpointing` | 15.45 | 13.90 | 16 | 12.27 | 32: 23.93 |
| 48 GiB, 64³, `--checkpointing` | 48 | 43.2 | 16 | 28.25 | 32: 55.89 |
| 48 GiB, `intermedio` o `referencia` | 48 | 43.2 | 4 | 31.58 | 8: 62.54 |

El presupuesto va en GiB, como los reporta torch, que es lo que lee `train3d`. En la 5070 Ti son
16 585 719 808 bytes, **15.45 GiB** («total» en el OOM de `runs/s1-plain/log.txt`, 2026-09-01);
nvidia-smi da 16 303 MiB, 15.92 GiB, porque cuenta también lo que reserva el driver. Los GB de la
etiqueta son binarios: la tarjeta se vende como de «16 GB», y 16 GB decimales serían 14.90 GiB,
menos de lo que tiene. Una de «48 GB» reportará algo menos de 48 GiB, no 44.7: `--vram-gb 44.7` es
sólo para una porción pactada en GB decimales.

Entrarían 12 parches (42.94 GiB), pero 12 no divide 32. Con 8 se usa ~60 % de la tarjeta, y está
bien: lo que cuenta es saturar el cómputo, no llenar la memoria. En la 5070 Ti, a 48³, el costo por
vóxel bajó 21 % de ×1 a ×2 y sólo 6 % de ×2 a ×4 (3.02 → 2.39 → 2.26 µs, `costo-tdv3d.txt`): a ×4
ya saturaba. Que en la tarjeta de destino ×8 sature igual es un supuesto. En la 5070 Ti `auto` da
2, no el 4 de antes: los 14.72 GB de ×4 cupieron en la grilla, pero sin el margen; quien lo quiera
lo pide explícito.

### El modelo de memoria es una extrapolación desde 16 GB

`pico = FIJA + n · P³ · POR_VOXEL[modo, checkpointing]`, en GiB de
`torch.cuda.max_memory_allocated`, del entrenamiento sin la evaluación. **Derivado** de la grilla de la 5070 Ti
(`docs/mediciones/costo-tdv3d.txt`, 2026-10-08) y **no medido en la tarjeta de destino**:

| constante | valor | de dónde |
|---|---|---|
| `FIJA` | 0.61 GiB | la recta de 48³ en `rapido` por n = 1 y 4 (4.14 y 14.72 GB), en n = 0. Por mínimos cuadrados con n = 2 serían 0.625 y 3.525: ninguna elección cambia |
| `rapido` | 3.19·10⁻⁵ GiB/vóxel | la pendiente de esa recta, 3.527 GiB por parche de 48³ |
| `intermedio` y `referencia` | 7.0·10⁻⁵ | por encima de los cuatro medidos (2.48 a 7.60 GB), que sobreestima 9–17 % |
| `rapido` con checkpointing | 3.4·10⁻⁶ | 64³ × 4: 4.11 GB; 96³ × 1: 3.33. Con los S pasos checkpointeados (abajo) |
| `referencia` con checkpointing | 5.6·10⁻⁶ | 64³ × 2: 3.56 GB; `intermedio` no se midió y usa este. Ídem |
| el paso 1 sin checkpoint | + `POR_VOXEL[modo, False]` / S: 6.6·10⁻⁶ en total en `rapido`, 1.26·10⁻⁵ en `referencia` | **derivado**, con `--checkpointing` y sin `--guard-amp`: un paso de los S sin checkpointing (abajo) |
| `MARGEN` | 10 % del presupuesto | **supuesto**: la evaluación, la fragmentación del allocator y los workspace de cuDNN |

Contra la grilla (`test_el_modelo_de_memoria_reproduce_la_grilla`, las filas con checkpointing con
la entrada con gradiente, que es como las midió el borrador): a 48³ y 64³ en `rapido` queda a
menos de 2.2 % de lo medido (el peor, 8.97 contra 8.78 a 64³ × 1); en toda la grilla no subestima
más que el margen —el peor caso es 32³ × 4, 8 % abajo, 4.79 contra 5.21—, y las 9 configuraciones
que dieron OOM las predice por encima de los 15.45 GiB que reporta torch (la más justa,
`referencia` 48³ × 2, en 16.09).

Lo que el modelo no sabe: otra arquitectura puede traer otros workspace de cuDNN y otro allocator,
y la evaluación a 128³ no está en el ajuste. **La sonda es lo que lo mide.**

### Con `--checkpointing` el paso 1 va sin checkpoint, y el modelo no lo contaba

La grilla salió del borrador `scripts/tdv3d_borrador/tdv3d.py`, que checkpointea los 10 pasos
(`if self.efficient:`). `ddr/model.py` checkpointea un paso sólo si su `x` lleva gradiente (`if
self.efficient and x.requires_grad:`), y en `train3d` y en la sonda el `x_0` lo lleva sólo con
`--guard-amp` finito. Con el default (`inf`), el paso 1 corre sin checkpoint y guarda sus
activaciones durante todo el backward; los otros 9 sí van checkpointeados, porque desde el paso 1
`x` depende de los pesos. Lo encontró una revisión el 2026-10-09. **Medido** ese día en CPU con el
`VNet3D` de `ddr` (features 32, eager, fp32; scripts descartables fuera del repo):

| `VNet3D` de `ddr`, forward + backward | pico de RSS, 48³ × 1 | bytes que autograd guarda fuera del checkpoint, por vóxel, 24³ × 1 |
|---|---|---|
| sin checkpointing | 9.42 GiB | 119 558 |
| checkpointing, entrada sin gradiente (`train3d` sin `--guard-amp`) | 2.41 GiB | 12 603 |
| checkpointing, entrada con gradiente (los 10 pasos, como el borrador) | 1.68 GiB | 124 |

El paso 1 sin checkpoint guarda **1.04 pasos medios** de los 10 de un backward sin checkpointing, y
en el pico de RSS son +43 % (la revisión había medido 9.51, 2.50 y 1.65 GiB). En la GPU, compilado
y en bf16, no está medido: `vram_predicha` le suma **derivado** `POR_VOXEL[modo, False] / S`, que en
`rapido` son 3.19·10⁻⁶ GiB/vóxel, del orden de la constante misma (3.4·10⁻⁶). Con `--guard-amp` la
entrada lleva gradiente, los 10 pasos van checkpointeados y las constantes valen tal cual; `train3d`
y la sonda le pasan a `vram_predicha` cuál de los dos casos es. La premisa la fija
`test_sin_guard_amp_el_paso_1_va_sin_checkpoint`: si `ddr/model.py` pasara a checkpointear también
el paso 1, falla, y la suma sobra.

Lo que cambia con eso:

- **`--checkpointing` baja la memoria ~5×, no ~9×** (3.19·10⁻⁵ contra 6.6·10⁻⁶ GiB/vóxel en
  `rapido`); ~9× sólo con `--guard-amp`. El ~9× de la entrada «El TDV33 3D en la 5070 Ti» es del
  borrador. El +27 % de tiempo también: con 9 pasos recomputados sería algo menos, sin medir.
- **En la 5070 Ti con `--checkpointing`, `auto` habría dado OOM**: elegía 32 con 12.64 GiB
  predichos, y el pico sería de ~23.9 (derivado). Ahora elige 16 (12.27). A 64³ en 48 GiB,
  igual: elegía 32 (29.13 predichos, ~55.9 derivados) y ahora 16 (28.25).
- **Los 3.59 GiB de la corrida de humo no hace falta atribuírselos a la evaluación**: era 48³ × 4
  con `--checkpointing` y sin `--guard-amp`, y el modelo predice ahora 3.53 para el entrenamiento
  solo (antes, 2.11). La evaluación sigue sin medirse por separado.
- La alternativa —checkpointear también el paso 1 en `ddr/model.py`, que con `use_reentrant=False`
  se puede sin entrada con gradiente— devolvería el ~9×, pero toca el modelo que comparte el 2D: no
  se hizo.

**Lo que se retira:** el ~9× de `--checkpointing` para `train3d` sin `--guard-amp` (fila nueva en
`## Descartado`), y dos afirmaciones publicadas que salen de él; esas entradas no se reescriben.
De «El entrenador 3D en la rama `3D-Training`», que los números del borrador «valen para este
modelo»: valen sin `--checkpointing` o con `--guard-amp`. De «El TDV33 3D en la 5070 Ti», los
~15 M vóxeles por micro-lote con checkpointing en una tarjeta de 64 GB (64 GB / 4.2 kB, 8 parches
de 120³): sin `--guard-amp`, con la misma cuenta y 6.6·10⁻⁶ GiB/vóxel ≈ 7.1 kB, son **~9 M, ~5
parches de 120³** (derivado). Y, de la primera versión de esta entrada, sin publicar, los
12.64 GiB predichos con `--checkpointing` en 48 GiB y la atribución a la evaluación de la
diferencia entre 3.59 y 2.11 GiB.

### La sonda, y una trampa de dynamo que la habría engañado

`--sonda-vram` corre `R.grad` compilado como el entrenamiento (`dynamic=False`), así que cada
micro-lote recompila. **Medido** en CPU el 2026-10-09 (torch 2.13, script descartable fuera del
repo): cada forma deja **dos** entradas en la caché de dynamo —el primer paso del esquema recibe
`x` sin gradiente y los demás con—: 2, 4 y 6 grafos tras n = 1, 2 y 4, con y sin checkpointing.
Con lote 32 serían hasta 12, y pasado `recompile_limit` (8) dynamo sigue en eager: la sonda habría
medido 16 y 32 sin compilar, que a 64³ es 61 % más memoria (14.16 contra 8.78 GB). La sonda sube el
límite a 2 × divisores + 2. El entrenamiento usa tres entradas (derivado: las dos del micro-lote y
la de la evaluación) y no lo necesita.

### Cómo se verificó

- **Suite: 143 pasados, 0 fallados** (CPU, `-k "not published_psnr"`: 4 deseleccionados); eran
  130, y los 13 nuevos son los de abajo.
- Tests nuevos en `tests/test_train3d.py`: los casos de `auto` (48³ en 48 y en 44.7 GiB da 8, en 16
  y en la 5070 Ti da 2, 64³ da 4, con checkpointing 32 en 48 GiB y 16 en la 5070 Ti y a 64³ —32
  con `--guard-amp`—, lote 24 da 12, error si ni 1 entra); que `auto` sea siempre el mayor divisor
  que entra, sobre una grilla de lotes, parches, modos, presupuestos y `--guard-amp`; el modelo
  contra la grilla medida y contra la corrida de humo; que sin `--guard-amp` el paso 1 vaya sin
  checkpoint; los errores de `--micro-lote` antes de tocar `--out`, con el paso 1 contado según
  `--guard-amp`; `auto` registrado como número en `config.json`, el estado y el log; reanudar con otro
  micro-lote (avisa y sigue a 7.8·10⁻⁷) y con otro lote (se rechaza); y `--sonda-vram` sin GPU, que
  termina con un mensaje y sin escribir nada.
- **Lo que no se probó:** la sonda en GPU, ni una sola iteración con lote 32 en GPU. Las GPUs
  estaban ocupadas por otra sesión, y la regla 3 pide no medir así.

**Consecuencia práctica:** en la tarjeta nueva, primero `--sonda-vram` y 200 iteraciones para el
s/it (enmienda del pre-registro del piloto, arriba); con esos dos números se lanza. Si la sonda
admite más que `auto`, `--micro-lote` explícito.

---------------------------------------------------------------------------------------------------

## 2026-10-09 — Cubos 100 % cerebro: hoy sólo el 4 % de los parches de 48³ cae entero en el cerebro

Rama `3D-Cerebro`, sacada de `3D-Training` a pedido del autor: entrenar con cubos que caigan enteros
dentro del cerebro, sin fondo ni cráneo, y con la mayor cantidad de cerebros posible. **No se
implementó**: el autor pausó el 3D para seguir con pruebas en 2D. Esta entrada guarda lo medido y
las decisiones que quedan abiertas. Crudos en `docs/mediciones/mascaras-cerebro-ldm.txt`,
`cobertura-cubos-cerebro.txt` y `verificacion-cubos-cerebro.txt`; las herramientas,
`scripts/mascaras_cerebro_ldm.py`, `cobertura_cubos_cerebro.py` y `verifica_cubos_cerebro.py`
(locales). n = 24 sujetos (los 12 de evaluación y 12 sorteados de `lista_train_8000.txt`).

### Cómo están hoy los parches

Con el criterio de `ddr/data3d.py` (48³ con al menos 35 % de vóxeles > 0.05), de las posiciones que
pasan **cae entero en el cerebro el 4.0 %** [2.8–6.2], y el parche medio es **60 % cerebro**: el
resto es cráneo, cuero cabelludo, aire y LCR externo. «Tejido» > 0.05 incluye la cabeza entera,
porque LDM_100k no está skull-stripped.

### Las máscaras de cerebro

En la máquina no había ninguna herramienta; se instalaron dos en entornos aparte (versiones en
`verificacion-cubos-cerebro.txt`): **deepbet 1.0.2** y **HD-BET 2.0.1**.

| | deepbet | HD-BET con TTA | HD-BET sin TTA |
|---|---|---|---|
| s por volumen, GPU (5070 Ti) | 0.16 | 2.53 | 0.51 |
| s por volumen, CPU (16 hilos) | 0.48 | 178 | 21.8 |

- **Dice entre las dos: 0.957** de mediana [0.940–0.976], y la diferencia va casi toda en un sentido:
  deepbet mete 69–178 ml fuera de HD-BET (LCR externo, la zona de los senos, la cisterna magna, un
  foco brillante de la base), hasta 10.7 mm. HD-BET sigue la superficie pial y en algunos sujetos
  toca la tabla interna en el vértice.
- **La intersección**, con un margen de 1 vóxel, quedó siempre sobre corteza en las vistas de borde
  miradas (PNG del verificador y de las dos etapas).
- Con margen 1, HD-BET sin TTA da casi los mismos cubos que con TTA (Jaccard de esquinas
  0.969–0.991, ~2 % menos de esquinas) y cuesta 5× menos.

### Qué ve el entrenamiento con cubos 100 % cerebro

Margen e = 1 sobre la intersección; «franja» = vóxeles a ≤ 6 mm del borde de la máscara (27.8 % del
cerebro), que es la cáscara externa: la corteza de los surcos hondos y la ínsula queda en «profundo».

| cubo | % del cerebro cubierto | % de la franja cubierta | % de lo profundo | esquinas por cerebro |
|---|---|---|---|---|
| 24³ | 85.5 | 49.6 | 99.3 | 423 k |
| 32³ | 80.1 | 38.1 | 96.2 | 250 k |
| 40³ | 74.1 | 28.3 | 92.0 | 134 k |
| **48³** | **66.3** | **18.5** | **84.4** | **60 k** |

- Los cubos que entran enteros se juntan en el centro: sólo el **0.25 %** [0.20–0.30] de lo que ve el
  entrenamiento cae en la franja (32³). Es lo que el autor llamó «tomar los núcleos».
- Un cubo 100 % cerebro tiene ~17 % de LCR (corte de una mezcla de tres gaussianas por sujeto), casi
  todo ventricular: excluirlo dejaría sin cubos (sólo el 0.1 % de los de 32³ tiene < 1 %).
- Con 56³ o más se pierde ≥ 37 % del cerebro, y con 64³ y e = 2 dos sujetos se quedan sin cubos.

### Almacenamiento

Recortado a la caja de su máscara (~133 × 174 × 134), un cerebro pesa **3.1 MB en uint8** y 6.2 MB en
float16, contra 10.4 MB del recorte actual. Cuantizar a 8 bits da un error RMS de 0.0011, el 1.15 %
de σ = 25/255. 8 000 cerebros: 25 GB; 20 000: 62 GB; 100 000: 310 GB (de disco). El costo de las
máscaras es derivado de los tiempos de arriba: con las dos herramientas sin TTA en GPU, 0.67 s por
cerebro, o sea ~1.5 h para 8 000, ~3.7 h para 20 000 y ~19 h para 100 000.

### T1 y T2

El autor preguntó si LDM_100k tiene T2. No: los 100 000 archivos son `_T1w` y **ninguno tiene el
contraste invertido** (en la caja central, razón ventrículos / sustancia blanca de 0.23 a 0.98, nunca
≥ 1; correlación con el promedio nunca negativa). La cola de baja correlación son T1 con
ventrículos muy chicos. Crudo en `docs/mediciones/contraste-t1-ldm100k.txt`. La polaridad de T2
(LCR > gris > blanca) se puede imitar con `--aug-signo`, porque K1 de media cero hace a la energía
invariante a un offset.

### Lo propuesto, sin decidir

1. Máscara por cerebro: deepbet ∩ HD-BET sin TTA, margen 1.
2. Cubos de 48³, como en `3D-Training`, para no cambiar dos cosas a la vez (32³ cubre más corteza,
   cuesta +23 % por vóxel y deja la escala más gruesa de TDV en 8³).
3. 20 000 cerebros en uint8, elegidos con la regla mezcla; o los 100 000 leyendo de disco.
4. Dónde correr las máscaras: acá con una GPU libre, o en la tarjeta de destino.

### Sobre la rama

El autor pidió esta rama como entrenamiento aparte. Lo recomendado fue usarla para desarrollar e
integrarla a `3D-Training` como un dataset nuevo más una bandera de muestreo, porque dos ramas vivas
con código distinto confunden cualquier comparación entre ellas (regla 4). Integrar o no lo decide
el autor.

---------------------------------------------------------------------------------------------------

## 2026-10-10 — VEREDICTO PARCIAL (brazo A) de `modo-y-tinit-100k-s1`: `rapido` reproduce `referencia` a 10⁵, y `λ` no

`rapido-100k-s1` terminó: `"terminada": true`, 100 000/100 000, **31.546075 dB**, 80 rechazos,
2 rebobinados, **12.648 h**, VRAM pico 5.07 GB. El brazo B sigue en vuelo (14.9 h), así que esto
califica **sólo las cuatro predicciones del brazo A**; el veredicto completo espera. Crudo en
`docs/mediciones/rapido-vs-referencia-100k.txt`, herramientas `scripts/measure_rapido_vs_ref.py` y
`scripts/measure_eigen_modo.py` (locales).

### Las cuatro predicciones, una por una

| # | predicción | resultado | veredicto |
|---|---|---|---|
| 1 | PSNR @100 000 en [31.22, 31.88] | **31.546075** | **CONFIRMADA** |
| 2 | termina en 13.0–13.8 h | **12.648 h** = 0.5059 h/4 000 it | **FALLA** (abajo) |
| 3 | rechazos en [82, 489] | **80** | **FALLA** (abajo, por 2) |
| 4 | VRAM pico ≈ 5.97 GB | **5.07 GB** | **FALLA** (abajo) |

**Las tres fallas van en la misma dirección: `rapido` salió más barato de lo predicho en los tres
ejes.** La cifra publicada de 0.52–0.55 h/4 000 it sobreestima para BSDS en 3 canales —lo medido es
**0.5059**— y la tabla de VRAM por modo (5.97 GB) también. La 3 falla por dos rechazos sobre un
umbral que yo mismo había puesto en 0.5× los 163 del control; el fundamento era la dispersión de 6×
entre las dos semillas del control, y 80/163 = 0.49× cae justo afuera. **No se arregla el umbral a
posteriori**: la predicción falló.

### El endpoint: 50 de 50 puntos pareados dentro del piso de detección

| | valor |
|---|---|
| puntos pareados (cada 2 000 it) | **50** |
| con \|Δ\| < 0.33 dB (piso con n = 1) | **50 de 50** |
| con \|Δ\| < 0.05 dB | **49 de 50** |
| Δ = rapido − referencia | media **+0.0033**, mediana **−0.0006**, sd 0.0318 |
| \|Δ\| mediana | **0.0022 dB** |
| extremos | **+0.2210** @6 000 y **−0.0257** @46 000 |
| endpoint | 31.546075 contra 31.547008 = **−0.00093 dB** |
| velocidad | **2.980×** (12.648 h contra 37.697) |

El único punto fuera de 0.05 dB es la it 6 000, y **no es el brazo A el que se mueve**: es el
control, que tiene ahí un pozo (31.005, bajando desde 31.172 y volviendo a 31.310 en la 8 000)
coincidente con su primera racha de rechazos, en las it 6 000–6 002. El outlier es un transitorio
del control.

### Las imágenes: indistinguibles

Convención de `docs/presentacion/denoise.py` (σ = 25, semilla 1234 + k, `model(z, z)[-1]` sin
recortar). Figura en `docs/presentacion/salida/img_rapido_vs_referencia.png`.

| imagen | referencia | rapido | Δ | PSNR entre las dos salidas | RMS | max |
|---|---|---|---|---|---|---|
| castillo (102061) | 32.492 | 32.487 | −0.004 | **48.58 dB** | 0.95/255 | 12.2/255 |
| cebras (253027) | 31.067 | 31.079 | +0.012 | **47.88 dB** | 1.03/255 | 14.2/255 |

`PSNR(rapido, referencia)` es el PSNR de una salida tomando la otra como referencia: ~48 dB es
~1 nivel de gris de diferencia cuadrática media sobre 255. El máximo de 12–14 niveles está en
píxeles aislados.

### Los pesos y los escalares: `λ` es lo único que se separa

| | referencia | rapido | cociente |
|---|---|---|---|
| `T` | 0.08264187 | 0.08632571 | 1.045 |
| `λ` | 0.12909329 | **0.02749158** | **0.213** |
| `w = λ/T` | 1.562081 | **0.318463** | **0.204** |
| `‖Δθ‖/‖θ‖` | — | — | **68.47 %** |

El 68.47 % **no es grande en este repo**: dos semillas de la misma configuración ya están a
**131.8 %** y dan el mismo PSNR (`## Descartado`, fila de `‖Δθ‖`). Los dos modos quedan más cerca
entre sí que dos semillas.

**`λ` nunca se recupera**, y el cociente cae monótono: 0.553 @2 000, 0.439 @10 000, 0.228 @50 000,
**0.213 @100 000**. Es la continuación del pozo medido el 10-09 —mínimo 0.013524 = 1.07·δ en la
it 2 848 contra 0.030028 = 2.37·δ del control— que nunca se cierra. Que no cueste PSNR es
consistente con lo ya medido: en denoising `λ` es casi plana, apagarla cuesta 0.033 dB.

**Pero `w = λ/T` queda 4.9× abajo**, del que

> **RETIRADO el 2026-10-12, dos veces.** `../tdv/wTDV_QSM_torch.py:188` usa la red entera (`vn(z·α, z·α)/α`) y **no lee `T` ni `λ`**. Y `w` es una coordenada de **gauge**: bajo `(T, K_N) → (cT, K_N/c)` el modelo no cambia y `w → w/c`, así que su spread entre corridas no mide el modelo. Lo invariante es `ρ = λ‖x−z‖ / T‖∇R‖`. Entrada del 10-12.

AGENTS.md ya decía que «no se traslada sin medir». Así que la respuesta a «son muy parecidos» tiene
dos mitades: para denoising sí, indistinguibles; para lo que la fase 3 va a usar, **no**.

### Los eigenvectores: la comparación directa NO contesta, y el control lo delata

Se calcularon las eigenfunciones de `R` del brazo A con `docs/presentacion/eigen_extra.sh` (CPU,
~22 min), sobre las mismas dos imágenes que usa la presentación, contra las del control que ya
estaban en disco. Crudo en `docs/mediciones/eigen-rapido-vs-referencia.txt`.

**El número directo engaña.** La correlación entre las dos eigenfunciones convergidas es 0.990
(img1) y 0.951 (img2). Pero cada una correlaciona con **su propio recorte de partida** casi igual:

| imagen | corr(ref, arranque) | corr(rapido, arranque) | corr(ref, rapido) |
|---|---|---|---|
| img1 (oso, 100007) | 0.98716 | 0.99053 | 0.99015 |
| img2 (textura, 100039) | 0.93736 | 0.94581 | 0.95126 |

A 1 000 iteraciones el iterado no se aleja lo suficiente del recorte común, así que el parecido
mide el punto de partida y no el modelo. **Esa comparación queda descartada como lectura**, y el
control que lo detecta está en el script.

**La medición limpia: `∇R` de los dos modelos en el MISMO punto.** Sin optimización y sin arranque,
`R.grad(x)` es propiedad del modelo. El coseno es la prueba de «misma función» porque es invariante
de escala, y la escala importa: el gauge `(T, K_N) → (cT, K_N/c)` da el mismo modelo con otro `∇R`.

| punto | coseno | `‖∇R‖` rap/ref | `Λ` ref | `Λ` rap |
|---|---|---|---|---|
| recorte img1 | 0.95296 | 0.671 | 0.017603 | 0.014374 |
| recorte img2 | 0.94998 | 0.733 | 0.23176 | 0.19980 |
| **img1 + ruido σ = 25** | **0.98247** | 0.802 | 1.5341 | 1.2939 |
| eigenfunción de referencia | **0.58015** | 1.146 | 0.0045325 | 0.0069558 |
| eigenfunción de rapido | **0.59672** | 0.595 | 0.0042243 | 0.0042556 |

- **En el régimen en que la red opera —imagen con ruido— los dos `∇R` apuntan igual a 0.982**, y en
  imagen limpia a 0.95. La magnitud de `rapido` queda 0.67–0.80× la del control.
- **En las eigenfunciones el coseno se cae a 0.58–0.60**: ahí los dos modelos discrepan fuerte, cada
  uno tiene las suyas. Pero ahí `‖∇R‖` vale **24–36 contra 1 611** en el punto ruidoso, ~50× menos:
  **los modelos difieren más justo donde el regularizador casi no actúa.** Eso es también la
  explicación de por qué la comparación de eigenfunciones no concluía.
- `Λ` comparada **en el mismo `x`** cae a 0.82–0.86× de forma consistente, o sea casi toda la
  diferencia es la escala de `‖∇R‖`, no la forma.

### Qué queda cerrado y qué no

**Cerrado:** «la comparación de modos sigue sin estar hecha a iteración pareada y larga», que era
el punto 3 de la cola de la fase 1 («¿`referencia` es de verdad peor?»). Ninguno de los dos modos
entrena mejor. Actualizado en `AGENTS.md`, `README.md` y `docs/metricas.md`.

**No cerrado, y hay que decirlo:** es **una semilla**. `rapido` no repite bit a bit, así que su
varianza entre corridas a 10⁵ sigue sin medir y esta corrida **no se puede replicar contra sí
misma**. Lo que está medido es que *esta* trayectoria sigue a la de `referencia` en 50 puntos, no
que otra semilla en `rapido` caiga en el mismo lugar.

### Cómo NO se leen estos resultados

- **No se lee «bf16 es equivalente a fp32 IEEE».** `rapido` cambia tres cosas a la vez (bf16,
  `cudnn.benchmark`, determinismo) y esto no atribuye nada a ninguna.
- **No se lee «`rapido` sirve para citar números».** Sigue sin repetir: para un número citado y
  para comparar dos corridas, `referencia`.
- **No se lee «`λ` da igual».** Da igual *en denoising*. El 4.9× de `w = λ/T` es exactamente lo que
  AGENTS.md marca como no trasladable a QSM.
- **Las tres predicciones que fallaron no se re-escriben.** Fallaron por debajo, o sea que el
  modelo de costo era pesimista; eso se corrige en la tabla de costos, no en la predicción.

---------------------------------------------------------------------------------------------------

## 2026-10-10 — Registro de lanzamiento de `t1p1delta-rapido-100k`: `T_init = 1.1·δ`, sondeo de la pared

**Registro de lanzamiento, no un pre-registro.** Pedido por el autor el 2026-10-10, textual: «solo
para ver que sale». No contesta una pregunta registrada y no tiene los ocho campos; sondea la pared
absorbente de `T`. Queda escrito como tal.

**Qué cambia.** `--t-init 0.013914021704740865` = **1.1·δ_adam** contra el 8·δ del control, en modo
`rapido`, semilla 1, 100 000 iteraciones, todo lo demás en default. Difiere del brazo A
(`rapido-100k-s1`, ya terminado) **sólo en `T_init`**, así que tiene un control pareado y terminado.

**Viola a propósito la regla 2-δ del repo**, donde `2·δ` es un **piso** y no un valor de trabajo.
Eso está dicho antes de lanzar, no después.

**Lo que se espera, escrito antes de los datos.** `1.1·δ` está **por debajo de la reserva exigida
medida**: 1.795·δ sobre la secuencia de `∂L/∂T` de `t8delta-100k-s0` y 1.638·δ sobre la de `-s1`
(`docs/mediciones/replay-T-2delta-100k.txt`). El replay predice que `T` **toca 0**: 517 y 443
iteraciones en cero respectivamente.

Con dos salvedades que valen tanto como la predicción:

1. **Ese replay es una traslación, no un contrafactual** —el paso de Adam sobre `T` no depende de
   `T`—, así que lo único que dice es `1.1·δ − 1.638·δ < 0`.
2. **La evidencia real apunta al revés.** Las dos corridas de `2·δ` dieron la vuelta enseguida: en
   `tinit-2delta-50k` el mínimo de `T` está en la **iteración 2** y de ahí sube, y `t2delta-100k-s1`
   va en 2.56·δ a la it 60 250, subiendo desde 2.00.

**El riesgo que NO es traslación**, y es el que de verdad pesa: a `1.1·δ` un solo pico cruza la
pared, «sea del tamaño que sea» (es la definición de `δ_adam`), y **el disyuntor no puede verlo**
porque `T → 0` no tiene pico y cada lote se ve sano. El precedente directo es `runs/tinit/B-larga`,
que murió con `T = 0` y 20.173 dB. Si muere así, **no es un fallo del disyuntor**.

**Una sola predicción, para que sea falsable:** `T` toca 0 antes de la it 100 000. Si no la toca, la
reserva exigida medida sobre gradientes de `T ≈ 0.08` no vale para `T ≈ 0.014`, y hay que rehacer
cómo se estima.

**Mecánica.** Unidad `tdv-t1p1delta-rapido-100k`, programada con `systemd-run --user
--on-active=51m` para disparar a las **13:58:29** —la hora que pidió el autor— en la **GPU 1**; la 0
tiene `t2delta-100k-s1` en vuelo. El `plan.sh` **no arranca si la GPU 1 no está libre** (umbral
1 500 MiB) y lo dice en el log: al momento de programarlo la tenía ocupada el autor con otro
proyecto, y con las GPUs ocupadas no se mide (regla 3).

**Ojo con una cosa de la máquina:** `loginctl show-user santicien -p Linger` da `Linger=no`, así que
si la sesión de usuario se cierra del todo antes de las 13:58 el timer muere con ella. Se relanza a
mano con `systemd-run --user --unit=tdv-t1p1delta-rapido-100k bash
runs/t1p1delta-rapido-100k/plan.sh`.

---------------------------------------------------------------------------------------------------

## 2026-10-10 — Observación en vuelo de `t1p1delta-rapido-100k`: `T` no se fue a la pared, y `T_init` mueve `w` más que el modo

**En vuelo, it 12 000 de 100 000.** No es un veredicto: la corrida tiene 88 000 iteraciones por
delante y todo esto puede darse vuelta. Se escribe ahora porque toca una afirmación publicada hoy
mismo. Arrancó a las 13:58:37 con la GPU 1 en 16 MiB (el guard del `plan.sh` la encontró libre).

### La predicción registrada va camino a FALLA, y es una falla informativa

`T` arrancó en 1.1·δ y **subió desde la primera medición**: 1.35·δ a la it 250, meseta en
1.44·δ hasta la it 6 000, y desde ahí sube despacio hasta **1.52·δ a la it 12 000**. No se acercó a
cero en ningún momento.

Eso es exactamente lo que hicieron las dos corridas de `2·δ` —dan la vuelta enseguida y suben— y
confirma la salvedad que quedó escrita antes de lanzar: **el replay de `T` es una traslación, no un
contrafactual**, así que la reserva exigida de 1.638–1.795·δ, medida sobre la secuencia de `∂L/∂T`
de corridas con `T ≈ 0.08`, **no se aplica a una corrida con `T ≈ 0.014`**. El gradiente de `T`
depende de `T`; la traslación lo trata como si no. Si la corrida termina sin tocar cero, lo que se
retira no es la corrida: es el método de estimar la reserva.

### Lo que sí fue a la pared es `λ`, y volvió

`λ` cae monótona desde 0.1 hasta **0.00831 = 0.657·δ** alrededor de la it 3 000 —por debajo de
`δ_adam`, dentro de la banda que el repo tiene marcada como letal para un escalar— y de ahí
**recupera sin interrupción** hasta 1.94·δ a la it 12 000.

Sobre la grilla de eval, los tres mínimos de `λ` caen en **la misma iteración, la 4 000**, y lo que
cambia es la profundidad:

| corrida | `λ_min / δ` | `T` en el tramo |
|---|---|---|
| `t8delta-100k-s1` (`referencia`, 8·δ) | **2.689** | 7.2–8.2·δ |
| `rapido-100k-s1` (`rapido`, 8·δ) | **1.287** | 7.2–8.2·δ |
| `t1p1delta-rapido-100k` (`rapido`, 1.1·δ) | **0.886** | 1.44–1.52·δ |

Es el mismo patrón que el del 10-09 entre `rapido` y `referencia`, ahora con un tercer peldaño:
**lo que cambia es la profundidad del pozo, no cuándo pasa.** Bajar `T_init` lo profundiza encima
de lo que ya lo profundiza bf16.

### El hallazgo que reordena la historia de ayer: `T_init` mueve `w` más que el modo

`w = λ/T`, y el veredicto del brazo A lo dejó como la

> **RETIRADO el 2026-10-12, dos veces.** `../tdv/wTDV_QSM_torch.py:188` usa la red entera (`vn(z·α, z·α)/α`) y **no lee `T` ni `λ`**. Y `w` es una coordenada de **gauge**: bajo `(T, K_N) → (cT, K_N/c)` el modelo no cambia y `w → w/c`, así que su spread entre corridas no mide el modelo. Lo invariante es `ρ = λ‖x−z‖ / T‖∇R‖`. Entrada del 10-12.

única cosa que no se traslada (`rapido` 4.9× por debajo de `referencia`). A iteración pareada:

| it | `referencia` 8·δ | `rapido` 8·δ | `rapido` 1.1·δ |
|---|---|---|---|
| 2 000 | 0.3554 | 0.1959 | **1.1177** |
| 4 000 | 0.3378 | 0.1613 | 0.6166 |
| 8 000 | 0.4414 | 0.2090 | 1.0758 |
| 12 000 | **0.5627** | **0.2381** | **1.2764** |
| 100 000 | 1.5621 | 0.3185 | *en vuelo* |

A la it 12 000, `rapido` con 1.1·δ da `w` **2.3× por encima** de `referencia` y **5.4× por encima**
de `rapido` con 8·δ. O sea que el `w` bajo del brazo A **no es una propiedad del modo**: el modo lo
mueve 2.4× (0.563 → 0.238 a `T_init` fijo) y `T_init` lo mueve 5.4× (0.238 → 1.276 a modo fijo).

> **ACOTADO el mismo día, it 26 000: los dos múltiplos son de la iteración, no del efecto.** `w` de
> `referencia` trepa rápido —0.355 @2 000, 0.563 @12 000, **1.143 @26 000**, 1.562 al final— mientras
> el de 1.1·δ se amesetó en 1.30–1.38. A la it 26 000 el modo mueve `w` **3.57×** y `T_init` **4.33×**,
> contra 2.4× y 5.4× a la it 12 000: se están juntando. **El «5.4× contra 2.4×» no se cita como
> tamaño de efecto.** Lo que sí aguanta, porque es cualitativo y vale en las trece iteraciones
> pareadas, es que **`rapido` a 1.1·δ alcanza un `w` del orden del de `referencia` y `rapido` a 8·δ
> nunca se le acerca** (0.32 contra 1.14 a la it 26 000).

**`T_init` es la perilla dominante sobre `w`, no la precisión.** Eso **no retira** el número del
brazo A —`rapido` a 8·δ sigue dando `w` 4.9× bajo al final— pero sí retira la lectura de que el
responsable sea bf16.

Queda sin decidir, y hay que decirlo: `w` todavía se mueve mucho a la it 12 000 en las tres
(`referencia` casi triplica de 12 k a 100 k), así que **esto no predice el `w` final de nada**.

### El costo, que es real y grande

| | rechazos @12 000 | rebobinados @12 000 |
|---|---|---|
| `referencia` 8·δ | 3 | 0 |
| `rapido` 8·δ | 5 | 0 |
| `rapido` 1.1·δ | **124** | **4** |

> **RETIRADO el mismo día, it 26 000: «y desde ahí se calma» es falso.** Lo escribí mirando hasta
> la it 12 000. El costo **no** está adelante: los rechazos entran a rachas durante toda la corrida
> —por tramo de 2 000 it: 87, 25, 4, 0, 4, 4, 10, **73**, 29, 42, 3, **58**, 6— y a la it 26 000 van
> **345 rechazos y 8 rebobinados** contra **35 y 1** de `referencia` y **37 y 1** de `rapido` a 8·δ.
> El múltiplo se sostiene en ~10× de rechazos y 8× de rebobinados, y no decae.

~25–40× más rechazos, y 4 rebobinados donde los controles no tuvieron ninguno; 87 de esos 124 y 3
de los 4 rebobinados caen antes de la it 1 750. **El disyuntor está haciendo
el trabajo que a 8·δ no hace falta hacer**, y es lo que separa esta corrida de `runs/tinit/B-larga`,
que murió con `T = 0` sin disyuntor.

### Y sin embargo el PSNR va pareado

| it | `referencia` 8·δ | `rapido` 8·δ | `rapido` 1.1·δ |
|---|---|---|---|
| 2 000 | 31.003 | 30.997 | 30.839 |
| 6 000 | 31.005 | 31.226 | 31.200 |
| 12 000 | 31.377 | 31.381 | **31.349** |

Los seis puntos pareados están dentro de **0.16 dB**, y los cuatro últimos dentro de **0.04**: por
debajo del piso de detección de 0.33 dB con n = 1. Es coherente con lo ya cerrado el 10-07 —`T_init`
de 2·δ a 10·δ no mueve el endpoint— y lo extiende un factor 7 hacia abajo, con la diferencia de que
acá el endpoint todavía no existe.

### Al pasar el primer halving del lr (it 26 000): dos cosas nuevas

**El déficit de PSNR es chico pero consistente, no ruido alrededor de cero.** Desde la it 8 000 las
diez evaluaciones pareadas contra `referencia` dan **el mismo signo**: −0.040, −0.029, −0.028,
−0.027, −0.043, −0.025, −0.026, −0.029, −0.024, −0.035, media **−0.031 dB**. La magnitud está un
orden por debajo del piso de detección de 0.33 dB y no cambia ninguna decisión; lo que no se puede
decir es que sea ruido. **Tampoco es un test de signos limpio**: los puntos de eval consecutivos de
una misma corrida están autocorrelacionados, así que diez signos iguales valen mucho menos que
`2⁻¹⁰`. La lectura prudente es «parece un sesgo real de ~0.03 dB, sin cuantificar».

**El pico más grande que vio el repo: `max‖g‖` = 2.39·10⁷ en la ventana de la it 24 000**, o sea
**~2 400× la cota absoluta** `GUARD_ABS = 1e4`. El disyuntor lo rechazó y la corrida siguió; `T`
**subió** a través del evento (1.639·δ @24 000 → 1.680·δ @26 000) y la ventana siguiente volvió a
`max‖g‖` = 96.2. Es la clase de evento que el disyuntor **sí** puede ver —un pico en `‖g‖`—, al
revés de la deriva `T → 0`, que es la que no tiene pico y sigue siendo el riesgo no cubierto.

### Al pasar el segundo halving (it 52 000): el hallazgo, y mi segunda corrección sobre lo mismo

**Primero la corrección, porque es la segunda sobre la misma frase.** Dije «se calma desde la
it 1 750» (falso), después «el múltiplo se sostiene en ~10× y no decae» (también falso). Lo cierto,
con el costo partido por fase de lr:

| | lr = 4e-4 | lr = 2e-4 | lr = 1e-4 | lr = 5e-5 |
|---|---|---|---|---|
| `referencia` 8·δ | 4 / 0 | 88 / 3 | 12 / 0 | 59 / 1 |
| `rapido` 8·δ | 37 / 1 | 43 / 1 | 0 / 0 | 0 / 0 |
| **`rapido` 1.1·δ** | **345 / 8** | **27 / 0** | **0 / 0** | *en vuelo* |

(rechazos / rebobinados por fase.) **El 93 % de los rechazos de 1.1·δ y sus ocho rebobinados caen
en la primera fase de lr**, y el primer hito los corta en seco. Las dos lecturas anteriores eran
artefactos de dónde terminaba mi ventana de datos: la primera cortaba en la it 12 000, la segunda
en la 26 000, justo encima del hito. **La frase robusta necesita la fase de lr, no una iteración.**

### `δ_Adam` no es una constante: se parte a la mitad en cada hito del lr

`train.py:51` calcula `DELTA_ADAM = LR / √(1−β₂)` **una vez**, desde el `LR` inicial. Pero el
desplazamiento que Adam imprime es proporcional al lr **vivo**, así que el calendario lo va
partiendo: 0.012649 → 0.006325 → 0.003162 → 0.001581 → 0.000791. La reserva de `T` medida contra el
δ vivo **se duplica en cada hito aunque `T` no se mueva**:

| reserva `T / δ_vivo` al cerrar cada fase | 4e-4 | 2e-4 | 1e-4 | 5e-5 | 2.5e-5 |
|---|---|---|---|---|---|
| `referencia` 8·δ | 6.41 | 12.60 | 25.67 | 52.24 | 104.53 |
| `rapido` 8·δ | 6.49 | 12.98 | 26.65 | 54.59 | 109.20 |
| **`rapido` 1.1·δ** | **1.65** | **3.84** | **7.86** | *en vuelo* | |

**Esto es lo que explica que la corrida esté viva.** `T_init = 1.1·δ` es 1.1·δ **sólo durante la
primera fase de lr**: a la it 26 000 esa misma `T` ya vale 3.36·δ_vivo y a la 52 000, 7.86. La
ventana de peligro de un `T_init` bajo **es finita y se cierra en el primer hito** — 25 000 de
100 000 iteraciones—, y el disyuntor sólo tiene que cubrir ese tramo. Es estructural: sale del
código y del calendario, no de esta corrida.

**Lo que esto NO dice.** No dice que la reserva prediga los rechazos: `referencia` tiene 88 en una
fase donde su reserva es 12.6·δ, o sea que los rechazos son un fenómeno de picos de `‖g‖` y no de
reserva. Lo que la reserva gobierna es **cuánto daño hace un pico**, no con qué frecuencia aparece.
Y tampoco se invoca acá el escalamiento `1/T`, que está en `## Descartado` desde el 10-05.

### Y `w` se dio vuelta: ahora el modo lo mueve más que `T_init`

| it | `referencia` 8·δ | `rapido` 8·δ | `rapido` 1.1·δ | manda |
|---|---|---|---|---|
| 12 000 | 0.5627 | 0.2381 | 1.2764 | `T_init` (5.4× contra 2.4×) |
| 26 000 | 1.1432 | 0.3199 | 1.3839 | `T_init` (4.33× contra 3.57×) |
| **52 000** | **1.4452** | **0.3306** | **1.1787** | **el modo** (4.37× contra 3.57×) |

El `w` de `referencia` siguió trepando y el de 1.1·δ se dio vuelta en la it ~32 000; se cruzaron
ahí. **Haber retirado el múltiplo a la it 26 000 en vez de reescribirlo fue lo correcto**, porque
el orden se invirtió 26 000 iteraciones después. Lo cualitativo que quedó en `AGENTS.md` aguanta en
las trece iteraciones pareadas: **`rapido` a 1.1·δ llega a un `w` del orden del de `referencia`
(1.18 contra 1.45) y `rapido` a 8·δ no se le acerca nunca (0.33).**

El déficit de PSNR sigue clavado: **−0.026 dB a la it 52 000**, mismo signo en las dieciséis
evaluaciones pareadas desde la it 8 000, media −0.030.

---------------------------------------------------------------------------------------------------

## 2026-10-11 — VEREDICTO de `t1p1delta-rapido-100k`: `T_init = 1.1·δ` sobrevive, la predicción FALLA, y cuesta 4.7× en rechazos

Terminó: `"terminada": true`, 100 000/100 000, **31.525357 dB**, **378 rechazos**, **8 rebobinados**,
**12.685 h**, VRAM pico 5.07 GB. Era el sondeo que pidió el autor el 10-10 —textual, «solo para ver
qué sale»— y tenía **una** predicción registrada, para que pudiera fallar. Falló. Crudo en
`docs/mediciones/t1p1delta-100k.txt`, herramienta `scripts/veredicto_1p1delta.py` (local).

### La predicción: FALLA, y es la falla informativa

| # | predicción | resultado | veredicto |
|---|---|---|---|
| 1 | `T` toca 0 antes de la it 100 000 | **nunca bajó de 1.437·δ**, y terminó en **2.228·δ** | **FALLA** |

`T` arrancó en 1.1·δ, subió desde la primera medición y **nunca volvió abajo**. El mínimo sobre la
grilla de eval es 1.437·δ en la it 4 000 y el final es 0.028186 = **2.228·δ** — o sea que la corrida
terminó *por encima* del piso de 2·δ del que había arrancado por debajo.

**Lo que se retira es el método, no la corrida.** La predicción venía del replay de Adam sobre la
secuencia de `∂L/∂T` de `t8delta-100k-s0/-s1`, que daba una reserva exigida de 1.638–1.795·δ. Esa
estimación usa gradientes de corridas con `T ≈ 0.08` para decidir sobre una con `T ≈ 0.014`, y el
gradiente de `T` depende de `T`. El replay es una traslación: lo único que decía era
`1.1·δ − 1.638·δ < 0`. **La salvedad estaba escrita antes de lanzar**, y resultó ser la que valía.

### Por qué sobrevivió: `δ_Adam` se parte a la mitad en cada hito del lr

Es el hallazgo de la corrida y es estructural —sale de `train.py:51` y del calendario, no de estos
datos—. `DELTA_ADAM` se calcula **una vez** desde el `LR` inicial, pero el desplazamiento que Adam
imprime va con el lr **vivo**, así que la reserva `T/δ_vivo` **se duplica en cada hito aunque `T` no
se mueva**:

| reserva al cerrar cada fase | 4e-4 | 2e-4 | 1e-4 | 5e-5 | 2.5e-5 |
|---|---|---|---|---|---|
| `referencia` 8·δ | 6.41 | 12.60 | 25.67 | 52.24 | 104.53 |
| `rapido` 8·δ | 6.49 | 12.98 | 26.65 | 54.59 | 109.20 |
| **`rapido` 1.1·δ** | **1.65** | **3.84** | 8.46 | 17.80 | 35.66 |

**El piso de 2·δ sólo ata en la primera fase de lr**, 25 000 de 100 000 iteraciones. Y ahí cae el
costo entero.

### El costo: 4.7× en rechazos, y el 91 % en la primera fase

| rechazos / rebobinados por fase | 4e-4 | 2e-4 | 1e-4 | 5e-5 | total | horas |
|---|---|---|---|---|---|---|
| `referencia` 8·δ | 4 / 0 | 88 / 3 | 12 / 0 | 59 / 1 | **163 / 4** | 37.697 |
| `rapido` 8·δ | 37 / 1 | 43 / 1 | 0 / 0 | 0 / 0 | **80 / 2** | 12.648 |
| **`rapido` 1.1·δ** | **345 / 8** | 27 / 0 | 3 / 0 | 3 / 0 | **378 / 8** | **12.685** |

**345 de 378 rechazos (91 %) y los 8 rebobinados caen en la primera fase de lr**, y el primer hito
los corta en seco. El reloj no lo nota: 12.685 h contra 12.648 del control, **+0.3 %** —un lote
rechazado ya pagó su forward y su backward, así que rechazar no ahorra tiempo—. Lo que sí hubo fue
un pico de `max‖g‖` = **2.39·10⁷** en la ventana de la it 24 000, ~2 400× la cota absoluta, que el
disyuntor rechazó y la corrida atravesó sin que `T` bajara.

**La reserva no predice los rechazos**, y hay que decirlo: `referencia` tiene 88 en una fase donde
su reserva es 12.6·δ. Lo que la reserva gobierna es cuánto daño hace un pico, no con qué frecuencia
aparece.

### El endpoint: 0.022 dB abajo, y el déficit es consistente

| | PSNR | `T` | `λ` | `w = λ/T` | rech | reb | horas |
|---|---|---|---|---|---|---|---|
| `referencia` 8·δ | **31.547008** | 6.533·δ | 10.206·δ | **1.5621** | 163 | 4 | 37.697 |
| `rapido` 8·δ | **31.546075** | 6.825·δ | 2.173·δ | **0.3185** | 80 | 2 | 12.648 |
| `rapido` 1.1·δ | **31.525357** | **2.228·δ** | 2.385·δ | **1.0702** | 378 | 8 | 12.685 |

−0.0217 dB contra `referencia` y −0.0207 contra `rapido` a 8·δ. Los **50 de 50** puntos pareados
están dentro del piso de 0.33 dB y 47 de 50 dentro de 0.05 — pero **49 de los 50 son negativos**, y
**los 47 desde la it 8 000 lo son todos**, con media −0.0263. La magnitud no cambia ninguna decisión;
el signo no es ruido. **No es un test de signos limpio**: los puntos de eval de una misma corrida
están autocorrelacionados, así que 47 signos iguales valen mucho menos que `2⁻⁴⁷`. La lectura
prudente es «un sesgo real de ~0.026 dB, sin cuantificar».

Esto **extiende** lo cerrado el 10-07 —de 2·δ a 10·δ el PSNR pareado se mueve ≤ 0.036 dB— un factor
7 hacia abajo: a 1.1·δ el endpoint se mueve 0.022 dB. `T_init` sigue sin mover el endpoint.

### `w` al final: manda el modo, no `T_init`

| | `w` final | contra `referencia` |
|---|---|---|
| `referencia` 8·δ | 1.5621 | 1.00× |
| `rapido` 8·δ | 0.3185 | **0.20×** |
| `rapido` 1.1·δ | 1.0702 | **0.69×** |

El modo mueve `w` **4.90×** a `T_init` fijo; `T_init` lo mueve **3.36×** a modo fijo. **Al final
manda el modo**, al revés de lo que parecía a la it 12 000 (2.4× contra 5.4×) y que ya había
quedado retirado. Lo útil igual queda: **bajar `T_init` recupera la mayor parte del `w` que
`rapido` pierde** —de 0.20× a 0.69× del de `referencia`—.

> **RETIRADO el 2026-10-12, dos veces.** `../tdv/wTDV_QSM_torch.py:188` usa la red entera (`vn(z·α, z·α)/α`) y **no lee `T` ni `λ`**. Y `w` es una coordenada de **gauge**: bajo `(T, K_N) → (cT, K_N/c)` el modelo no cambia y `w → w/c`, así que su spread entre corridas no mide el modelo. Lo invariante es `ρ = λ‖x−z‖ / T‖∇R‖`. Entrada del 10-12.

Si la fase 3 necesita velocidad **y** un `w` del orden del oficial, ésta es la perilla, al precio de
4.7× en rechazos y 0.022 dB.

### Cómo NO se leen estos resultados

- **No se lee «1.1·δ es seguro».** Es **n = 1**, en `rapido`, que no repite bit a bit. Sobrevivió
  con 8 rebobinados: sin disyuntor esta corrida no existe, y el precedente `runs/tinit/B-larga`
  murió con `T = 0` desde 2·δ.
- **No se lee «el piso de 2·δ está de más».** Lo que se midió es que el piso **ata sólo en la
  primera fase de lr**, y que ahí abajo el disyuntor trabaja 10×. El default sigue siendo 8·δ.
- **No se lee «el déficit de 0.026 dB es significativo».** Está un orden debajo del piso de
  detección y la prueba de signos no es limpia.
- **La predicción que falló no se re-escribe.** Falló el método de estimar la reserva, y eso se
  corrige en el método.

---------------------------------------------------------------------------------------------------

## 2026-10-11 — VEREDICTO COMPLETO de `modo-y-tinit-100k-s1`: 7 CONFIRMADAS, 3 FALLAS, y ninguna de las dos perillas mueve el endpoint

Terminaron las dos. El brazo A se calificó el 10-10 (entrada propia, 1 CONFIRMADA y 3 FALLAS);
esto cierra el **brazo B** y la tanda. Crudo en `docs/mediciones/modo-y-tinit-100k-s1.txt`,
herramienta `scripts/veredicto_modo_tinit.py` (local).

| | PSNR @100 000 | `T` | `λ` | `w = λ/T` | rech | reb | horas | VRAM |
|---|---|---|---|---|---|---|---|---|
| control (`referencia`, 8·δ) | **31.547008** | 6.533·δ | 10.206·δ | 1.5621 | 163 | 4 | 37.697 | 9.94 |
| **A** (`rapido`, 8·δ) | **31.546075** | 6.825·δ | 2.173·δ | 0.3185 | 80 | 2 | 12.648 | 5.07 |
| **B** (`referencia`, 2·δ) | **31.535652** | 2.764·δ | 10.709·δ | **3.8747** | 325 | 5 | 37.428 | 9.94 |
| *sondeo* (`rapido`, 1.1·δ) | *31.525357* | *2.228·δ* | *2.385·δ* | *1.0702* | *378* | *8* | *12.685* | *5.07* |

### Las seis predicciones del brazo B, una por una

| # | predicción | resultado | veredicto |
|---|---|---|---|
| 5 | `T` no toca 0 en las 100 000 | mínima **sobre la traza** 0.024733 = **1.955·δ**, en la **it 3** | **CONFIRMADA** |
| 6 | `T` @100 000 > 0.0300 | **0.034960** = 2.764·δ | **CONFIRMADA** |
| 7 | rechazos > 163 | **325** = 1.99× el control | **CONFIRMADA** |
| 8 | rebobinados > 4 | **5** | **CONFIRMADA** |
| 9 | PSNR @100 000 en [31.22, 31.88] | **31.535652** | **CONFIRMADA** |
| 10 | ningún rebobinado repite respaldo | respaldos 2503, 3753, 7503, 9753, 22760 — **5 distintos de 5** | **CONFIRMADA** |

**La 10 se confirma pero no prueba el arreglo, y eso hay que decirlo.** La predicción decía «si
alguno lo hace, es la primera prueba en vivo del arreglo de hoy». Ninguno lo hizo, así que el
arreglo del alias del respaldo **sigue sin test en vivo**: sumando las cuatro corridas de 10⁵ hay
**19 rebobinados y los 19 van a respaldos distintos**. El defecto sólo aparece con dos rebobinados
al **mismo** respaldo, o sea dos dentro de las 250 iteraciones sanas que separan un refresco del
siguiente. Los tests de `tests/test_disyuntor.py` lo cubren; una corrida todavía no.

**La 5 también desmiente al replay por segunda vez.** El replay predecía un mínimo de 0.205·δ y
0.362·δ; el real es **1.955·δ**, un factor 5–10. Es el mismo método que ya quedó en `## Descartado`
el 10-11 por la corrida de 1.1·δ.

### El endpoint: ninguna de las dos perillas lo mueve

| brazo | Δ contra el control | 50 puntos pareados |
|---|---|---|
| **A** (modo) | **−0.00093 dB** | 50/50 dentro de 0.33; 49/50 dentro de 0.05 |
| **B** (`T_init`) | **−0.01136 dB** | 49/50 dentro de 0.33; 48/50 dentro de 0.05 |

Los dos están **un orden y medio por debajo** del piso de detección de 0.33 dB. El brazo B tiene
49 de 50 puntos negativos y media −0.0237, el mismo sesgo chico y consistente que el sondeo de
1.1·δ; **vale la misma advertencia**: la magnitud no cambia nada y el test de signos no es limpio.

**Los dos puntos del brazo B fuera de 0.1 dB están explicados y ninguno es el brazo separándose:**

- **it 10 000, −0.7224.** La evaluación cae **19 iteraciones dentro** de la racha de 25 rechazos
  seguidos que va de la it 9 982 a la 10 006, con `‖g‖` hasta 4.47·10⁴ — o sea que mide la corrida
  **adentro del evento**, antes de que el rebobinado de la it 10 006 la saque. A la it 12 000 ya
  está en 31.453.
- **it 6 000, +0.2237.** Es el pozo del **control**, el mismo que sacó de rango al brazo A.

### El costo se concentra donde la reserva es fina, y eso vale para las cuatro

| rechazos / rebobinados por fase de lr | 4e-4 | 2e-4 | 1e-4 | 5e-5 | % en la 1ª fase |
|---|---|---|---|---|---|
| control 8·δ (reserva 6.41·δ) | 4 / 0 | 88 / 3 | 12 / 0 | 59 / 1 | **2 %** |
| A `rapido` 8·δ (6.49·δ) | 37 / 1 | 43 / 1 | 0 / 0 | 0 / 0 | **46 %** |
| **B 2·δ (2.18·δ)** | **311 / 5** | 8 / 0 | 1 / 0 | 5 / 0 | **96 %** |
| *sondeo 1.1·δ (1.65·δ)* | *345 / 8* | *27 / 0* | *3 / 0* | *3 / 0* | ***91 %*** |

**Las dos corridas de `T_init` bajo ponen más del 90 % de su costo en la primera fase de lr; las
dos de 8·δ no.** Es la contracara del hallazgo del 10-11: `δ_Adam` se parte a la mitad en cada
hito, así que la reserva `T/δ_vivo` se duplica sola y la ventana de peligro es finita.

**Sigue sin ser cierto que la reserva prediga los rechazos**: el control tiene 88 en una fase donde
su reserva es 12.6·δ. Lo que la reserva predice es **dónde se concentran**, no cuántos hay.

### `T*` no es un atractor

La predicción 6 tenía una rama alternativa escrita: «si queda en [0.024, 0.030], `T*` **es** un
atractor, y sería el primer resultado positivo del repo sobre `T*`». No quedó ahí.

| | `T_init / δ` | `T` final `/ δ` | cociente |
|---|---|---|---|
| sondeo 1.1·δ | 1.100 | 2.228 | **2.026** |
| B | 2.000 | 2.764 | **1.382** |
| control | 8.000 | 6.533 | **0.817** |
| A | 8.000 | 6.825 | **0.853** |

`T* = 0.0247 = 1.953·δ`, y **ninguna de las cuatro termina ahí**. Las cuatro se mueven hacia una
banda del medio —las de abajo suben, las de arriba bajan— pero a 10⁵ iteraciones el **orden por
`T_init` se conserva intacto**, y con el lr ya en 2.5·10⁻⁵ no se van a mover más. `T_init` manda
sobre `T` final. Es lo que pide el gauge `(T, K_N) → (cT, K_N/c)`: el modelo no puede preferir una
`T`, así que no hay nada que atraiga.

### `w = λ/T`: las dos perillas lo mueven, en direcciones opuestas

| | `w` | contra el control |
|---|---|---|
| **B** (`referencia`, 2·δ) | **3.8747** | **2.48×** |
| control (`referencia`, 8·δ) | 1.5621 | 1.00× |
| *sondeo* (`rapido`, 1.1·δ) | *1.0702* | *0.69×* |
| **A** (`rapido`, 8·δ) | **0.3185** | **0.20×** |

**Bajar `T_init` sube `w`; pasar a `rapido` lo baja.** El brazo B tiene el `w` más alto de las
cuatro, 2.5× el del control.

> **RETIRADO el 2026-10-12, dos veces.** `../tdv/wTDV_QSM_torch.py:188` usa la red entera (`vn(z·α, z·α)/α`) y **no lee `T` ni `λ`**. Y `w` es una coordenada de **gauge**: bajo `(T, K_N) → (cT, K_N/c)` el modelo no cambia y `w → w/c`, así que su spread entre corridas no mide el modelo. Lo invariante es `ρ = λ‖x−z‖ / T‖∇R‖`. Entrada del 10-12.

Del que `AGENTS.md` ya decía
que «no se traslada sin medir»: ahora se sabe que **las dos perillas de la receta lo mueven un
factor 2.5–5 sin tocar el PSNR**, o sea que el escalar que la fase 3 va a usar es el que **menos**
determina el entrenamiento de denoising y el que **más** se mueve entre configuraciones.

### Qué queda cerrado, y qué no

**Cerrado.** Ninguna de las dos perillas mueve el endpoint a 10⁵: el modo 0.001 dB, `T_init` (×4)
0.011 dB, y el sondeo (×7) 0.022 dB, contra un piso de 0.33. Y `T*` no es un atractor.

**No cerrado, y es lo mismo de siempre.** Es **n = 1 por brazo**, y el campo 6 del pre-registro lo
dijo antes de ver los datos: para eventos no hay poder, y **ninguno de los dos brazos puede
establecer equivalencia**. Lo medido es que *estas* trayectorias siguen a la del control, no que
otra semilla caiga en el mismo lugar.

### Cómo NO se leen estos resultados

- **No se lee «2·δ es tan bueno como 8·δ».** El endpoint sí; la exposición no: **1.99× los
  rechazos** del control y el 96 % de ellos en la primera fase de lr. El default sigue siendo 8·δ.
- **No se lee «el disyuntor está probado».** Los 19 rebobinados de las cuatro corridas van a
  respaldos distintos: el arreglo del alias **no tuvo prueba en vivo**.
- **No se lee «`w` da igual».** Da igual *para el PSNR de denoising*. Es justo el escalar de QSM.
- **Las tres predicciones del brazo A que fallaron no se re-escriben**, y las seis del B que se
  confirmaron tampoco se festejan: cinco de las seis tenían umbral flojo a propósito.

---------------------------------------------------------------------------------------------------

## 2026-10-12 — RETIRO dos afirmaciones mías: la «regla 2-δ» y que el solver de QSM use `w = λ/T`

Las dos las cuestionó el autor en el mismo turno —«no sé de dónde lo sacas, me tinca que lo pusiste
con datos random»— y las dos caen. La primera por su propio registro; la segunda es un error mío
peor, porque la publiqué en cuatro archivos sin leer el solver. Medición nueva en
`docs/mediciones/gauge-4-corridas.txt`, herramienta `scripts/measure_gauge_4runs.py`.

### 1. La «regla 2-δ» no es una regla: `δ_Adam` es real, el factor 2 era una elección

Lo que **sí** está derivado y medido es `δ_Adam = lr/√(1−β₂)`: el desplazamiento acumulado que Adam
imprime desde un pico aislado, `Σ_k lr(1−β₁)β₁ᵏG/(√(1−β₂)G)` con `G` cancelándose, con 3–6 % de
error y **sin depender del tamaño del pico sobre cinco órdenes de magnitud** (entrada del 09-01,
`docs/mediciones/adam-respuesta-al-pico.txt`). Eso no se toca.

Lo que no se sostiene es el **2**. Su justificación textual, en la entrada «Corrección adoptada»
del 09-01, es: «después de un pico del peor caso, `T` aterriza en el umbral, no en el clamp» — o
sea 1·δ consumido por el pico más 1·δ de sobra. **Nada dice que 1·δ de sobra sea seguro**: el 2 es
una elección, no un umbral medido. Y la evidencia del propio repo va al revés, en los dos sentidos:

| corrida | `T_init` | desenlace |
|---|---|---|
| `runs/tinit/B-larga` | **exactamente 2·δ** | **murió** con `T = 0` en la it 32 750 |
| `t1p1delta-rapido-100k` | **1.1·δ**, debajo | **terminó las 100 000** con 31.525 dB |

O sea que el piso **no predijo ninguno de los dos casos que lo tocan**. Y hay un precedente directo
que ya estaba escrito: el mismo razonamiento aplicado a `λ` quedó **refutado el 09-06** —«la regla
2-δ da 2.00·δ y hace falta 2.2·δ; 10 de 12 siguen tocando»—. Encima, desde el 10-11 se sabe que
`δ_Adam` **se parte a la mitad en cada hito del lr**, así que «2·δ del lr inicial» no es ni
siquiera una cantidad fija de seguridad: son 4·δ_vivo después del primer hito y 8 después del
segundo.

**Lo que queda en su lugar** no es un umbral sino un costo continuo, medido a 10⁵ con cuatro
corridas: 8·δ → 31.547 dB y 163 rechazos; 2·δ → −0.011 dB y 1.99×; 1.1·δ → −0.022 dB y 4.7×, con
>90 % del costo antes del primer hito en las dos de `T_init` bajo. El default sigue siendo 8·δ
**porque es el de la réplica**. `tests/test_recipe.py::test_regla_2delta` se queda: assertea que
los *defaults* despejen `δ`, que es higiene de procedencia, y su propio docstring ya decía «no como
corrección de un modo de fallo».

### 2. El solver ADMM de QSM NO usa `w = λ/T`. Lo afirmé en cuatro archivos sin leerlo

`../tdv/wTDV_QSM_torch.py:184-197` usa la red **entera** como denoiser Plug-and-Play:

```python
batch_alpha = batch * alpha        # alpha = target_std / qsm_std, calibrado por paciente
x_batch = vn(batch_alpha, batch_alpha)
outputs_perm = (outputs / alpha)
```

**No lee `T` ni `λ` en ningún momento.** `T` y `λ` actúan, pero sólo adentro de `vn(z, z)`, y la
única perilla que el solver toca es `alpha`, que reescala los datos al σ con el que se entrenó.
Lo mismo en `wTDV_QSM_torch_3d.py:128,161`. La frase «`w = λ/T` es el escalar que usa el solver
ADMM de QSM» es mía, del 10-10, y está retirada de `AGENTS.md`, `README.md` y de las entradas del
10-10 y el 10-11 con banner.

### 3. Y además `w` es una coordenada de gauge, así que su spread no significaba lo que dije

El esquema de `ddr/model.py:86-99` es, con `τ = T/S` y `S = 10` pasos desde `x₀ = z`:

```
x ← x − (1/S)·[ T·∇R(x) + λ·(x − z) ]
```

o sea `S` pasos de descenso sobre `E(x) = T·R(x) + (λ/2)‖x − z‖²`. El gauge
`(T, K_N) → (cT, K_N/c)` deja el modelo **idéntico** —`K_N` entra lineal en `R`, así que `R → R/c`
y `T·∇R` queda igual— pero manda `w = λ/T → w/c`. **`w` no es propiedad de la función aprendida.**

Lo invariante son las dos **fuerzas**, porque las dos lo son: `T·‖∇R(x)‖` y `λ·‖x − z‖`. Medidas en
el paso `S/2` del esquema, sobre las dos imágenes de la presentación con ruido σ = 25:

| | `w = λ/T` | `T·‖∇R‖` | `λ·‖x−z‖` | `ρ = dato/reg` |
|---|---|---|---|---|
| `referencia` 8·δ | 1.5621 | 10.53 / 11.70 | 1.718 / 1.716 | **0.163 / 0.147** |
| `rapido` 8·δ | 0.3185 | 9.35 / 11.52 | 0.367 / 0.358 | **0.039 / 0.031** |
| `referencia` 2·δ | 3.8796 | 10.02 / 10.65 | 1.820 / 1.802 | **0.182 / 0.169** |
| `rapido` 1.1·δ | 1.0702 | 10.09 / 11.41 | 0.397 / 0.387 | **0.039 / 0.034** |

**`w` abre 12.2× y `T·‖∇R‖` queda dentro de 1.13×.** Casi todo el spread de `w` era gauge, y las
lecturas del 10-10 y el 10-11 —«`T_init` mueve `w` más que el modo», «al final manda el modo»—
estaban midiendo la coordenada, no el modelo. Los cuatro `∇R` tienen además **coseno 0.978–0.987**
entre sí en los seis pares: **las cuatro corridas aprendieron el mismo regularizador.**

**Lo que sí es real, y separa por modo y no por `T_init`:** la fuerza del término de dato. `ρ` vale
0.15–0.18 en las dos de `referencia` y **0.031–0.039** en las dos de `rapido`, un factor **4.8×**,
con las dos corridas de cada modo cayendo juntas. `bf16` aprendió un término de dato más débil, y
`T_init` no lo mueve.

Y aun así los cuatro denoisers son casi el mismo: `PSNR` entre salidas **42.2–45.6 dB**, y los
cuatro dentro de **0.08 dB** contra la imagen limpia. Es coherente: con un término de dato que pesa
3–18 % del regularizador, multiplicarlo por 4.8 mueve poco.

### Cómo NO se lee esto

- **No se lee «`T_init` da igual».** Da igual para el endpoint y para el regularizador aprendido;
  no da igual para la exposición, que va 1.99× y 4.7×.
- **No se lee «`rapido` y `referencia` son el mismo modelo».** Tienen el mismo `R` y distinto peso
  del dato, 4.8×. Que eso cueste 0.08 dB en denoising **no dice nada sobre PnP en QSM**, donde el
  denoiser se aplica decenas de veces en un lazo.
- **No se lee «`w` no sirve para nada».** Sirve adentro de una corrida, donde el gauge está fijo.
  Lo que no se puede es compararlo **entre** corridas, que es exactamente lo que hice.

---------------------------------------------------------------------------------------------------

## 2026-10-12 — PRE-REGISTRO de `calendario-lr-corto`: dos calendarios comprimidos contra el control, SIN LANZAR (REVISADO antes de lanzar)

Pedido por el autor: «me parece el C, me gusta. El A cambialo a 40.000 iteraciones, así toma 3
tandas de 5.000. Cuando veas las 2 gpu desocupadas, tira las 2 runs». Escrito **antes** de lanzar;
al momento de escribirlo las dos GPUs están ocupadas con trabajo del autor (`grid_search_wtdv.py`),
así que los `plan.sh` llevan guardia y no arrancan si no están libres (regla 3).

**1. Qué variable cambia, y la prueba de que es una sola.** El **calendario del lr**, y nada más.
Los dos brazos salen del control `t8delta-100k-s1` cambiando sólo `--iterations`,
`--lr-halve-start` y `--lr-halve-every`; mismo `--mode referencia`, misma `--seed 1`, mismo
`--t-init` (8·δ, el default), mismo todo. **Ojo: los dos brazos NO son comparables entre sí** —
difieren en dos cosas (el paso del halving y el largo)—, así que cada uno se lee **sólo contra el
control** (regla 4).

| | `--iterations` | `--lr-halve-start` | `--lr-halve-every` | hitos | lr final |
|---|---|---|---|---|---|
| control (archivado) | 100 000 | — | 25 000 | 25k / 50k / 75k / 100k | 2.5e-5 |
| **brazo C** | **35 000** | 25 000 | **2 500** | 25k / 27.5k / 30k / 32.5k / 35k | **2.5e-5** |
| **brazo A′** | **40 000** | 25 000 | **5 000** | 25k / 30k / 35k / 40k | **5e-5** |

**2. Control: archivado, y además PAREADO POR CONSTRUCCIÓN.** `t8delta-100k-s1`, 31.547008 dB. Y
hay una propiedad de diseño que vale más que el pareo estadístico: **en `referencia` las primeras
25 000 iteraciones de los dos brazos tienen que ser bit a bit las del control**, porque nada en la
config difiere antes del primer hito —`MultiStepLR` deja `lr = 4e-4` hasta la it 25 000 en los
tres, `evaluate_denoising` usa su propio generador (`evaluate.py:45`) y no toca el RNG de
entrenamiento, y `--iterations` sólo cambia el tamaño del array de traza y la lista de hitos—. Eso
es la predicción 1, y si falla, el resto de la tanda no se lee.

> ### REVISIÓN ANTES DE LANZAR, y hay que escribirla porque corrige un argumento mío
>
> El plan salió de una afirmación que hice y que **no se sostiene**: «la mitad de la ganancia de
> las últimas tres fases son los saltos de los hitos, así que halvear más seguido la cosecha en
> menos iteraciones». Verificada contra las cuatro corridas:
>
> 1. **No es la mitad, es el 27 %.** Mis +0.030 / +0.010 / +0.007 son las ventanas **crudas** de
>    2 000 it posteriores a cada hito, y la corrida **ya venía subiendo**. Descontando la pendiente
>    previa (media de las tres ventanas de 2 000 que cierran en el hito), el exceso atribuible al
>    halving es **+0.0182 / +0.0052 / +0.0064 = +0.0298**, o sea **31.8 %** de los 0.0939 en
>    `t8delta`; sobre las cuatro corridas, **26.6 %**. Moler es el 73 %, no el 50 %.
> 2. **El exceso es real pero minúsculo.** 12 de 12 positivos (4 corridas × 3 hitos), test de
>    signos **p = 2.4·10⁻⁴**: el mecanismo existe. Pero suma **0.0246 dB = 0.15 sd** de la sd entre
>    semillas, y **0 de 12** superan los 0.026 dB de acuerdo entre semillas a iteración pareada. Y
>    el 3er hito paga **más** que el 2º, así que no hay ley de la que extrapolar al 4º y 5º.
> 3. **Hay precedente en el repo y va en contra.** `runs/t1-rapido-30k` es la única corrida con
>    calendario comprimido (hitos en 20 000 y 25 000). Su **primer** halving pagó como en BSDS
>    (exceso **+0.0097**); el **segundo, encimado 5 000 después, no pagó nada: exceso −0.0053**, y
>    la corrida se congeló (+0.0047 en sus últimas 2 000). **Los halvings no se apilan**, que es
>    justo la condición que mi inferencia necesitaba. La bitácora ya lo leía así en la línea 5004.
> 4. **El piso de 2 500 ya estaba registrado** (líneas 4033 y 4435): `1/(1−β₂) = 1 000` iteraciones
>    es la memoria del segundo momento de Adam, y un tramo de 2 500 son 2.5 memorias — `v` todavía
>    arrastra gradientes del lr anterior cuando llega el hito siguiente. **El brazo C queda justo
>    en ese piso.** Y la misma entrada ya midió que la longitud del tramo casi no importa: «2 000 y
>    5 000 iteraciones por tramo cobran casi lo mismo… 0.004 dB».
> 5. **Mi predicción era el número del control.** 31.49 a la it 35 000 está medido **cuatro veces**
>    con el calendario del paper truncado. El valor marginal de los halvings extra sobre truncar y
>    parar es, sumando excesos medidos y restando lo que se deja de moler a un lr más bajo, **entre
>    +0.010 y −0.013 dB**.
>
> **Se lanza igual, pero midiendo otra cosa.** El autor lo pidió, y el diseño sigue siendo
> informativo — sólo que el endpoint no es lo medible. Ver el punto 4 y el 5 corregidos.

**3. La pregunta principal, corregida.** No es «¿35 000 iteraciones alcanzan los 0.1 dB?» —eso ya
está contestado y la respuesta es que sí sin tocar nada: el control parado en la it 35 000 está en
**31.4967** y en la 40 000 en **31.5062**—. Y tampoco puede ser «¿cuánto da el endpoint?», porque
la cantidad en disputa (0.09 dB = 0.56 sd) está **un orden por debajo** del mínimo detectable de
0.33 dB con n = 1: salga lo que salga, es compatible con las dos hipótesis. La pregunta que **sí**
es medible, porque se lee **dentro de cada corrida** y ahí el signo fue 12 de 12:

> **¿Un halving encimado paga, o no paga? ¿Y a partir de cuál deja de pagar?**
>
> `t1-rapido-30k` dice que el 2º encimado a 5 000 ya no paga. El brazo A′ pone **tres** a 5 000 y
> el brazo C pone **cuatro a 2 500**, que es el piso. Cada hito da su propio exceso contra la
> pendiente previa, dentro de la misma corrida, sin necesitar réplicas.

El brazo C llega a la it 32 500 con el lr en `LR/16` y termina en `LR/32`, contra `LR/2` del
control. **En el repo no hay ninguna medición de PSNR a 2.5e-5 ni a 1.25e-5**: el 4º hito de las
corridas de 10⁵ cae *en* la it 100 000, que es el último eval. O sea que el brazo C gasta 5 000 de
sus 35 000 iteraciones en dos regímenes de lr **sin payoff medido**. Eso es parte de lo que mide.

No lo contesta ningún replay: `scripts/replay_lambda_adam.py` re-simula un escalar bajo otro
`init`, no una trayectoria de 388 k pesos bajo otro calendario.

**4. Endpoint.** PSNR en CBSD68 a **iteración pareada contra el control**, que es la regla de la
fase 1. Los dos números de referencia son el control **a la misma iteración** (31.498 @35k,
31.5062 @40k), **no** los 31.547 de la it 100 000.

**5. Predicciones, con umbral numérico.** Las dos primeras son las que la tanda **puede** decidir;
las otras quedan registradas pero con el aviso de que el instrumento no llega.

1. **Las primeras 25 000 filas de `traza.npy` son bit a bit idénticas a las del control**, en los
   dos brazos. Si no lo son, hay una dependencia de `--iterations` que no vi y la tanda no se lee.
2. **MEDIBLE — el 2º halving encimado da exceso ≤ 0**, en los dos brazos, calculado como en la
   revisión: incremento de las 2 000 it posteriores al hito menos la media de las tres ventanas de
   2 000 que cierran en él. Fundamento: 12 de 12 positivos con hitos a 25 000 de separación, pero
   el único caso encimado medido (`t1-rapido-30k`, 5 000) dio **−0.0053**. Si sale positivo en los
   dos, **los halvings sí se apilan** y el precedente era del dominio, no del espaciado.
3. **MEDIBLE — el brazo C se congela antes de terminar**: sus últimas 2 500 iteraciones (lr
   1.25e-5) ganan **< 0.005 dB**. Fundamento: a 5e-5 la molienda medida es 0.00015–0.00030 dB/1000
   it, y `t1-rapido-30k` hizo +0.0047 en sus últimas 2 000. Si gana más, hay payoff debajo de 5e-5
   y el repo no lo tenía medido.
4. **Brazo C @35 000 en [31.488, 31.504]**, central **31.492**. Control a la misma iteración:
   **31.4967**. ⚠️ **Sin poder**: la banda entera mide 0.016 dB contra un mínimo detectable de
   0.33. Esto no se puede confirmar ni refutar con n = 1; queda escrito para que el número esté
   fijado antes y no después.
5. **Brazo A′ @40 000 en [31.492, 31.515]**, central **31.505**. Control a la misma iteración:
   **31.5062**. Mismo aviso.
6. **Ninguno llega a 31.547** (el endpoint de 10⁵). Si alguno lo alcanza, el comprimido no es
   «casi igual por menos»: es **mejor**, y eso cambia la receta. Umbral flojo a propósito.
7. **`T` no toca 0 y no hay más de 2 rebobinados por brazo.** El control lleva 1 a la it 25 000 y
   4 en las 100 000, y el calendario no toca la exposición temprana.
8. **Ningún rebobinado cruza un hito.** Y si alguno lo cruza, **es la primera prueba en vivo del
   arreglo del 2026-10-09**, que `AGENTS.md` marca como pendiente: el brazo C tiene **5 hitos en
   sus últimas 10 000 iteraciones** y la ventana expuesta es ~500 it por hito, así que es el
   escenario más probable que haya habido. Si el lr queda al doble del calendario hasta el final,
   la corrida no mide lo que se pidió y hay que relanzarla.

**6. Potencia, dicha sin maquillaje.** Para el **endpoint no hay potencia y no la va a haber**: la
cantidad en disputa es 0.0918 dB = **0.56 sd** entre semillas, y los excesos de los halvings suman
0.0246 dB = **0.15 sd**, los dos muy por debajo del mínimo detectable de 0.33 dB con n = 1
(regla 5). Las predicciones 4 y 5 **no se pueden decidir** y están marcadas.

Lo que sí tiene poder son las predicciones 2 y 3, y por dos motivos distintos:

- El pareo hasta la it 25 000 es **exacto por construcción** (predicción 1), así que lo que pase
  después viene del calendario o de nada: no hay ruido de trayectoria que descontar.
- El exceso de cada hito se lee **dentro de la misma corrida**, contra su propia pendiente previa.
  Ahí el signo fue **12 de 12** sobre las cuatro corridas archivadas, o sea detectable sin
  réplicas. Dos brazos dan **7 hitos encimados** (4 en C, 3 en A′) contra el único que existe hoy.

O sea: la tanda **no mide cuánto PSNR cuesta comprimir** —eso necesitaría ~300 semillas—, mide
**si un halving encimado paga**, que es la pieza de mecanismo que falta y la que se traslada al 3D.

**7. Costo.** Brazo C **13.19 h**, brazo A′ **15.08 h**, a 0.37697 h/1000 it medido en `referencia`
sobre el control. Concurrentes, una por tarjeta: **~15.1 h de reloj** y 28.3 h de GPU, con la
contención medida de ~0.5 % (`docs/mediciones/dos-gpu.txt`). Contra las 37.7 h del control, los dos
juntos cuestan menos que una corrida completa.

**Por qué `referencia` y no `rapido`**, que sería 3× más barato: porque `rapido` no repite bit a
bit y perdería la propiedad del punto 2, que es lo que hace exacta la comparación. El pareo vale
más que las 10 h.

**8. Cómo NO se leen los resultados.**

- **No se lee «35 000 iteraciones alcanzan»** si el brazo C queda dentro de 0.1 dB: eso ya lo hacía
  el control sin cambiar nada. Lo único que esta tanda puede acreditar es el **delta contra el
  control a la misma iteración**.
- **No se lee «sirve para el 3D»** salga lo que salga. Es BSDS 2D con `S = 10`, 388 k parámetros y
  un `δ_Adam` que sale de `LR = 4e-4` y `β₂ = 0.999`. Lo que se traslada al 3D hay que derivarlo
  del código del 3D, no de este número.
- **No se lee un ~1 como equivalencia.** Con n = 1 y un piso de 0.33 dB, «no se separó» es
  «no se midió».
- **Las predicciones que fallen no se re-escriben.**

---------------------------------------------------------------------------------------------------

## 2026-10-12 — ¿El recorte del calendario se traslada al 3D? El mecanismo sí, los números no, y el diseño correcto ya está pre-registrado

Pregunta del autor: «mi idea con esto es aplicarlo al entrenamiento 3D, me imagino que aplicaría
este mismo método no». Contestada leyendo la rama `3D-Training`, sin lanzar nada.

### Lo que se traslada POR CONSTRUCCIÓN, o sea sin medir un solo dato 3D

**`train.py` es byte-idéntico en las dos ramas** (`git diff main 3D-Training -- train.py` vacío), y
`train3d.py:42-47` **importa** `LR`, `BETAS`, `DELTA_ADAM`, `LR_HALVE_EVERY`, `T_INIT`,
`LAMBDA_INIT`, `SIGMA`, `STEPS` y el disyuntor entero. No son valores parecidos: son **los mismos
objetos**, y `tests/test_train3d.py:204-211` lo exige con `is`.

1. **`δ_Adam` y su partición en cada hito valen igual.** `δ = LR/√(1−β₂) = 0.012649…` sale de las
   mismas dos constantes, y `lr_en(it, hitos) = LR·0.5^Σ(h ≤ it−1)` (`train3d.py:170-177`) replica
   la semántica de `MultiStepLR` iteración por iteración. **La reserva `T/δ_vivo` se duplica sola
   en cada hito también en 3D.**
2. **La pared absorbente de `T` en 0 es la misma.** `vnet_config3d` delega en `train.vnet_config`
   y sólo cambia `R["type"]`, así que `T` queda con `min: 0.0`; `VNet3D` hereda `_scalar` y
   `project_` sin tocarlos.
3. **En 3D el recorte es una bandera, y encima más limpia.** `--hitos` acepta una lista arbitraria,
   y el lr es una **función pura de `it` reescrita en `param_groups` cada iteración** — o sea que
   ni un respaldo ni un estado reanudado pueden traer un lr viejo: la clase de defecto que en 2D
   hubo que arreglar a mano el 10-09 **no puede existir** en `train3d.py`.
4. **El 3D ya está «recortado» de hecho.** Con su default `--iterations 40000` los hitos son
   `range(25000, 40001, 25000)` = **[25000]**: un solo halving, 25 k a 4e-4 y 15 k a 2e-4. No hay
   un plan de 10⁵ en 3D que recortar.

### Lo que NO se traslada, y es casi todo lo numérico

- **Los números son de BSDS.** El codo en la it ~22 000, «+11.283 de +11.377 dB», «el 76 % del
  costo por 0.094 dB»: todo de CBSD68 en color con lote 32 y `S = 10`. **La traza 3D más larga del
  repo tiene 60 iteraciones** (`runs/humo3d/tronco/estado.json`).
- **La iteración no es la misma unidad.** `PATCH_SIZE` pasa de 93 a **48**, así que el contenido
  por paso va de 93²·3 = 25 947 valores a **48³ = 110 592**, o sea **4.26×** con el mismo lote de
  32 parches. Y el modelo pasa de 387 970 a **1 162 114** parámetros (3.0×). Un codo «a la 22 000»
  en 2D no es un codo «a la 22 000» en 3D ni en parches vistos ni en vóxeles.
- **La contabilidad por fases tampoco.** Es una afirmación sobre una corrida de cuatro fases a 10⁵.

### El diseño correcto ya está pre-registrado en `3D-Training`, y es mejor que el de esta tanda

La entrada del piloto 3D propone **tronco de 40 000 a lr completo + tres ramas de 5 000** reanudadas
en k = 10 000, 20 000 y 30 000, cada una con su halving, leídas como **rama − tronco a la iteración
pareada k + 5 000**. Eso es un **contrafactual de verdad**: el tronco a lr completo **es** el
control de «la pendiente que ya traía», que es justo lo que a las historias de 10⁵ de BSDS les
falta y lo que hizo que yo atribuyera mal el salto del hito. El diseño de ramas mide bien lo que en
2D quedó mal medido.

**Una corrección para ese pre-registro, que no se toca desde acá** (el 3D vive en su rama y no se
integra sin pedirlo): su **predicción 2** dice «cada rama queda entre **+0.01 y +0.06 dB** sobre el
tronco», calibrada con «BSDS, primer halving: +0.030 y +0.033». Esos son los incrementos **crudos**
de la ventana posterior al hito, que incluyen la pendiente que la corrida ya traía. Descontándola,
el primer halving de BSDS vale **+0.0065 a +0.0182** sobre las cuatro corridas de 10⁵. La banda
honesta es **+0.007 a +0.018**, y el techo de +0.06 está **~3× alto**.

### Entonces, la respuesta corta

**El método se traslada; la tabla de números, no.** Lo que hay que hacer en 3D no es copiar «cortar
en la 22 000»: es correr el piloto de tronco + ramas que ya está escrito, con la banda de la
predicción 2 corregida, y **medir dónde está el codo en 3D**. La parte del hallazgo que sí viaja
gratis es la estructural: `δ_Adam` se parte a la mitad en cada hito, así que el riesgo de la pared
de `T` se concentra antes del primer hito, y comprimir **después** de ese punto no toca la parte
peligrosa.

---------------------------------------------------------------------------------------------------

## 2026-10-12 — `calendario-lr-corto` LANZADA 13:39:56, y la predicción 1 ya está CONFIRMADA

Las dos GPUs se liberaron a las 13:39:56 (829 y 16 MiB) y el lanzador disparó las dos unidades.
Hitos verificados en el log de arranque: `[25000, 27500, 30000, 32500, 35000]` en el brazo C y
`[25000, 30000, 35000, 40000]` en el A′, los dos en `referencia`, semilla 1, `T_init = 0.101193`.

**Predicción 1 CONFIRMADA al primer eval.** Las 2 000 primeras filas de `traza.npy` son **bit a
bit** las del control en los dos brazos, y el eval coincide hasta el último dígito: **31.003129 dB**,
`T` = 0.10312238, `λ` = 0.03665400 en los tres. O sea que el pareo exacto hasta el primer hito es
real y la comparación posterior no va a estar confundida por la trayectoria.

**Un tropiezo propio que vale anotar**, porque el patrón se repite: mi primer chequeo dijo «NO
idénticas, 2 000 filas de 2 000 distintas, columnas 6 y 7». Era `np.array_equal` sin `equal_nan`:
la columna 7 (`amp`) es **NaN en las 100 000 filas** del control —sólo se mide con `--guard-amp`
finito— y `NaN != NaN`. Las filas impresas eran idénticas a simple vista, que es lo que delató el
bug. Un comparador de igualdad sobre una traza con columnas opcionales necesita `equal_nan=True`.

El chequeo completo de la predicción 1 —las 25 000 filas hasta el hito— se hace en el eval de la
it 26 000, ~10 h después del arranque.

---------------------------------------------------------------------------------------------------

## 2026-10-12 — `calendario-lr-corto` MUERTA a pedido en la it ~17 000, SIN VEREDICTO

El usuario la cortó: «dado que esos runs no nos sirven tanto para la run en 3D, ya no me
interesan». Las dos unidades se pararon a las ~17:50, con el brazo C en la it 16 750 y el A′ en la
17 000. Las GPUs quedaron en 0 %.

**La tanda se cierra sin veredicto.** De las ocho predicciones registradas (bitácora 5997), siete
quedan SIN RESULTADO: las dos que tenían poder de verdad —la 2 (¿paga un halving encimado?) y la 3
(¿se congela el brazo C en el último tramo a lr/32?)— necesitaban pasar la it 25 000, y nunca
llegamos ahí. No se relanza: la pregunta era si el recorte del calendario se traslada al 3D, y esa
pregunta se contesta ahora en 3D directamente.

**Lo único que sí quedó medido, y vale guardarlo, es la predicción 1, ahora sobre 16 000 filas.**
`scripts/verifica_pareo_bit.py` sobre las trazas archivadas:

```
C  35k: 16000 filas escritas, comparo las primeras 16000
  bit a bit identicas al control: SI
A' 40k: 16000 filas escritas, comparo las primeras 16000
  bit a bit identicas al control: SI
```

y los evals coinciden con el control en los tres (31.401 y 31.411 dB a las mismas iteraciones).
O sea: **`--iterations`, `--lr-halve-start` y `--lr-halve-every` no tocan nada de la trayectoria
antes del primer hito.** Eso es lo que hace barato probar calendarios —se puede arrancar un brazo
nuevo sabiendo que hasta su primer hito es el control, bit a bit— y se traslada tal cual al 3D,
donde `--hitos` toma una lista arbitraria. Las 9 000 filas que faltaban hasta la 25 000 no se
probaron, pero no hay mecanismo por el que pudieran diferir: el `lr` vale 4e-4 en los tres hasta la
25 000 y `MultiStepLR` no mira el total de iteraciones.

Los dos `runs/` quedan en disco con 16 000 filas de traza y su `best.pth`; `runs/` está
gitignoreado, así que no entran al repo.

---------------------------------------------------------------------------------------------------

## 2026-10-12 — POR QUÉ los checkpoints propios rinden peor como prior PnP de QSM: el paso desenrollado se vuelve inestable 4× abajo de donde entrena

El usuario reportó que su checkpoint `t8delta-100k-s1` «daba muy mal» dentro de
`../tdv/wTDV_QSM_torch.py`, que `1.1δ` y `2δ` daban «un poco mejor» y que los tres quedaban peor
que el oficial. Corrí el ADMM yo mismo con los cuatro, ajustes idénticos
(`scripts/corre_qsm_admm.py`, copia fiel: μ = 0.008, μ₂ = 1, tolUpdate = 0.39, `target_std` = 0.06
→ α = 4.8591, autocast bf16, tres cortes adyacentes como RGB; sin `torch.compile`, sin escribir en
`results/`, `cudnn.benchmark = False`):

| checkpoint | RMSE final | mejor en el camino | iteraciones |
|---|---|---|---|
| oficial `tdv3-3-25-f32-color` | **27.3447 %** | 27.3619 % (it 31) | **33, convergió** |
| `t8delta-100k-s1` (8δ) | 28.3557 % | 28.1935 % (it 62) | 100, no convergió |
| `t1p1delta-rapido-100k` (1.1δ) | 28.4749 % | 28.4244 % (it 59) | 100, no convergió |
| `t2delta-100k-s1` (2δ) | 28.5002 % | 28.4855 % (it 80) | 100, no convergió |

Los tres nuestros se agrupan en 28.36–28.50 % y **ninguno converge**; el oficial converge en 33 y
queda 1.0–1.15 puntos mejor. El de `8δ` además **empeora** después de la it 62 (28.1935 → 28.3557)
y su curva oscila en diente de sierra desde la it ~15, cosa que las otras tres no hacen.

### Antes que nada: esto YA estaba medido, el 2026-10-09, y no lo miré

La rama `worktree-wtdv-channels-flag` del repo `../tdv` trae el commit `331491c` del **9 de
octubre**: «Registro del Jacobiano del paso TDV: t8delta amplifica 4.47x (oficial 1.38x) con
target_std=0.06», con la nota «con target_std=0.15 t8delta baja a 1.46x» y la explicación «donde
los datos no sujetan (fondo, cono del dipolo) el ADMM reaplica el denoiser y esa ganancia no deja
que la solución se asiente».

O sea que el mecanismo —el paso amplifica, el nuestro mucho más que el oficial, y subir
`target_std` lo baja— estaba encontrado tres días antes. Todo lo que sigue es una **re-derivación
independiente** que no miró esa rama. Vale como verificación cruzada, no como hallazgo: dos
métodos distintos (ellos, autovalor dominante del jacobiano del paso por diferencias finitas en
fp32 en la iteración 0; acá, `(T/S)·λ_max(∇²R)` por iteración de potencia sobre el Hessiano en
float64, más la Lipschitz del denoiser completo) dan la misma conclusión y el mismo signo en
`target_std`. Los valores absolutos no son comparables porque no son la misma cantidad.

**La lección de proceso es mía**: antes de investigar algo del solver de QSM hay que mirar
`git log --all` de `../tdv`, no sólo el working tree. El working tree de `master` no tenía nada de
esto; estaba todo en una rama de worktree.

### La causa, medida: el umbral de estabilidad del paso explícito

El esquema desenrollado es descenso de gradiente **explícito** sobre `T·R`, y un paso explícito es
estable mientras

    (T/S)·λ_max(∇²R) < 2

`scripts/diag_qsm_estabilidad.py` lo mide por iteración de potencia sobre el producto
Hessiano-vector (diferencias centradas sobre `R.grad`, que es la adjunta exacta escrita a mano, no
autograd), en float64:

| modelo | a σ = 25 (donde **entrena**) | a std = 0.06 (donde lo usa **QSM**) |
|---|---|---|
| oficial | 1.3778 | **0.9942** |
| 1.1δ | 1.6063 | **5.7623** |
| 2δ | 1.8941 | **2.7906** |
| final s1 (8δ) | 2.0134 | **23.2169** |

El umbral exacto no es 2 sino **`2 + λ/S`**: el jacobiano del paso compuesto —gradiente más prox—
es `(I − (T/S)H)/(1 + λ/S)`, y es no expansivo mientras `(T/S)·λ_max < 2 + λ/S`. Con las `λ` de las
cuatro eso da 2.003–2.014, o sea que el 2 redondo se queda corto por menos de 0.7 %. Los márgenes:

| margen `2 + λ/S − (T/S)λ_max` | oficial | 1.1δ | 2δ | final s1 |
|---|---|---|---|---|
| a σ = 25 | +0.633 | +0.397 | +0.119 | **−0.0005** |
| a std = 0.06 | +1.016 | −3.759 | −0.777 | **−21.20** |

El `−0.0005` del de `8δ` **no se puede leer como «inestable»** —es n = 1, una tanda de parches, y
está dentro del ruido de la iteración de potencia—, pero sí dice que quedó **exactamente sobre el
umbral a su propia amplitud de entrenamiento**, que es el único de los cuatro, y es coherente con
que sea el que oscila en el ADMM y el que no converge ni siquiera a `target_std = 0.25`.

**A la amplitud de entrenamiento los cuatro caen en 1.38–2.01**, pegados al umbral y por debajo:
el entrenamiento lo fija ahí, porque si lo cruzara entrenaría mal. **Qué pasa 4× más abajo no lo
controla nada**, y va para lados opuestos: el oficial **baja** a 0.99 —su modo más rígido queda
aniquilado en un solo paso— y los tres nuestros **suben** por encima de 2, el de `8δ` 11.5×.

Eso explica las tres cosas que se observan: que ninguno converja, que el de `8δ` sea el que oscila
y se degrada, y que el orden de «qué tan mal se porta» sea el del número de estabilidad
(23.2 > 5.8 > 2.8 > 0.99) aunque las RMSE finales queden apiladas.

**Y explica por qué la PSNR de denoising es ciega a esto**: se mide exactamente a la amplitud donde
el entrenamiento fijó la estabilidad. Ahí los cuatro están bien por construcción, y coinciden en
0.027 dB.

### Lo que NO lo explica, y lo tenía como hipótesis

**Mi predicción falló.** Antes de leer el barrido registré: «si el problema es la amplitud, subir
`target_std` hacia la de entrenamiento (0.25) tiene que mejorar a `8δ` por debajo de 28.00 %».
Medido:

| `target_std` | 0.03 | **0.06** | 0.09 | 0.12 | 0.25 |
|---|---|---|---|---|---|
| `8δ` | 29.7479 % | **28.3557 %** | — | 29.9664 % | **31.2121 %** |
| `1.1δ` | 30.1255 % | **28.4749 %** | — | — | — |

A 0.25 da **31.21 %**, mucho peor. **FALLA.** El knob no mueve sólo la estabilidad: subir
`target_std` mete la red en saturación y el prior se debilita —el movimiento relativo cae de 0.51 a
0.34 entre std 0.06 y 0.25—, así que se gana estabilidad y se pierde regularización. `0.06` resulta
ser el óptimo **de todos los checkpoints**, no sólo del oficial, lo que también retira la idea de
que el grid-search del solver estuviera atado al checkpoint oficial.

Y el barrido del **oficial** sobre sí mismo dice cuán afilado es ese óptimo: 27.3447 % a 0.06,
**35.0622 %** a 0.12 (100 it, sin converger) y **41.8488 %** a 0.25. Se degrada mucho más que los
nuestros al moverse — o sea que `0.06` no es un ajuste cómodo sino un punto estrecho que fija el
dato.

**Y acá hay que frenar una afirmación mía de más.** Llegué a escribir que el criterio de
estabilidad «predice la convergencia en todos los puntos probados». **No es cierto fuera del punto
de operación**, y lo muestran dos casos del propio barrido: el oficial a `target_std = 0.12` tiene
el número en **0.95** —estable— y **no converge** en 100; y `1.1δ` a 0.12 lo tiene alto y
**converge en 30**. Mover `target_std` cambia la estabilidad **y** la fuerza del prior a la vez, y
la convergencia del ADMM depende además de su propio acoplamiento (`μ = 0.008`) y de la variable
dual. Lo que sí está sostenido, y es lo que importa, es el punto de operación: **a
`target_std = 0.06` el único estable es el oficial y es el único que converge, cuatro de cuatro.**

(La curva de estabilidad contra amplitud —`scripts/diag_qsm_curva_estabilidad.py`, 30 iteraciones
de potencia— da valores **más chicos** que la medición fina de 40 iteraciones: 3.91 contra 5.7623
para `1.1δ` en el mismo punto. La iteración de potencia converge por abajo, así que esa curva es
**cota inferior** y sirve para ver la forma, no para comparar números finos.)

Otras dos hipótesis mías que murieron en el camino, las dos por medición propia:

- **«El régimen de amplitud los separa».** En el punto exacto del solver los cuatro mueven el dato
  dentro de **5 %** (0.5302 / 0.5213 / 0.5069 / 0.5081) — `scripts/diag_qsm_checkpoints.py`. No
  alcanza.
- **«`T·‖K_N‖` es el predictor».** Es invariante de gauge y vale 0.0260 / 0.0260 / 0.0316 / 0.0659,
  que ordena bien — pero `1.1δ` tiene **exactamente** el del oficial y aun así su `λ_max(∇²R)` sale
  **5.1×** mayor (2044 contra 402). El Hessiano depende de toda la cadena, no de `‖K_N‖`.

Y una advertencia sobre la constante de Lipschitz del denoiser que medí primero: usa una dirección
**aleatoria**, así que es **cota inferior** de la norma del jacobiano. Leerla como «`1.1δ` es
contractivo, está bien» es un error: por el criterio del Hessiano está en 5.76.

Dicho eso, **medida bien —float64, ε hasta 1e-4— es la foto más limpia que salió de todo esto**:

| ‖D(x+e) − D(x)‖/‖e‖, ε → 0 | oficial | 1.1δ | 2δ | final s1 |
|---|---|---|---|---|
| a σ = 25, **donde los comparamos siempre** | 0.243 | 0.243 | 0.243 | 0.242 |
| a std = 0.06, **donde los usa QSM** | 0.085 | 0.108 | 0.675 | **244.589** |

Los cuatro coinciden **a tres decimales** a la amplitud de entrenamiento y se abren **2 900×** a la
de uso. No hay mejor resumen de por qué la evaluación de denoising no podía anticipar nada de esto.

### Y la receta ya está demostrada: es tu propio checkpoint reentrenado

Medidos con el mismo protocolo fino —40 iteraciones de potencia, float64, el `α = 4.8591` del
solver— los dos checkpoints de **1 canal**, en el punto de operación:

| en `target_std = 0.06` | `T` | `λ_max(∇²R)` | `(T/S)·λ_max` | umbral | margen |
|---|---|---|---|---|---|
| `tdv3-3-25-f32-gray` (oficial) | 0.01722 | 609.78 | **1.0502** | 2.0217 | **+0.972** |
| `checkpoint_qsm_rapido` (reentrenado sobre QSM, 30 000 it) | 0.09561 | 118.97 | **1.1376** | 2.0130 | **+0.875** |

**El reentrenado sobre QSM queda en 1.14, estable, justo donde el solver lo usa.** Y lo importante
es *cómo* llegó ahí: su `T = 0.0956` es prácticamente el de nuestro `8δ` (0.0826) — no es que tenga
el escalar más chico. Lo que bajó es `λ_max`, **118.97 contra 2809.34, un factor de 23.6**. O sea
que reentrenar en el dominio y a la amplitud de uso cambia **el regularizador**, no la coordenada.
*[Nota del 2026-10-15: `checkpoint_qsm_rapido.pth` no se reentrenó sobre QSM. Es byte a byte
`runs/t1-rapido-30k/best.pth` (comprobado con `cmp`): cortes 2D de T1 de `datasets/ldm100k` en
[0, 1], 1 canal, σ = 25, `rapido`, 30 000 iteraciones con lr/2 en la 20 000 y la 25 000. No vio QSM
ni la amplitud de uso. Lo medido sigue en pie: 1.1376 en el punto de operación y un `λ_max` 23.6×
menor que el de `8δ`, con una `T` parecida. La causa, no: contra `8δ` cambian a la vez el dominio
(T1 contra BSDS), los canales (1 contra 3), el modo, las iteraciones y el calendario, y el oficial
en gris, que los autores entrenaron en BSDS, también queda estable (1.05). Se retiran «la receta
ya está demostrada» y «reentrenar en el dominio y a la amplitud de uso cambia el regularizador».
Crudo: `docs/mediciones/qsm-prior-4corridas.txt`, bloque 7.]*

Con eso la separación en el punto de operación es total: los tres que funcionan están en
**0.99 / 1.05 / 1.14** y los tres que no, en **2.79 / 5.76 / 23.2**. Es exactamente lo que el
criterio pide, y es lo que el barrido de `target_std` no podía lograr desde afuera.

El problema práctico es otro: **ese checkpoint es de 1 canal y `wTDV_QSM_torch.py` no lo puede
correr**, porque arma el batch con tres cortes adyacentes como RGB. Las dos salidas son adaptar el
solver —meter los cortes por la dimensión de batch en vez de la de canal, perdiendo el prior
through-plane— o reentrenar con `--channels 3` sobre QSM. *[Nota del 2026-10-15: los de 1.05 y
1.14 no se corrieron en este ADMM, así que «funcionan» es una inferencia desde el número. En
`../tdv`, `checkpoint_qsm_rapido` corrido corte a corte con μ = 0.008 y `target_std` = 0.06 no
converge (Update ~17, RMSE ~30 %), y su mejor combinación de la grilla (μ = 0.02, `target_std` =
0.1) da 30.35 % contra 27.34 % del color (`../tdv/wTDV_QSM_torch.py`, «LOG DE OPTIMIZACIONES»,
punto 6, del commit `cc1bdd8` del 2026-10-08, en la rama `worktree-wtdv-channels-flag`). Es el
mismo hueco que mostró `ventana-3d-12h`: estable no es convergente. Esa rama entró en el `master`
de `../tdv` el 2026-10-13: desde entonces el solver corre modelos de 1 canal con `--channels 1`,
corte a corte.]*

### Lo que esto significa para el 3D

**La PSNR de denoising no es criterio de selección para un checkpoint destinado a PnP.** Dos
modelos que coinciden en 0.027 dB tienen números de estabilidad de 0.99 y 23.2 en el punto donde
se los va a usar. El criterio correcto es barato y no necesita reentrenar: medir
`(T/S)·λ_max(∇²R)` **a la amplitud de uso**, sobre un puñado de parches. Eso es
`scripts/diag_qsm_estabilidad.py`, y el único insumo es el checkpoint.

Dos detalles del solver que valen por sí solos:

- **No le pasa una imagen**: apila **tres cortes adyacentes como R, G, B**
  (`wTDV_QSM_torch.py:176`). Usa el acoplamiento de color aprendido en BSDS como prior
  *through-plane*.
- **`checkpoint_qsm_rapido.pth` y `tdv3-3-25-f32-gray.pth` son de 1 canal y no corren ahí**: el
  batch es `(B, 3, H, W)` y revientan con
  «expected input[2, 3, 50, 50] to have 1 channels». O sea que el baseline «repo original» contra
  el que se compara sólo puede ser el color.

---------------------------------------------------------------------------------------------------

## 2026-10-12 — El checkpoint 3D no carga exacto en QSM-TDV, la cruz de 7 cubos no entra en el cerebro, y `--mascara-cerebro` queda implementada (apagada)

El autor fijó el objetivo del 3D: **un checkpoint que `../QSM-TDV` use como regularizador sobre el
volumen entero, de una vez**, sin cortes. Preguntó además si conviene muestrear por cerebro una cruz
de 7 cubos (un central más uno por cara) en vez de cubos sueltos, y pidió la máscara de cerebro como
bandera. Todo en CPU: las dos GPUs estaban ocupadas por otra sesión.

### Compatibilidad con `../QSM-TDV` (5a7b5c6)

Crudo en `docs/mediciones/compat-qsm-tdv.txt` y, el control independiente del revisor, en
`compat-qsm-tdv-simetrico.txt`. Scripts: `scripts/compat_qsm_tdv/` y `scripts/compat_qsm_*.py`
(locales).

- **No carga directo:** `_load_regularizer` da `ValueError: Checkpoint does not contain TDVEnergy3D
  parameters`. Nuestras claves son las del port de VLOGroup (`R.K1`, `R.mb…`, `R.KN`, bajo `model`);
  las de ellos, `analysis.`, `macro_blocks.`, `energy_head.` (`evaluate_regularizers.py:341`).
- **Con los 56 tensores mapeados 1:1, `strict=True` carga, pero ∇R difiere 1.2–3.6 %**
  (‖Δ∇R‖/‖∇R‖, float64 y float32). La diferencia vive en las caras del volumen: baja de 1e-3 recién a
  13–15 vóxeles del borde.
- **Una sola pieza difiere: la bajada/subida de escala.** Nosotros desenfocamos el kernel (3³ con el
  binomial 5³ da 7³) y espejamos la entrada 3 vóxeles, como VLOGroup; ellos convolucionan el 3³ y
  desenfocan la salida, espejada 2. Con nuestra bajada/subida puesta dentro de su modelo, R coincide y
  ∇R a 1.5e-16. K1, micro-bloque, Student-t, 5 micro-bloques, 3 escalas, estado entre macro-bloques,
  K_N, el 1/m y el signo coinciden. No hay re-pesado que lo arregle (derivado: igualar el interior
  fija el kernel, y con ese kernel el borde difiere).
- **Memoria del volumen entero** (CPU fp32, medido; en GPU fp32 es la cota, en bf16 menos): nuestro ∇R
  ocupa 1 651 B/vóxel, **8.94 GiB y 60 s a 160×224×160**; a 256×256×176 se predicen 17.8 GiB. El de
  QSM-TDV, 3 487 B/vóxel (2.1×). Los dos aceptan tamaños impares (37×41×43: adjunto a 1.6e-16).
- **Lo que agregó el revisor:** `strict=True` no prueba el mapeo (uno cruzado carga y difiere 70–78 %);
  con el prox de `ddr/model.py:104` el paso efectivo de ∇R es (T/S)/(1 + λ/S) = T/(S + λ), no T/S;
  QSM-TDV corre en float16 (`use_amp`), no medido con nuestros pesos; y quedan brechas de dominio sin
  medir (borde de la máscara en χ, signo de χ, vóxel anisotrópico).
- **Opciones, sin decidir (las decide el autor, QSM-TDV es otro repositorio):** (a) una variante de la
  bajada/subida en QSM-TDV detrás de una bandera, ~30–40 líneas, más el conversor
  `scripts/compat_qsm_tdv/a_qsm_tdv.py`: exacta por el control de arriba, y no toca el entrenamiento
  (la recomendada); (b) cambiar `ddr/conv3d.py` a la semántica de ellos: rompe
  `tests/test_conv3d.py:105`, aparta el 3D del port y obliga a re-medir la memoria; (c) entrenar en
  `train3d.py` el `TDVEnergy3D` de ellos: la más cara, y K1 quedaría sin proyectar.

### La cruz de 7 cubos no entra en el cerebro

Crudo en `docs/mediciones/cruz-siete-cubos.txt` (script `scripts/cruz_siete_cubos.py`, local); los
24 sujetos y la máscara de la entrada «Cubos 100 % cerebro» (deepbet ∩ HD-BET sin TTA, margen 1).
Reproduce los 16 valores de `cobertura-cubos-cerebro.txt` antes de medir nada nuevo.

| P | sujetos con al menos una cruz 100 % cerebro | % del cerebro: cruces / cubos sueltos | cruz literal (en el centroide): % cerebro, peor cubo |
|---|---|---|---|
| 24 | 24 de 24 | 58.3 / 85.4 | 100.0, 99.9 |
| 32 | 5 de 24 (mediana 0 cruces) | 0.0 / 79.7 | 98.1, 86.9 |
| 40 | 0 | — / 73.9 | 94.8, 66.1 |
| 48 | 0 | — / 65.4 | 83.6, 49.3 |

- Lo que la impide es el eje vertical: el cubo inferior cae en la base del cráneo (con P = 32 falla en
  los 24 sujetos). La cruz mide 3P + 2 por eje; la caja del cerebro, 133 × 173 × 134 de mediana.
- Redundancia (indicador grueso, P = 24): la correlación intraclase de la intensidad media es 0.397
  dentro de una cruz, 0.387 entre cubos sueltos del mismo cerebro y −0.02 entre cerebros. **Lo que se
  repite es el cerebro, no la contigüidad.** Derivado (efecto de diseño de Kish): 32 cubos en cruces
  valen como 7–24 cubos independientes, según el rasgo.
- La red procesa cada cubo por separado, con espejo en sus caras: que sean adyacentes no le da
  contexto. Por eso la cruz no aporta frente a cubos sueltos de cerebros distintos.

### Campo receptivo del TDV entrenado (2D)

Crudo en `docs/mediciones/campo-receptivo-tdv2d.txt` (script `scripts/campo_receptivo_tdv2d.py`,
local), float64 en CPU, con `t8delta-100k-s1` (BSDS color) y `t1-rapido-30k` (T1).

- El esquema completo concentra el 90 % de |∂x_S[c]/∂y| en un semiancho de 17–24 px y el 99 % en
  35–44 px; un solo paso de ∇R, r99 = 18–28 px. El soporte teórico es 219–220 px por paso.
- **El efecto de cortar** (salida de un recorte contra la de la imagen entera), en % de σ: 23–29 en la
  cara, 7–10 a 4 px, 4–5 a 8 px y < 1 desde 21–24 px. Es casi igual con recortes de 32 a 96.
- Traducción a 3D (derivada: supone que el 3D entrenado tiene por eje el campo del 2D): los vóxeles a
  21 o más de todas las caras son el 0.2 % de un cubo de 48³, 4.1 % de 64³, 17.8 % de 96³ y 30.3 % de
  128³; el parche 2D del paper (93²), 30.1 %. El borde pesa poco salvo en sus primeros vóxeles, pero
  un cubo de 48³ casi no tiene vóxeles con todo su contexto. Cubos más grandes chocan con la máscara
  (a 56³ o más se pierde ≥ 37 % del cerebro) y con el costo. Queda abierto.

### `--mascara-cerebro`, implementada y apagada

- `ddr/data3d.py`: con máscaras, `PatchDataset3D` sortea sólo esquinas cuyo cubo (P + 2)³ cae entero
  en la máscara (`MARGEN_CEREBRO = 1`). Sin máscaras, los parches 0..49 de tres configuraciones dan el
  mismo sha256 que el código de 23b07ec (test).
- `train3d.py --mascara-cerebro` es receta (no se cambia al reanudar) y lee
  `datasets/ldm100k3d/mascaras/<split>/<sujeto>.npy`. La evaluación no cambia.
- `datasets/mascaras_ldm100k3d.py`: `deepbet` y `hdbet` corren cada uno en su venv; `combina` hace la
  intersección y recorta con la misma caja que la cache. Las 12 de evaluación quedaron en
  `mascaras/val`: el `.npy` cacheado es bit a bit `normaliza(NIfTI)[caja]` y las esquinas coinciden con
  `cobertura.json` (`docs/mediciones/mascaras-val-ldm100k3d.txt`). **Faltan las de entrenamiento:**
  ~22 min de GPU para 2 000 (derivado, 0.67 s por cerebro).
- La sonda de VRAM suma una columna de s/parche por micro-lote, para ver en la tarjeta de destino si
  un micro-lote más grande acelera. No se probó en GPU.
- Revisión (`docs/mediciones/revision-mascara-cerebro.txt`): sin bugs en 1 200 casos de fuerza bruta,
  1 500 parches reales y 3 alineaciones. Dos hallazgos quedan pendientes porque el hook de la regla 2
  bloquea `train3d.py` mientras corren `lr-corto-A-40k` y `lr-corto-C-35k`: la evaluación no tiene un
  PSNR dentro del cerebro (medio), y los estados de `runs/humo3d` no se pueden reanudar por la clave
  nueva (`train3d.py:437`, bajo).
- Las herramientas de máscara se movieron del directorio temporal del job a
  `~/.local/share/training-tdv/herramientas` (venvs de deepbet y HD-BET con sus pesos; importan bien
  ahí), y las máscaras crudas de los 24 sujetos a `datasets/ldm100k3d/cerebro24/`. Las rutas citadas
  en `docs/mediciones/` y en `scripts/` se corrigieron.

---------------------------------------------------------------------------------------------------

## 2026-10-12 — Evaluación QSM y seguimiento W&B: los módulos reproducen el 26.89 % del autor; falta integrarlos en `train3d.py`

El autor pidió que el entrenamiento 3D evalúe periódicamente el modelo **como denoiser dentro de su
reconstrucción QSM**, y que todo vaya a comet_ml o W&B (pérdidas y métricas de entrenamiento y
validación, el NRMSE, imágenes de resultados parciales). Se eligió W&B: ninguna de las dos estaba
instalada ni tenía credenciales, y W&B corre offline sin cuenta (se sube después con `wandb sync`).
`train3d.py` no tiene épocas (sortea parches): la evaluación QSM irá en cada validación.

### `ddr/qsm3d.py`: el ADMM del autor con el denoiser 3D

- El protocolo es el de `../tdv/wTDV_QSM_torch_3d.py` sobre los datos COSMOS de `../tdv/data_mat`
  (160³; verdad de terreno `chi_cosmos`, 933 481 vóxeles): ADMM con el TDV como denoiser, μ = 0.008,
  μ₂ = 1.0, target_std = 0.06, hasta 100 iteraciones con su criterio de parada. `admm_wtdv` acepta k
  denoisers: k = 3 es su pseudo-3D (la red 2D en tres ejes), k = 1 la red 3D sobre el volumen entero.
  La métrica es la suya: NRMSE = 100·‖χ − χ_GT‖/‖χ_GT‖ en la máscara.
- **Reproduce el 26.89 % publicado** (`../tdv/README.md:207`): con su red 2D color en los tres ejes, en
  CPU float32, **26.8898 %**, α = 4.8591 (el de su log) y el corte en la misma iteración (t = 26). Por
  iteración se separa de su log ≤ 0.0026 pp; él corrió en GPU con bf16. Contra su bucle transcripto en
  `tests/test_qsm3d.py` (20³ sintético) es idéntico bit a bit, y `carga_cosmos` reproduce exacto su
  preprocesamiento sobre los `.mat` reales. Crudo en `docs/mediciones/qsm-eval-cosmos.txt`; scripts
  en `scripts/qsm_eval/` (locales).
- **El 3D sobre el volumen entero** (`runs/humo3d`, 60 iteraciones de entrenamiento, así que su NRMSE no
  significa nada): en CPU fp32, 466–478 s por iteración de ADMM y 7.77 GB de pico de RAM. En GPU no está
  medido; el supuesto del agente es ~3 s por iteración en la 5070 Ti en `rapido`.
- Láminas uint8 para el seguimiento: los tres cortes centrales de GT | reconstrucción | error, y de
  limpia | ruidosa | salida | 5·|error| para el denoising.
- **Abierto:** `TARGET_STD = 0.06` se calibró para la red 2D color, no para una 3D entrenada en T1
  (sin medir); y un modelo poco entrenado puede no cortar nunca y correr las 100 iteraciones.

### `ddr/seguimiento.py`: W&B que nunca corta el entrenamiento

- Funciones `abre`, `registra`, `registra_imagenes`, `resumen` y `cierra`. Modo `auto`: online si hay
  credencial y offline si no, con el comando para subirla. Con `no` es un no-op y ni importa wandb: la
  suite del `.venv` principal, que no lo tiene, salta esos tests y pasa.
- Toda llamada a wandb va protegida: el primer error se imprime una vez y el seguimiento se apaga
  para el resto de la corrida. Es la regla de `train.py` de no cortar nunca el bucle.
- El id de la corrida vive en `<out>/seguimiento.json`. Reanudar en el mismo `--out` reabre la misma
  corrida. Una rama es otra corrida, con `group` igual al tronco.
- **Costo medido** (`docs/mediciones/costo-seguimiento.txt`): 0.19 ms por fila de métricas, 1.3 kB por
  fila en disco, 53 MB por corrida de 40 000 iteraciones (derivado). W&B descarta un paso menor que el
  último: por eso, tras un corte que vuelve a un estado anterior, se abre otra corrida.
- **La revisión arregló tres cosas, con tests** (`docs/mediciones/seguimiento-cuelgues.txt`):
  - con una entrada de `~/.netrc` sin password, `auto` se quedaba en el prompt de login de wandb
    hasta el timeout;
  - lo mismo pasaba con un `WANDB_BASE_URL` que apunta a otro host;
  - una rama lanzada en el mismo `--out` reabría la corrida vieja.

  Ahora cae a offline en 0.5 s. También arregló las láminas, que con tensores bf16 levantaban
  `TypeError`.

### Lo que falta

**Integrar en `train3d.py`.** Hay que registrar las columnas de la traza cada `--log-every`, y la
validación (PSNR, PSNR en cerebro, NRMSE de denoising y QSM, láminas) cada `--eval-every`. El hook
de la regla 2 lo bloquea mientras corren `lr-corto-A-40k` y `lr-corto-C-35k`. Condiciones de la
integración, de la revisión:

- un OOM de la evaluación QSM sobre 160³ no puede matar el entrenamiento;
- la evaluación QSM no consume el generador del ruido del entrenamiento;
- las banderas del seguimiento y de la evaluación son `LIBRES`, no receta.

`wandb` 0.30.0 entra como dependencia de la rama (`pyproject.toml`, `uv.lock`; sólo agrega
paquetes). Para no tocar el `.venv` compartido con `main`, el worktree tiene su propio `.venv`.

---------------------------------------------------------------------------------------------------

## 2026-10-12 — Evaluación QSM y seguimiento W&B integrados en `train3d.py`: con y sin ellos, la traza sale bit a bit igual

Cierra el «Lo que falta» de la entrada anterior. Todo se hizo en CPU y en `referencia`, sin GPU:
el autor pidió no usarlas por 4 horas. La validación de punta a punta en GPU queda pendiente.

### Qué hace ahora cada validación

`train3d.py` no tiene épocas, porque sortea parches. La «época» que pidió el autor es la
validación: cada `--eval-every` iteraciones (2 000 por defecto) y en la última.

- **Denoising** (`evaluate_denoising3d`). Además de los PSNR de siempre, mide el NRMSE en el
  recorte y en tejido. Si están las 12 máscaras de `datasets/ldm100k3d/mascaras/val/`, mide
  también el PSNR en el cerebro, con la máscara recortada con la misma ventana. Es la región que
  ve el entrenamiento con `--mascara-cerebro`; el recorte entero tiene cráneo y aire. En esta
  máquina las 12 están.
- **QSM** (`evalua_qsm`). Corre `admm_wtdv` con k = 1 sobre los 160³ de COSMOS, en el modo de la
  corrida. Por defecto va en cada validación; con `--eval-qsm-cada N`, sólo en las de iteración
  múltiplo de N y en la última. El resultado va al log y a una línea de `qsm.jsonl`, y el estado
  lleva las filas.
- **W&B** (`--seguimiento auto|online|offline|no`). Registra `train/*` cada `--log-every`,
  `val/*` y `qsm/*` en cada validación, las láminas `val/denoising` y `qsm/reconstruccion`, y el
  resumen al final.

COSMOS va en `datasets/qsm_cosmos/`. En esta máquina son dos enlaces, a
`../tdv/data_mat/params.mat` y `chi_cosmos.mat`, y el worktree enlaza la carpeta. **En otra
máquina hay que copiar los dos `.mat`**, porque los enlaces no viajan. Sin ellos, un AVISO al
arrancar y el entrenamiento sigue sin evaluación QSM.

### Decisiones que la especificación no fijaba

- **Frecuencia de W&B.** Registra cada `--log-every`, como pide la especificación, y no en cada
  iteración, como proponía el bloque final de `ddr/seguimiento.py`. Para no perder un estallido
  entre dos muestras se suma `train/grad_norm_max`, el `max|g|` del log en el intervalo. La traza
  sigue teniendo todas las filas.
- **Grupo de W&B.** Lo elige `abre` y no se le pasa. Así una rama de una rama queda en el grupo del
  tronco original. Para una rama directa da lo mismo que pasar el nombre del `--out` de origen.
- **Orden.** La evaluación QSM corre después de guardar `history.json`, `last.pth` y `best.pth`, y
  antes del estado. Un corte duro a mitad del ADMM no pierde la validación, y el estado lleva la
  fila.
- **Las filas de QSM viajan en el estado** (clave `qsm`). Al reanudar, `qsm.jsonl` se reescribe
  desde ahí, como `history.json`, y el resumen final cuenta también lo anterior al corte. Un
  estado sin evaluaciones borra `qsm.jsonl`.
- **Apagado.** `--qsm-datos ''` apaga la evaluación QSM sin aviso. La usan los tests que no la
  miran: con la carpeta por defecto, cada test cargaría el COSMOS real.
- **`qsm/fallo`** vale 0 en las evaluaciones que terminan, no sólo 1 en las que fallan.
- **`estado.json` suma las claves `qsm_*`:** evaluaciones, fallos, el último NRMSE final con su
  iteración y motivo, y el menor con la suya. El resumen de W&B lleva lo mismo que `estado.json`.
- **La config de W&B lleva `commit`:** el HEAD, con `+cambios` si lo versionado tiene cambios, o
  None sin git.
- **Reanudar un estado viejo.** La bandera de receta que le falta a un estado se compara con su
  default (`p.get_default`), no con None. Los estados de `runs/humo3d`, sin `mascara_cerebro`,
  vuelven a poder reanudarse.

### Tests

En CPU y `referencia`, con un COSMOS de juguete de 24³ escrito con `scipy.io.savemat`. Son once
tests nuevos en `tests/test_train3d.py`, sección SEGUIMIENTO Y EVALUACIÓN QSM (los dos últimos de
la tabla, de la revisión de abajo), y pasan todos:

| qué se pide | resultado |
|---|---|
| con y sin evaluación QSM: traza, `history.json`, log, pesos y Adam finales, generador global y del ruido | bit a bit iguales |
| con W&B offline, evaluación QSM y validación cada 2 en vez de 3, contra la corrida sin nada | traza bit a bit igual |
| un ADMM que levanta `torch.cuda.OutOfMemoryError` en cada evaluación | la corrida termina, con la traza de la corrida sin QSM; 2 fallos en el log, en `qsm.jsonl`, en `estado.json` y como `qsm/fallo` = 1 |
| carpeta de COSMOS sin `chi_cosmos.mat`, con `--seguimiento no` | aviso que nombra el archivo y `../tdv/data_mat`; termina; wandb no se importa |
| reanudar desde la 3 con evaluación QSM | traza de la corrida seguida; `qsm.jsonl` reescrito desde el estado y seguido |
| estado sin `mascara_cerebro`, sin las banderas nuevas y sin `qsm` | se reanuda; con `--mascara-cerebro` se rechaza |
| máscaras de cerebro en `VolumeDataset3D` | mismo recorte, mismos PSNR y NRMSE que sin ellas; PSNR en cerebro contra el cálculo directo |
| W&B offline: pasos, nombres y valores | `train/*` igual a `metricas_traza`; `val/*` en 2, 4 y 6; `qsm/*` sólo en 4 y 6 con `--eval-qsm-cada 4`; resumen igual a `estado.json` |
| armar la lámina de QSM levanta | las métricas QSM son las de la corrida sin la falla, sin `qsm/fallo`; se pierde sólo la imagen; sin W&B la lámina ni se arma |
| W&B al reanudar, offline: extender la terminada, seguir desde la 3 con pasos registrados hasta la 6, rama desde la 3 | la misma corrida (pasos 2, 4, 6, 8); otra con la vieja en `descartadas` (pasos 4 y 6); otra en el grupo del tronco, con su id de padre |

Suite completa, `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 <python> -m pytest -k "not
published_psnr"`:

- `.venv` del worktree, con wandb 0.30.0: **193 passed**, 4 deselected (237 s). Antes eran 182.
- `.venv` principal, sin wandb: **179 passed, 14 skipped**, 4 deselected (217 s). Los 14
  saltados son los que piden wandb: 12 de `test_seguimiento.py` y los dos de W&B de
  `test_train3d.py` (la corrida offline y qué corrida abre `--reanudar`).

Son los números tras la revisión de abajo; antes de ella, 191 y 178 + 13.

### Revisión adversarial (el mismo día, en CPU)

- **Defecto, corregido:** `evalua_qsm` armaba la lámina `qsm/reconstruccion` dentro de su
  `try`. Si `lamina_qsm` levantaba, la evaluación entera contaba como fallo y se perdían las
  métricas de un ADMM que había terminado; además se armaba también sin W&B, contra lo que decía
  el bloque final. Ahora `evalua_qsm` devuelve `chi` y la lámina se arma en `registra_lamina`.
  El test nuevo con `lamina_qsm` que levanta falla con el código anterior (las filas de
  `qsm.jsonl` salían como `fallo`) y pasa con el nuevo.
- **Afirmación retirada (regla 10):** el README decía que reanudar en el mismo `--out` sigue la
  misma corrida de W&B. Eso pasa sólo si no se registró nada después del estado. Tras un corte
  común, W&B ya tiene pasos posteriores al último estado (registra cada `--log-every`; los
  estados van cada `--guardar-cada`), así que `abre` abre otra corrida y deja la vieja en
  `descartadas`. Corregidos el README, el bloque final de `train3d.py` (SEGUIMIENTO) y el
  docstring de `ddr/seguimiento.py`, que además seguía diciendo que `registra` va en cada
  iteración.
- **Compilado, la traza tampoco cambia:** con `R.grad` compilado (los tests corren sin
  compilar), en `referencia` y CPU, la traza con y sin evaluación QSM sale bit a bit igual
  (`docs/mediciones/traza-compilada-qsm.txt`).
- **Los estados de `runs/humo3d`:** con las banderas de su `config` y la regla nueva, `--reanudar`
  no encuentra ninguna distinta (con la vieja, `mascara_cerebro`), y el estado tiene todas las
  claves que lee `main`. Simulado sobre `estado-000030.pth` del tronco, sin correrlo: cargaría los
  2 000 volúmenes.
- **Sin cambios:** las máscaras de `mascaras/val` tienen la forma de su volumen en los 12 sujetos,
  y el recorte de 128³ deja adentro entre el 88.5 y el 92.7 % del cerebro.

### Lo que no está verificado

- **En GPU, de punta a punta:** ninguna corrida con la integración pasó todavía por una tarjeta.
  El costo de la evaluación QSM está medido aparte, en `docs/mediciones/qsm-eval-gpu.txt`.
- **Online:** esta máquina no tiene credencial de W&B. Offline sí está probado.
- **Las entradas de dynamo:** son 4 contra el límite de 8, y volver a entrenar no recompila. Está
  medido en CPU con formas chicas (`docs/mediciones/dynamo-entradas-qsm.txt`); que en GPU dé lo
  mismo es un supuesto.
- **`TARGET_STD = 0.06`** sigue calibrado para la red 2D color, no para la 3D.

**Consecuencia práctica:** `train3d.py` trae la evaluación QSM y W&B prendidos por defecto; sin
credencial, W&B queda offline. Antes de lanzar en la tarjeta nueva hay que:

1. copiar los dos `.mat` a `datasets/qsm_cosmos/` y las máscaras de `mascaras/val/`;
2. correr `wandb login`, si se quiere seguir la corrida en vivo;
3. hacer una corrida corta en GPU que llegue a una evaluación QSM entera.

---------------------------------------------------------------------------------------------------

## 2026-10-12 — El número de estabilidad `(T/S)·λ_max(∇²R)` entra en cada validación de `train3d.py`: exacto contra la Hessiana, y con él la traza no cambia

**Procedencia:** `main`, entrada del mismo día «POR QUÉ los checkpoints propios rinden peor como
prior PnP de QSM: el paso desenrollado se vuelve inestable 4× abajo de donde entrena». Allá se
midió, con `scripts/diag_qsm_estabilidad.py`, que el número de estabilidad del paso explícito,
`(T/S)·λ_max(∇²R)` (estable si < 2), da 1.38–2.01 en los cuatro checkpoints 2D a σ = 25 y, a la
amplitud de uso del ADMM (std 0.06), 0.99 en el oficial y 2.8–23.2 en los propios, que son los que
no convergen. El PSNR no los distingue. Esta rama lo mide ahora en el 3D, en cada validación. Todo
en CPU: las GPUs estaban reservadas por el autor.

### Qué mide y dónde

`ddr/qsm3d.estabilidad` hace la iteración de potencia con el producto Hessiano-vector EXACTO: la
VJP por autograd de `R.grad`, sin diferencias finitas. Va en float32 IEEE (sin autocast ni TF32),
con un generador propio y con el `R.grad` de la clase, no el compilado: con `R.grad` compilado no
suma grafos de dynamo, y el compilado da lo mismo bit a bit después
(`scripts/dynamo_estabilidad.py`, CPU, `docs/mediciones/dynamo-estabilidad.txt`). Lo mide en dos
puntos, con la cadencia de la evaluación QSM (`--eval-qsm-cada`):

- **`val`**, siempre: el cubo central de `--estabilidad-lado`³ (96³) del primer recorte de
  evaluación, con el ruido de la validación (σ = `--sigma`, semilla 1234).
- **`qsm`**, con COSMOS: `punto_qsm3d`, la entrada del primer denoiser del ADMM 3D (α·χ₀),
  recortada a un cubo centrado en la máscara (en COSMOS, el 77.6 % del cubo de 96³ cae en ella).
  Se obtiene corriendo `admm_wtdv` una iteración con un denoiser que guarda su entrada, así que no
  puede divergir del ADMM.

El resultado va al log, a `history.json` (`val`) o `qsm.jsonl` (`qsm`) con las claves
`estabilidad`, `lambda_max` y `estabilidad_cambio`, y a W&B como `val/*` y `qsm/*`. Una excepción,
también un OOM, queda como `estabilidad_fallo` y la corrida sigue.

### El método, contra la Hessiana entera

`tests/test_qsm3d.py`: red de 4 features sobre 6³, en float64. La referencia es `eigvalsh` de la
Hessiana de la ENERGÍA, armada entera con autograd sin pasar por `R.grad`. Con 200 iteraciones,
la potencia da el módulo del autovalor dominante (`λ_max` = `‖Hv‖`) a ~1e-15 relativo, y el
cociente de Rayleigh su signo, con un dominante positivo (semilla 1: 3.61) y con uno negativo
(semilla 0: −5.35). El test exige 1e-9. Un tercer caso, con dos dominantes de signo opuesto casi
empatados (semilla 3 a amplitud 0.1: −2.6731 y +2.6705), exige que `λ_max` quede entre los dos
módulos; ahí el cociente de Rayleigh no converge (revisión, abajo).

Con las 40 iteraciones de `main` (`scripts/convergencia_estabilidad.py`, crudo en
`docs/mediciones/convergencia-estabilidad.txt`), el error depende del cociente entre los dos
mayores módulos:

| cociente \|μ₂\|/\|μ₁\| | 0.78–0.80 | 0.86 | 0.92 | 0.94 | 0.98 |
|---|---|---|---|---|---|
| error relativo con 40 iteraciones | 2e-9 – 7e-9 | 1.7e-7 | 3.1e-5 | 1.3e-3 | **9.0e-3** |

`estabilidad_cambio` avisa, pero subestima: el error fue de 1.7 a 5.3 veces el cambio con
cociente ≤ 0.94, y 41 veces con 0.98. En la red de humo de 32 features el cambio da 2e-4 a
1.6e-3. (La tabla es la rehecha en la revisión, con `λ_max` = `‖Hv‖`; la primera versión medía el
cociente de Rayleigh y daba 0.87 de error con 0.98, por la falla de abajo, no por convergencia
lenta.)

### Costo: 96³ entra

`scripts/mide_costo_estabilidad.py`, crudo en `docs/mediciones/costo-estabilidad.txt`. CPU
compartida, 8 hilos, float32, con el modelo de `runs/humo3d/tronco/best.pth`:

| lado | por validación (dos puntos), CPU | estabilidad / denoiser |
|---|---|---|
| 32³ | 13.9 s | 3.7–4.0 |
| 64³ | 208 s | 5.6–7.4 |

**Derivado** para la GPU a 96³, por dos caminos (el cociente estabilidad/denoiser con el fp32 de
`qsm-eval-gpu.txt`, y el cociente CPU/GPU del denoiser a 160³): **19–36 s por punto**, 39–73 s
por validación. Queda debajo del tope de 60 s por punto que fijó la especificación, así que el
default es 96³. Es el 0.2–0.5 % de un intervalo de 2 000 iteraciones. La memoria, derivada de la
RSS, es de ~9 GB a 96³ en float32, sin medir en GPU.

El número depende del lado: con el modelo de humo, el punto de COSMOS da 0.85 a 32³ y 1.27 a 64³.
Se compara al mismo lado.

### Decisiones que la especificación no fijaba

- ~~**`lambda_max` es el cociente de Rayleigh, con su signo.**~~ Retirada en la revisión (abajo):
  `lambda_max` es `‖Hv‖`, como en el script de `main`, y el cociente de Rayleigh va sólo al log.
- **TF32 apagado durante la medición** (`_sin_tf32`). `rapido` lo prende y corta la mantisa a 10
  bits; el contexto deja los flags como estaban, también si algo levanta (test en CPU). En GPU no
  está probado.
- **El punto `val` lleva el ruido de la validación**: semilla 1234 sobre el recorte de 128³ entero,
  y después el corte. `SEMILLA_VAL` pasa a ser explícita en la llamada a `evaluate_denoising3d`,
  con el mismo valor que su default.
- **El orden.** `val` va después de guardar la validación y reescribe `history.json` con la
  entrada completada. `qsm` va antes del ADMM, en la misma fila de `qsm.jsonl`.
- **No se guarda el reloj** en los JSON, porque cambiaría la fila al reanudar. Los segundos y `T`
  van al log, y un número de 2 o más se marca `*INESTABLE*`.
- **`--estabilidad-lado`** (LIBRE, 0 la apaga) es la única bandera nueva. La usan los tests (12³)
  y sirve para bajar la memoria.

### Tests

Nuevos: 4 en `tests/test_qsm3d.py`, 6 casos (la Hessiana con los dos signos y con el empate; no toca pesos,
`.grad`, generador ni autocast, y no llama al `R.grad` de la instancia; `_sin_tf32`;
`punto_qsm3d` bit a bit contra la primera entrada de un ADMM de cuatro iteraciones y a 1e-5 de la
fórmula cerrada) y 2 en `tests/test_train3d.py`:

| qué se pide | resultado |
|---|---|
| con y sin la medición (`--estabilidad-lado 0`), en `referencia` y CPU: traza, log, pesos, Adam y generadores finales | bit a bit iguales; `history.json` igual salvo las claves nuevas |
| la estabilidad recalculada fuera de la corrida, con los pesos del estado final, en los dos puntos declarados | la de `history.json` y `qsm.jsonl` a 1e-6 |

Además se extendieron seis: las claves de `qsm.jsonl` y `history.json`; un OOM simulado en
`estabilidad` (dos `FALLÓ` por punto, `estabilidad_fallo` en los JSON, nada en W&B, la traza de la
corrida sin QSM); W&B offline (`val/*` y `qsm/*` sólo en las validaciones de la cadencia QSM); un
estado sin la bandera nueva se reanuda; y el default de 96.

Suite completa antes de la revisión (después, abajo), `CUDA_VISIBLE_DEVICES=""
OMP_NUM_THREADS=8 <python> -m pytest -k "not published_psnr" -p no:cacheprovider`:

- `.venv` del worktree, con wandb: **200 passed**, 4 deselected (265 s). Antes eran 193.
- `.venv` principal, sin wandb: **186 passed, 14 skipped**, 4 deselected (264 s). Antes, 179 + 14;
  los 14 saltados son los mismos, los que piden wandb.

### Lo que no está verificado

- **En GPU, nada.** Ni el costo (es derivado), ni la memoria, ni el apagado de TF32 dentro de una
  corrida `rapido`.
- **Que 40 iteraciones alcancen en una red entrenada de 32 features.** En la de humo el cambio da
  ≤ 1.6e-3.
- **Que el umbral 2 ordene a los modelos 3D como ordenó a los 2D.** En `main` está medido en 2D,
  con tres cortes como RGB; acá el punto es un cubo de 96³ y el ADMM es el 3D.

**Consecuencia práctica:** cada evaluación QSM trae ahora `qsm/estabilidad`, el número en el punto
de uso, y la validación trae `val/estabilidad`, el del punto de entrenamiento. Para elegir un
checkpoint 3D destinado al ADMM, el de `qsm` se mira junto al NRMSE QSM, no el PSNR solo. Que el
umbral 2 separe en 3D lo que separó en 2D falta medirlo.

### Revisión, el mismo día: `λ_max` vuelve a ser `‖Hv‖`, como en `main`

Una revisión adversarial, toda en CPU, rehízo por su cuenta las tres cosas que sostienen el número
y encontró una falla en la segunda.

- **`punto_qsm3d` es la entrada del primer denoiser.** Contra la fórmula del t = 0 de `admm_wtdv`
  en float64 sobre el COSMOS real, a 1.3e-7 relativo en 32³, 64³, 96³ y 160³, y bit a bit igual a
  lo que recibe el primer denoiser de un ADMM de dos iteraciones. α = 4.85908, el 4.8591 del
  solver de `main` (con `np.std` en vez de `torch.std`, 5e-7 de diferencia). La ventana de 96³ es
  [32:128, 34:130, 26:122], con el 77.6 % en la máscara.
- **El cociente de Rayleigh con signo no es un autovalor** cuando los dos de mayor módulo tienen
  signo opuesto y módulo parecido. En una red de 4 features al azar (semilla 3, 6³ a amplitud 0.1,
  extremos −2.6731 y +2.6705), el `lambda_max` de entonces daba 0.97 con 40 iteraciones, 0.57 con
  200 y −1.32 con 1000. Su `cambio_relativo` era de 1.5e-3 a 4e-3, el mismo orden que en el
  modelo de humo, así que no lo avisaba. El «0.87 de error con cociente 0.98» de la primera tabla
  era esta falla (semilla 0, 7³: −8.15 y +7.98), no convergencia lenta. Además, con un dominante
  negativo el número salía negativo: se leía «estable» y nunca marcaba `*INESTABLE*`.
- **El script de `main` devolvía `‖Hv‖`**, que converge desde abajo a max|μ|. Ese número no es menor
  que el autovalor mayor con signo, y en el empate queda entre los dos módulos (2.67057 con 40
  iteraciones). Si `paso` ≥ 2, el paso amplifica con cualquier signo: con μ ≤ −2S/T,
  |1 − (T/S)·μ| ≥ 3. Ninguno de los dos da el autovalor mayor con signo cuando el dominante es
  negativo; `‖Hv‖` lo acota por arriba.

**Corregido:** `lambda_max` es ahora `‖Hv‖`, comparable con `main` sin traducción, y `estabilidad`
devuelve además `rayleigh`, que va al log para el signo. Las claves de los JSON y de W&B no
cambian. El test de la Hessiana tiene tres casos: los dos signos y el empate.
`docs/mediciones/convergencia-estabilidad.txt` está rehecha. En ella el error con 40 iteraciones
es 1.7 a 5.3 veces el cambio con cociente ≤ 0.94, y 41 veces con 0.98 (0.9 % contra 2.2e-4). El
error va como `cambio·r²/(1 − r²)`, así que un cambio de 1e-3 puede esconder algunos %.

En el modelo de humo el dominante es positivo en los cuatro puntos (32³ y 64³, val y qsm), y los
dos estimadores difieren en ≤ 5e-4: los números de `costo-estabilidad.txt` siguen valiendo (nota
al pie del archivo). También se tocaron el README y los bloques finales de `ddr/qsm3d.py` y
`train3d.py`. En `train3d.py` entra una frase en MEMORIA: la estabilidad pide ~9 GB derivados,
más que la evaluación QSM, y puede ser el pico de `estado.json`.

Suite después de la revisión, mismo comando:

- `.venv` del worktree: **201 passed**, 4 deselected (303 s).
- `.venv` principal: **187 passed, 14 skipped**, 4 deselected (296 s). Las dos corrieron a la vez.

Sigue sin verificar lo de arriba: GPU, 40 iteraciones en una red entrenada (la tabla dice cuánto puede
esconder el cambio, no cuánto esconde) y que el umbral 2 separe en 3D. Además, el umbral exacto con el prox es
`2 + λ/S`, según la entrada de `main`; el 2 redondo se queda corto por < 1 %.

---------------------------------------------------------------------------------------------------

## 2026-10-13 — PRE-REGISTRO de `ventana-3d-12h`: el 3D de punta a punta en GPU, y si `--aug-ganancia 4` arregla la estabilidad PnP a la amplitud de QSM

El autor dio 12 h con las dos 5070 Ti («iteres sobre lo que quieras») antes de pasar el repo a una
máquina más potente para el entrenamiento largo. Escrito ANTES de lanzar, con el código en
`e71b0ea`. Después, también antes de lanzar, se sumó `best_qsm.pth`: el estado con menor NRMSE QSM
entre las validaciones con estabilidad en COSMOS < 2 (2 justo queda afuera; un empate conserva el
anterior), que viaja en el estado como `mejor_qsm` y `estado.json` resume en `best_qsm`. No toca
la aritmética (test: traza bit a bit la de la corrida sin QSM) y el estado pasa a ~51 MB en la
40 000 (derivado). Los brazos corren el commit que imprime su `plan.sh`.

### Por qué esto y no otra cosa

En `main` se midió el 2026-10-12 («POR QUÉ los checkpoints propios rinden peor como prior PnP de
QSM») que los checkpoints 2D propios quedan 1.0–1.15 puntos peor que el oficial en el ADMM del autor
y no convergen, y la causa: el número de estabilidad del paso explícito, `(T/S)·λ_max(∇²R)`, a la
amplitud de USO (std 0.06) da 0.99 en el oficial y 2.8–23.2 en los nuestros (estable si < 2),
mientras que a la de entrenamiento los cuatro dan 1.38–2.01 y el PSNR no los distingue. «El
entrenamiento fija la estabilidad donde entrena; 4× más abajo no lo controla nada.» Reentrenar a la
amplitud de uso lo arregló (`checkpoint_qsm_rapido`: 1.08). *[Nota del 2026-10-15, al integrar
`3D-Training` en `main`: en `main` se rehízo con el protocolo fino —40 iteraciones de potencia,
float64— y da 1.1376 (6190bfd; entrada «POR QUÉ los checkpoints propios rinden peor…»): sigue debajo
de 2.]* *[Otra nota del 2026-10-15: ese checkpoint no se entrenó a la amplitud de uso ni sobre
QSM. Es `runs/t1-rapido-30k/best.pth`, cortes 2D de T1 en [0, 1] a σ = 25 (`cmp`). El precedente
que motivaba `--aug-ganancia` no existía: ningún checkpoint del repo se había entrenado a la
amplitud de uso. La Q2 se planteó sobre ese rótulo, y su resultado, abajo, fue en dirección
contraria en COSMOS.]*
`train3d.py` ahora mide ese número en cada validación (`qsm/estabilidad`, `val/estabilidad`)
junto con el NRMSE de la reconstrucción QSM. `--aug-ganancia G` (implementada y apagada desde
`9a94eaf`) multiplica cada parche limpio por un factor log-uniforme en [1/G, G] antes del ruido: con
G = 4 el entrenamiento ve amplitudes de ¼ a 4×, que cubren la de uso. Es una variable sola contra el
control (regla 4).

### Configuración (común salvo lo dicho)

`train3d.py --mode rapido --seed 1 --lote 32 --patch 48 --micro-lote auto --hitos --iterations 3500
--eval-every 500 --guardar-cada 500 --train-list datasets/ldm100k3d/lista_train_1000.txt
--seguimiento auto` (W&B
offline: no hay credencial en esta máquina). `--hitos` sola = lr constante 4e-4, el tronco del
piloto. 1 000 cerebros (subconjunto mezcla de los 2 000: las primeras 500 líneas, voraces, y las
líneas 1001–1500, sorteadas) y no 2 000, porque dos corridas con 2 000 pasarían ~54 GB de RAM en una
máquina de 62 sin swap. Evaluación QSM y estabilidad en cada validación.

| brazo | GPU | única diferencia |
|---|---|---|
| `v12-control` | 0 | — |
| `v12-ganancia4` | 1 | `--aug-ganancia 4` |

Antes: una corrida de humo de punta a punta en la GPU 0 (`--iterations 40 --eval-every 20`) y la
sonda de VRAM; en la GPU 1, las máscaras de cerebro de `lista_train_2000.txt` (~22 min), que no usa
ninguno de los dos brazos: quedan para que el autor decida `--mascara-cerebro` en la máquina nueva.
El humo fija una sola cosa antes de lanzar: el lado de la medición de estabilidad. Son 96³ por
defecto (~9 GB derivados) si entra en los 16 GB junto al entrenamiento, y 64³ si no. Lo que se elija
se escribe abajo, en «Lado de la estabilidad», ANTES de lanzar los brazos, y vale para los dos.
Después, una grilla del ADMM (target_std 0.045 / 0.06 / 0.09 × μ 0.004 / 0.008 / 0.016) sobre el
estado final de cada brazo (`scripts/qsm_eval/grilla_qsm3d.py`, local).

### Preguntas

**Principal (Q1):** ¿`train3d.py` corre de punta a punta en GPU a la configuración real, con la
validación nueva, y cómo es la trayectoria temprana del 3D (PSNR, NRMSE QSM, estabilidad, ‖g‖,
rechazos, T, λ)? Es lo que necesita la corrida larga y no depende de la potencia.

**Secundaria (Q2, exploratoria):** ¿`--aug-ganancia 4` baja `qsm/estabilidad` y mejora el NRMSE QSM
frente al control, y a qué costo en PSNR a σ = 25?

### Predicciones

1. Ninguno de los dos brazos se cae, y ninguna evaluación QSM ni de estabilidad registra un fallo.
2. Fuera de las validaciones, 7.6–9.4 s/it en los dos (8.48 derivado: 0.265 s/parche con micro-lote
   2 × 32, `costo-tdv3d.txt`), con `auto` eligiendo micro-lote 2.
3. Rechazos ≤ 1 % de las iteraciones en el control (la predicción 3 del piloto).
4. PSNR del recorte del control en la 3 500 entre 30.5 y 33.0 dB (base: 29.585 a la 60 con lote 8;
   el T1 2D llegó a 32.766 a la 30 000, en cortes).
5. **El control hereda la inestabilidad del 2D:** `qsm/estabilidad` del control > 2 en la 3 500.
6. **`--aug-ganancia 4` la baja:** `qsm/estabilidad` de `v12-ganancia4` < la del control en todas las
   validaciones desde la 1 500, y < 2 en la 3 500.
7. NRMSE QSM final de `v12-ganancia4` en la 3 500 ≤ el del control.
8. El PSNR a σ = 25 de `v12-ganancia4` en la 3 500 queda dentro de 0.3 dB del control (sin dirección).

### Potencia y lectura

Una semilla por brazo. En PSNR lo mínimo detectable con una semilla por brazo es ~0.33 dB (regla 5,
medido en 2D): la 8 sólo puede fallar por mucho. La estabilidad es determinista dada una trayectoria,
pero su variación entre semillas no está medida: la 6 se lee sólo como diferencia grande (≥ 2×), y
una diferencia chica no es un resultado. El NRMSE QSM de una semilla tampoco tiene barra de error. Si
la 6 se confirma, lo que habilita es proponer `--aug-ganancia 4` para la corrida larga como
decisión del autor, no adoptarlo. Si un brazo tiene un evento, no se re-corre la semilla (regla 6).
La cota absoluta del disyuntor (4.26e4 a 48³) no se reescala con la ganancia: en `v12-ganancia4` los
rechazos se separan en los que pasan la cota y los que pasan 5× la mediana, y no se comparan crudos
con los del control.

### Lado de la estabilidad

**96³**, el default, en los dos brazos. Completado con el humo, antes de lanzar.

Humo `runs/v12-humo` (GPU 0, 2026-10-13 00:20, `bb0ed93`): la configuración del control con
`--iterations 40 --eval-every 20 --guardar-cada 20`. Corrió de punta a punta: VRAM pico 8.16 GiB
con las evaluaciones (predicho 7.67 sin ellas), 2 evaluaciones QSM y 4 de estabilidad sin fallo,
`best_qsm.pth` escrito, W&B offline. Estabilidad a 96³: 31–40 s por punto. QSM: 100 iteraciones
de ADMM sin cortar en las dos (`max_iter`), 380 y 344 s; NRMSE final 84.4 % en la 20 y 46.6 % en
la 40. En la 40: PSNR del recorte 23.18 dB y estabilidad 1.09 en validación y 0.43 en COSMOS.

Un evento en la 13 (sin re-correr, regla 6): sanas hasta la 12 (`loss` de 1 072 a 177, `‖g‖`
480–2 200). La actualización aceptada de la 12 dejó el modelo con `loss` ~1 350 y `‖g‖` de
5–7·10⁴ en cualquier lote. De la 13 a la 37 hubo 25 rechazos seguidos, todos por encima de la cota
de 4.26·10⁴. En la 37 rebobinó al respaldo de la 3 (`GUARD_RESPALDO` = 250: hasta la 250 el
respaldo es el de la 3). De la 38 a la 40 volvió a lo sano. 25 rechazos de 40; iteración efectiva
6. La predicción 3 no se toca: si al control le pasa lo mismo, ya gasta 25 de los 30 rechazos que
permite el 1 % de 3 000.

Sondas (`--sonda-vram`, `runs/v12-sonda` y `runs/v12-sonda-gpu1`), micro-lote 2 en las dos:

| GPU | s/parche | s/it (32 parches) | VRAM medida | micro 4 |
|---|---|---|---|---|
| 0 | 0.2907 | 9.30 | 7.74 GiB | OOM |
| 1 | 0.2799 | 8.96 | 7.75 GiB | 14.65 GiB > 13.92 útiles |

La de la GPU 0 corrió con deepbet en la 1. En 3D la GPU 1 no es la lenta, al revés que en 2D
`rapido`. *[Nota del 2026-10-15, al integrar `3D-Training` en `main`: el «al revés que en 2D
`rapido`» se apoya en el «~1.3× más lenta la GPU 1», que `main` retiró el 2026-10-08: era arranque
de `torch.compile`, no régimen (`## Descartado`, «que la GPU 1 entrene más lento…»;
`docs/mediciones/velocidad-dos-tarjetas.txt`). Esta sonda tampoco compara tarjetas (regla 3): tres
iteraciones cronometradas en cada una, con deepbet en la GPU 1 mientras corría la de la 0.]* La
predicción 2 (7.6–9.4 s/it) no se toca.

### Enmienda antes de lanzar: 3 000 iteraciones, no 3 500

A 9.3 s/it y ~7.3 min por validación sin corte del ADMM, 3 500 iteraciones son ~9.9 h por brazo.
Con la grilla posterior (~1 h) quedan ~30 min antes de la entrega de la máquina (~12:15), sin
contar que dos corridas cargando datos a la vez no se midieron. **Los dos brazos van a
`--iterations 3000`** (6 validaciones: 500, …, 3 000; ~8.5 h derivado). Todo lo que arriba dice «la
3 500» se lee «la 3 000, la última». El resto, sin cambios.

### Segunda enmienda antes de lanzar: 400 iteraciones, sin grilla

La sesión del agente quedó detenida de 00:44 a 10:13: un `cp` a `runs/` esperó una aprobación. A
las 10:15 quedan ~2 h de ventana. **Los dos brazos van a `--iterations 400 --eval-every 200
--guardar-cada 200`** (validaciones en la 200 y la 400; ~77 min derivado), y no hay grilla. Q1 se
contesta igual, en la trayectoria temprana. Q2 y las predicciones 4–8, que se escribieron para la
3 500, se leen en la 400 sólo como dirección, sin veredicto: la 6 decía «desde la 1 500», y a la
400 no se juzga. Las predicciones 1–3 valen tal cual: el 1 % de 400 son 4 rechazos.

---------------------------------------------------------------------------------------------------

## 2026-10-13 — RESULTADO de `ventana-3d-12h`: el 3D corre de punta a punta; la ganancia baja la estabilidad a la amplitud de entrenamiento y la sube en COSMOS; estable no es convergente

Pre-registro arriba, con sus dos enmiendas: dos brazos de 400 iteraciones, validaciones en la 200
y la 400, una semilla cada uno, sin grilla. Corrieron a la vez de 10:14 a 11:30, con el código en
`f584d6f`: control en la GPU 0, ganancia 4 en la 1. Salida cruda:
`docs/mediciones/ventana-3d-12h.txt` (`scripts/resumen_ventana_3d.py`, local).

### Números

| | `v12-control` | `v12-ganancia4` |
|---|---|---|
| rechazos / rebobinados / efectiva final | 50 (12.5 %) / 2, en la 37 y la 72 / 331 | 1 (0.25 %), en la 268 / 0 / 400 |
| s/it, todo incluido salvo ADMM y estabilidad | 9.24 | 9.17 |
| T y λ en la 400 | 0.1108, 0.0742 | 0.1031, 0.0815 |
| PSNR del recorte, 200 → 400 | 30.223 → 31.003 dB | 30.441 → 31.096 dB |
| PSNR en cerebro, 200 → 400 | 31.859 → 32.419 dB | 32.005 → 32.495 dB |
| estabilidad, validación, 200 → 400 | 2.04 → 2.40 | 0.29 → 0.41 |
| estabilidad, COSMOS, 200 → 400 | 1.54 → 2.79 | 3.63 → 3.29 |
| NRMSE QSM final, 200 → 400 | 144.0 → 102.5 % | 37.8 → 69.7 % |
| NRMSE QSM mínimo del ADMM (iteración) | 34.2 % (6) → 34.3 % (5) | 32.2 % (24) → 33.0 % (6) |
| ADMM | 100 iteraciones sin cortar, las dos | 100 sin cortar, las dos |
| `best_qsm.pth` | el de la 200: 144.0 %, estabilidad 1.54 | no hay: nunca estable en COSMOS |

El s/it incluye el arranque, la compilación y el PSNR de validación. Cada validación costó ~8 min:
341–357 s de ADMM, ~62 s de estabilidad (dos puntos a 96³) y unos segundos de PSNR. El pico de VRAM
fue 8.16 GiB en los dos; la RAM de la máquina, 35 GB usados con las dos corridas (10 sin ellas). La `loss` y `‖g‖` de
ganancia4 no se comparan con las del control: la ganancia escala el limpio.

### Q1 — de punta a punta: sí

Las dos corridas terminaron (`"terminada": true`). No hubo fallos de QSM ni de estabilidad, y W&B
quedó offline con métricas e imágenes. El control repitió el evento del humo en la 13: misma
semilla y mismos lotes, `‖g‖` 48 297 contra 48 386. Hubo otro en la 48, la iteración efectiva 14
tras el rebobinado. Los dos fueron rachas de 25 rechazos y rebobinaron al respaldo de la 3, que es
el único hasta la 253 (`GUARD_RESPALDO` = 250). De la 73 a la 400 no hubo eventos. *[Nota del
2026-10-15, al integrar `3D-Training` en `main`: el segundo es un rebobinado repetido al mismo
respaldo, el caso del defecto del alias, y es la primera vez que el arreglo de `rebobina()` se
ejerce en una corrida real; en ningún log de `runs/` posterior al arreglo hay otro.]* Costaron 69
iteraciones de 400. Es un evento temprano contenido por el disyuntor, visto en una semilla: no se
re-corre (regla 6) y no es una tasa. En ganancia4 hubo un solo rechazo suelto, por el criterio
relativo (`‖g‖` 2 543 contra una mediana de 226).

A la 400, ningún ADMM corta en ningún brazo. Todos tocan un mínimo de 32–34 % entre las
iteraciones 5 y 24 y después se alejan, contra el 26.89 % del oficial 2D (pseudo-3D). A 400
iteraciones la red no sirve todavía como prior, que es lo esperable.

### Q2 — `--aug-ganancia 4`: dirección contraria en COSMOS, sin veredicto

- **Amplitud de entrenamiento (punto de validación):** la ganancia baja la estabilidad 7× en la 200
  y 5.8× en la 400 (0.29 y 0.41, contra 2.04 y 2.40 del control). El control ya es inestable ahí:
  en `main`, los 2D daban 1.38–2.01 a su amplitud de entrenamiento.
- **Amplitud de uso (COSMOS):** la sube, a 3.63 y 3.29 contra 1.54 y 2.79 del control. La
  predicción 6 decía lo contrario, en las dos validaciones.
- **NRMSE QSM final:** menor en ganancia4 (37.8 y 69.7 % contra 144.0 y 102.5 %), pero con ADMMs que
  no cortan es dónde cae la iteración 100, no un endpoint. Los mínimos difieren 1–2 puntos.
- **PSNR:** +0.09 dB en la 400 para ganancia4. La diferencia está dentro del ruido de una semilla
  (~0.33 dB, regla 5), y por eso sólo se lee como «no cuesta PSNR visible».

Una semilla, 400 iteraciones de 3 500 previstas. No se adopta ni se descarta, y no va a
`## Descartado`. Lo que se puede decir es que, a esta altura, la ganancia no lleva la estabilidad
en COSMOS por debajo de 2.

### Estable no es convergente: el criterio de `best_qsm.pth` tiene un hueco

En la 200 del control, la estabilidad en COSMOS dio 1.54 (< 2) y aun así el ADMM divergió: 144 %
al final, peor que devolver cero (100 %). El número se mide en un solo punto, la entrada del
**primer** denoiser del ADMM; los iterados siguientes pasan por otros. Por eso `best_qsm.pth` del
control es ese estado de 144 %: el único estable. El criterio sigue siendo el menor NRMSE entre los
estables, y en una corrida larga un estado mejor lo reemplaza. Pero «estable» no basta para
«usable». Antes de pasarlo a QSM-TDV hay que mirar su `nrmse_final_pct`, que viaja en el archivo y
en `estado.json`. Exigir además que el ADMM corte (`motivo == "corte"`) dejaría afuera a los 2D
propios de `main`, que no cortaban y daban 28.4 %. Qué condición agregar lo decide el autor; el
código no cambió.

### Predicciones

1. Sin caídas ni fallos: **se cumple**.
2. 7.6–9.4 s/it: **se cumple** (9.24 y 9.17; sondas, 9.30 y 8.96).
3. Rechazos ≤ 1 % en el control: **falla**, 12.5 %, por dos rachas tempranas.
4. PSNR del control en la última, entre 30.5 y 33.0 dB: 31.003, adentro aun en la 400 (sin
   veredicto).
5. Estabilidad del control en COSMOS > 2 en la última: 2.79 en la 400 (1.54 en la 200). En
   dirección, sin veredicto.
6. Ganancia4 < control en COSMOS y < 2: **dirección contraria** en las dos validaciones.
7. NRMSE final de ganancia4 ≤ control: en dirección sí, pero no es un endpoint con ADMMs sin
   corte.
8. PSNR dentro de 0.3 dB: 0.09 dB.

### Lo que esto cambia

- **Para la corrida larga:** la configuración corre en GPU. En la 5070 Ti cuesta 9.2 s/it, y ~8 min
  cada validación mientras el ADMM no corte. Con `--eval-every 2000`, ~3 % (derivado: 480 s por cada 18 400).
- **`--aug-ganancia 4`:** esta evidencia no la sostiene como arreglo de la estabilidad en COSMOS.
  Se decide con la corrida larga o con otra tanda, y lo decide el autor.
- **`best_qsm.pth`:** se mira su NRMSE antes de usarlo (README). La condición extra está abierta.
- **Abierto:** por qué el punto de COSMOS no anticipa la divergencia del ADMM. Para contestarlo
  habría que medir la estabilidad sobre los iterados del ADMM (`al_iterar`), no sólo en la entrada
  del primero.

---------------------------------------------------------------------------------------------------

## 2026-10-15 — Decisiones para la corrida larga: Blackwell de 96 GB, 8·δ, 2 000 cerebros, cubos 100 % cerebro

El autor fija lo que faltaba antes de llevar la rama a la máquina nueva:

- **La tarjeta es una RTX PRO 6000 Blackwell Max-Q de 96 GB, entera**, con RAM de sobra. Antes se
  hablaba de 48 GB de una tarjeta compartida (entrada del 2026-10-09). `VRAM_GB` pasa de 48 a 96,
  y con eso `auto` da micro-lote 16, con 57.1 GiB predichos. Esa es la cifra de «~56 GB» que
  planteó el autor. Con 8 serían 28.8 GiB, y con 32, 113.5, que no entran. Son extrapolaciones de
  la 5070 Ti. **Si 16 es más rápido que 8 no está medido**: en la 5070 Ti, doblar el micro-lote
  bajaba el costo por parche 2–6 %. La sonda lo mide en la tarjeta, y si 16 no gana claramente,
  `--micro-lote 8`. `--checkpointing` no hace falta: era para entrar en 16 GB y cuesta +22–27 %
  de tiempo. Test: `auto` da 16 con 96 GiB y con 96 GB decimales.
- **`T_init = 8·δ_Adam`** (0.1012). Es el default desde la fase 1: no cambia nada.
- **2 000 cerebros** (`lista_train_2000.txt`, el default): son los que están cacheados y tienen
  máscara. Por paso, 32 cubos de 48³ independientes: cada uno de un volumen al azar, en una
  posición y con una de las 48 orientaciones del cubo al azar. No es «un cubo grande» ni la cruz
  de 7: la cruz no entra entera en el cerebro a 48³ en ninguno de los 24 sujetos medidos, y sus
  cubos se parecen entre sí tanto como dos cubos sueltos del mismo cerebro (ICC 0.397 contra
  0.387, `docs/mediciones/cruz-siete-cubos.txt`). En 40 000 iteraciones son 640 cubos por cerebro
  (derivado).
- **`--mascara-cerebro`, prendida en el comando de la corrida larga**, no como default. La duda del
  autor era el tiempo. Por iteración no cuesta nada: la esquina sale de una lista. Al arrancar,
  leer las máscaras lleva 0.06 s por volumen, ~2 min para los 2 000, y las esquinas ocupan
  ~0.4 GB de RAM (derivado de los 0.06 s y ~0.2 MB por volumen del README, «Cubos 100 %
  cerebro»). *[Nota del 2026-10-16: el README reescrito ya no tiene esa subsección; las dos
  cifras están en `ddr/data3d.py`, bloque final, «CUBOS DE CEREBRO».]*
  El precio es otro, y es de receta: la periferia. A 48³ los cubos cubren el 66 % del cerebro y el 18.5 % de la franja cortical
  (≤ 6 mm del borde). La red no ve el borde del cerebro, y el ADMM sí se lo pasa: lo que eso le
  hace al NRMSE QSM no está medido. La evaluación QSM de cada validación lo va a mostrar.
- **`--aug-ganancia` queda apagada**: la `ventana-3d-12h` no la sostiene como arreglo de la
  estabilidad en COSMOS.
- **Abierto:** las iteraciones y los `--hitos`. Se fijan con el s/it de la tarjeta, que no está
  medido, para que entren en 48–72 h. *[Nota del 2026-10-15, más tarde: el autor habla
  ahora de 72–96 h, con el calendario `--iterations N --hitos N−15000 N−10000 N−5000`: tres lr/2,
  a 15 000, 10 000 y 5 000 iteraciones del final (con N = 40 000, `--hitos 25000 30000 35000`). N
  se fija después de medir el s/it en la tarjeta: N ≈ horas · 3 600 / (s/it), menos las
  validaciones.]*
- **Seguimiento: W&B**, que el autor dio como opción junto a Comet ML. Registra:
  - `train/*` cada `--log-every`;
  - la validación (`val/*`), la QSM (`qsm/*`) y la estabilidad;
  - las imágenes `val/denoising` y `qsm/reconstruccion`.
  En la máquina nueva hace falta `wandb login`; sin credencial queda offline y se sube con
  `wandb sync`.

Comando completo: README, «Llevar el entrenamiento 3D a otra máquina».

---------------------------------------------------------------------------------------------------

## 2026-10-15 — Integración de `3D-Training` en `main`

A pedido del autor, la rama 3D se integró en `main`. En GitHub, `main` queda como un solo commit,
«incorporar 3D», encima de `6bb5b88`. La historia detallada queda en dos etiquetas locales, sin
subir: `historia-3D-Training` (`aad554e`) y `historia-main` (`2256f3d`). Los hashes que cita esta
bitácora están ahí.

- **El 3D es aditivo.** Respecto de la base común `e88512d`: 47 archivos nuevos y 13 modificados,
  ninguno borrado ni renombrado. Comprobado en CPU:
  - los cinco checkpoints 2D (dos oficiales y tres propios) cargan con `strict=True`, con las
    mismas 102 claves, y dan la misma salida bit a bit;
  - dos iteraciones de `train.py` en `referencia` dan traza, pesos y Adam idénticos;
  - `train.py` es el mismo blob en las dos ramas: el arreglo de `rebobina()` llegó idéntico por
    las dos (`78fcf82` y `81b3701`).
- **Conflictos:**
  - `AGENTS.md`, el README y esta bitácora: las entradas de los dos lados, en orden cronológico,
    y las filas de las tablas, unidas;
  - `tests/test_disyuntor.py`: los mismos dos tests, con los docstrings de `main`;
  - `docs/mediciones/alias-respaldo-adam.txt`: queda la fecha de `main`.
- **Verificación:** ninguna línea agregada por un lado se perdió (99 archivos revisados contra la
  base). La suite pasa con los dos entornos: 205 tests con wandb, y 191 más 14 saltados sin él.
- **Retiros de `main` que el 3D seguía afirmando:** quedan como notas fechadas junto a cada
  afirmación. Son la regla 2-δ, la banda de la predicción 2 del piloto, la GPU 1 «lenta» y el
  1.08 de `checkpoint_qsm_rapido` (1.1376 con el protocolo fino).
- **Pendiente:**
  - wandb pasa a ser dependencia obligatoria de `main`: el primer `uv sync` instala ~25 paquetes.
    Se propone pasarlo a un extra opcional.
  - `checkpoint_qsm_rapido.pth` es byte a byte `runs/t1-rapido-30k/best.pth` (cortes 2D de T1,
    30 000 iteraciones, lr/2 en la 20 000 y la 25 000; comprobado con `cmp`). No se reentrenó
    sobre QSM, como dice la entrada «POR QUÉ los checkpoints propios rinden peor…». Se corrige en
    la revisión de los `.md`. *[Hecho el 2026-10-15: notas en esa entrada y en el pre-registro de
    `ventana-3d-12h`, y fila en `## Descartado`. `docs/mediciones/qsm-prior-4corridas.txt`, que
    repite el rótulo, no se edita: lo corrige la nota de la entrada, que lo cita.]*

---------------------------------------------------------------------------------------------------

## 2026-10-16 — README y AGENTS.md reescritos: guía para personas y contexto para agentes

- **Qué cambió.** El README pasa de 86 kB a ~26 kB y queda como guía de uso para un estudiante, con
  la estructura del repositorio al principio y sin «Estado y lo que sigue» ni las listas de lo
  descartado, que siguen acá (`## Cómo retomar`, `## Descartado`). `AGENTS.md` pasa de 22.5 kB a
  ~20 kB y absorbe lo operativo. Las citas a títulos que el README ya no tiene apuntan a los nuevos
  o, en una entrada vieja, llevan una nota fechada.
- **Cómo se juzgó.** Un «estudiante» siguió el README con datos propios, 2D y 3D, en una copia
  limpia, y quedó conforme con reparos; un «académico» revisó la precisión. Se aplicaron los
  reparos de los dos.
- **Medido de paso.** Una corrida 3D de prueba a 24³ con lote 8 (`rapido`, 300 iteraciones, en la
  5070 Ti) tuvo un pico de VRAM de **8.16 GiB** contra **4.14** predichos (su `estado.json`): la
  predicción cubre sólo el entrenamiento, no la validación (recortes de 128³, estabilidad en 96³).
  Los dos brazos de `ventana-3d-12h`, a 48³ y con 7.67 predichos, marcaron el mismo 8.16. Y una
  rama 3D reanudada desde `estado-000010.pth` dio una traza idéntica bit a bit, en CPU y
  `referencia` (lo informó el estudiante; la comparación no quedó en disco).
- **Regla 5 corregida en `AGENTS.md`.** Con sd 0.164 dB entre semillas, lo mínimo detectable entre
  dos corridas de una semilla es ≈ 0.45 dB al 95 % (1.96·√2·σ), no 0.33, que era 2σ sin el √2.
  Las salidas crudas viejas que dicen 0.33 no se reescriben. Fila en `## Descartado`.
- **Pendiente en `train.py`, que no se toca con una tanda en vuelo (regla 2):**
  - la línea 51 comenta `DELTA_ADAM` como «el piso de la regla 2-delta», que se retiró el
    2026-10-12;
  - el `--help` de `--checkpointing` (~:191) dice −70 % de memoria y +30 % de tiempo. El banco
    medido da 10.1 → 2.2 GB y 1.00 → 1.31 s/it, o sea −78 % y +31 %, que es lo que dice el README.
    Ese banco está en el README de `ff4aa3c`, no en `docs/mediciones/`.
- **El «0.33 dB» sigue en `docs/presentacion/`** (`estado_100k.py`, `guion.txt`), que se usa para
  presentar. Se corrige cuando se vuelva a generar la presentación.
