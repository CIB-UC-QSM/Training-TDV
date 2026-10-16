# Decisiones del entrenamiento 3D frente al 2D

Inventario de la rama `3D-Training` a `897fb8b` (2026-10-12): cada diferencia entre `train3d.py` y
sus módulos y el `train.py` 2D, con lo que es en cada uno, si es consecuencia mecánica de pasar a 3D,
una decisión distinta o algo nuevo, su porqué con la fuente, si está medido, derivado o es un
supuesto, y si viene activo por defecto. Los punteros `archivo:línea` valen para `897fb8b`.

**Integrada en `main` el 2026-10-15.** Lo que aquí dice «main» y «la rama», y los punteros
`archivo:línea`, valen para `897fb8b` y el `main` de ese día; la bitácora unida tiene otra
numeración. Lo que los ítems dan como corrección de `main` que la rama no tiene ya está en el mismo
árbol, y las afirmaciones de la rama que `main` había retirado llevan una nota en su sitio
(bitácora, entrada «Integración de `3D-Training` en `main`»).

Lo armaron tres agentes por área leyendo el código, la bitácora y los bloques finales. Un cuarto
verificó las afirmaciones con números y buscó lo que faltaba: sus correcciones están en la primera
sección y **mandan sobre el texto de los ítems**. No es una fuente primaria: ante una duda vale el
código o la entrada de la bitácora citada.

**Posterior al inventario:** la evaluación QSM y el seguimiento W&B quedaron integrados en
`train3d.py` (bitácora, «Evaluación QSM y seguimiento W&B integrados en `train3d.py`»). Lo que los
ítems dan como pendiente o «no integrado» describe `897fb8b`:

- la evaluación QSM corre en las validaciones (`--qsm-datos`, `--eval-qsm-cada`, `--qsm-max-iter`);
- W&B registra cada `--log-every`, lo que resuelve la discrepancia del ítem «Seguimiento W&B:
  frecuencia, métricas e imágenes»;
- la validación mide el NRMSE y, con las máscaras de `mascaras/val`, el PSNR en el cerebro;
- las banderas nuevas son `LIBRES`;
- un estado sin una bandera de receta, como los de `runs/humo3d`, se reanuda con su default;
- `best_qsm.pth`, el checkpoint para QSM (menor NRMSE QSM entre los estados con estabilidad < 2),
  viaja en el estado: el ítem «Estado completo» pesa ~51 MB en la 40 000, no ~46 (derivado);
- la tarjeta de destino es una RTX PRO 6000 Blackwell Max-Q de 96 GB, entera, y no 48 GB de una
  compartida (2026-10-15). `VRAM_GB` = 96 y `auto` da 16 (57.1 GiB predichos), no 8. Los ítems
  «Tope de VRAM» y «Micro-lote» dicen 48 y 8.

Lo vigente está en las entradas SEGUIMIENTO y EVALUACIÓN QSM del bloque final de `train3d.py`.

## Correcciones del verificador

- **Decía:** Datos de entrenamiento (grupo entrenador): «la mezcla de 2 000 cubre las colas mejor que un sorteo de 8 000: máximo 5.987 contra 7.921»
  **Error:** 5.987 y 7.921 son de la selección vieja (`--z 6 --n 500 2000 8000`, docs/mediciones/variedad-ldm100k.txt:23,25; bitácora l.5017-5023). No son las listas versionadas que usa train3d.py.
  **Correcto:** Las listas versionadas salen con los defaults (`--z 10 --n 2000 8000`). Ahí la mezcla de 2 000 tiene máximo 5.999 y el sorteo de 8 000, 8.378 (docs/mediciones/seleccion-ldm100k3d.txt:21,23; datasets/ldm100k3d/README.md:64-72). La conclusión no cambia.

- **Decía:** Modelo, esquema desenrollado: «el checkpointing baja la memoria ~9× por +27 % de tiempo (docs/bitacora.md:4890-4925)»
  **Error:** Ese ~9× se retiró el 2026-10-09 (bitácora l.5445-5447; fila en ## Descartado, l.1006). Venía del borrador, que checkpointeaba los 10 pasos.
  **Correcto:** En train3d sin --guard-amp el paso 1 va sin checkpoint (ddr/model.py:98; 1.04 pasos medios, medido en CPU). La memoria baja ~5× (derivado: 3.19e-5 contra 6.6e-6 GiB/vóxel en rapido) y ~9× sólo con --guard-amp. El +27 % también es del borrador: con 9 pasos recomputados sería algo menos, sin medir (bitácora l.5447).

- **Decía:** Estado del 3D: «Ninguna corrida larga. Sólo un humo en GPU de 60 it»
  **Error:** Incompleto: en GPU corrieron dos cosas, no una, y ninguna con los defaults actuales.
  **Correcto:** Corrieron el tronco de 60 it y una rama reanudada desde estado-000030.pth con --hitos 30 (lr 2e-4 desde la 31). La rama dio 29.490 dB en la 60, contra 29.585 del tronco, con VRAM pico 3.62 GB (docs/mediciones/humo-train3d.txt; bitácora l.5169-5172: «no es un efecto»). Las dos usaron lote 8 = 2×4 y --checkpointing. Con lote 32 y micro-lote auto no corrió ninguna iteración en GPU (l.5492-5493).

- **Decía:** Lo que NO está integrado en train3d.py: «Verificado hoy con systemctl: tdv-lr-corto-A-40k y tdv-lr-corto-C-35k están activos», y por eso el hook de la regla 2 bloquea la integración
  **Error:** Ya no es así. Lo verifiqué el 2026-10-12 a las 20:10.
  **Correcto:** `systemctl --user list-units --all 'tdv-*'` no lista ninguna unidad, y no hay ningún proceso train(3d).py vivo (pgrep anclado y ps). El salida.log de runs/lr-corto-A-40k termina en la it 17 000 (mtime 20:01) y el de C-35k en la 16 750 (19:57). Ninguno tiene estado.json. Con eso el hook no bloquea escribir train3d.py ahora. No sé por qué se cortaron. Según AGENTS.md, una corrida cortada lleva un estado.json escrito a mano con false y el motivo.

- **Decía:** Selección, control de calidad: «saca un sujeto si su distancia al centro en el espacio PCA está a más de 10»
  **Error:** El umbral de 10 no se aplica a la distancia cruda.
  **Correcto:** Se aplica al z robusto de la distancia. Con d = ‖X − mediana(X)‖, sale si (d − mediana(d)) / (MAD(d)·1.4826) > 10 (datasets/cache_ldm100k3d.py:210-212). Los escalares usan la misma escala robusta por columna, con los saturados en log1p (:206-209).

- **Decía:** Compatibilidad con ../QSM-TDV: «con los 56 tensores mapeados 1:1, ∇R difiere 1.2–3.6 %»
  **Error:** Ese rango vale sólo para entradas T1+ruido con los pesos de humo3d. Con entradas uniformes la diferencia es mayor.
  **Correcto:** Con T1+ruido, ‖d‖/‖g‖ da 2.115e-2 a 40³, 3.604e-2 a 41×45×39 y 1.223e-2 a 96³. Con x~U(0,1) da 6.58e-2 a 40³ y 9.475e-2 a 41×45×39, y max|d|/max|g| llega a 0.24–0.25 (docs/mediciones/compat-qsm-tdv.txt:33-47). Con la bajada/subida de ddr dentro de su modelo, la diferencia es 1.5e-16 (l.52).

- **Decía:** Seguimiento W&B: «La revisión arregló dos cuelgues en el prompt de login»
  **Error:** Incompleto: la revisión arregló tres cosas más las láminas.
  **Correcto:** Según la bitácora l.5732-5739, la revisión arregló tres cosas. Dos eran cuelgues: una entrada de ~/.netrc sin password y un WANDB_BASE_URL que apuntaba a otro host. La tercera: una rama lanzada en el mismo --out reabría la corrida vieja. Ahora cae a offline en 0.5 s. Además arregló el TypeError de las láminas con tensores bf16.

- **Decía:** Precisión de la evaluación: «AGENTS.md avisa que rapido contra referencia no está medido»
  **Error:** Lo dice el AGENTS.md de esta rama, pero en main ya está medido, en 2D.
  **Correcto:** En main (bitácora 2026-10-10, l.5296-5330) rapido-100k-s1 da 31.546075 dB contra 31.547008 de referencia (−0.00093 dB), con 50 de 50 puntos pareados dentro de 0.33 dB y 12.648 h. El 2026-10-12, también en main, se midió que rapido aprende un término de dato 4.8× más débil (ρ 0.031–0.039 contra 0.15–0.18; main bitácora l.19 y 954). Esto es la precisión del ENTRENAMIENTO, en BSDS. Evaluar en bf16 contra fp32 en 3D sigue sin medir.

- **Decía:** T_init y λ_init: el respaldo citado es test_t_init_respeta_la_regla_2delta (tests/test_train3d.py:222) y la predicción 5 del pre-registro (T no baja de 2·δ = 0.0253)
  **Error:** La «regla 2-δ» se retiró en main el 2026-10-12, y la rama no recoge ese retiro.
  **Correcto:** Main retiró la regla porque el factor 2 era una elección y nunca predijo: tinit/B-larga arrancó en 2·δ y murió, y t1p1delta arrancó en 1.1·δ y terminó las 100 000 (main docs/bitacora.md l.953 y 5897; tests/test_recipe.py de main). La rama sigue nombrando la regla en el test, en AGENTS.md («2·δ es un piso») y en la predicción 5. El test sigue siendo útil como tripwire de procedencia, pero el umbral de la predicción 5 es arbitrario. train.py es byte-idéntico en las dos ramas (git diff HEAD main -- train.py vacío).

- **Decía:** Modelo, sin invariant ni mask: «el state_dict 3D tiene 56 claves contra 100 del TDV 2D»
  **Error:** Esos números son sólo del regularizador R, no del checkpoint.
  **Correcto:** El state_dict de la VNet (el que va en last.pth/best.pth) tiene 58 claves en 3D y 102 en 2D, contando T y lmbda. La diferencia es la misma: 44 «.mask». Buffers: 1 500 elementos en 3D (12 blur de 125) y 688 en 2D. Lo conté en CPU con VNet3D(vnet_config3d(T_INIT)) y VNet2D(vnet_config(T_INIT, 1|3)).

- **Decía:** Tamaño de parche: «se midió con la GPU 0 ocupada (l.4893-4894)»
  **Error:** Falta decir en qué tarjeta se midió.
  **Correcto:** La grilla se midió en la GPU 1, con la GPU 0 ocupada por t1-rapido-30k (bitácora l.4892-4894: «estimación de costo en la GPU 1»). AGENTS.md pedía entonces comparar en la 0. [2026-10-15: retirado en `main` el 2026-10-08 —el «~1.3× más lenta la GPU 1» era arranque de `torch.compile`—; la regla 3 pide comparar un tiempo en la misma tarjeta y a la misma iteración.] Sirve para decidir, no como factor publicable.

- **Decía:** Calendario del lr, default y semántica de --hitos: «No encontré por qué el default de 40 000 it se combina con un halving en la 25 000»
  **Error:** Hay una mención, aunque tampoco justifica la elección.
  **Correcto:** La entrada de main del 2026-10-12, «¿El recorte del calendario se traslada al 3D?» (l.6150-6200), la registra: «con su default --iterations 40000 los hitos son range(25000, 40001, 25000) = [25000]: un solo halving, 25 k a 4e-4 y 15 k a 2e-4». Lo describe y no dice por qué. Sigue sin fuente.


## Datos y muestreo

### Fuente de datos

*consecuencia mecánica del 3D* · evidencia: **medido** · por defecto: activo

- **2D:** Réplica: BSDS400 en color, `train`+`test` de BSDS500 (400 PNG de ~481x321), CBSD68 reservado en `val` (train.py:69-72, ddr/data.py:73-89). Fase T1: cortes 2D de LDM_100k ya cacheados en `datasets/ldm100k` (datasets/cache_ldm100k.py).
- **3D:** Volúmenes enteros de LDM_100k: 100 000 T1w sintéticas uint16 de 160x224x160 a 1 mm isotrópico, alineadas entre sí, con cráneo y defacing (datasets/cache_ldm100k3d.py:36-38; datasets/ldm100k3d/README.md).
- **Por qué:** Frente a T1-2D es la misma fuente, pero leída como volumen, que es lo que pide un parche 3D. Frente a BSDS, el cambio de dominio ya se había decidido en la fase T1 (AGENTS.md: «cortes de T1 y después QSM»). Lo que se midió de la fuente: `participants.tsv` pesa 0 bytes (no hay edad, sexo ni volúmenes), tiene cráneo y el tejido ocupa ~47 % del volumen (bitácora 2026-10-08 «LDM_100k para 3D»).

### Unidad que se guarda por sujeto

*consecuencia mecánica del 3D* · evidencia: **derivado** · por defecto: activo

- **2D:** T1-2D: un corte fijo por plano (sag, cor, axi) por sujeto, elegido una sola vez al cachear en el 60 % central del eje (int(0.20n)..int(0.80n)) y con ≥ 35 % de tejido; 8 000 sujetos = 24 000 cortes float32, 3.1 GB (datasets/cache_ldm100k.py:42-51, 54-73; datasets/ldm100k/README.md). Cada iteración saca parches de esos cortes fijos.
- **3D:** El volumen entero recortado a la caja del tejido. Los parches se sortean en cada iteración en cualquier posición de esa caja, incluidos el cráneo, la base y el vértice (ddr/data3d.py:248-255).
- **Por qué:** Un parche 3D necesita el volumen. En 2D, el porqué de guardar un corte por plano era que «la variedad está en los sujetos, no en los cortes vecinos de un mismo cerebro» (datasets/cache_ldm100k.py:8-9). En 3D no hay ninguna frase que justifique usar todo el volumen: es implícito.

### Selección: escaneo de los 100 000

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: no aplica (offline; las listas versionadas ya lo incorporan)

- **2D:** No existe. T1-2D sortea sujetos al azar, sin describirlos (datasets/cache_ldm100k.py:98-103).
- **3D:** `escanea` describe cada sujeto con una miniatura float16 de 20x28x20 (promedios en bloques BLOQUE = 8 vóxeles) y 7 escalares: p99.5 crudo, fracción de tejido, fracción oscura OSCURO = (0.02, 0.3) y cuantiles 10/50/90 en la caja central (40:120, 56:168, 40:120), y vóxeles saturados ≥ SATURADO = 65 000 (datasets/cache_ldm100k3d.py:39-44, 66-68, 88-108). Duró 1 529 s con 8 procesos; leyó 100 000 de 100 000.
- **Por qué:** Como no hay metadatos, la variedad se mide en las imágenes (datasets/cache_ldm100k3d.py:342-345; bitácora «LDM_100k para 3D»). Que falten los metadatos está medido. Ninguna fuente justifica BLOQUE = 8 ni la elección de estos 7 escalares; el código sólo dice «comparables entre sujetos» (línea 39).

### Selección: PCA de las miniaturas

*nuevo (el 2D no lo tiene)* · evidencia: **sin fuente** · por defecto: no aplica (offline)

- **2D:** No existe.
- **3D:** PCA sin blanquear, con COMPONENTES = 32, ajustada sobre MUESTRA = 20 000 sujetos y proyectada sobre los 100 000 (datasets/cache_ldm100k3d.py:52, 56, 187-201).
- **Por qué:** Sin blanquear porque así «la distancia en el espacio PCA aproxima la L2 entre volúmenes de 8 mm» (datasets/cache_ldm100k3d.py:188). Las 32 componentes explican el 69.3 % de la varianza, y la primera el 12.0 % (docs/mediciones/seleccion-ldm100k3d.txt). No encontré por qué 32, por qué 20 000, ni por qué ITER_KMEANS = 50.

### Selección: control de calidad

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: activo (con z = 10 no excluye a nadie)

- **2D:** No existe. T1-2D sólo descarta un volumen que no se puede leer (datasets/cache_ldm100k.py:60-63).
- **3D:** Saca un sujeto si algún escalar está a más de Z_CALIDAD = 10 desviaciones robustas de la mediana (MAD·1.4826; los saturados se juzgan en log1p) o si su distancia al centro en el espacio PCA está a más de 10 (datasets/cache_ldm100k3d.py:55, 58, 203-221).
- **Por qué:** «Un volumen generado mal es lo primero que elegiría un muestreo por distancia máxima» (datasets/cache_ldm100k3d.py:204). Con z = 6 salían 31 (14 por escalares, 22 por PCA); se miraron 8 y tenían anatomía plausible, casi todos con ventrículos muy grandes, que es la cola que se busca. Como ninguno pasa de z = 8.1, con 10 no sale ninguno y el control queda como salvaguarda (líneas 53-55; bitácora «LDM_100k para 3D»).

### Selección del conjunto de evaluación

*decisión distinta del 2D* · evidencia: **supuesto** · por defecto: activo

- **2D:** Réplica: las 68 de CBSD68 (train.py:72). T1-2D: 68 sujetos al azar, los primeros del sorteo, con un corte axial cada uno. «68 iguala el cardinal de CBSD68» y «un solo plano deja el benchmark homogéneo» (datasets/cache_ldm100k.py:81-85, 103).
- **3D:** N_EVAL = 12 sujetos: el más cercano a cada centroide de un k-means de 12 grupos (Lloyd, 50 iteraciones, sobre una muestra de 20 000) en el espacio PCA. Son disjuntos del entrenamiento por construcción (datasets/cache_ldm100k3d.py:51, 166-173, 223-229; datasets/ldm100k3d/lista_eval.txt, 12 líneas).
- **Por qué:** Que sean «representativos de los modos, no extremos» (datasets/cache_ldm100k3d.py:224) no se midió. El número 12 sale del costo: «12 recortes de 128^3: ≤ 1 min de evaluación a 2.26 µs/vóxel» (línea 51), una cuenta derivada.

### Selección del conjunto de entrenamiento (regla mezcla)

*decisión distinta del 2D* · evidencia: **medido** · por defecto: activo

- **2D:** T1-2D: sorteo uniforme sin reposición de 8 000 sujetos, semilla 20261007 (datasets/cache_ldm100k.py:87, 98-103). BSDS: las 400 fijas.
- **3D:** Regla `mezcla`: la mitad de N por k-center voraz (FPS, que arranca en el sujeto más cercano a la media) y la otra mitad sorteada entre los que el voraz no tomó. Semilla 20261008, con un solo generador consumido en orden. N_TRAIN = (2000, 8000). Las dos listas no están anidadas: comparten la mitad voraz (datasets/cache_ldm100k3d.py:49-50, 143-151, 231-248, 357-372).
- **Por qué:** Medido como cobertura, es decir, la distancia PCA de cada sujeto del corpus a su elegido más cercano. La mezcla de 2 000 da mediana 3.996, p99 5.374 y máximo 5.999; un sorteo de 8 000 da máximo 8.378, así que la mezcla de 2 000 cubre las colas mejor (docs/mediciones/seleccion-ldm100k3d.txt; datasets/ldm100k3d/README.md). Ojo: la cobertura es de cerebros en miniaturas de 8 mm, no de parches, y en T1-2D los lotes resultaron 1.7–3.7x MENOS heterogéneos que los de BSDS (bitácora:4508-4517). Su efecto sobre el entrenamiento no está medido.

### Lista versionada y orden de los volúmenes

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: activo

- **2D:** Se entrena con todos los archivos de los splits, ordenados por nombre con `list_images` (ddr/data.py:73-89; train.py:222-225).
- **3D:** Se entrena con los archivos de `--train-list` en el orden de la lista, por defecto `datasets/ldm100k3d/lista_train_2000.txt` (train3d.py:92, 486-487). El volumen se sortea por posición, así que una lista permutada da otros parches. `PatchDataset3D.from_root` también existe, pero ordena por nombre.
- **Por qué:** Repetida con BLAS en 16 y en 32 hilos, la lista de 8 000 sale con dos pares de vecinos del voraz en otro orden (X difiere en ≤ 2.9e-6). Por eso «la que vale es la lista versionada, no el comando» (datasets/cache_ldm100k3d.py:379-382; docs/mediciones/seleccion-ldm100k3d.txt).

### Tamaño del conjunto de entrenamiento

*decisión distinta del 2D* · evidencia: **medido** · por defecto: activo (2000)

- **2D:** BSDS: 400 imágenes, que «entran holgadas en RAM» (ddr/data.py:121). T1-2D: 8 000 sujetos y 24 000 cortes en float32, 3.1 GB.
- **3D:** 2 000 volúmenes por defecto (TRAIN_LIST, train3d.py:92). Hay una lista de 8 000 que no se usa por defecto.
- **Por qué:** Es la RAM: un volumen cacheado pesa 9.9 MB de mediana en float16, así que 2 000 son ~19 GB y 8 000 ~78 GB, que no entran en los 62 GB de la máquina (ddr/data3d.py:316-318; docs/mediciones/tejido-ldm100k3d.txt). La corrida de humo tuvo un pico de RAM de 26.8 GB (bitácora «El entrenador 3D»). La lista de 8 000 está para «igualar los del cache 2D» (datasets/cache_ldm100k3d.py:50).

### Normalización de intensidad

*consecuencia mecánica del 3D* · evidencia: **medido** · por defecto: activo

- **2D:** BSDS: PNG/255 (ddr/data.py:39). T1-2D: `normaliza`, que divide por el percentil P_NORM = 99.5 de los vóxeles positivos y recorta a [0, 1] (datasets/cache_ldm100k.py:28, 32-39).
- **3D:** La misma `normaliza`, importada de `cache_ldm100k` (datasets/cache_ldm100k3d.py:33, 260). El escaneo la reescribe para obtener además el p995 (línea 97-100).
- **Por qué:** Respecto de T1-2D no cambia. Se usa el percentil y no el máximo porque hay vóxeles calientes en 6.55e4, el techo del uint16 (datasets/cache_ldm100k.py:35). T y λ se aprenden para la escala [0, 1] (ddr/data.py:6-7). Que la reescritura del escaneo da los mismos bits se verificó sobre 20 volúmenes sintéticos, con 0 diferencias (datasets/cache_ldm100k3d.py:350-353).

### Recorte a la caja del tejido con MARGEN

*nuevo (el 2D no lo tiene)* · evidencia: **supuesto** · por defecto: activo

- **2D:** T1-2D guarda el corte entero del plano, sin recortar (datasets/cache_ldm100k.py:48-50).
- **3D:** Recorta a la caja de los vóxeles > UMBRAL_TEJIDO = 0.05, agrandada MARGEN = 4 vóxeles y acotada al campo (datasets/cache_ldm100k3d.py:61, 71-79, 263-266). Las máscaras de cerebro usan la misma caja, y si cambia MARGEN hay que rehacerlas (datasets/mascaras_ldm100k3d.py:35-36).
- **Por qué:** Medido sobre 200 volúmenes: la caja mide 144–160 x 195–223 x 144–154 y toca el borde del campo en el eje 2 en los 200; fuera de la caja el p99 es 0.0098, o sea aire. El recorte ahorra poco: 9.9 MB de mediana contra 10.9 MB entero y 9.1 MB sin margen. Con margen, el filtro de tejido acepta el 93.5 % de los parches; sin margen, el 96.3 % (datasets/cache_ldm100k3d.py:391-397; docs/mediciones/tejido-ldm100k3d.txt). Que valga la pena «dejar aire entre la cabeza y el borde del parche» (línea 395) no está medido, y nada dice por qué 4 y no 0 u 8.

### Precisión de almacenamiento (disco y RAM)

*decisión distinta del 2D* · evidencia: **derivado** · por defecto: activo

- **2D:** float32 en RAM (ddr/data.py:39, 50, 122). T1-2D también guarda float32 en disco (datasets/cache_ldm100k.py:50).
- **3D:** float16 en disco (datasets/cache_ldm100k3d.py:266) y en RAM (ddr/data3d.py:199-201). El parche pasa a float32 después de orientarlo (ddr/data3d.py:266). Los recortes de evaluación se guardan en float32: 12 de 128³ son 100 MB (ddr/data3d.py:290, 378).
- **Por qué:** La mitad de RAM que float32. Con 11 bits de mantisa, el error de redondeo en [0.5, 1] es ≤ 2.4e-4, 400 veces menor que el ruido de σ = 25 (0.098). Y el limpio ES el valor guardado, así que no sesga (datasets/cache_ldm100k3d.py:389-390).

### Tamaño de parche

*decisión distinta del 2D* · evidencia: **medido** · por defecto: activo

- **2D:** PATCH_SIZE = 93, o sea 93x93, el del paper (train.py:35; ddr/data.py:119).
- **3D:** PATCH_SIZE = 48, o sea 48³ (train3d.py:54); se cambia con `--patch`.
- **Por qué:** Es el menor costo por vóxel de la grilla medida en `rapido` en la 5070 Ti: 2.26 µs a 48³x4, contra 2.77 a 32³x4 y 2.36 a 64³x1; 64³x2 no entra en 16 GB (train3d.py:758-760; docs/mediciones/costo-tdv3d.txt). En la tarjeta de destino no se midió. Consecuencias que no estaban en esa decisión: con máscara, los cubos de 48³ cubren el 66.3 % del cerebro y el 18.5 % de la franja cortical; con 32³, el 80.1 % y el 38.1 %. Además, sólo el 0.2 % de los vóxeles de un cubo de 48³ tiene todo su contexto, contra el 30.1 % de 93² (bitácora 2026-10-12, «Campo receptivo», derivado). Está escrito como pregunta abierta.

### Lote: parches y valores por iteración

*decisión distinta del 2D* · evidencia: **supuesto** · por defecto: activo

- **2D:** BATCH_SIZE = 32 parches, en micro-lotes de 16 (train.py:36, 47). BSDS: 93²x3 = 25 947 valores por parche y 830 304 por iteración. T1-2D (gris): 8 649 por parche y 276 768 por iteración.
- **3D:** LOTE = 32 parches de 48³ = 110 592 valores cada uno, 3 538 944 por iteración: 4.3x BSDS y 12.8x T1-2D. El micro-lote es `auto` (train3d.py:55-56).
- **Por qué:** El autor lo decidió el 2026-10-09: «el número del paper, no su contenido». Antes era 8, que daba 885 k valores, del orden de BSDS. No está medido cuál entrena mejor por hora; lo contesta el piloto con B_simple, columnas 5 y 10 de la traza (train3d.py:761-768; bitácora 2026-10-09 «Lote 32»).

### Canales

*consecuencia mecánica del 3D* · evidencia: **derivado** · por defecto: activo

- **2D:** CHANNELS = 3 por defecto, con `--channels` (train.py:73, 161). T1-2D corre con `--channels 1`, que da 387 394 parámetros.
- **3D:** CHANNELS = 1 fijo, sin bandera (train3d.py:94, 115).
- **Por qué:** T1 es gris (train3d.py:94). Un efecto lateral: la cota absoluta del disyuntor 3D se escala tomando como referencia BSDS en color, VALORES_BSDS = 93²x3 = 25 947, no T1-2D en gris; da 4.26e4 a 48³ (train3d.py:87, 121-123). Es una cuenta derivada, no calibrada (train3d.py:927-937).

### Criterio de tejido por parche

*decisión distinta del 2D* · evidencia: **medido** · por defecto: activo (sin --mascara-cerebro)

- **2D:** PatchDataset2D no filtra el parche: lo sortea en cualquier posición (ddr/data.py:131-142). BSDS no tiene filtro. T1-2D filtra el CORTE al cachear (≥ 0.35 de píxeles > 0.05, hasta 12 intentos), no el parche (datasets/cache_ldm100k.py:42-51, 86).
- **3D:** Filtra cada parche al sortearlo: UMBRAL_TEJIDO = 0.05, MIN_TEJIDO = 0.35 e INTENTOS = 12. Si fallan los 12, se queda el último, así que nunca cuelga (ddr/data3d.py:45-47, 248-255).
- **Por qué:** «El filtro no es del paper: en BSDS todo parche tiene contenido, y acá un tercio del cubo puede ser aire. Es el análogo del 35 % que los cortes 2D ya exigían» (ddr/data3d.py:322-323). Los valores se copiaron del 2D; no se optimizaron. Medido: un parche 48³ uniforme tiene fracción de tejido de mediana 0.90, p10 0.44 y p1 0.12. El filtro rechaza el 6.5 % (13.2 % en el peor volumen), y la probabilidad de que fallen los 12 es ≤ 3e-11 (ddr/data3d.py:319-321; docs/mediciones/tejido-ldm100k3d.txt). Ojo: «tejido» incluye cráneo y cuero cabelludo. De los parches que pasan, sólo el 4.0 % [2.8–6.2] cae entero en el cerebro, y el parche medio es 60 % cerebro (bitácora 2026-10-09).

### Augmentación geométrica: 48 orientaciones contra 8

*consecuencia mecánica del 3D* · evidencia: **derivado** · por defecto: activo

- **2D:** Flip vertical con p = 0.5, flip horizontal con p = 0.5 y rot90 con k ∈ {0, 1, 2, 3}: 16 combinaciones que cubren las 8 simetrías del cuadrado, cada una dos veces (ddr/data.py:144-151).
- **3D:** Un k uniforme entre 48 ORIENTACIONES: 6 permutaciones de ejes por 8 combinaciones de flips; primero permuta, después invierte (ddr/data3d.py:133-149, 264-265).
- **Por qué:** Las 48 son el grupo octaédrico completo, y un k uniforme lo recorre de manera uniforme. Permutar ejes vale porque los volúmenes son isotrópicos a 1 mm (ddr/data3d.py:358-363; tests/test_data3d.py:53). Advertencia escrita: con el dipolo deja de valer, porque el eje de B0 no es intercambiable (bitácora «Fase 2 en 3D»).

### Semilla por índice global y offset (reanudación)

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: activo

- **2D:** rng = default_rng((seed, index)) (ddr/data.py:134); length = iterations·32 (train.py:224). «train.py no reanuda» (train.py:506).
- **3D:** rng = default_rng((seed, offset + index)), con offset = it0·lote y length = (iterations − it0)·lote (ddr/data3d.py:246; train3d.py:505-508). Las semillas son las mismas: parches seed·100 y ruido seed·100 + 1 (train3d.py:507, 525; train.py:224, 243).
- **Por qué:** Para reanudar alcanza con `offset = it·lote`, sin guardar estado del muestreo y sin que un worker dependa de otro (ddr/data3d.py:12-13, 376-377). Verificado en `referencia` y CPU: 12 iteraciones seguidas contra 6+6 y 9+3 reanudadas dan traza, history, log y estado idénticos bit a bit (train3d.py:878-884; tests/test_train3d.py).

### Orden fijo de los sorteos dentro de un parche

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: activo

- **2D:** Imagen, esquina, flip, flip, rot (ddr/data.py:134-151). No hay aumentos de intensidad.
- **3D:** 1 volumen, 2 tres enteros por intento, 3 orientación, 4 signo, 5 ganancia. La intensidad va al final. Con máscara, 1 y 2 son otros sorteos y de 3 a 5 siguen igual (ddr/data3d.py:368-376).
- **Por qué:** Prender un aumento de intensidad no cambia la geometría: un brazo y su control ven los mismos parches. Sin máscara, el flujo es bit a bit el de antes: un test fija el sha256 de los parches 0..49 en tres configuraciones (tests/test_data3d.py:132-169).

### Aumento de signo

*nuevo (el 2D no lo tiene)* · evidencia: **derivado** · por defecto: apagado

- **2D:** No existe.
- **3D:** `--aug-signo`: x → −x con p = 1/2 (ddr/data3d.py:269-270; train3d.py:348-349).
- **Por qué:** QSM tiene susceptibilidades de los dos signos y T1 es positiva. Con padding simétrico y K1 de media cero, la energía ya es invariante a un offset; lo que falta cubrir es el signo (bitácora «Fase 2 en 3D», derivado). También imitaría la polaridad de T2: está medido que LDM_100k no tiene T2 (docs/mediciones/contraste-t1-ldm100k.txt). Su efecto no está medido.

### Aumento de ganancia

*nuevo (el 2D no lo tiene)* · evidencia: **supuesto** · por defecto: apagado

- **2D:** No existe.
- **3D:** `--aug-ganancia g`: multiplica por un factor log-uniforme en [1/g, g]. El default es 1.0, o sea apagado, y g < 1 es un error (ddr/data3d.py:192-193, 271-273; train3d.py:350-351).
- **Por qué:** Está medido que la red no es escala-equivariante (15–28 %). Entrenar con ganancias podría volver menos crítico el `alpha` del ADMM, pero eso es un supuesto (bitácora «Fase 2 en 3D»).

### Ruido: σ fijo contra ciego

*nuevo (el 2D no lo tiene)* · evidencia: **supuesto** · por defecto: apagado

- **2D:** `add_noise`: σ fijo = 25/255 (train.py:41, 280; ddr/data.py:94-98).
- **3D:** Por defecto, el mismo `add_noise` reusado con σ = 25 (ddr/data3d.py:27; train3d.py:600). Opcionalmente, `add_noise_ciego`: σ ~ U(--sigma-min, --sigma-max) por parche, primero los σ y después el ruido, del mismo generador (ddr/data3d.py:154-169; train3d.py:595-598). SIGMA_MIN = 0.0 y `--sigma-max` = 0.0, apagado (train3d.py:95, 345-347). La columna 13 de la traza guarda el σ medio del lote. La evaluación va siempre a `--sigma` fijo (train3d.py:939-944).
- **Por qué:** Es la mejora candidata de más valor para QSM: quitaría el reescalado de `--sigma-ref` y el `alpha` del solver (AGENTS.md, «Mejoras candidatas»; ddr/data3d.py:384-389). No está medida; también es supuesto que la PSNR a σ = 25 baje con un solo T (bitácora «Fase 2 en 3D»). Dos detalles: la convención U(0, 55) pide pasar `--sigma-max 55`, porque el default no la activa; y con σ_min = σ_max el ruido no es el de `add_noise` (ddr/data3d.py:159-160).

### Ruido por micro-lote y su estado

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: activo

- **2D:** Micro-lote fijo de 16 y generador `seed·100 + 1` que no se guarda (train.py:47, 243).
- **3D:** El ruido se genera en CPU, micro-lote por micro-lote. El estado del generador viaja en `estado-<it>.pth` y se restaura al reanudar (train3d.py:536, 716).
- **Por qué:** Con σ fijo y P³ múltiplo de 16, `torch.randn` da los mismos valores con cualquier partición. Medido con torch 2.13: 4 parches de 48³ juntos, 2+2 y 1+1+1+1 son idénticos bit a bit; a 15³ no. Con `--sigma-max` deja de valer y el AVISO al reanudar lo dice (train3d.py:870-877; bitácora 2026-10-09).

### Máscara de cerebro (fuente)

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: apagado

- **2D:** No existe.
- **3D:** `--mascara-cerebro` lee `<data-root>/mascaras/train/<sujeto>.npy` (train3d.py:352-354, 488-513). La máscara es deepbet 1.0.2 ∩ HD-BET 2.0.1 sin TTA, con `binary_fill_holes`, recortada con la misma caja que la cache. Se verifica que el `.npy` cacheado sea bit a bit `normaliza(NIfTI)[caja]` (datasets/mascaras_ldm100k3d.py:178-202). Las máscaras se leen una por vez con un generador, porque 2 000 juntas serían ~10 GB (train3d.py:502-504).
- **Por qué:** LDM_100k no está skull-stripped: con el filtro de tejido, sólo el 4.0 % de los parches cae entero en el cerebro. deepbet mete 69–178 ml que HD-BET deja afuera; la intersección mide 1 387 ml contra 1 389 de HD-BET y quedó sobre corteza en todas las vistas de borde miradas. Sin TTA, HD-BET cuesta 5x menos y su Dice contra la versión con TTA es 0.9972 (n = 24; datasets/mascaras_ldm100k3d.py:286-314; bitácora 2026-10-09). Las máscaras de entrenamiento no están generadas: son ~22 min de GPU para 2 000 (bitácora 2026-10-12).

### Máscara de cerebro (muestreo, MARGEN_CEREBRO, esquinas)

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: apagado

- **2D:** No existe.
- **3D:** MARGEN_CEREBRO = 1. Una esquina vale si el cubo (P + 2)³ cae entero en la máscara, y el borde del volumen cuenta como afuera. Las esquinas se cuentan con una suma acumulada 3D sobre la caja de la máscara y se guardan como índices planos uint32 (~0.2 MB por volumen); la máscara se descarta. El sorteo es uniforme entre los volúmenes con alguna esquina y después uniforme entre sus esquinas, sin filtro de tejido ni intentos. Un volumen sin esquinas queda fuera, con un aviso (ddr/data3d.py:53, 96-127, 210-234, 256-262).
- **Por qué:** El borde de la máscara es incierto en un vóxel: con y sin TTA, HD-BET difiere en una cáscara de 1–1.4 mm. Con margen 0, el 0.3 % [0.1–1.7] de los cubos toca algo que la versión con TTA excluye; con 1, el 0.0 % [0.0–0.3]; con 2 se pierde otro 22 % de esquinas y la franja cubierta baja de 18.5 a 8.6 % (ddr/data3d.py:341-345; docs/mediciones/cobertura-cubos-cerebro.txt). Hay ~55 000 esquinas por cerebro a 48³. No encontré por qué se pesa uniforme por volumen y no por número de esquinas.

### Evaluación: qué se evalúa

*decisión distinta del 2D* · evidencia: **derivado** · por defecto: activo

- **2D:** Imágenes 2D enteras: las 68 de CBSD68 o los 68 cortes axiales de T1-2D (ddr/data.py:156-178; train.py:229-230).
- **3D:** 12 recortes FIJOS de EVAL_RECORTE = 128³, centrados en la caja del tejido del volumen cacheado, es decir en la cabeza, y corridos hacia adentro si se salen. Un volumen menor que 128 es un error, no se rellena (ddr/data3d.py:48, 83-91, 277-304). `--eval-recorte` existe sólo para tests (train3d.py:378-380).
- **Por qué:** 128³ son 2.1 M vóxeles: ≤ 4.8 s por volumen a 2.26 µs/vóxel, y entra en la caja de los 200 volúmenes medidos (el mínimo por eje es 144, 195, 144) (ddr/data3d.py:324-326). No se rellena para no medir PSNR sobre ceros inventados (ddr/data3d.py:280-281). Inferencia mía, no escrita: el recorte no mide el uso final que fijó el autor (el volumen entero en QSM-TDV, bitácora 2026-10-12), y sus caras tienen el efecto de corte medido en 2D (23–29 % de σ en la cara).

### Evaluación: PSNR en tejido

*nuevo (el 2D no lo tiene)* · evidencia: **sin fuente** · por defecto: activo

- **2D:** Sólo PSNR sobre la imagen entera (evaluate.py:37-62).
- **3D:** Además del PSNR de todo el recorte, el PSNR sobre los vóxeles con limpio > 0.05. Va a `history.json` como `psnr_tejido` (evaluate.py:65-101; train3d.py:686).
- **Por qué:** La máscara sale del limpio porque la del ruidoso «cambiaría con cada realización» (evaluate.py:84). No encontré escrito por qué se agrega; el contexto es el «~38 % de fondo casi plano» de T1 (AGENTS.md). «Tejido» incluye el cráneo. La revisión dejó pendiente un PSNR dentro del cerebro, con prioridad media (bitácora 2026-10-12).

### Evaluación: precisión y σ

*decisión distinta del 2D* · evidencia: **supuesto** · por defecto: activo

- **2D:** Sin autocast, en fp32 aun en `rapido`, y con reescalado σ_ref/σ, que en el entrenamiento vale 1 (evaluate.py:41, 49; train.py:364-366). El modo por defecto es `referencia`.
- **3D:** Con el autocast del modo de la corrida: bf16 en `rapido`, que es el default 3D. Sin reescalado de σ y siempre a `--sigma`, aunque se entrene ciego (evaluate.py:66-83; train3d.py:680-682). La semilla del ruido de evaluación es la misma que en 2D: 1234 + i.
- **Por qué:** «Se evalúa como se entrena» (evaluate.py:71-72). No está medido cuánto cambia el PSNR entre evaluar en bf16 y en fp32: un PSNR 3D no se compara con el 31.547 de la réplica, que es fp32 en `referencia`.

### Workers del DataLoader

*decisión distinta del 2D* · evidencia: **supuesto** · por defecto: activo

- **2D:** 16 workers, prefetch 4, persistentes (train.py:226-228).
- **3D:** WORKERS = 2; prefetch 4 y persistentes sólo si workers > 0 (train3d.py:100, 515-518).
- **Por qué:** Un parche cuesta 0.31 ms con la CPU libre y 0.81 ms cargada: los 32 suman 26 ms en el peor caso, contra ~8 s de GPU por iteración. Con uno alcanzaría; dos «porque la tarjeta de destino es otra máquina con otra CPU». El costo está medido; la elección de 2 es un supuesto (train3d.py:96-100; docs/mediciones/tejido-ldm100k3d.txt).


## Modelo, evaluación y herramientas

### Modelo: convolución base

*consecuencia mecánica del 3D* · evidencia: **medido** · por defecto: activo

- **2D:** Conv2d con pesos (out,in,3,3), F.conv2d / conv_transpose2d, output_padding en 2 ejes (ddr/conv.py:58-170).
- **3D:** Conv3d con pesos (out,in,3,3,3), F.conv3d / conv_transpose3d, output_padding calculado eje por eje en los 3 ejes (ddr/conv3d.py:55-143). Bias-free, zero_mean/bound_norm instalan weight.proj() igual que en 2D; reduction_dim (1,2,3,4) contra (1,2,3).
- **Por qué:** El docstring dice que es un espejo de conv.py con un eje más (ddr/conv3d.py:1-11). En la bitácora figura como «Port mecánico» (docs/bitacora.md:4896-4902). En CPU y float64 las adjuntas cierran a ~1e-15 y grad contra autograd da 4.8e-16.

### Modelo: inicialización de pesos

*consecuencia mecánica del 3D* · evidencia: **sin fuente** · por defecto: activo

- **2D:** N(0, sqrt(1/(C_in·k²))) (ddr/conv.py:102).
- **3D:** N(0, sqrt(1/(C_in·k³))) (ddr/conv3d.py:90). weight.L_init = 1e4 en los dos.
- **Por qué:** Es la misma regla 1/fan-in con un eje más, pero en el repo no hay ningún porqué escrito para la versión 3D. No encontré ninguna medición específica de esta inicialización en 3D.

### Modelo: padding simétrico 3D

*consecuencia mecánica del 3D* · evidencia: **medido** · por defecto: activo

- **2D:** symmetric_pad2d / symmetric_pad2d_transpose sobre 2 ejes, con pad (izq,der,arriba,abajo) (ddr/conv.py:22-55).
- **3D:** symmetric_pad3d espeja primero el eje de profundidad y después delega a symmetric_pad2d. pad = (izq,der,arriba,abajo,adelante,atrás), en el orden de F.pad. La transpuesta va en orden inverso (ddr/conv3d.py:29-52).
- **Por qué:** El reflect de PyTorch no equivale: el simétrico repite el borde (ddr/conv3d.py:7-8, igual que conv.py:7-8). Hay un test nuevo que el 2D no tiene: compara contra np.pad(mode='symmetric'), porque un reflect pasaría el test de adjunta (tests/test_conv3d.py:29-38). La adjunta va a rel 1e-10 con 3 pads × 3 formas, con pads asimétricos y ejes pares e impares mezclados (tests/test_conv3d.py:21-50).

### Modelo: antialias binomial y bajada/subida de escala

*consecuencia mecánica del 3D* · evidencia: **medido** · por defecto: activo

- **2D:** Blur [1,4,6,4,1] como producto externo 5×5 normalizado por la suma (/256), aplicado al kernel 3×3 con padding=4 → kernel 7×7, y después stride 2. Buffer blur (1,1,5,5) (ddr/conv.py:218-236).
- **3D:** Blur [1,4,6,4,1] con einsum i,j,k → 5×5×5 normalizado (/4096), aplicado al kernel 3³ con padding=4 → 7³ (kernel_size + 2·stride), y después stride 2. Buffer blur (1,1,5,5,5). conv_down tiene forma (32,32,7,7,7). Bajada y subida no comparten pesos (ddr/conv3d.py:162-218).
- **Por qué:** Es la semántica del port de VLOGroup llevada a 3D (ddr/conv3d.py:9-11; docs/bitacora.md:4896-4900). Un test exige que sea el binomial y no una caja ones/125, que también suma 1 (tests/test_conv3d.py:95-106). Es la ÚNICA pieza que difiere de ../QSM-TDV: ellos convolucionan el 3³ y desenfocan la salida, con espejo de 2 (docs/bitacora.md:5608-5613).

### Modelo: sin opción invariant ni buffer mask

*decisión distinta del 2D* · evidencia: **medido** · por defecto: activo

- **2D:** Conv2d acepta invariant=True (pesos 1×3 expandidos a 3×3 con máscara [1,4,4]) y registra siempre un buffer mask, que la proyección de K1 usa para ponderar (ddr/conv.py:91-100, 110-117, 126-141).
- **3D:** Conv3d no tiene invariant ni mask, y get_weight devuelve el peso tal cual (ddr/conv3d.py:86-113). El state_dict 3D tiene 56 claves contra 100 del TDV 2D: son las 44 '.mask' menos. Lo conté hoy en CPU.
- **Por qué:** TDV no usa invariant en ningún kernel, y sin esa opción mask es todo unos: sacarla no cambia ninguna cuenta y deja una clave menos por convolución (ddr/conv3d.py:13-15). Un test exige que las claves 3D sean las 2D sin '.mask' y con un eje espacial más en cada kernel (tests/test_gradient3d.py:232-243).

### Modelo: micro/macro-bloques y Student-t

*consecuencia mecánica del 3D* · evidencia: **medido** · por defecto: activo

- **2D:** MicroBlock v = u + K2·φ(K1 u) y MacroBlock con 5 micro-bloques (2 por escala fina y 1 en la más gruesa). El estado multiescala persiste entre macro-bloques. La caché act_prime es local, no un estado del módulo (ddr/tdv.py:53-180). Student-t φ(a) = log(1+α a²)/(2α) con α = 1.0; si la entrada viene en half se promueve a float32 (ddr/tdv.py:33-50).
- **3D:** Es la misma topología con Conv3d/ConvScale3d/ConvScaleTranspose3d y sin el parámetro invariant (ddr/tdv3d.py:28-142). StudentT2 y MacroCache se importan de tdv.py, no se copian (ddr/tdv3d.py:23, 35).
- **Por qué:** La Student-t actúa vóxel a vóxel, así que lo único que cambia son los operadores (ddr/tdv3d.py:9-12). En float64 el Hessiano por vector y la segunda derivada en los pesos dan < 1e-9 contra autograd (tests/test_gradient3d.py:152-185). En el compat se midió que QSM-TDV evalúa la Student-t siempre en float32, con un efecto de 4.175e-08 (docs/mediciones/compat-qsm-tdv.txt, sección 4).

### Modelo: hiperparámetros de arquitectura (TDV33) y config

*decisión distinta del 2D* · evidencia: **supuesto** · por defecto: activo

- **2D:** vnet_config(t_init, channels=CHANNELS=3): S=10, num_features 32, num_scales 3, num_mb 3, multiplier 1, zero_mean True. K1 con bound_norm, K_N 1×1. T init 8·δ = 0.1012 y λ init 0.1, los dos learned con min 0 y max 1000 (train.py:30-41, 85-97). El 2D en T1 usaba --channels 1.
- **3D:** vnet_config3d = vnet_config(t_init, CHANNELS=1), más R.type = 'tdv3d' y num_features configurable sólo para los tests (train3d.py:94, 112-118). Todo lo demás es idéntico e importado de train.py (train3d.py:43-46).
- **Por qué:** Se mantiene TDV33 en vez de achicarlo para 3D: la receta se importa y un test lo verifica por identidad ('is') (tests/test_train3d.py:204-219). En «Lo que no se toca» está escrito que más features o bloques no se tocan y que «En 3D la red ya tiene 3× los parámetros sin agregar nada» (docs/bitacora.md:5094-5097). in_channels=1 porque T1 es gris (train3d.py:94). No se midió si TDV33 es el tamaño adecuado en 3D.

### Modelo: número de parámetros y buffers

*consecuencia mecánica del 3D* · evidencia: **medido** · por defecto: activo

- **2D:** TDV33 en gris: 387 392 parámetros (387 394 con T y λ). En color: 387 968 (387 970). K1 tiene 288 pesos en gris y K_N 32. Buffers: 12 blur de 25 y 44 mask, 688 elementos en total.
- **3D:** TDV33 en gris: 1 162 112 parámetros (1 162 114 con T y λ). K1 tiene 1·32·27 = 864. Cada macro-bloque suma 5 micro-bloques de 2·32·32·27 = 55 296 más 4 cambios de escala de 32·32·27 = 27 648, o sea 387 072. K_N tiene 32. Buffers: 12 blur de 125, 1 500 elementos (tests/test_gradient3d.py:28-31; ddr/tdv3d.py:146-148).
- **Por qué:** Son 27 pesos por filtro contra 9, así que sale 3.0× salvo K_N, que es 1×1(×1) en los dos (docs/bitacora.md:4900-4902). Lo verifican tests/test_gradient3d.py:141-145 y tests/test_train3d.py:246-247, y lo volví a contar hoy en CPU.

### Modelo: VNet3D y cambio en VNet2D

*decisión distinta del 2D* · evidencia: **medido** · por defecto: activo

- **2D:** Antes de la rama, VNet2D instanciaba TotalDeepVariation2D directo en __init__ (git diff e88512d..HEAD -- ddr/model.py).
- **3D:** VNet2D tiene ahora un atributo de clase TDV = TotalDeepVariation2D y construye self.R = self.TDV(...). VNet3D(VNet2D) cambia sólo TDV = TotalDeepVariation3D (ddr/model.py:49-50, 68, 126-129). El prox, T, λ, S y project_ son los mismos. OJO: esto modifica un archivo compartido del 2D.
- **Por qué:** Un atributo de clase no es un submódulo, así que no entra en el state_dict (ddr/model.py:49). Fuera del regularizador nada sabe de dimensiones (ddr/model.py:13-14). Se verificó que el state_dict 2D no cambia: huella SHA-256 de los 5 checkpoints con strict=True, antes y después (docs/bitacora.md:5146-5148). Además test_vnet2d_sigue_cargando_el_oficial (tests/test_gradient3d.py:246-253), que se saltea sin checkpoints/Oficiales.

### Modelo: esquema desenrollado, checkpointing y compile

*consecuencia mecánica del 3D* · evidencia: **medido** · por defecto: activo (compile on; checkpointing off salvo --checkpointing)

- **2D:** x_{s+1} = prox(x_s − (T/S)∇R, z, λ/S) con S = 10. efficient=True hace checkpointing de R.grad. torch.compile(model.R.grad, dynamic=False) siempre (ddr/model.py:86-109; train.py:219).
- **3D:** Es el mismo forward. torch.compile también se aplica al método y no al módulo, pero se puede apagar con --sin-compilar, una bandera sólo para tests en CPU (train3d.py:473-476, 376).
- **Por qué:** En 3D el checkpointing es exacto: < 1e-10 en float64 contra el gradiente sin él (tests/test_gradient3d.py:211-229). En 3D se midió que compile ahorra 35 % de tiempo y 38 % de memoria a 64³, y que el checkpointing baja la memoria ~9× por +27 % de tiempo (docs/bitacora.md:4890-4925). El revisor notó que con el prox el paso efectivo de ∇R es T/(S+λ), no T/S (docs/bitacora.md:5617-5620). Eso vale igual para el 2D.

### Evaluación: conjunto y recorte

*decisión distinta del 2D* · evidencia: **derivado** · por defecto: activo

- **2D:** CBSD68: 68 imágenes completas de val (train.py:71-72; ImageDataset2D, ddr/data.py:156-178). La salida trae num_images/per_image.
- **3D:** 12 volúmenes de lista_eval.txt, uno por centroide de un k-means (train3d.py:93; docs/bitacora.md:5024-5025). Cada uno se recorta FIJO a EVAL_RECORTE = 128³, centrado en la caja del tejido. Si el volumen es menor que el recorte es un error, no se rellena (ddr/data3d.py:48, 277-306). La salida trae num_volumenes/per_volume. --eval-recorte existe sólo para los tests (train3d.py:380).
- **Por qué:** 128³ son 2.1 M vóxeles, ≤ 4.8 s por volumen a 2.26 µs/vóxel, y entra en la caja de los 200 volúmenes medidos, cuyo mínimo por eje es 144, 195 y 144 (ddr/data3d.py:48, 324-326). Medido en el humo: 23 s por evaluación, y 59 s la primera porque recompila R.grad para 128³ (docs/bitacora.md:5234-5236). Derivado del campo receptivo 2D: sólo el 30.3 % de los vóxeles de un 128³ tienen contexto completo (docs/bitacora.md, entrada del 2026-10-12, «Campo receptivo»).

### Evaluación: sin reescalado de σ

*decisión distinta del 2D* · evidencia: **sin fuente** · por defecto: activo

- **2D:** evaluate_denoising(model, ds, sigma, sigma_ref) reescala la entrada por sigma_ref/sigma (sec. 5.2 del paper) (evaluate.py:38-49). En el entrenamiento los dos valen args.sigma (train.py:364-366).
- **3D:** evaluate_denoising3d no tiene sigma_ref: evalúa siempre al --sigma del entrenamiento (evaluate.py:65-73; train3d.py:320-321). Con --sigma-max (ruido ciego) también se evalúa a --sigma fijo (train3d.py:940-944).
- **Por qué:** Sólo está escrito que «en 3D se evalúa siempre al sigma del entrenamiento» (evaluate.py:72). No encontré por qué se sacó el parámetro. El contexto conocido es que la red no es escala-equivariante (15–28 %, evaluate.py:6-9).

### Evaluación: precisión (autocast)

*decisión distinta del 2D* · evidencia: **supuesto** · por defecto: activo

- **2D:** La evaluación corre siempre en fp32, sin autocast, aunque se entrene en rapido con bf16 (evaluate.py:49; train.py:365-366).
- **3D:** Evalúa bajo el autocast del modo de la corrida: bf16 en rapido, que es el default 3D (evaluate.py:71-72, 82-83; train3d.py:680-682). El denoiser QSM acepta autocast_dtype de la misma forma (ddr/qsm3d.py:225-234).
- **Por qué:** «se evalúa como se entrena» (evaluate.py:71-72). No está medido cuánto cambia el PSNR 3D entre bf16 y fp32. AGENTS.md avisa que rapido contra referencia no está medido. [corregido arriba, «Correcciones del verificador»: en `main` `rapido` reproduce `referencia` a 10⁵ (−0.00093 dB)]

### Evaluación: PSNR en tejido

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: activo

- **2D:** Sólo PSNR sobre la imagen completa (evaluate.py:50-54).
- **3D:** Además de psnr_noisy y psnr_denoised en todo el recorte, calcula psnr_noisy_tejido, psnr_denoised_tejido y fraccion_tejido por volumen. Tejido son los vóxeles LIMPIOS > UMBRAL_TEJIDO = 0.05, con la máscara sacada del limpio (evaluate.py:84-93; ddr/data3d.py:45). history.json suma psnr_tejido y efectiva (train3d.py:685-687), contra iteration, psnr, T y λ en el 2D (train.py:369-370).
- **Por qué:** T1 tiene fondo: el 38–51 % de los vóxeles crudos vale exactamente 0 (ddr/data3d.py:313; docs/mediciones/tejido-ldm100k3d.txt). La máscara sale del limpio porque la del ruidoso cambiaría con cada realización (evaluate.py:84). El test: con T = λ = 0 la salida es la entrada y el PSNR en tejido es el esperado (tests/test_train3d.py:834-861). Límite: tejido no es cerebro, incluye cráneo y cuero cabelludo. Hay un hallazgo pendiente, «la evaluación no tiene un PSNR dentro del cerebro», severidad medio (docs/bitacora.md:5681-5682).

### Evaluación: dónde viven los tensores y criterio de best.pth

*decisión distinta del 2D* · evidencia: **sin fuente** · por defecto: activo

- **2D:** Pasa clean y noisy al device y deja la salida en el device (evaluate.py:45-49). best.pth es la de mejor PSNR y se escribe con torch.save directo (train.py:373-379).
- **3D:** clean y noisy quedan en CPU: sólo la entrada va al device, y la salida vuelve en float a CPU (evaluate.py:77-83). best.pth es la de mejor PSNR en TODO el recorte, no en tejido. Se escribe a un temporal más replace, y el estado guarda una copia 'mejor' para reponerla al reanudar (train3d.py:690-698, 977).
- **Por qué:** No encontré escrito por qué los tensores quedan en CPU ni por qué best se elige por el PSNR total y no por el de tejido. El pre-registro fija como endpoint los dos PSNR, total y en tejido (docs/bitacora.md:5200-5203).

### Evaluación: CLI y cargador de checkpoints 3D

*nuevo (el 2D no lo tiene)* · evidencia: **sin fuente** · por defecto: no aplica (no implementado)

- **2D:** evaluate.py main() y load_vnet cargan cualquier .pth 2D con strict=True y miden CBSD68 (evaluate.py:28-34, 104-135).
- **3D:** No hay CLI ni cargador 3D: load_vnet construye siempre VNet2D (evaluate.py:32). vnet_config3d escribe R.type = 'tdv3d' para que «un cargador» sepa qué VNet construir (train3d.py:113-116), pero nada lo lee: un grep sólo lo encuentra en tests.
- **Por qué:** La intención está en train3d.py:113-114. No encontré por qué no se implementó el cargador.

### Evaluación QSM: ddr/qsm3d.py (ADMM wTDV del autor)

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: no integrado (train3d.py no lo importa)

- **2D:** El entrenamiento 2D nunca evaluó QSM. El protocolo vive en ../tdv/wTDV_QSM_torch_3d.py, fuera del repo.
- **3D:** admm_wtdv generaliza ese ADMM a k denoisers: k = 3 es el pseudo-3D (la red 2D en los 3 ejes, cada split pesa μ/k) y k = 1 es VNet3D sobre el volumen entero (denoiser_3d). Constantes del autor: MU = 0.008, MU2 = 1.0, TARGET_STD = 0.06, MAX_ITER = 100, TOL = 1.05, CORTE = 1.4, y para la máscara UMBRAL_SOLIDO 0.05, FRACCION_MEDIA 0.05, ITER_CIERRE 2, SIGMA_SUAVIZADO 1.0, UMBRAL_SUAVIZADO 0.5 (ddr/qsm3d.py:42-58, 132-234). Métrica: NRMSE = 100·‖χ−χ_GT‖/‖χ_GT‖ en chi≠0.
- **Por qué:** El autor pidió evaluar periódicamente el modelo como denoiser dentro de su reconstrucción QSM (docs/bitacora.md:5691-5698). Reproduce el 26.89 % publicado: 26.8898 % con k = 3 y la red 2D color, CPU fp32, α = 4.8591 y corte en t = 26. Contra el log del autor difiere a lo sumo 0.0026 pp de NRMSE y 0.0094 de update (docs/mediciones/qsm-eval-cosmos.txt; ddr/qsm3d.py:332-337). Contra el bucle transcripto da bit a bit sobre un sintético de 20³. k = 1 contra k = 3 con el mismo denoiser difiere 2.5e-8.

### Evaluación QSM: calibración alpha/TARGET_STD para la red 3D

*decisión distinta del 2D* · evidencia: **supuesto** · por defecto: no integrado

- **2D:** TARGET_STD = 0.06 se calibró por grid search para la red 2D color del autor (ddr/qsm3d.py:47).
- **3D:** Se hereda sin recalibrar: «No tocar: es la referencia» (ddr/qsm3d.py:43-44). α se fija en t = 0 con χ_0, independiente de la red (ddr/qsm3d.py:179-181, 323-324).
- **Por qué:** Está abierto: TARGET_STD se calibró para la red 2D color, no para una 3D entrenada en T1, y no está medido (docs/bitacora.md:5717-5718). α existe porque la red conoce un solo σ (ddr/qsm3d.py:10-12).

### Evaluación QSM: corte por no finito y costo 3D

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: no integrado

- **2D:** No aplica.
- **3D:** Agrega motivo 'no_finito', que corta si el update deja de ser finito, algo que el script del autor no hace (ddr/qsm3d.py:196-199). Costo del 3D sobre COSMOS 160³ de una vez, en CPU fp32 con 8 hilos: 466 y 478 s por iteración, 7.77 GB de pico de RSS (ddr/qsm3d.py:341-347).
- **Por qué:** Sin el corte, una red muerta correría las 100 iteraciones con NaN (ddr/qsm3d.py:196). El test termina en 2 iteraciones (tests/test_qsm3d.py:225-229). El costo está medido en CPU (docs/mediciones/qsm-eval-cosmos.txt, sección d). En GPU no está medido: el agente supuso ~3 s por iteración en la 5070 Ti (docs/bitacora.md:5713-5714). Un modelo poco entrenado puede no cortar nunca.

### Evaluación QSM: láminas uint8

*nuevo (el 2D no lo tiene)* · evidencia: **derivado** · por defecto: no integrado

- **2D:** No hay imágenes de evaluación.
- **3D:** cortes_centrales, lamina_qsm (GT | reconstrucción | error en la máscara, rango ±0.1 ppm) y lamina_denoising (limpia | ruidosa | salida | 5·|error|, rango [0, 1]). AMPLIACION_ERROR = 5.0, ORIENTACION_COSMOS = (-90,-90,90), ORIENTACION_T1 = (-90,-90,-90), SEPARACION = 2 (ddr/qsm3d.py:60-70, 237-312).
- **Por qué:** La ampliación ×5 es derivada: el ruido σ = 25/255 queda en 0.49 y un error de 30 dB en 0.16 (ddr/qsm3d.py:63-65). Las orientaciones se miraron a ojo el 2026-10-12, y la de COSMOS es la de imshow_3d del autor (ddr/qsm3d.py:66-69). Se arregló un TypeError con bf16 (ddr/qsm3d.py:275).

### Seguimiento W&B: ddr/seguimiento.py

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: no integrado

- **2D:** No hay seguimiento externo: las fuentes son traza.npy, history.json y estado.json.
- **3D:** abre, registra, registra_imagenes, resumen y cierra. Modos auto/online/offline/no. auto mira en orden WANDB_MODE, después si la corrida previa era offline, después si hay credencial; si no hay credencial va offline, y si init online falla cae a offline. PROYECTO = 'training-tdv3d', INIT_TIMEOUT = 30 s, console='off'. Toda llamada pasa por _protegido: el primer error apaga el seguimiento y la corrida sigue (ddr/seguimiento.py:1-17, 40-286).
- **Por qué:** El autor pidió comet_ml o W&B. Se eligió W&B porque corre offline sin cuenta y se sube después con wandb sync (docs/bitacora.md:5695-5697). La invariante de que el seguimiento nunca corta el entrenamiento replica que train.py no corta el bucle. PROYECTO e INIT_TIMEOUT son supuestos (ddr/seguimiento.py:42-47). La revisión arregló dos cuelgues en el prompt de login (docs/mediciones/seguimiento-cuelgues.txt). Esta máquina no tiene credencial, así que auto sale offline.

### Seguimiento W&B: identidad de la corrida y reanudación

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: no integrado

- **2D:** train.py no reanuda (train.py:506).
- **3D:** El id vive en <out>/seguimiento.json. Hay 4 casos: sin json, corrida nueva; mismo proyecto con desde ≥ paso, se reabre la misma; desde < paso, corrida nueva y el id viejo pasa a 'descartadas'; padre en otro directorio, es una rama: corrida nueva en el group del tronco (ddr/seguimiento.py:115-139, 360-383).
- **Por qué:** W&B descarta sin aviso un paso menor que el último. Medido offline: con los pasos 5, 3, 6 guarda [5, 6] (docs/mediciones/costo-seguimiento.txt). Sin verificar, porque no hay servidor: que online acepte una config cambiada al reabrir y que los extremos del resumen sobrevivan (ddr/seguimiento.py:382-383).

### Seguimiento W&B: frecuencia, métricas e imágenes

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: no integrado

- **2D:** El log va cada --log-every (250).
- **3D:** Está diseñado para registrar en CADA iteración las métricas train/* de las 14 columnas de la traza (COLUMNAS_TRAZA), más grad_razon = ‖g‖/mediana y rechazos y rebobinados acumulados. EXTREMOS guarda el máximo de val/psnr y el mínimo de val/nrmse_pct y qsm/*. Las imágenes tienen que ser uint8: si no, es un error que apaga el seguimiento (ddr/seguimiento.py:248-322, 394-441).
- **Por qué:** Se registra por iteración porque un estallido dura una iteración (‖g‖ de 1e8–1e13). Medido: 0.185 ms de mediana y 0.335 ms de p99 por llamada, y 1 337 B por fila. Derivado: 53 MB a 40 000 iteraciones y 56 MB de métricas de sistema en 89 h (ddr/seguimiento.py:398-404; docs/mediciones/costo-seguimiento.txt). Las imágenes van en uint8 porque wandb 0.30 pasa un float a uint8 con astype y la imagen saldría negra (ddr/seguimiento.py:257-261). DISCREPANCIA sin resolver: la bitácora dice «registrar las columnas de la traza cada --log-every» (docs/bitacora.md:5743), mientras que el módulo dice que es por iteración.

### Compatibilidad con ../QSM-TDV

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: no aplica

- **2D:** El 2D es compatible con el repo de VLOGroup: los oficiales cargan con strict=True y reproducen los 4 PSNR publicados.
- **3D:** El checkpoint 3D NO carga directo en QSM-TDV: ValueError 'Checkpoint does not contain TDVEnergy3D parameters', porque las claves son R.K1/R.mb/R.KN contra analysis./macro_blocks./energy_head.. Con los 56 tensores mapeados 1:1 carga con strict=True, pero ∇R difiere 1.2–3.6 %: 2.115e-2 a 40³, 3.604e-2 a 41×45×39 y 1.223e-2 a 96³. La diferencia vive en las caras, y recién a 13–15 vóxeles del borde baja de 1e-3 (docs/mediciones/compat-qsm-tdv.txt).
- **Por qué:** La única pieza distinta es la bajada/subida: nosotros desenfocamos el kernel y espejamos 3, ellos convolucionan y desenfocan la salida, espejo de 2. El control: con nuestra bajada/subida dentro de su modelo, ∇R coincide a 1.5e-16. strict=True no prueba el mapeo, porque uno cruzado carga y difiere 70–78 % (docs/mediciones/compat-qsm-tdv-simetrico.txt; docs/bitacora.md:5602-5627). Las opciones (a), (b) y (c) están sin decidir; la recomendada es (a), una bandera en QSM-TDV de ~30–40 líneas. QSM-TDV corre en float16 (use_amp), y eso no se midió con nuestros pesos.

### Compatibilidad: memoria de ∇R sobre el volumen entero

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: no aplica

- **2D:** No aplica.
- **3D:** En ddr: 0.067 GiB + 1.5378 GiB por millón de vóxeles (1 651 B/vóxel). Medido 8.94 GiB y 60 s a 160×224×160 en CPU fp32; a 256×256×176 se predicen 17.81 GiB. En QSM-TDV: 3 487 B/vóxel, 2.1×. Los dos aceptan tamaños impares (37×41×43: adjunta a ~1e-16) (docs/mediciones/compat-qsm-tdv.txt).
- **Por qué:** El objetivo del autor es un checkpoint que QSM-TDV use sobre el volumen entero, de una vez (docs/bitacora.md:5591-5593). Sólo está medido en CPU. Para GPU fp32 es una cota derivada, y bf16 no está medido.

### Dependencias

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: activo (declarada; se importa sólo con modo distinto de 'no')

- **2D:** pyproject sin wandb. El .venv principal se comparte con main.
- **3D:** pyproject.toml suma 'wandb>=0.30.0' en una línea y uv.lock suma 626 líneas, sólo paquetes agregados. El worktree tiene su propio .venv, con wandb 0.30 instalado. La suite del .venv principal saltea los tests de wandb con importorskip.
- **Por qué:** Es para no tocar el .venv que se comparte con main (docs/bitacora.md:5752-5754). Con modo 'no', wandb ni se importa (ddr/seguimiento.py:7).

### Tests: tests/test_conv3d.py

*consecuencia mecánica del 3D* · evidencia: **medido** · por defecto: activo

- **2D:** tests/test_conv.py: adjunta del pad, de conv, de bajada y de subida, con formas 32×32 y 31×29 y pad 1, más el blur binomial.
- **3D:** Adjunta <Kx,y> = <x,Kᵀy> en float64 para pad3d (3 pads: (1…), (3…) para kernels 7³ y el asimétrico (1,2,0,3,2,1)) × 3 formas con ejes pares e impares mezclados (12×10×8, 11×9×7, 12×9×10). También para Conv3d (k = 1 y 3), ConvScale3d y ConvScaleTranspose3d, que debe recuperar la forma exacta. Más pad = np.pad symmetric, blur binomial y no caja, proyección de K1 con in_channels 1 y 3, y que proj sólo se instale con restricciones (tests/test_conv3d.py:21-143).
- **Por qué:** Las formas mezclan ejes porque el output_padding se calcula eje por eje y un cubo no vería un error en uno solo (tests/test_conv3d.py:1-6). Es el único sostén del 3D: no hay checkpoint oficial contra el que compararlo (ddr/__init__.py:5-7). El test de np.pad es nuevo respecto del 2D.

### Tests: tests/test_gradient3d.py

*consecuencia mecánica del 3D* · evidencia: **medido** · por defecto: activo

- **2D:** tests/test_gradient.py: grad contra autograd, derivada direccional, repetibilidad, densidad, proyección, sin bias, Hessiano por vector y segunda derivada en pesos.
- **3D:** Lo mismo sobre 2×1×13×14×15, con dos ejes impares, que bajan a 7×7×8 y 4×4×4 (atol 1e-9). Además, lo que el 2D no tiene: invariancia a offsets grad(x+0.37) = grad(x) con atol 1e-12, conteo de 1 162 112 parámetros, VNet3D con S pasos igual al prox paso a paso, checkpointing exacto (< 1e-10), claves 3D = claves 2D sin mask, y que VNet2D siga cargando el oficial con strict (tests/test_gradient3d.py:1-253).
- **Por qué:** La segunda derivada se testea porque el backward del entrenamiento atraviesa la cadena adjunta escrita a mano (tests/test_gradient3d.py:148-150). Mutaciones: 8 de 8 en el modelo ponen algún test en rojo (docs/bitacora.md:5164-5165).

### Tests: tests/test_train3d.py (receta, reanudación bit a bit, evaluación)

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: activo

- **2D:** No hay tests de reanudación, porque train.py no reanuda. La receta se testea en tests/test_recipe.py.
- **3D:** 22 nombres de la receta son el MISMO objeto que en train.py. Los defaults son la receta: patch 48, lote 32, micro 'auto', vram 48, features 32, eval_recorte 128, mode rapido y todo lo candidato apagado. Reanudación: 12 iteraciones seguidas contra 6+6 y 9+3, en subprocesos, CPU y referencia, dan traza.npy idéntica en bits, más history, log, estado completo y last/best.pth, con disyuntor forzado, ruido ciego y aumentos. También que el bucle no tenga break y que la evaluación mida recorte y tejido (tests/test_train3d.py:1-861).
- **Por qué:** Las corridas largas necesitan sobrevivir cortes y ramas (docs/bitacora.md:5036-5075). Suite del 2026-10-08: 130 pasados, y 17 de 19 mutaciones de train3d.py en rojo, las otras 2 equivalentes (docs/bitacora.md:5156-5165). Estos tests lanzan train3d.py en subprocesos, así que el hook de la regla 2 los bloquea con una tanda viva (AGENTS.md).

### Tests: tests/test_data3d.py (hashes)

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: activo

- **2D:** No hay hashes de parches.
- **3D:** Guarda el sha256 de los bytes float32 de los parches 0..49 de cada configuración (receta, aumentos, sin aumentos), calculado el 2026-10-12 con el ddr/data3d.py de 23b07ec. Garantiza que el camino sin máscara no cambió al agregar --mascara-cerebro (tests/test_data3d.py:132-166). Prueba además las herramientas de cache y máscaras sobre NIfTI sintéticos.
- **Por qué:** Es para que agregar la máscara no cambie un bit de las corridas sin ella. Una revisión independiente lo confirmó sobre 6 configuraciones × 200 parches (docs/mediciones/revision-mascara-cerebro.txt, sección 3).

### Tests: tests/test_qsm3d.py y tests/test_seguimiento.py

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: activo (los de wandb, sólo con wandb instalado)

- **2D:** No existen.
- **3D:** test_qsm3d: ADMM con k = 3 bit a bit contra el bucle del autor transcripto, sobre un sintético de 20³ con 3 denoisers de juguete NO lineales y distintos, para que un alpha mal aplicado o splits cruzados fallen. Además la semántica del corte, k = 1 ≡ k = 3, la red muerta, sin GT, carga_cosmos exacto sobre los .mat reales (se saltea sin ../tdv/data_mat), denoiser_3d, bf16 → float32, NRMSE y láminas. test_seguimiento: todo offline en tmp_path, leyendo el .wandb; los 4 casos de identidad; que 'no' no importe wandb; que los errores no lleguen al bucle; los fallbacks; que no toque los RNG de torch, numpy ni random; que las columnas coincidan con COLUMNAS = 14 de train3d; y que NRMSE y PSNR sean la misma información por volumen.
- **Por qué:** Lo dicen los docstrings (tests/test_qsm3d.py:1-7; tests/test_seguimiento.py:1-7). El test de los RNG garantiza que prender el seguimiento no cambia la aritmética de la corrida (ddr/seguimiento.py:410-412).

### Herramientas: --sonda-vram y banderas de test

*nuevo (el 2D no lo tiene)* · evidencia: **supuesto** · por defecto: apagado

- **2D:** No hay sonda. train.py no tiene --features ni --sin-compilar, y compila siempre.
- **3D:** --sonda-vram mide en la GPU el pico y los s/parche de cada micro-lote posible, escribe sonda-vram.json y sale sin entrenar. Usa SONDA_ITERACIONES = 4 y SONDA_DESCARTADAS = 1 (train3d.py:76-81, 229-310, 338-340). --features, --eval-recorte y --sin-compilar existen sólo para achicar la red en CPU (train3d.py:376-380).
- **Por qué:** El modelo de memoria se ajustó en la 5070 Ti y en otra tarjeta es una extrapolación (train3d.py:61-64). Los 4/1 de la sonda son supuestos, sin dispersión medida (train3d.py:78-81). La columna s/parche no se probó en GPU (docs/bitacora.md, entrada del 2026-10-12). Un test exige que las banderas de test no cambien los defaults (tests/test_train3d.py:228-247).

### Lo que NO está integrado en train3d.py

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: no integrado

- **2D:** No aplica.
- **3D:** train3d.py no importa qsm3d ni seguimiento (grep: 0 coincidencias). Todavía faltan: la evaluación QSM en cada validación; W&B, con abre después de iter(loader), registra por iteración, val/* e imágenes, y resumen y cierra al final; val/nrmse_pct en evaluate_denoising3d, que hoy no lo calcula (ddr/seguimiento.py:427-429); un PSNR dentro del cerebro; un cargador/CLI 3D; y que los estados de runs/humo3d se puedan reanudar, porque fallan por la clave nueva (train3d.py:437, severidad bajo).
- **Por qué:** El hook scripts/guardia_tanda.py (regla 2) bloquea escribir train3d.py mientras haya un train.py vivo. Verificado hoy con systemctl: tdv-lr-corto-A-40k y tdv-lr-corto-C-35k están activos. Condiciones de la revisión para integrarlo: un OOM del QSM sobre 160³ no puede matar el entrenamiento, el QSM no puede consumir el generador del ruido, y las banderas nuevas van a LIBRES (docs/bitacora.md:5741-5751; ddr/seguimiento.py:394-409).


## Entrenador

### Estado del 3D (qué se corrió)

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: no aplica

- **2D:** Fase 1 cerrada: t8delta-100k-s1 llega a 31.547 dB en CBSD68 (σ=25) contra 31.53 publicado (AGENTS.md, intro).
- **3D:** Ninguna corrida larga. Sólo un humo en GPU de 60 it (rapido, lote 8 de 48³ = 2×4, --checkpointing): 28.759 dB en la 30 y 29.585 en la 60, 0 rechazos, VRAM pico 3.59 GB, RAM 26.8 GB (bitácora l.5166-5172; docs/mediciones/humo-train3d.txt). El piloto (tronco de 40 000 + 3 ramas) es un pre-registro en borrador, sin lanzar (bitácora l.5179-5183). El lote 32 y la sonda nunca corrieron en GPU (l.5492-5493, 5677-5678). Ninguna traza 3D pasa de 80 it (l.5198).
- **Por qué:** Todo lo que sigue está verificado en tests de CPU y decenas de iteraciones. La tasa de rechazos, el codo del lr y B_simple los contesta el piloto (bitácora l.5174-5175).

### Receta importada (lo que no cambia)

*decisión distinta del 2D* · evidencia: **supuesto** · por defecto: activo

- **2D:** LR=4e-4, BETAS=(0.9,0.999), SIGMA=25, STEPS S=10, TDV33 (FEATURES=32, SCALES=3, MACRO_BLOCKS=3), NOISE_SCALE=255, DATA_RANGE=1, --clip inf, Adam de un solo grupo, project_() tras cada paso, semillas torch=seed, parches=seed·100, ruido=seed·100+1, bucle sin break (train.py:35-43, 74-75, 172, 200, 224, 235, 243, 317).
- **3D:** Los mismos objetos, importados de train.py y no copiados (train3d.py:42-47). tests/test_train3d.py:204-219 exige `is`. Mismo Adam (train3d.py:524), mismas semillas (460, 507, 525), mismo project_ (638) y mismo --clip inf (355). El bucle tampoco corta nunca (test_el_bucle_no_corta_nunca, tests/test_train3d.py:826).
- **Por qué:** Una sola fuente: tocar train.py mueve los dos (train3d.py:50-53; README.md:376-378). Que estos valores sirvan en 3D/T1 no está medido. AGENTS.md («Lo que ya se sabe») dice que todo es de BSDS color y que otro dominio pide volver a medirlo.

### T_init y λ_init

*decisión distinta del 2D* · evidencia: **supuesto** · por defecto: activo

- **2D:** T_INIT = 8·δ_Adam = 8·0.01265 ≈ 0.1012; LAMBDA_INIT = 0.1; --t-init con default T_INIT (train.py:51-56, 150).
- **3D:** Los mismos, importados; --t-init con default T_INIT (train3d.py:44-46, 319). vnet_config3d los pasa intactos (112-118). test_t_init_respeta_la_regla_2delta (tests/test_train3d.py:222).
- **Por qué:** En 2D están medidos: 8·δ llega a 31.547 dB y T_init no mueve el endpoint. En 3D es un supuesto: la predicción 5 del pre-registro dice que T no baja de 2·δ = 0.0253 en 40 000 it (bitácora l.5221-5222), y no hay corrida que lo pruebe. [la «regla 2-δ» se retiró en `main` el 2026-10-12: ver «Correcciones del verificador»]

### Calendario del lr: mecanismo

*decisión distinta del 2D* · evidencia: **derivado** · por defecto: activo

- **2D:** MultiStepLR(gamma=0.5) con hitos = range(primero, iterations+1, --lr-halve-every). Defaults: --lr-halve-every 25 000, --lr-halve-start 0 (= every), --lr-halvings 0 (sin tope). sched.step() va después de opt.step() y el lr vive en opt.param_groups (train.py:165-169, 236-242, 316).
- **3D:** Sin scheduler (train3d.py:523). lr_en(it, hitos) = 4e-4·0.5^#{h ≤ it−1} se escribe en el optimizador al principio de CADA iteración (176-183, 587-588). Las tres banderas --lr-halve-* desaparecen y las reemplaza --hitos, una lista explícita con nargs='*' (341-343, 420-421).
- **Por qué:** Que una rama pueda reanudar con otro calendario y que ningún estado restaurado (respaldo o estado-<it>.pth) traiga un lr viejo (train3d.py:585-586, 841-849; bitácora «Fase 2 en 3D», punto 2, l.5058-5060). La equivalencia iteración por iteración con MultiStepLR la verifican tests/test_train3d.py:487 (test_lr_en_reproduce_multisteplr) y 503.

### Calendario del lr: default y semántica de --hitos

*decisión distinta del 2D* · evidencia: **sin fuente** · por defecto: activo: sin --hitos hay un halving en la 25 000

- **2D:** El default de 10 000 it (ITERACIONES_TANDA) no alcanza el primer hito, así que el lr queda constante en 4e-4 (train.py:55, 149; bloque final l.490-491).
- **3D:** Sin --hitos: múltiplos de 25 000 hasta --iterations. Con el default de 40 000 eso da [25000]: lr 2e-4 desde la 25 001. «--hitos» sola da [] (lr constante, el tronco del piloto). «--hitos 10000» da lr/2 desde la 10 001 (train3d.py:17-18, 341-343, 420-421).
- **Por qué:** La regla «sin bandera = calendario del paper» está documentada en el help (l.342-343) y el docstring la advierte con «Ojo con --hitos» (l.17-18). No encontré por qué el default de 40 000 it se combina con un halving en la 25 000 cuando todos los comandos del piloto pasan --hitos (README.md:406, 409, 416; bitácora l.5255).

### Iteraciones por defecto

*decisión distinta del 2D* · evidencia: **derivado** · por defecto: activo

- **2D:** ITERATIONS = 100 000 (la receta, no es el default) y --iterations con default ITERACIONES_TANDA = 10 000 (train.py:37, 55, 149).
- **3D:** --iterations con default ITERACIONES_TANDA = 40 000 (train3d.py:57, 318). No hay constante de 100 000 en 3D.
- **Por qué:** Es el tronco del piloto (ramas desde la 10 000, 20 000 y 30 000, de 5 000 it cada una). Costo derivado: 0.250 s/parche × 32 × 40 000 = 89 h en la 5070 Ti con micro-lote 4, 94 h con el 2 de auto (0.265 s). En la tarjeta de destino no está medido (train3d.py:774-776; bitácora l.5278-5281). [2026-10-15: la sonda del 2026-10-13 midió en la 5070 Ti 0.280–0.291 s/parche con el micro-lote 2, y la tanda `ventana-3d-12h` corrió a 9.17–9.24 s/it (docs/mediciones/ventana-3d-12h.txt): el costo se lee en s/it, y las horas salen del s/it de cada tarjeta.]

### Lote (parches por paso de Adam)

*decisión distinta del 2D* · evidencia: **supuesto** · por defecto: activo

- **2D:** BATCH_SIZE = 32 parches de 93×93×3, 830 k valores por paso, constante sin bandera (train.py:36).
- **3D:** LOTE = 32 parches de 48³, 3.54 M vóxeles por paso (4.3×), con bandera --lote (train3d.py:55, 329). Viaja en el estado y --reanudar rechaza cambiarlo. Hasta el 2026-10-09 era 8 (885 k, del orden de BSDS en valores).
- **Por qué:** Decisión del autor del 2026-10-09: igual al «mini batch size of 32» del paper en número de parches, no en contenido (train3d.py:761-768; bitácora l.5306-5312, 5351-5360). Cuál lote entrena mejor por hora no está medido. Lo contestaría B_simple con las columnas 5 y 10 de la traza del piloto.

### Tamaño de parche

*decisión distinta del 2D* · evidencia: **medido** · por defecto: activo

- **2D:** PATCH_SIZE = 93 (93×93), fijo y sin bandera (train.py:35).
- **3D:** PATCH_SIZE = 48 (48³), con bandera --patch (train3d.py:54, 328).
- **Por qué:** Es el menor costo por vóxel de la grilla en rapido en la 5070 Ti: 2.26 µs a 48³×4, contra 2.77 a 32³×4 y 2.36 a 64³×1; 64³×2 no entra en 16 GB (docs/mediciones/costo-tdv3d.txt; train3d.py:758-760; bitácora l.4904-4918). Se midió con la GPU 0 ocupada (l.4893-4894): sirve para decidir, no como factor publicable. No se eligió por campo receptivo. Derivado: sólo el 0.2 % de un cubo de 48³ queda a ≥21 vóxeles de todas las caras (bitácora l.5659-5663, «queda abierto»).

### Micro-lote y acumulación

*nuevo (el 2D no lo tiene)* · evidencia: **derivado** · por defecto: activo

- **2D:** MICRO_BATCH = 16 fijo y ACUMULACION = 2, sin bandera. Se eligió para que intermedio y referencia entren sin checkpointing en 16 GB (train.py:45-48).
- **3D:** --micro-lote acepta 'auto' (default) o un entero que divida a --lote (train3d.py:56, 163-173, 330-332, 404-413). auto = micro_lote_auto: el mayor divisor de 32 con pico predicho ≤ presupuesto·(1−0.10) (144-160). Da 8 en 48 GiB (28.83 GiB predichos; 12 entrarían pero no dividen 32, y 16 predice 57.06), 2 en la 5070 Ti, 32 con --checkpointing en 48 GiB y 4 en intermedio/referencia en 48 GiB. config.json y el estado guardan el número efectivo, no 'auto'.
- **Por qué:** El micro-lote es hardware, no receta: con acumulación el gradiente es el mismo para cualquier partición, porque no hay normalizaciones. Cambiar la partición al reanudar se aparta a ≤7.8·10⁻⁷ relativo, sin ser bit a bit (medido en CPU/referencia, tests/test_train3d.py:420; bitácora l.5334-5342). Usar ~60 % de 48 GiB es derivado: en la 5070 Ti, de ×2 a ×4 el costo por vóxel bajaba sólo 6 % (2.39→2.26 µs). Que ×8 sature en la tarjeta de destino es un supuesto (bitácora l.5385-5390).

### Modelo de memoria (FIJA, POR_VOXEL, MARGEN)

*nuevo (el 2D no lo tiene)* · evidencia: **derivado** · por defecto: activo (lo usa auto)

- **2D:** No existe.
- **3D:** pico GiB = FIJA + n·P³·POR_VOXEL[modo, checkpointing] (train3d.py:61-77, 126-137). FIJA = 0.61. POR_VOXEL: rapido 3.19e-5; intermedio y referencia 7.0e-5; rapido con ckpt 3.4e-6; intermedio/referencia con ckpt 5.6e-6. Con --checkpointing y sin --guard-amp suma POR_VOXEL[modo,False]/S (el paso 1 va sin checkpoint): 6.6e-6 en total en rapido y 1.26e-5 en referencia. MARGEN = 0.10.
- **Por qué:** Ajustado a la grilla de la 5070 Ti (costo-tdv3d.txt, 2026-10-08). A 48³ y 64³ en rapido queda a menos de 2.2 % de lo medido; a 32³×4 subestima 8 % (4.79 contra 5.21); las 9 configuraciones que dieron OOM las predice por encima de 15.45 GiB (tests/test_train3d.py:288). En 48 GiB es una extrapolación. intermedio con ckpt no se midió y usa el valor de referencia. MARGEN es un supuesto que cubre evaluación a 128³, fragmentación y cuDNN (train3d.py:787-822; bitácora l.5392-5416).

### Tope de VRAM (--vram-gb, set_per_process_memory_fraction)

*nuevo (el 2D no lo tiene)* · evidencia: **supuesto** · por defecto: activo: sólo limita en tarjetas de más de 48 GiB

- **2D:** No existe: usa la tarjeta entera.
- **3D:** --vram-gb con default VRAM_GB = 48.0 GiB (train3d.py:65, 333-337). presupuesto = min(vram_gb, total de la tarjeta); si queda por debajo, torch.cuda.set_per_process_memory_fraction(presupuesto/total) (398-403). En CPU el presupuesto es --vram-gb, para que los tests no dependan de la máquina. 44.7 sólo para una porción pactada en GB decimales.
- **Por qué:** El autor puede usar 48 GB de una Blackwell, quizás compartida, y «mejor un OOM propio que pisar a otro proceso» (train3d.py:402, 812-816; bitácora l.5308-5321). Va en GiB como los reporta torch: la 5070 Ti da 16 585 719 808 B = 15.45 GiB (medido en un OOM, bitácora l.5378-5383).

### --sonda-vram

*nuevo (el 2D no lo tiene)* · evidencia: **supuesto** · por defecto: apagado

- **2D:** No existe.
- **3D:** Bandera store_true. En GPU corre SONDA_ITERACIONES = 4 de forward+backward por cada divisor de --lote, de menor a mayor, y no cronometra SONDA_DESCARTADAS = 1. Para en el primero que no entra o da OOM, escribe sonda-vram.json (predicho contra medido, s/parche y el más rápido) y sale sin cargar datos ni entrenar (train3d.py:78-82, 229-310, 338-340, 479-481). Sube recompile_limit de dynamo a 2·#divisores+2 (241-242). Sin GPU sale con error (391-393).
- **Por qué:** El modelo de memoria es una extrapolación desde 16 GB. La trampa de dynamo se midió en CPU: cada forma deja 2 entradas en la caché, y pasado el límite de 8 sigue en eager, con 61 % más memoria a 64³ (bitácora l.5468-5477). Las cuentas 4/1 son un supuesto, «sin medir su dispersión» (train3d.py:78-80). Eran 2 iteraciones (bitácora l.5322); pasaron a 4 con la columna de s/parche en 0eb0074. Nunca se corrió en GPU.

### Modo de ejecución por defecto

*decisión distinta del 2D* · evidencia: **medido** · por defecto: activo

- **2D:** --mode con default 'referencia': fp32 IEEE, determinista, benchmark apagado (train.py:78-82, 148).
- **3D:** --mode con default 'rapido': autocast bf16, benchmark encendido, no determinista. El dict MODES es el mismo, importado (train3d.py:44-46, 316-317).
- **Por qué:** En 3D referencia cuesta ~21× (47.9 contra 2.26 µs/vóxel; en 2D era ~3×), intermedio 2.8×, y referencia 48³×2 no entró en 16 GB (medido; costo-tdv3d.txt; bitácora l.4919-4938; train3d.py:781-783). Una corrida 3D no repite bit a bit: se lee la tasa sobre semillas. La excepción es rama contra tronco desde el mismo estado (README.md:421-426). referencia queda para tests cortos con --micro-lote 1 (train3d.py:772-773).

### Precisión de la evaluación

*decisión distinta del 2D* · evidencia: **sin fuente** · por defecto: activo

- **2D:** evaluate_denoising no usa autocast: en rapido se entrena en bf16 pero se evalúa en fp32 (evaluate.py:38-63; train.py:365-366).
- **3D:** evaluate_denoising3d(..., amp=mode['amp']): en rapido la evaluación corre en bf16 (evaluate.py:66-73; train3d.py:680-682).
- **Por qué:** La fuente dice «se evalúa como se entrena» (evaluate.py:72-73; train3d.py:971). No encontré medición del efecto de bf16 sobre el PSNR de evaluación, ni una razón para apartarse del 2D.

### torch.compile y --sin-compilar

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: activo (compila); --sin-compilar apagada

- **2D:** Siempre compila el método: model.R.grad = torch.compile(model.R.grad, dynamic=False), sin bandera (train.py:216-219).
- **3D:** Lo mismo (train3d.py:475-477), salvo con --sin-compilar (376), pensada para tests en CPU.
- **Por qué:** Compilar R.grad ahorra 35 % de tiempo y 38 % de memoria a 64³: 0.619 contra 0.952 s y 8.78 contra 14.16 GB (medido, bitácora l.4911-4913, 4922). Compilar el método y no el módulo deja el state_dict sin el prefijo _orig_mod., igual que en 2D (train3d.py:475).

### --checkpointing

*consecuencia mecánica del 3D* · evidencia: **derivado** · por defecto: apagado

- **2D:** Bandera apagada; el help dice «-70 % de memoria, +30 % de tiempo» (train.py:191-192). VNet2D.forward checkpointea un paso sólo si su x lleva gradiente (ddr/model.py:98-99).
- **3D:** La misma bandera y el mismo mecanismo: VNet3D hereda forward (ddr/model.py:126-129). El help dice «~5× menos memoria (~9× con --guard-amp), +27 % de tiempo» (train3d.py:369-371), y vram_predicha suma el paso 1 sin checkpoint (126-137).
- **Por qué:** Sin --guard-amp, x_0 no lleva gradiente y el paso 1 guarda sus activaciones: 1.04 pasos medios, medido en CPU. El ~5× es derivado. El ~9× venía del borrador, que checkpointeaba los 10 pasos, y quedó retirado (bitácora l.5418-5466; fila en ## Descartado, l.1006). El +27 % se midió sobre el borrador (l.4923). Checkpointear también el paso 1 tocaría el modelo compartido con 2D, y no se hizo (l.5454-5456). Lo mismo pasa en 2D, porque el forward es el mismo; no revisé de dónde sale el «-70 %» de train.py.

### Disyuntor: razón, ventana, rachas, respaldo

*decisión distinta del 2D* · evidencia: **supuesto** · por defecto: activo

- **2D:** GUARD_RATIO = 5.0, GUARD_VENTANA = 100, GUARD_SEGUIDOS = 25, GUARD_RESPALDO = 250, GUARD_LIMPIAS = 3, con rechaza_lote, respalda y rebobina (train.py:61-66, 100-142).
- **3D:** Los mismos valores y las mismas tres funciones, importados (train3d.py:44-47, 356-366, 627-628, 645-662).
- **Por qué:** El guard relativo es invariante de escala: reescalar ‖g‖ no cambia el set de rechazos (medido, docs/mediciones/umbral-guard-escala.txt; train3d.py:928-929). GUARD_SEGUIDOS = 25 sale de la racha sana más larga de BSDS (18); en 3D «la tiene que medir el tronco» (train3d.py:934-935). La predicción 3 del pre-registro (tasa sana ≤ 1 %) es un supuesto (bitácora l.5214-5216).

### Disyuntor: cota absoluta escalada (cota_absoluta)

*decisión distinta del 2D* · evidencia: **derivado** · por defecto: activo

- **2D:** --guard-abs con default GUARD_ABS = 1e4 (train.py:62, 177-178).
- **3D:** --guard-abs con default None, que pasa a cota_absoluta(patch) = 1e4·P³/(93²·3) = 1e4·P³/25 947 (VALORES_BSDS): 4.26e4 a 48³ y 1.01e5 a 64³ (train3d.py:87, 121-123, 358-359, 387-388). Un valor explícito la reemplaza.
- **Por qué:** La pérdida es una suma por parche, así que ‖g‖ escala con los valores sumados. En 80 it con una semilla, el arranque sano 3D cruzó 1e4: 32³×4 llegó a 10 329 y 64³×1 a 15 985 (2 cruces); 48³ a 8 367 (medido; arranque-g-3d.txt; bitácora l.4945-4977). La cota queda 5.1× y 6.3× sobre esos máximos, pero es «derivada, no calibrada», y que el hueco [6 113, 1.97·10⁵] de BSDS escale igual es un supuesto (train3d.py:930-935). No encontré si debería escalar también con --sigma-max o --aug-ganancia, que cambian el tamaño de la pérdida.

### rebobina(): el arreglo 78fcf82

*decisión distinta del 2D* · evidencia: **medido** · por defecto: activo (sólo actúa al rebobinar)

- **2D:** Arreglado en train.py:130-142: opt.load_state_dict(copy.deepcopy(respaldo[1])), conservando el lr vigente. En 3D-Training entró como 78fcf82 (2026-10-08) y en main como 81b3701 (2026-10-09). Las corridas de la fase 1 (t8delta-100k-s1) corrieron sin él.
- **3D:** La misma función, importada (train3d.py:46-47, 646). Conservar el lr es redundante acá, porque lr_en lo reescribe en cada iteración (587-588). Lo cubren test_un_rebobinado_no_cambia_el_lr (tests/test_train3d.py:510) y test_el_respaldo_del_estado_es_el_del_momento_en_que_se_tomo (808).
- **Por qué:** Sin la copia, load_state_dict adopta los tensores del respaldo, y el segundo rebobinado al mismo respaldo no restaura Adam. Eso armaba el lazo de t1-guard-2delta-5k: 45 rebobinados, 41 al mismo respaldo (docs/mediciones/alias-respaldo-adam.txt). Restaurar el lr deshacía un halving cruzado. Sin rebobinados es inerte: 120 it en referencia con traza.npy idéntica bit a bit (bitácora l.5117-5134). Hoy no es una diferencia entre 2D y 3D. Nota mía: el comentario de train3d.py:429-430 («el estado de Adam y el respaldo [comparten memoria] tras un rebobinado») parece anterior al arreglo; con la copia ya no comparten (derivado de leer el código, no verificado).

### Iteración efectiva (contador descartadas, columna 12 de la traza)

*nuevo (el 2D no lo tiene)* · evidencia: **derivado** · por defecto: activo

- **2D:** No se cuenta: estado.json sólo lleva rebobinados (train.py:387-392). La efectiva se reconstruía a posteriori (docs/mediciones/rebobinados-t1.txt).
- **3D:** En cada rebobinado, descartadas += min(it − it_respaldo, it − it_rebobinado) (train3d.py:647-650). Va a la columna 12 (it − descartadas, l.664-665), a history.json ('efectiva'), a estado.json (iteracion_efectiva) y al log ('ef=').
- **Por qué:** Con disyuntor, el endpoint es el PSNR a iteración pareada, la efectiva si hay muchos rebobinados (AGENTS.md, «Lo que ya se sabe»). En T1 2D a 2·δ el rebobinado descartó el 72 % de las iteraciones (bitácora 2026-10-08, l.4578). El min evita contar dos veces lo ya descartado al mismo respaldo (train3d.py:647-649); lo cubre test_la_iteracion_efectiva_descuenta_lo_rebobinado (tests/test_train3d.py:547).

### Traza: columnas 0-9

*decisión distinta del 2D* · evidencia: **derivado** · por defecto: activo

- **2D:** 10 columnas float64: 0 loss, 1 T, 2 ∂L/∂T, 3 λ, 4 ∂L/∂λ, 5 ‖g‖ sin recortar, 6 mediana de la ventana, 7 amp (NaN sin --guard-amp), 8 rechazado, 9 rebobinado (train.py:264-266, 309-310, 346, 510-512).
- **3D:** COLUMNAS = 14; las 10 primeras son idénticas y van en el mismo orden (train3d.py:109, 532-533, 631-632, 664-665, 906-910).
- **Por qué:** Ocho scripts/ las leen por posición (train.py:264; train3d.py:907-908). Lo cubre test_la_traza_conserva_las_diez_columnas_de_train (tests/test_train3d.py:539).

### Traza: columna 10 (‖g‖ del primer micro-lote)

*nuevo (el 2D no lo tiene)* · evidencia: **derivado** · por defecto: activo

- **2D:** No existe.
- **3D:** g_micro = acumulacion·get_total_norm(.grad), leído tras el backward del micro-lote 0 (train3d.py:36, 613-617).
- **Por qué:** Junto con la columna 5 estima offline la escala de ruido del gradiente, B_simple = tr(Σ)/|G|² (McCandlish et al., 2018), para saber si el lote 32 pasa esa escala (train3d.py:912-919; bitácora l.5065-5070, 5287-5292). Leer .grad entre backwards no cambia ningún bit (test_leer_el_gradiente_entre_backwards_no_cambia_un_bit, tests/test_train3d.py:557). Con micro-lote = lote (por ejemplo --checkpointing en 48 GiB, donde auto da 32) es igual a la 5 y no estima nada: ahí va --micro-lote 16. B_simple todavía no se midió.

### Traza: columna 11 (lr)

*nuevo (el 2D no lo tiene)* · evidencia: **derivado** · por defecto: activo

- **2D:** No está en la traza: sólo aparece en el log cada --log-every, vía sched.get_last_lr() (train.py:356).
- **3D:** El lr leído de opt.param_groups[0] antes del paso (train3d.py:634-635, 664-665, 920).
- **Por qué:** «Se lee del optimizador y no del calendario: es el que de verdad usa el paso» (train3d.py:634). Hace visible el calendario de cada rama.

### Traza: columna 13 (σ media del lote)

*nuevo (el 2D no lo tiene)* · evidencia: **sin fuente** · por defecto: activo (vale 25 constante sin ruido ciego)

- **2D:** No existe: σ es único.
- **3D:** suma_sigma/lote: vale --sigma, o la media de los σ sorteados con --sigma-max (train3d.py:592, 595-601, 665, 923).
- **Por qué:** La fuente sólo la describe. Su uso con ruido ciego (la pérdida escala con σ² y el guard relativo lo va a ver) es un supuesto de la bitácora (l.5076-5079), no un porqué escrito para la columna.

### Escritura atómica de traza y .pth

*nuevo (el 2D no lo tiene)* · evidencia: **derivado** · por defecto: activo

- **2D:** np.save(traza.npy) y torch.save(last.pth/best.pth) escriben directo, sólo en cada evaluación (train.py:373-379).
- **3D:** guarda_traza y guarda_pth escriben a un temporal y después hacen replace (train3d.py:190-202). La traza se reescribe al arrancar, en cada evaluación y en cada estado (553, 690, 707).
- **Por qué:** Un corte a mitad de escritura no deja un archivo roto con el nombre bueno ni un estado del que no se pueda reanudar (train3d.py:198-199, 981-982). Lo cubre test_un_corte_al_guardar_no_deja_un_pth_roto (tests/test_train3d.py:765).

### Estado completo (estado-<it>.pth, --guardar-cada)

*nuevo (el 2D no lo tiene)* · evidencia: **derivado** · por defecto: activo

- **2D:** No existe: «train.py no reanuda» (train.py:506). last.pth guarda sólo config, model, iteration, psnr y optimizer.
- **3D:** --guardar-cada con default GUARDAR_CADA = 10 000, y siempre en la última iteración; 0 lo apaga (train3d.py:58, 372-373, 704-720). Guarda pesos, Adam, la ventana escala, el respaldo con it_respaldo, los contadores (rechazos, seguidos, max_seguidos, rebobinados, limpias, descartadas, it_rebobinado), history, best_psnr, mejor (el payload de best.pth), la traza hasta it, max_grad, noise_gen, los rng de CPU y CUDA, horas, args y config.
- **Por qué:** Para sobrevivir un corte en una máquina remota (ya hubo uno en t8delta-100k) y medir el halving con ramas desde un tronco en vez de adivinarlo (bitácora l.5061-5064, 5106-5111). 10 000 son los puntos de rama del piloto. El estado pesa ~46 MB en la 40 000 (derivado: pesos, m y v tres veces, más 4.5 MB de traza; train3d.py:777-780).

### --reanudar, ramas y LIBRES

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: apagado (sólo con --reanudar)

- **2D:** No existe.
- **3D:** --reanudar <estado>: con el mismo --out sigue la corrida, con otro --out es una rama (train3d.py:433). LIBRES = {out, reanudar, iterations, log_every, eval_every, guardar_cada, workers, micro_lote, vram_gb, sonda_vram}; una rama puede cambiar además 'hitos'. Cualquier otra bandera distinta es un error (107-108, 434-442). --iterations tiene que ser ≥ it0 (444-445). Otro micro_lote da un AVISO que nombra qué cambia (446-457).
- **Por qué:** Lo demás cambiaría la aritmética, y se rechaza en vez de ignorarse (train3d.py:865-866). Medido en CPU/referencia: 12 it seguidas contra 6+6 y 9+3 reanudadas dan traza, history, log y estado idénticos bit a bit, con ruido ciego, aumentos y rebobinados a los dos lados del corte (train3d.py:878-884; tests/test_train3d.py:594). Defecto conocido y pendiente (bloqueado por el hook de la regla 2): un estado anterior a una bandera nueva no se puede reanudar, porque estado['args'].get('mascara_cerebro') = None ≠ False (train3d.py:436-437; bitácora l.5681-5683, «bajo»). Las banderas de W&B y QSM que falta integrar tienen que ir en LIBRES (bitácora l.5750).

### Offset de datos y estado de los generadores

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: activo

- **2D:** El parche i sale de rng((seed·100, i)) y length = iterations·32 (data.py:134; train.py:222-225). El ruido sale de un generador CPU con semilla seed·100+1 (train.py:243). No se guarda ningún estado de RNG.
- **3D:** El parche i sale de rng((seed·100, offset + i)), con offset = it0·lote y length = (iterations − it0)·lote (data3d.py:246; train3d.py:505-507). Los estados de noise_gen, torch CPU y CUDA viajan en el estado; los de torch se restauran después de iter(loader) (train3d.py:536, 573-580, 716-718).
- **Por qué:** Reanudar en it0 es empezar en el parche it0·lote, y el DataLoader no guarda nada (train3d.py:484-485, 862-864). iter(loader) saca una semilla del generador global y la corrida de origen lo hizo una sola vez (574-575). Lo cubre el test de reanudación bit a bit.

### Directorio de salida al arrancar (aparta)

*nuevo (el 2D no lo tiene)* · evidencia: **derivado** · por defecto: activo

- **2D:** Escribe config.json al arrancar y no limpia nada; estado.json se escribe sólo al terminar (train.py:246-247, 387-392).
- **3D:** Al arrancar borra estado.json y reescribe config.json (con el micro-lote efectivo), history.json y traza.npy hasta it0. aparta() renombra a <nombre>-descartada-<it>.pth (o -ilegible, -2, -3…) los last.pth/best.pth que no son del estado, y best.pth se repone desde el estado (train3d.py:205-226, 546-557, 888-902).
- **Por qué:** Seguir desde un estado anterior no puede dejar un estado.json «terminada» de la corrida corta, ni un best.pth de iteraciones que no están en la traza (train3d.py:898-902). Nada apartado se borra ni se pisa. Tests en tests/test_train3d.py:695-765.

### estado.json, history.json y log

*nuevo (el 2D no lo tiene)* · evidencia: **derivado** · por defecto: activo

- **2D:** estado.json: terminada, iteraciones_pedidas, iteraciones_hechas, mejor_psnr, psnr_final, rechazos, max_seguidos, rebobinados, horas, vram_pico_gb (train.py:387-392). history: iteration, psnr, T, lambda (369-370). Log: loss, lr, T, λ, |g|, max|g|, corr, rech, reb, horas restantes (355-359).
- **3D:** estado.json agrega iteracion_efectiva, reanudada_de y vram_predicha_gb, y horas acumula los tramos previos (train3d.py:722-730). history agrega efectiva y psnr_tejido (685-687). El log agrega 'ef=' y toma el lr del optimizador; el ETA se calcula sobre it − it0 (671-675). El arranque imprime dispositivo, GiB, presupuesto, GiB útiles, micro-lote y pico predicho (415-419) y, con máscara, cuántos volúmenes tienen cubos y la mediana de esquinas (510-514). El cierre imprime el pico predicho y la iteración efectiva (731-734).
- **Por qué:** Lo piden la reanudación (horas, reanudada_de), el modelo de memoria (predicho junto al medido) y la iteración efectiva. No hay medición propia.

### Defaults de log y evaluación

*decisión distinta del 2D* · evidencia: **medido** · por defecto: activo

- **2D:** --log-every 250 y --eval-every 2000 (train.py:153-154).
- **3D:** Los mismos: --log-every 250 y --eval-every 2000 (train3d.py:323-324). Los comandos del piloto pasan --eval-every 1000 (README.md:406; bitácora l.5255, 5262-5263).
- **Por qué:** La frecuencia no toca la aritmética y es LIBRE al reanudar. El piloto usa 1000 para tener el par exacto en k+5 000 (bitácora l.5262-5263). Una evaluación 3D tardó 23 s, y la primera 59 s porque recompila para 128³; se midió una sola vez, con la GPU compartida (bitácora l.5233-5236).

### Conjunto y forma de la evaluación

*decisión distinta del 2D* · evidencia: **medido** · por defecto: activo

- **2D:** CBSD68: 68 imágenes enteras en color, ruido con semilla 1234+i y sigma_ref = sigma, así que el reescalado no actúa (train.py:69-72, 229-230, 365-366; evaluate.py:38-63).
- **3D:** 12 volúmenes de lista_eval.txt (uno por centroide de un k-means), cada uno con un recorte fijo de --eval-recorte = EVAL_RECORTE = 128³ centrado en la caja del tejido. PSNR en todo el recorte y en tejido (limpio > 0.05), sin sigma_ref. best.pth se elige por el PSNR de todo el recorte (train3d.py:93, 380, 519-520, 680-696; data3d.py:48; evaluate.py:66-101).
- **Por qué:** 128 entra en la caja del tejido de los 200 volúmenes medidos (mínimo por eje 144, 195, 144; data3d.py:48). Los 12 representan los modos, no los extremos (bitácora l.5023-5024). El volumen entero no se evalúa, y falta un PSNR dentro del cerebro (hallazgo pendiente, bitácora l.5681-5682). --eval-recorte existe para los tests en CPU (train3d.py:378-380).

### Datos de entrenamiento

*decisión distinta del 2D* · evidencia: **medido** · por defecto: activo

- **2D:** BSDS400 color: datasets/bsds500, splits train+test, 400 imágenes en RAM; banderas --data-root, --train-splits, --eval-split, --eval-list (train.py:69-72, 155-160).
- **3D:** LDM_100k, T1 sintética con cráneo, en datasets/ldm100k3d. --train-list con default lista_train_2000.txt (2 000 volúmenes en float16 en RAM, ~19 GB; 8 000 no entran) y --eval-list lista_eval.txt; no hay --eval-split (train3d.py:90-93, 325-327, 486-487; data3d.py:199-201, 317).
- **Por qué:** 2 000 entran en RAM. La selección «mezcla» (mitad k-center, mitad sorteo) de 2 000 cubre las colas mejor que un sorteo de 8 000: máximo 5.987 contra 7.921, en el PCA de las miniaturas (medido, bitácora l.5009-5024). La cobertura es de cerebros, no de parches (l.5028-5030). La RAM pico del humo fue 26.8 GB (medido).

### Muestreo de parches (orientaciones y filtro de tejido)

*decisión distinta del 2D* · evidencia: **medido** · por defecto: activo

- **2D:** Flips y rot90: 8 orientaciones, sin filtro de contenido (data.py:134-151).
- **3D:** Las 48 simetrías del cubo (orienta). Filtro: al menos MIN_TEJIDO = 35 % de vóxeles > UMBRAL_TEJIDO = 0.05, con hasta INTENTOS = 12 sorteos; si ninguno pasa queda el último (data3d.py:45-47, 246-262).
- **Por qué:** Las 48 extienden flips y rotaciones; valen porque LDM es isotrópica a 1 mm y dejan de valer con el dipolo (bitácora l.5043-5046). El 35 % es el --min-cerebro de los cortes 2D de T1 y rechaza el 6.5 % de los parches de 48³ (medido, tejido-ldm100k3d.txt). P(fallan los 12) = 3e-11 en el peor volumen. Con este filtro sólo el 4.0 % de los parches cae entero en el cerebro (medido, bitácora l.5513-5516).

### Workers del DataLoader

*decisión distinta del 2D* · evidencia: **supuesto** · por defecto: activo

- **2D:** num_workers=16, persistent_workers y prefetch_factor=4, fijos (train.py:226-228).
- **3D:** --workers con default WORKERS = 2 (era 1 hasta el 2026-10-09). persistent_workers y prefetch_factor=4 sólo si es > 0 (train3d.py:100, 377, 515-518).
- **Por qué:** Un parche cuesta 0.31 ms con la CPU libre y 0.81 ms con la CPU cargada (medido, tejido-ldm100k3d.txt). Los 32 de una iteración suman 26 ms contra ~8 s de GPU: uno alcanzaría, y dos son por si la máquina de destino tiene otra CPU, sin medir (train3d.py:96-100; bitácora l.5327-5329). El «cada una suma 16 workers» de AGENTS.md:36 vale sólo para train.py.

### Canales, parámetros y escala de la pérdida

*consecuencia mecánica del 3D* · evidencia: **derivado** · por defecto: activo

- **2D:** CHANNELS = 3, con --channels (train.py:73, 161-162). TDV33 color: 387 970 parámetros (387 394 en gris). La pérdida suma 93²·3 = 25 947 valores por parche.
- **3D:** CHANNELS = 1 fijo, sin bandera (train3d.py:94). vnet_config3d es vnet_config en gris con R.type = 'tdv3d' (112-118). R tiene 1 162 112 parámetros (tests/test_train3d.py:245), 3.0× el 2D, o sea 27/9. La pérdida suma 48³ = 110 592 valores por parche.
- **Por qué:** T1 es gris y los kernels pasan de 3² a 3³ (bitácora l.4898-4902). Derivado por mí: el nivel de «no hacer nada» de la loss sube a (25/255)²·110 592 ≈ 1 063, contra 83.1 en T1 2D y 249.4 en BSDS color.

### --features

*nuevo (el 2D no lo tiene)* · evidencia: **derivado** · por defecto: no aplica (default = receta)

- **2D:** No existe: FEATURES = 32 fijo (train.py:43).
- **3D:** --features con default FEATURES = 32 (train3d.py:378-379), sólo para achicar la red en tests de CPU. test_los_defaults_son_la_receta lo fija en 32 (tests/test_train3d.py:228-245).
- **Por qué:** «Sólo para los tests en CPU: con los defaults la red y la evaluación son las de la receta» (train3d.py:378; vnet_config3d, l.113-114).

### --guard-amp en 3D

*consecuencia mecánica del 3D* · evidencia: **derivado** · por defecto: apagado

- **2D:** Default inf (apagada). pS se normaliza con MICRO_BATCH·ACUMULACION² (train.py:189-190, 296-299).
- **3D:** Default inf. pS se normaliza con args.micro_lote·acumulacion² (train3d.py:367-368, 618-621). Con la bandera, x_0 lleva gradiente y los 10 pasos se checkpointean (~9× de memoria con --checkpointing); vram_predicha y la sonda distinguen el caso. amp suma normas micro-lote por micro-lote, así que depende de la partición y cambia si se reanuda con otro micro-lote (train3d.py:875-877).
- **Por qué:** Consecuencia del micro-lote variable y del forward compartido (ddr/model.py:98). Que dependa de la partición está escrito en train3d.py:875-877 y en la bitácora (l.5347-5349).

### Ruido ciego (--sigma-max, --sigma-min)

*nuevo (el 2D no lo tiene)* · evidencia: **supuesto** · por defecto: apagado

- **2D:** Sólo --sigma fijo (default 25) con add_noise (train.py:151, 280; data.py:94-98).
- **3D:** --sigma-max con default 0.0 (apagado). Si es > 0, sortea σ ~ U(--sigma-min, --sigma-max) por parche con add_noise_ciego, del mismo generador: primero los σ y después el ruido. --sigma-min tiene default SIGMA_MIN = 0.0 (train3d.py:95, 345-347, 595-601; data3d.py:154-171). La evaluación sigue a --sigma fijo.
- **Por qué:** Es la candidata de más valor para QSM: elimina el reescalado de evaluate.py --sigma-ref y el alpha del ADMM (AGENTS.md, «Mejoras candidatas»). U(0, 55) es la convención citada de DnCNN-B. El efecto es un supuesto sin medir: con un solo T, la PSNR a σ=25 baja; y como la pérdida escala con σ², la dispersión de ‖g‖ sube (bitácora l.5076-5079). Con ruido ciego, otro micro-lote al reanudar cambia el ruido de cada parche (train3d.py:874-875).

### --aug-signo

*nuevo (el 2D no lo tiene)* · evidencia: **supuesto** · por defecto: apagado

- **2D:** No existe.
- **3D:** Bandera store_true: multiplica el parche por −1 con p = 1/2, después de elegir parche y orientación (train3d.py:348-349; data3d.py:269-270).
- **Por qué:** QSM tiene susceptibilidades de los dos signos y T1 es positiva. Derivado: con padding simétrico y K1 de media cero, la energía ya es invariante a un offset, y faltan el signo y la escala (bitácora l.5080-5083). También imitaría la polaridad de T2 (l.5568-5570). Se aplica al final para que brazo y control vean los mismos parches (train3d.py:942-943). Que sirva no está medido.

### --aug-ganancia

*nuevo (el 2D no lo tiene)* · evidencia: **supuesto** · por defecto: apagado

- **2D:** No existe.
- **3D:** --aug-ganancia con default 1.0 (apagada). Con g > 1 multiplica el parche por exp(U(−ln g, ln g)) (train3d.py:350-351; data3d.py:271-273).
- **Por qué:** Está medido que la red no es escala-equivariante (15–28 %, docs/mediciones/escala-equivariancia.txt). Que entrenar con ganancias vuelva menos crítico el alpha es un supuesto (bitácora l.5084-5085).

### --mascara-cerebro

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: apagado

- **2D:** No existe.
- **3D:** Bandera store_true. Lee <data-root>/mascaras/train/<sujeto>.npy (deepbet ∩ HD-BET sin TTA), y cada parche pasa a ser un cubo entero dentro del cerebro, a MARGEN_CEREBRO = 1 vóxel o más del borde. Si falta alguna máscara se niega antes de tocar --out y da los 3 comandos. Es receta: no está en LIBRES (train3d.py:352-354, 488-514, 948-963; data3d.py:53).
- **Por qué:** Sin la máscara, sólo el 4.0 % [2.8–6.2] de los parches de 48³ cae entero en el cerebro y el parche medio es 60 % cerebro (medido, 24 sujetos). El precio: a 48³ los cubos cubren el 66 % del cerebro y el 18.5 % de la franja cortical (≤ 6 mm); a 32³, el 80 % y el 38 % (medido; bitácora l.5501-5553). Las máscaras de entrenamiento todavía no existen: ~22 min de GPU para 2 000 (derivado; l.5675-5676). Prenderla lo decide el autor.

### Evaluación QSM y seguimiento W&B (no integrados)

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: no aplica (no integrado)

- **2D:** No existen.
- **3D:** ddr/qsm3d.py (ADMM wTDV con μ=0.008, μ₂=1.0, target_std=0.06 y hasta 100 iteraciones) y ddr/seguimiento.py (W&B en modo auto, offline o no) están en 897fb8b, pero train3d.py no los llama.
- **Por qué:** Lo pidió el autor. La integración está bloqueada por el hook de la regla 2 mientras corren lr-corto-A-40k y lr-corto-C-35k, que son 2D. El módulo QSM reproduce 26.8898 % contra el 26.89 % publicado (medido, CPU fp32). TARGET_STD se calibró para la red 2D color y no se midió en 3D (bitácora l.5691-5750).


## Lo que faltaba (agregado por el verificador)

### Qué lr imprime el log en un hito

*decisión distinta del 2D* · evidencia: **derivado** · por defecto: activo

- **2D:** El log imprime sched.get_last_lr()[0] DESPUÉS de sched.step() (train.py:316, 356). En la it 25 000, que es múltiplo de --log-every 250, muestra 2.00e-04 aunque ese paso usó 4e-4: el log va una iteración adelante. El lr no está en la traza.
- **3D:** lr = opt.param_groups[0]['lr'] se lee ANTES del paso (train3d.py:635) y va al log (672) y a la columna 11. En la 25 000 muestra 4.00e-04, y 2e-4 desde la 25 001.
- **Por qué:** «Se lee del optimizador y no del calendario: es el que de verdad usa el paso» (train3d.py:634). Al comparar logs 2D y 3D en un hito, la línea del hito difiere a propósito. La diferencia del log 2D la derivé leyendo el código; no está escrita en ningún lado.

### Piloto 3D pre-registrado (contenido) y enmienda por el lote 32

*nuevo (el 2D no lo tiene)* · evidencia: **supuesto** · por defecto: apagado (no lanzado; lo lanza el autor)

- **2D:** En 2D el calendario se midió con corridas de 10⁵ en BSDS (bitácora l.4332), y las tandas se lanzan con plan.sh.
- **3D:** Borrador sin lanzar (bitácora l.5179-5302). Diseño: un tronco de 40 000 a lr constante (--hitos, --eval-every 1000) y tres ramas de 5 000 con lr/2 desde k = 10 000, 20 000 y 30 000. Endpoint: rama − tronco a k+5 000, pareado, sobre los 12 recortes (todo el recorte y tejido), a iteración efectiva si hay rebobinados. Predicciones: (1) en la 14 000 el tronco ya cobró ≥75 % de la ganancia; (2) la rama queda entre +0.01 y +0.06 dB [banda corregida por `main` a +0.007 a +0.018: «Corrección de main a la predicción 2»]; (3) rechazos ≤1 %; (4) todo ‖g‖ > 4.26e4 cae en una racha que termina en rebobinado; (5) T ≥ 2·δ = 0.0253 [umbral de la regla 2-δ, retirada]. Regla original sobre B_simple: mediana ≥16 → lote ≥16; <8 → lote 8 (l.5224-5226). La ENMIENDA del 2026-10-09 (l.5265-5292) fija el lote en 32 por decisión del autor: B_simple deja de decidir el lote (se lee si ≥32), las predicciones 1 y 2 quedan como supuestos más débiles, y el costo derivado en la 5070 Ti sube a 89 h el tronco y 33 h las ramas. Antes de lanzar: --sonda-vram y 200 it para medir s/it.
- **Por qué:** Medir con ramas el codo del lr en 3D, en vez de adivinarlo. Hay una sola semilla, y la regla 5 lo admite porque la pregunta principal está escrita antes (l.5228-5230). Todas las predicciones son supuestos.

### Corrección de main a la predicción 2 del piloto, que la rama no tiene

*decisión distinta del 2D* · evidencia: **derivado** · por defecto: no aplica

- **2D:** Main, 2026-10-12 (l.6150-6200): descontando la pendiente que la corrida ya traía, el primer halving en BSDS vale +0.0065 a +0.0182 dB sobre las 4 corridas de 10⁵.
- **3D:** El pre-registro de 3D-Training sigue con «+0.01 a +0.06 dB», calibrado con los incrementos crudos +0.030/+0.033 (bitácora l.5211-5213). Según main, la banda honesta es +0.007 a +0.018, y el techo de +0.06 está ~3× alto.
- **Por qué:** Main corrigió esa predicción sin tocar la rama: «el 3D vive en su rama y no se integra sin pedirlo» (main l.6190-6195). Si se lanza el piloto tal como está escrito, la predicción 2 usa una banda ya refutada en 2D.

### δ_Adam vivo y exposición de T en el tronco a lr constante

*decisión distinta del 2D* · evidencia: **derivado** · por defecto: activo (el tronco del piloto usa --hitos)

- **2D:** Main (2026-10-11): δ_Adam se parte a la mitad en cada hito, así que la reserva T/δ_vivo se duplica sola. El 91–96 % de los rechazos cae en la primera fase de lr (main bitácora l.21 y 42).
- **3D:** lr_en (train3d.py:176-183) aplica el mismo mecanismo. El tronco del piloto corre con --hitos sin valores, o sea lr 4e-4 constante: δ_vivo queda en 0.01265 las 40 000 it, y toda la corrida está en la fase que en 2D concentró los rechazos. Con el default sin --hitos, esa fase dura 25 000 it.
- **Por qué:** Derivado mío, juntando el hallazgo de main con los comandos del piloto (README.md:406; bitácora l.5255). No está escrito en la rama. Es relevante para las predicciones 3 a 5 del pre-registro.

### Cambios a archivos que comparte el 2D

*decisión distinta del 2D* · evidencia: **medido** · por defecto: activo

- **2D:** Base e88512d.
- **3D:** train.py: rebobina copia el respaldo y conserva el lr (+6/−2 líneas; hoy byte-idéntico a main). ddr/model.py: atributo de clase TDV y VNet3D. evaluate.py: suma evaluate_denoising3d e importa UMBRAL_TEJIDO de ddr.data3d (evaluate.py:22), así que el evaluate 2D, que usa train.py, ahora carga ddr/data3d.py. ddr/__init__.py: exporta lo 3D. tests/test_disyuntor.py: +2 tests del arreglo de rebobina. AGENTS.md: el patrón de pgrep pasa a train(3d)?\.py y el hook también cubre train3d.py. docs/resultados/t1-guard-2delta-5k.md: 2.14× pasa a 2.16×. pyproject: wandb>=0.30.0. .gitignore: versiona las listas 3D.
- **Por qué:** Que el state_dict 2D no cambie está medido: huella SHA-256 de los 5 checkpoints con strict=True, antes y después (bitácora l.5146-5148). La dependencia nueva de evaluate.py hacia data3d no la registra ninguna fuente: es inerte, pero acopla el 2D al módulo 3D.

### Cota absoluta: sobre qué se midió y margen a 32³

*decisión distinta del 2D* · evidencia: **derivado** · por defecto: activo

- **2D:** Main no recalibra GUARD_ABS = 1e4 por dominio: en T1-2D en gris (8 649 valores por parche) quedó en 1e4, a 5.0× sobre el mayor ‖g‖ aceptado (6.0× en BSDS) (AGENTS.md de main; cotas-disyuntor-dominio.txt). La fórmula 3D, aplicada a T1-2D, daría 3 333.
- **3D:** cota_absoluta usa BSDS color como referencia. Los máximos que la justifican salieron de parches con ≥50 % de tejido (bitácora l.4949), pero el entrenamiento filtra al 35 %. A --patch 32, que la bitácora propone como alternativa, cota_absoluta(32) = 12 629 contra un máximo sano de 10 329 a 32³×4: margen de 1.22×, no los 5.1× de 48³. No se reescala ni con --mascara-cerebro ni con --sigma-max.
- **Por qué:** Las dos cuentas (12 629 y 1.22×) las calculé yo en CPU con cota_absoluta. Que el filtro de la medición fuera otro está escrito en la bitácora (l.4949), pero no en el porqué de train3d.py:927-935.

### Decisiones de datos propuestas sin decidir (Cubos 100 % cerebro)

*decisión distinta del 2D* · evidencia: **medido** · por defecto: apagado

- **2D:** T1-2D: 8 000 sujetos, 24 000 cortes en float32, 3.1 GB, sorteo uniforme.
- **3D:** Lo propuesto en la bitácora l.5572-5578: (1) máscara deepbet ∩ HD-BET sin TTA con margen 1: implementada como bandera apagada (0eb0074), pero faltan las 2 000 de entrenamiento, ~22 min de GPU. (2) Cubos de 48³ «para no cambiar dos cosas a la vez». 32³ cubre el 80.1 % del cerebro y el 38.1 % de la franja cortical (48³: 66.3 % y 18.5 %), cuesta +23 % por vóxel (2.77 contra 2.26 µs) y deja la escala más gruesa en 8³; desde 56³ se pierde ≥37 % del cerebro. (3) 20 000 cerebros en uint8 elegidos con la regla mezcla (3.1 MB por cerebro recortado a su máscara, error RMS 0.0011 = 1.15 % de σ, 62 GB de disco), o los 100 000 leyendo de disco. Hoy son 2 000 en float16 en RAM. (4) Dónde correr las máscaras. La rama 3D-Cerebro sigue existiendo, y se recomendó integrarla como bandera (l.5582-5585).
- **Por qué:** El autor pausó el 3D para seguir con pruebas en 2D (l.5504-5505). Las coberturas, tiempos y tamaños están medidos con n = 24; el costo de las máscaras es derivado. La elección es del autor.

### Muestreo en cruz de 7 cubos (descartado)

*decisión distinta del 2D* · evidencia: **medido** · por defecto: no aplica

- **2D:** No existe.
- **3D:** No se implementó. A 48³ ninguno de los 24 sujetos tiene una cruz entera dentro del cerebro; a 32³, 5 de 24 (mediana 0); a 24³, 24 de 24. Lo impide el cubo inferior, que cae en la base del cráneo. La correlación intraclase de la intensidad media es 0.397 dentro de una cruz y 0.387 entre cubos sueltos del mismo cerebro: lo que se repite es el cerebro, no la contigüidad (bitácora l.5628-5648; docs/mediciones/cruz-siete-cubos.txt).
- **Por qué:** La red procesa cada cubo por separado, con espejo en las caras, así que la adyacencia no le da contexto. Que 32 cubos en cruces valgan como 7–24 independientes es derivado (efecto de diseño de Kish).

### Candidatas y etapas no implementadas en 3D

*nuevo (el 2D no lo tiene)* · evidencia: **supuesto** · por defecto: no aplica (no implementado)

- **2D:** AGENTS.md, «Mejoras candidatas»: EMA, channels_last, AdamW con T y λ fuera del weight decay, otros calendarios.
- **3D:** Ninguna está en train3d.py. La bitácora l.5086-5092 las da como no implementadas: EMA (su estado tendría que viajar con el respaldo), evaluar el DENOISING sobre los χ de COSMOS y Sim2 como chequeo de transferencia (qsm3d evalúa la reconstrucción ADMM, no eso) y la etapa con el término de datos del dipolo. También quedan brechas de dominio sin medir (l.5619-5620): el borde de la máscara en χ, el signo de χ, el vóxel anisotrópico, y que QSM-TDV corre en float16 (use_amp) con nuestros pesos.
- **Por qué:** Una variable por vez (regla 4). Para el dipolo, la cita es Milovic et al. (ISMRM 2021) y lo demás es derivado. Nada de esto está medido.

### Errores antes de tocar --out, y main(argv)

*nuevo (el 2D no lo tiene)* · evidencia: **medido** · por defecto: activo

- **2D:** main() sin argv (train.py:145). Crea --out y escribe config.json (train.py:246-247) sin validar nada más que argparse.
- **3D:** opciones() y main(argv) separados (train3d.py:313, 384), para llamarlo desde los tests. Todo p.error y toda carga de datos van antes de out.mkdir (train3d.py:549): micro-lote imposible, lote que no divide, --reanudar con banderas distintas o --iterations < it0, máscaras faltantes y sonda sin GPU. Los tests lo exigen (tests/test_train3d.py:382-404; bitácora l.5488-5491).
- **Por qué:** Una invocación mal armada no puede ensuciar el directorio de una corrida ni de un linaje (bloque final, DIRECTORIO DE SALIDA, train3d.py:888-902).

### Reanudación: herencia de la rama y RNG de CUDA

*nuevo (el 2D no lo tiene)* · evidencia: **derivado** · por defecto: apagado (sólo con --reanudar)

- **2D:** No reanuda (train.py:506).
- **3D:** Una rama hereda history, best_psnr y best.pth del tronco (train3d.py:537, 900-901): su best.pth puede ser del tronco, de antes de k. El RNG de CUDA se restaura sólo si torch.cuda.device_count() coincide con el del estado (train3d.py:578). Si no coincide, se saltea sin aviso, y el bucle no usa el generador global (train3d.py:863).
- **Por qué:** Lo de la herencia está escrito en train3d.py:900-901. Que el salteo del RNG de CUDA sea silencioso lo derivé leyendo el código; la bitácora no lo menciona.

### La evaluación no mide lo que entrena --mascara-cerebro

*decisión distinta del 2D* · evidencia: **derivado** · por defecto: activo (con --mascara-cerebro)

- **2D:** No aplica.
- **3D:** Con la bandera se entrena sólo con cubos 100 % cerebro, pero la evaluación sigue sobre recortes de 128³ que incluyen cráneo y aire, y «tejido» (> 0.05) también incluye el cráneo (train3d.py:962-963; evaluate.py:84-93). train3d.py dice que así «una corrida con la bandera y una sin ella se miden sobre lo mismo».
- **Por qué:** Que medir sobre lo mismo favorezca a la corrida sin máscara es derivado mío. La revisión dejó pendiente un PSNR dentro del cerebro, con severidad media (bitácora l.5681-5682). Hay máscaras de los 12 de evaluación en mascaras/val (l.5673-5675), pero la evaluación no las lee.
