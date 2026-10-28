## 2026-10-28 — Líneas base del abstract con el protocolo de los desafíos

Texto para integrar a `docs/bitacora.md` (lo escribió la sesión del abstract en el worktree
`abstract-ismrm`, que no toca la bitácora del checkout principal). Scripts en
`herramientas_abstract/`; salidas en `runs/abstract-ismrm/`.

**Pregunta** (del autor, para el abstract ISMRM): con el mismo ADMM lineal, término de datos, pesos,
máscaras de los desafíos (COSMOS `msk.mat`, Sim2 `MaskBrainExtracted` de Sim2Snr1) y 100
iteraciones, ¿cuánto mejor que TV y que la red 2D de TDV-QSM (Milovic et al., ISMRM 2021)
reconstruye la 14 880? Predicción del autor: una mejora importante sobre la 2D.

**Protocolo** (`sujetos.py`): el peso de Sim2 es √W² de `sim2.npz` en la máscara del desafío,
renormalizado a media 1 en ella y al cuadrado (igual que DIRECTOR, 28-10). NRMSE y XSIM en t = 100,
con la máscara entera y con la erosionada 4 vóxeles (6-conexa); RMSE por capa (`grid_tdv2d.py`,
funciones copiadas de `scripts/grid_msk_27_10.py`). Cada método, a sus mejores parámetros por
sujeto: el menor NRMSE en t = 100 de su grid.

**TV** (`tv_fansi.py`, CPU): el `wTV.m` lineal (split z = ∇χ con diferencias hacia adelante
periódicas; z2 = D∗χ con el mismo peso; μ₂ = 1), con umbral suave por componente (anisótropo) o, con
`ISO=1`, sobre la norma del gradiente (isótropo). Grid anisótropo α ∈ {1e-5, 3e-5, 1e-4, 3e-4, 1e-3,
3e-3, 1e-2} × k = μ₁/α ∈ {10, 30, 100, 300}, refinado en {5e-5, 1e-4, 2e-4} × {300, 1000, 3000};
isótropo {3e-5, 5e-5, 1e-4, 2e-4} × {300, 1000} (Sim2, más {1.4e-4, 3e-4, 4e-4} × 300). 300
iteraciones, leído en t = 100.
- COSMOS: isótropo α 1e-4, k 300, **28.64 % / 0.814** (sin el borde 26.90 / 0.821; update 0.18 %);
  el anisótropo, α 5e-5, k 1000, 29.30 / 0.810 (27.56 / 0.815).
- Sim2: anisótropo α 1e-4, k 300, **35.36 % / 0.816** (33.40 / 0.827; update 1.00 %); el
  isótropo, α 2e-4, k 300, 35.46 / 0.801 (33.16 / 0.817).
- La primera corrida de Sim2 con `mask_final` se descartó (`*-mask_final-descartado.*`).

**La red 2D oficial** (`grid_tdv2d.py`, GPU 1): `tdv3-3-25-f32-color.pth`, tríos axiales
consecutivos como RGB, cada corte promediado sobre los tríos que lo contienen (`process_axis` de
`../tdv/ddr/utils.py`), dentro de `qsm3d.admm_wtdv` (k = 1).
- **El puerto reproduce `../tdv`** (máscara del autor, target 0.06, μ 0.008): 2D 26.5744 % en 25
  iteraciones (`../tdv/README.md`: 26.57 %), pseudo-3D 25.9392 % en 100 (25.94 %). Las 4 primeras
  iteraciones de `grid_tdv2d.admm` coinciden con `admm_wtdv` a 0 pp.
- Diagonal k fijo por (0.06, 0.008), s ∈ {0.71, 1, 1.41}:
  - COSMOS (0.06, 0.008): 26.23 % / 0.8466 (sin el borde 24.59 / 0.8522; update 0.31 %);
    (0.0426, 0.00568): **26.17 % / 0.8452** (24.51 / 0.8516; update 2.93 %); (0.0846, 0.01128):
    26.51 / 0.8456 (24.87 / 0.8509; update 0.41 %). El mínimo caía en el borde de abajo: se agregó
    (0.03, 0.004) (`SUFIJO=-ext`, `28-10-grid-tdv2d-COSMOS-ext.json`): 26.27 / 0.8438 (24.60 /
    0.8504; update 6.29 %). **El mejor de la 2D en COSMOS es (0.0426, 0.00568)**, encerrado.
  - Sim2, en la diagonal (k fijo): 0.06/0.008 → 42.45 % / 0.7638 (40.08 / 0.7748); 0.0426/0.00568
    → 40.85 / 0.7760 (38.37 / 0.7878); 0.03/0.004 → 39.56 / 0.7848 (37.09 / 0.7967). Con μ ×2,
    0.0426/0.01136 → **36.73 / 0.8050** (34.07 / 0.8207; update 0.20 %): la 2D de Sim2 quiere un μ
    más alto que la diagonal de COSMOS. Se pararon a mano, sin terminar, 0.0846/0.01128,
    0.12/0.016, 0.0213/0.00284, 0.0426/0.00284 y 0.015/0.002 (la búsqueda cambió de dirección), y a
    las 14:58, por decisión del autor (no más corridas para el abstract), 0.0426/0.0227, 0.03/0.008
    y 0.06/0.016. **El mejor medido de la 2D en Sim2 es (0.0426, 0.01136), sin encerrar**: μ más
    alto quedó sin probar, y el abstract lo dice («did not bracket its Sim2 optimum»).
- **Veredicto, Sim2:** con lo medido, la 3D (34.79 / 0.819) tiene menos error que TV (35.36 /
  0.816) y que la 2D (36.73 / 0.805); contra la 2D la diferencia puede achicarse con más búsqueda.
  Por capa (RMSE de la 3D sobre el de la 2D a sus mejores puntos medidos, `fig_profundidad.py` con
  `espec/fig5v3.json`): COSMOS −2.0 a +0.5 % en todas las capas; Sim2 −8.2 a −3.9 %, la 3D con
  menos error en todas, también en el borde. Sin el borde (erosión 4, `fig_cuantitativa.py`): COSMOS
  2D 24.5 / 0.852 contra 3D 24.5 / 0.848; Sim2 2D 34.1 / 0.821 contra 3D 32.5 / 0.831.

**La 14 880 contra las líneas base** (mismos números de `runs/grid-mu-alfa/28-10-abstract.json`):
COSMOS 26.09 / 0.843 (24.51 / 0.848; update 6.68 %), Sim2 34.79 / 0.819 (32.50 / 0.831; update
1.17 %). Grids: 17 puntos en COSMOS, 18 en Sim2.
- **Veredicto, COSMOS: la 3D empata con la 2D** (−0.14 pp de NRMSE, −0.004 de XSIM) y le gana a TV
  (−2.55 pp, +0.029). La predicción del autor no se cumple en COSMOS. Los −2 pp contra el 28.12 %
  publicado de TDV-QSM son del protocolo: la misma red 2D da 26.23 % con el nuestro.
- **En t = 100 el ADMM de la 3D oscila entre dos estados** en COSMOS (NRMSE 26.00 y 26.08 % en
  iteraciones alternas desde la ~40); de ahí su update de 6.7 %. La 2D y TV no oscilan.
- **Por capa** (RMSE de la 3D sobre el de cada base, en t = 100): contra TV, 0.90–0.92 en todas
  las capas de COSMOS; en Sim2, 1.005–1.032 en las capas 0–5, 0.927–0.995 de la 6 a la 11 y 0.971
  en el interior. Contra la 2D, 0.984–1.004 en todas las capas de COSMOS: no hay borde peor ni
  interior mejor.
- **La 2D no deja error de corte a corte** (`anisotropia_error.py`, COSMOS, máscara erosionada 4):
  la energía de la diferencia del error entre cortes axiales vecinos, sobre la de los otros dos
  ejes, da 0.932 en la 2D, 0.936 en la 3D y 0.952 en TV. La 3D deja ~7 % menos energía de alta
  frecuencia en el error que la 2D en los tres ejes, con el mismo NRMSE.

**¿Crece la ventaja de la 3D al sacar el borde?** (`erosion_ventaja.py`, NRMSE en la máscara
erosionada e vóxeles, t = 100, cada método a su mejor punto; ventaja = base − 3D, en puntos):
- Sim2 contra TV lineal: +0.56 (e 0), +0.72 (2), +0.90 (4), +0.99 (6), +1.04 (8); la 3D no lineal a
  300 iteraciones (DIRECTOR), +2.75, +3.13, +3.47, +3.71, +3.88. Contra TV lineal sí crece.
- Sim2 contra TV no lineal a 300 (su mejor punto, 4e-4 / 15): −0.05 (e 0) y −0.16 (e 4). No crece:
  TV no lineal baja tanto como la 3D al sacar el borde.
- Sim2 contra la 2D: +1.94 → +1.39 (lineal, se achica); no lineal, +4.12 → +4.23 (casi igual).
- COSMOS contra TV lineal: +2.55 → +2.25 (se achica un poco); contra la 2D, +0.07 → −0.06. No crece.
Con TV no lineal no hay ventaja que crezca: el abstract sólo muestra el χ de la 3D sin el borde
(Fig. 4) y da el número de TV al lado.

**Rayas de la calcificación de Sim2** (DIRECTOR, `compute_calcification_metrics_v2` con la
segmentación oficial, etiqueta 16, y `MaskBrainExtracted`; streaking normalizado / sin normalizar):
TV 0.072 / 0.0288, 2D a (0.06, 0.008) 0.103 / 0.0375, 2D a (0.0426, 0.01136) 0.085 / 0.0305, 3D
0.098 / 0.0301. TV deja menos rayas que
la 3D, aunque en el corte sagital de los insertos la 3D muestra una sola raya y TV una X: un corte
no alcanza para decir que la 3D raya menos.

**TV no lineal** (`tv_fansi.py` con `NL=1 ISO=1`: el z2 de `qsm3d._z2_nolineal`, fase en radianes y
peso con máximo 1, como `admm_wtdv(datos_modelo="nolineal")`; 300 iteraciones, leído en t = 300;
28-10 16:10–16:25, CPU 16–31; `28-10-tv-fansi-{COSMOS,Sim2}-fino-iso-nl.json`), 9 puntos por sujeto.
COSMOS, α ∈ {1e-4, 2e-4, 3e-4} × k ∈ {2, 5, 15}: el mejor, α 2e-4, k 15, **29.32 / 0.804** (sin
borde 27.48 / 0.813); con 3e-4 sube a 31.4. Sim2, α ∈ {2e-4, 4e-4, 8e-4} × {2, 5, 15}: el mejor,
α 4e-4, k 15, **32.56 / 0.802** (sin borde 29.76 / 0.816; streaking 0.024, momento −33.7, con
`metricas_npz.py` de DIRECTOR). El segundo, 2e-4 / 5, casi empata (32.63) y difiere en el resto:
XSIM 0.825, sin borde 30.35 / 0.833, streaking 0.038. Con 8e-4 sube a 37.6–38.0, y 4e-4 / 5 oscila
(mínimo 32.39 en t = 269, 33.29 en t = 300). En los dos sujetos el NRMSE todavía baja en t = 300 y
el mejor k es el borde de la grilla. Contra la 3D no lineal: en COSMOS la 3D gana por 3.4 pp y 0.042
de XSIM (25.95 / 0.846); en Sim2 empatan en NRMSE (32.61 contra 32.56; sin borde, 29.92 contra
29.76), la 3D gana en XSIM (0.822 contra 0.802; sin borde, 0.833 contra 0.816) y TV en la
calcificación (0.024 contra 0.062).

**La 3D con el término no lineal** (DIRECTOR, 28-10 15:45–16:05; 14 880, peso de FANSI, máscaras
del desafío): Sim2 a (0.0102, 0.00055) seguía bajando en t = 100 (35.44 %) y se llevó a 300:
**32.61 / 0.822** (sin borde 29.92 / 0.833; streaking 0.062, momento −28.3), contra el lineal a 300,
34.73 / 0.818 (32.43 / 0.830; 0.097). COSMOS a (0.0031, 0.00054), a 300: **25.95 / 0.846** (24.37 /
0.850), contra el lineal 26.09 / 0.843; a 100 daba 27.79. Decisión del autor (16:05): el abstract
pasa al no lineal en los dos sujetos, la 3D a 300 iteraciones, TV rehecho con el no lineal
(isótropo, a 300) y la 2D lineal, rotulada. Por capa en Sim2, la 3D no lineal queda +2.3, −0.5, +1.9
y +1.2 % contra TV lineal en las capas 0–3 y entre −1.3 y −12.4 % más adentro; contra la 2D, entre
−8 y −13 % en todas (`capas_npz.py`).

**Decisiones del autor para el abstract** (28-10, vía DIRECTOR y en directo, vigentes a las
16:30): término no lineal de FANSI y 300 iteraciones para TV (isótropo) y la 3D; la 2D, lineal a
100 y rotulada; la 3D lineal a 100 como el par justo de la 2D; lo publicado por TDV-QSM sólo como
contexto y con los dos sujetos (28.1 y 31.0 %, su protocolo), sin la 23 312 y sin comparar lineal
contra no lineal como resultado. Figuras sólo de cerebro: entrenamiento (cubos y denoising a SNR
50), COSMOS, Sim2, un solo χ de la 3D no lineal en Sim2 sin los 4 vóxeles externos (`fig_erosion.py`,
en lugar de las curvas del borde) y mapas de error; las rehace `rehacer_nl.py`. Sacar el borde baja
el error de todos (TV 32.56 → 29.76; la 2D 36.73 → 34.07), así que el texto da el de TV al lado y
no atribuye la baja a la cobertura del entrenamiento.

**Cobertura de los cubos de entrenamiento** (`cobertura_cubos.py`, los 12 T1 de validación, 48³
con `MARGEN_CEREBRO` = 1): el 65.3 % del cerebro, el 19.5 % de los 6 mm externos; por capa de
erosión 0…11, 2.8, 5.9, 11.8, 18.2, 25.4, 32.8, 40.4, 47.9, 54.4, 59.8, 64.6 y 68.2 %; interior,
90.7 % (`28-10-cobertura-por-capa.json`). Con margen 0 serían 72.2 y 32.4 %; con 2, 58.6 y 9.4 %.
