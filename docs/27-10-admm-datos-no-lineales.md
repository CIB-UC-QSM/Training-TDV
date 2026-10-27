# El término de datos no lineal y L1 en el ADMM (rama `qsm-datos-no-lineales`)

Pedido del autor (27-10): usar el término de datos no lineal de `nlTV.m` y, como segunda opción, el
L1 de L1-QSM (`Inversion/L1-QSM/nlL1TV.m`), los dos del FANSI-toolbox de Carlos Milovic
(gitlab.com/cmilovic/FANSI-toolbox), con la fase en radianes, y elegir con un flag. Esta página dice
qué funciones se tocan, qué líneas cambian y dónde exactamente se reemplaza el ADMM lineal.

## Qué no cambia

El ADMM sigue siendo `admm_wtdv` de `ddr/qsm3d.py`, con el mismo x-update cerrado en Fourier y el
mismo paso del prior (la red TDV, z = M·D(α·M·(χ + s))/α). En FANSI, `nlTV.m` y `nlL1TV.m`
comparten con el lineal el x-update (`nlTV.m:123`, `nlL1TV.m:202`) y difieren del TV en el paso
del prior (el umbral de los gradientes, `nlTV.m:139-141`), que aquí es la red. **Lo único que
cambia es el paso del término de datos: el z2 (z2 ≈ D·χ) y su dual s2**, más, en L1, otra variable
partida z3 y su dual s3.

Con `datos_modelo="lineal"` (el default) el bucle es, bit a bit, el de antes
(`test_lineal_es_el_de_siempre` y los tests del script del autor, que no cambiaron).

## Las tres opciones

| `datos_modelo` | término de datos | paso de z2 | de dónde |
|---|---|---|---|
| `"lineal"` | ½‖W(z2 − φ)‖² | cerrado: z2 = (W²φ + μ₂(Dχ + s2))/(W² + μ₂) | `../tdv/wTDV_QSM_torch_3d.py` (el del autor) |
| `"nolineal"` | ½‖W(e^{i·z2} − e^{i·φ})‖² = W²(1 − cos(z2 − φ)) | Newton sobre W²·sin(z2 − φ) + μ₂·z2 − μ₂(Dχ + s2) = 0, hasta 10 pasos o paso relativo ≤ 10⁻⁶, desde z2 = Dχ + s2 | `nlTV.m:149-163` |
| `"l1"` | ‖W₁(e^{i·z2} − e^{i·φ})‖₁ | z3 = prox L1 complejo de e^{i·z2} − e^{i·φ} + s3 con umbral W₁/μ₃; z2 por Newton (hasta 4 pasos, \|denominador\| ≥ 0.05, corta si el paso crece) hacia e^{i·z2} ≈ e^{i·φ} + z3 − s3; s3 += e^{i·z2} − e^{i·φ} − z3 | `nlL1TV.m:231-261` |

- **Pesos.** Los dos no lineales pesan como FANSI, con la magnitud en [0, 1]: el `W2` del lineal
  (la magnitud normalizada a media 1 en el cerebro, al cuadrado) pasa a máximo 1, W² = W2/max W2,
  como `W = weight.^2` de `nlTV.m:274`. No es cosmético: con W² ≤ μ₂ = 1 el denominador del Newton
  de nlTV, W²·cos(z2 − φ) + μ₂, no se anula; con el W2 de media 1 (máximo ~10) se anula donde
  |z2 − φ| > π/2 y el Newton diverge (lo muestra `test_z2_nolineal_es_nltv_de_fansi`). Por eso,
  con fases chicas, el no lineal es el lineal **con ese mismo peso**, no el lineal del autor: el dato
  pesa ~10 veces menos frente a μ₂ y μ, y el (target, μ) se recalibra. En "l1", W₁ =
  `lambda_l1`·√W² (`nlL1TV.m:18-23`; λ < 1 rechaza más vóxeles inconsistentes); μ₃ = 1
  (`nlL1TV.m:58-62`).
- **Radianes.** Los dos no lineales trabajan con la fase en radianes, φ·γ·B0·TE, como
  `params.input = phase_use*phs_scale` de los scripts de FANSI: con ppm (±0.1) sin y cos son
  lineales y no cambia nada. `fase_por_ppm` es obligatorio en esos modos:
  - COSMOS: `FASE_POR_PPM_COSMOS` = 2π·42.58·2.8936·0.025 = 19.354 rad/ppm, los B0 y TE con que
    FANSI simuló `params.mat` (`script_cosmos_phantom.m:27-31`; la fase de `params.mat` está acotada
    en ±π/19.354 ppm, bitácora 26-10).
  - Sim2: `FASE_POR_PPM_SIM2` = 2π·42.58·7·0.008 = 14.98 rad/ppm: B0 = 7 T (Marques et al., MRM
    2021) y TE = 8 ms, el ΔTE de las diferencias de fase del mapa de campo del challenge 2.0
    (indicación del supervisor, «TE 8 ms en vez de 4 ms»).
  - El NRMSE, `chi`, `chi0` y `alpha` del resultado vuelven a ppm. La red ve lo mismo en las tres
    opciones, porque α·χ no depende de la escala (α = target/std(χ₀) se ajusta).
- **μ = k·α** (`mu_por_alfa=k`): μ se fija desde t = 1 como k·α, con α en ppm. χ₀ (t = 0) sale con
  `mu`, porque α sale de χ₀. Es la relación μ/α fija que pidió el supervisor (en FANSI,
  `mu1 = 100*alpha1`, `nlTV.m:12` y `:207`).

## Funciones y líneas (`ddr/qsm3d.py` de esta rama)

| dónde | qué | líneas |
|---|---|---|
| docstring del módulo | menciona las opciones | 5 |
| `__all__` | exporta `DATOS_MODELOS`, `FASE_POR_PPM_COSMOS`, `FASE_POR_PPM_SIM2`, `MU3` | 46 |
| bloque «TÉRMINO DE DATOS NO LINEAL (FANSI)» | constantes nuevas, cada una con su línea de FANSI | 78-92 |
| `_z2_nolineal` (nueva) | `nlTV.m:149-163`, el Newton del z2 | 264-277 |
| `_prox_l1_complejo` (nueva) | `nlL1TV.m:232`, max(\|Y\| − u, 0)·Y/\|Y\| | 280-282 |
| `_z2_l1` (nueva) | `nlL1TV.m:234-256`, el Newton del z2 en L1 | 285-306 |
| `admm_wtdv`, firma | `datos_modelo`, `fase_por_ppm`, `mu3`, `lambda_l1`, `mu_por_alfa` | 309-316 |
| `admm_wtdv`, docstring | las opciones, los pesos, los radianes y μ = k·α | 340-353 |
| `admm_wtdv`, validación | modelo conocido y `fase_por_ppm` si no es lineal; `escala` | 357-361 |
| `admm_wtdv`, arranque | fase en radianes, W² con máximo 1 y W²φ; en L1, W₁, e^{iφ}, s3 = 0 y z2 = W₁φ/max W₁ (`nlL1TV.m:108, 127, 134`); si no, el z2 de siempre (= `nlTV.m:57`) | 375-389 |
| `admm_wtdv`, t = 0 | χ₀ en ppm; μ = k·α si hay `mu_por_alfa` | 409-416 |
| `admm_wtdv`, NRMSE | sobre χ en ppm | 422 |
| **`admm_wtdv`, el paso de datos** | **el reemplazo del ADMM lineal**: rama "lineal" (la línea de siempre), "nolineal" y "l1"; s2 igual en las tres; s3 en L1 | **455-466** |
| `admm_wtdv`, resultado | χ y α en ppm; `mu`, `datos_modelo`, `fase_por_ppm` | 473-480 |

En `main`, el paso de datos que se reemplaza son las líneas 350-352 de `ddr/qsm3d.py`:

```python
        # --- z2 Y SU DUAL: el término de datos pesado por W² ---------------------------------
        z2 = (Wy + mu2 * (FhDFx + s2)) / weight_mu2
        s2 += FhDFx - z2
```

y el arranque del z2, la línea 288 (`z2 = Wy / weight_mu2`).

## Tests (`tests/test_qsm3d.py`, sección «TÉRMINO DE DATOS NO LINEAL Y L1 (FANSI)»)

- `test_z2_nolineal_es_nltv_de_fansi` y `test_z2_l1_es_nll1tv_de_fansi`: transcripciones textuales
  de `nlTV.m:149-163` y `nlL1TV.m:231-261`, bit a bit contra las funciones, con fases de hasta
  ~2 rad; el z2 anula el gradiente del subproblema; el prox complejo encoge el módulo y no toca la
  fase.
- `test_lineal_es_el_de_siempre`: el lineal no cambia, bit a bit.
- `test_no_lineal_con_fase_chica_es_el_lineal`: con radianes chicos el no lineal sigue al lineal
  con el peso de máximo 1.
- `test_no_lineales_corren_lejos_del_regimen_lineal`: con fases de ~1 rad, finitos y mejoran χ₀.
- `test_no_lineal_pide_la_escala` y `test_mu_por_alfa`.

## Falta

- **El flag en `train3d.py`** (`--qsm-datos-modelo lineal|nolineal|l1` y `--qsm-mu-por-alfa k`,
  que pasan a `admm_wtdv` con `fase_por_ppm` = `FASE_POR_PPM_COSMOS` o `_SIM2` según el sujeto): no
  se puede escribir mientras corre `ciego-epocas-l32-30k` (regla 2; el hook lo bloquea también en
  un worktree). Entra cuando termine.
- **Medirlo con la red** en COSMOS y Sim2 (GPU): los tres modelos al mismo (target, μ).
