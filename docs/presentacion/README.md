# Presentación de la fase 1: figuras, fórmulas, guion y brief

Reemplaza a [`docs/avance/`](../avance/) (presentación del 2026-10-02). Las figuras de allá
explotaban porque dibujaban lo que pasa después del estallido; acá **toda corrida que se pierde se
corta en su muerte** (✕ estallido, ◆ T → 0) y lo que cae fuera de escala va con una flecha y su
valor.

| archivo | qué es |
|---|---|
| [`guion.txt`](guion.txt) | lo que se dice, lámina por lámina, con número y fuente; preguntas probables |
| [`brief_para_claude.txt`](brief_para_claude.txt) | el pedido para que Claude arme las diapositivas con estas figuras |
| [`conceptos_y_parametros.md`](conceptos_y_parametros.md) | glosario y todos los parámetros usados: T, λ, disyuntor, receta |
| [`por_implementar.md`](por_implementar.md) | la meseta 50–60k: qué se puede y qué no, y cómo recuperar esos estados |
| `salida/numeros.md` | números finales y predicciones registradas, ✔/✘ |
| `salida/metricas_sigma.md` | PSNR, NRMSE y SNR a σ = 15/25/50 (sin re-entrenar) |
| `salida/meseta.md`, `salida/salto_lr.md` | cambio entre estados guardados; el escalón de cada halving |
| `salida/*.png` | figuras (inventario en el brief) |

## Estado al 2026-10-07

La final **s1 terminó** (100 000 it, 31.547 dB, 4 rebobinados) y la **s0 se cortó a mano en la
66k** (31.526 en la 64k; commit 5419800). Todo el material se regenera con:

```bash
bash docs/presentacion/rehacer.sh eigen    # todo, con los checkpoints finales (~1–2 h de CPU)
```

Después, completar los `⟦…⟧` que queden en `guion.txt` con `salida/*.md` y pasarle el brief a
Claude. Ojo: `best.pth` es el de **mejor PSNR** (s1: la 100k; s0: la 64k).

## Procesos en segundo plano (CPU)

- `eigen_cola.sh` calcula todas las eigenfunciones. Es idempotente: si se corta, se relanza con
  `nohup setsid nice -n 5 bash docs/presentacion/eigen_cola.sh >> docs/presentacion/salida/eig/cola.log 2>&1 &`.
- `metricas_sigma.py` se paraleliza por modelo con `MODELO="final s1"` (una caché por modelo).
- `vigia_instantaneas.py` copiaba `last.pth` de las finales cada 2 000 it. **Se cortó en la 68k**
  con la sesión del 06-10: los estados de la 70k–98k de la s1 no existen (`por_implementar.md`).
- El 06-10 los lotes de eigenfunciones se cortaron tres veces con SIGTERM (código 144 /
  «Terminated»); no se pudo atribuir. Relanzar la cola los retoma.

## Scripts

| script | produce | tiempo |
|---|---|---|
| `estilo.py` | paleta, rcParams, cargadores, eje de capas (no se corre solo) | — |
| `trayectorias.py` | catálogo de estados de una misma trayectoria; crea los init | segundos |
| `figuras.py` | `a1`–`a3`, `b1`, `b2_*`, `c1` | ~1 min |
| `formulas.py` | `formula_*` | segundos |
| `dentro_trazas.py` | `g1` (adentro, sana vs estallada), `g2` (estallido ampliado), `g3` (magnitud) | segundos |
| `dentro.py` | `g4` (salida por estado), `g5` (10 pasos), `g7` (paso efectivo), `g8` (capas); `dentro.json` | ~10 min |
| `meseta.py` | `m1`–`m3`, `meseta.md` | ~5 min |
| `salto_lr.py` | `s1` (halvings en la traza), `s2_*` (con pesos, si existen los estados), `salto_lr.md` | ~5 min |
| `metricas_sigma.py` | `h1`, `metricas_sigma.md` (caché `metricas_sigma_<modelo>.json`) | ~10 min por modelo y σ |
| `denoise.py` | `img_denoise_*`, `img_denoise.json` | ~2 min |
| `aprendido.py` | `d1_filtros_K1`, `d2_movimiento_capas` | segundos |
| `fig_sigma.py` | `d3_sigma_max` | segundos |
| `eigen_lote.sh`, `eigen_extra.sh`, `eigen_cola.sh` | `salida/eig/eig_*.npz` (con `docs/avance/eigen.py`) | ~10–35 min por estado |
| `fig_eigen.py` | `e1_eigen_comparacion`, `g6_eigen_trayectorias` | segundos |
| `estado_100k.py` | `img_estado_100k_4modelos`, `estado_100k.md`: los cuatro al término de las 100k | ~4 min |
| `qsm_admm.py` | `img_qsm_admm_4modelos`: los mismos cuatro dentro del ADMM de QSM | segundos |
| `numeros.py` | `numeros.md`, `numeros.json` | segundos |
| `vigia_instantaneas.py` | `salida/ckpt/inst/*/it*.pth` (gitignorado) | hasta que terminan las corridas |

Todo corre en **CPU** (`CUDA_VISIBLE_DEVICES=""`). Las eigenfunciones del oficial, v4 y
`referencia-100k` sobre la imagen 1 se copiaron de `docs/avance/salida/` (mismo script e imagen).

## Decisiones de las figuras que no son obvias

- **Muertes de v1, v5 y v6 (◆) salen de la bitácora, no del log**: son muertes por T → 0; sus
  logs muestrean cada 250 it, v1 no guarda ‖g‖ y v6 ya pasa 10⁴ en la ventana [1, 250] sin morir.
- **Con disyuntor, la muerte se decide evento por evento** (`figuras.py`, `muerte()`): cuenta el
  primer evento después del cual la mediana de ‖g‖ aceptado (300–800 it después) no baja de 10³, o
  no hay lotes aceptados (trabada). Un evento rebobinado o contenido no es muerte.
- **«Mejor PSNR» de `a1` es el sano**: sólo evaluaciones antes de la muerte (`referencia-100k` y v6
  no tienen ninguna).
- **`T/δ(t)` usa el δ del lr vigente**: por eso las finales saltan ×2 en 25k y 50k.
- **Las 55 corridas cortas** excluyen las de diagnóstico de 300 it y la truncada `clip/clip-seed1`.
- **Trayectorias**: `t8delta-100k-corte-s1` es la final s1 hasta la 18k (bit a bit, E3);
  `referencia-100k` es `lambdainit/B-seed0` extendida; el `best.pth` de `corte-s0` es la **16k**.
- **El paso efectivo** usa la definición de `scripts/measure_adjoint_chain.py`: ‖∇R‖ de lote
  promediada sobre los 10 x_s. Medida sólo en z da ~3× más.
