# Por implementar: la meseta y los halvings, estado por estado

Pregunta: entre la it 52 000 y la 60 000 las finales suben **< 0.01 dB** (+0.002 s0, +0.007 s1;
de la 50k a la 60k, +0.012 y +0.015, casi todo el escalón del lr ÷ 2 en la 50k). ¿Se ve una
evolución en lo que aprende la red en ese tramo, y en los escalones de cada halving?

## Lo que ya se contestó (sin GPU)

- **Traza por iteración** (`m1_ganancia_por_tramo.png`, `g1`, `s1_salto_lr_traza.png`): desde la
  30k la PSNR gana ~0.01 dB por cada 10 000 it y no sigue bajando; `w = λ/T` sigue subiendo
  (s0: 1.29 → 1.36 entre la 50k y la 60k; s1: 1.42 → 1.49, medianas de 500 it); la pérdida de
  entrenamiento casi no cambia (s0: mediana 20.56 en 40–50k, 20.51 en 50–60k, 20.49 en 60–66k).
- **Cada halving** (25k, 50k y 75k): ‖g‖ cae de golpe 14–25 % en la iteración exacta y la PSNR
  sube un escalón (+0.03, +0.01, +0.0075 dB); `T` no se mueve (`salida/salto_lr.md`).
- **Con pesos, en la meseta** (`m2`, `m3`, `salida/meseta.md`): en 2 000 it la salida cambia
  ~0.5 niveles de gris (RMS) contra ~1.0 entre la 16k y la 18k. Menos de 1 nivel de gris no se ve
  a ojo. Las dos semillas difieren entre sí ~1.3, casi lo mismo que la final contra el oficial.

## Lo que NO existe, y por qué

| estados | por qué no están |
|---|---|
| 20k–60k de las dos semillas (incluido el salto 50k → 52k) | `train.py` guarda sólo `best.pth` y `last.pth` y los pisa en cada evaluación (train.py:347-350) |
| s0 después de la 66k | se cortó a mano en la 66k (PSNR estancada; commit 5419800) |
| s1 entre la 70k y la 98k (incluido el salto 74k → 76k) | `vigia_instantaneas.py` los copiaba, pero se cortó en la 68k junto con la sesión del 06-10 |

Lo que sí hay: s0 16k, 18k, 64k, 66k; s1 18k, 62k, 64k, 66k, 68k y 100k (`salida/ckpt/inst/` y
`runs/t8delta-100k-*`). Por eso **no hay imágenes de antes y después de un halving**.

## Para recuperarlos (necesita GPU)

`referencia` es bit-reproducible: re-correr una semilla con el mismo código da la misma
trayectoria —verificado en esta tanda: las cortadas por el corte de luz coinciden con las finales
en las 18 000 filas de la traza, bit a bit (E3/E4, `salida/numeros.md`)—. Basta con guardar estados
en el camino.

1. **Agregar a `train.py` una bandera `--guardar-cada N`** que, en el bloque de evaluación
   (train.py:333-350), haga `torch.save(payload, out / f"it{it:06d}.pth")` cuando `it % N == 0`.
   Regla 2 de AGENTS.md: sólo con ninguna tanda en vuelo.
2. **Verificar que es inerte**: 120 it en `referencia` con y sin la bandera, traza idéntica bit a
   bit (el mismo control que se hizo con `--guard inf`). `torch.save` no consume el generador.
3. **Re-correr la s1** con `--guardar-cada 2000` y la línea de comando de
   `runs/t8delta-100k-s1/config.json`, en otro `--out`, hasta la **78 000**: cubre la meseta 50–60k
   y los halvings de la 50k (estados 48k–54k) y de la 75k (72k–78k). Al terminar, confirmar que su
   `traza.npy` es idéntica bit a bit a la archivada en esas 78 000 filas; si no lo es, los estados
   **no son de la misma trayectoria** y no se pueden usar.
4. Correr `meseta.py` y `salto_lr.py` (este último ya arma, con esos estados, la figura `s2` con
   pesos por capa, diferencias de imagen y la razón ΔPSNR/‖ΔW‖ que distingue «asentarse» de «sólo
   dar pasos más cortos»).

**Costo**: 78 000 it × 1.36 s/it ≈ **29.5 h** en una GPU. Disco: 39 estados × 4.7 MB = 183 MB.
Con la s1 alcanza: es la única que pasó por la 75k.

## Qué no hacer

- **No** «aproximar» la 50k interpolando entre la 18k y la 64k: los pesos no se mueven en línea
  recta y las salidas intermedias no serían de ningún estado real.
- **No** reanudar desde un `last.pth` intermedio para «reconstruir» la trayectoria: el payload no
  guarda el estado del disyuntor (ventana, respaldo) ni de los generadores de datos y ruido, así
  que la continuación no sería bit a bit la original.
- **No** usar `best.pth` de `t8delta-100k-corte-s0` como la 18k: es la **16k** (s0 tuvo 31.393 en
  la 18k, menos que 31.419 en la 16k).
