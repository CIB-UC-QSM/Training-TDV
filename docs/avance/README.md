# Avance 2026-10-02: figuras y vectores de la presentación

Copia local de la página publicada en <https://claude.ai/artifact/3npGXmXrSxtxg9j7G7NyQT>
(`avance.html`, autocontenida: se abre directo en el navegador).

## Qué hay en `salida/`

| archivo | qué es |
|---|---|
| `f1`–`f5` | PSNR, `T`, `λ`, trayectoria (`T`, `λ`) con flechas y `‖g‖` de v4, v1, `referencia-100k` y las 24 corridas de `runs/precision/` |
| `f6`, `f7` | filtros de `K1` y `‖W − W₀‖/‖W₀‖` por capa contra el init de la semilla 0 |
| `f8` | `σ_max` del operador adjunto, desde `docs/mediciones/adjunta-sigma-max.txt` |
| `f9` | eigenfunciones no lineales (`∇R(x) = Λx`) de siete modelos |
| `trayectorias.npz` | los vectores de `f1`–`f5`: `T`, `λ`, `‖g‖` y PSNR por corrida (`*_traza` = columnas `loss T dT lambda dlambda \|g\|` de `traza.npy`) |
| `movimiento.json` | distancia relativa al init por tensor, para cuatro modelos |
| `eig_<modelo>.npz` | eigenfunciones: recorte inicial (`start`), estados del flujo (`it<k>`) y `log` = (iteración, residuo, Λ) |

## Cómo se regenera

Desde la raíz del repo, en este orden (`eigen.py` usa la GPU si hay una):

```bash
uv run python docs/avance/movimiento.py
uv run python docs/avance/figuras.py
uv run python docs/avance/fig_sigma.py
uv run python docs/avance/eigen.py checkpoints/Oficiales/tdv3-3-25-f32-color.pth oraculo \
    datasets/bsds500/images/test/100007.jpg docs/avance/salida 1000     # uno por modelo
uv run python docs/avance/fig_eigen.py
uv run python docs/avance/armar.py                            # plantilla.html -> avance.html
```

Los números del texto de la página que no están en la bitácora (recuperación 0/22 contra 2/6, `K1`
casi quieto, clip de v4 sin actuar) son de esta sesión y todavía no pasaron por `docs/bitacora.md`.
