#!/bin/bash
# Regenera todo el material de la presentación con el estado actual de las corridas finales.
#   bash docs/presentacion/rehacer.sh          # figuras, números, métricas, imágenes (~40 min, CPU)
#   bash docs/presentacion/rehacer.sh eigen    # además rehace las eigenfunciones de las finales
# CPU a propósito: se puede correr con las GPUs entrenando. Los pasos son independientes: si uno
# falla, los demás siguen y al final se listan los que fallaron.
set -u
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=""
for f in ../../runs/t8delta-100k-s0 ../../runs/t8delta-100k-s1; do
  [[ -f $f/estado.json ]] && grep -q '"terminada": true' $f/estado.json \
    && echo "$(basename $f): terminada" || echo "$(basename $f): EN VUELO — las cifras saldrán PROVISIONAL"
done
fallas=()
paso() { echo "== $*"; "$@" || fallas+=("$*"); }
paso uv run python figuras.py
paso uv run python formulas.py
paso uv run python numeros.py
paso uv run python dentro_trazas.py
paso uv run python aprendido.py
paso uv run python fig_sigma.py
paso uv run python denoise.py
paso uv run python dentro.py
paso uv run python meseta.py
paso uv run python salto_lr.py
paso uv run python tinit_2delta.py
paso uv run python metricas_sigma.py final
if [[ "${1:-}" == eigen ]]; then
  rm -f salida/eig/eig_final_s*.npz
  paso bash eigen_extra.sh final_s0=runs/t8delta-100k-s0/best.pth final_s1=runs/t8delta-100k-s1/best.pth
fi
paso uv run python fig_eigen.py
echo; [[ ${#fallas[@]} -eq 0 ]] && echo "todo bien" || printf 'FALLÓ: %s\n' "${fallas[@]}"
