#!/bin/bash
# Eigenfunciones (docs/avance/eigen.py) en CPU, para los pares tag=checkpoint que se pasen.
#   bash docs/presentacion/eigen_extra.sh final_s0=runs/t8delta-100k-s0/best.pth ...
# Calcula las dos imágenes de cada uno y salta las que ya existen (borrar el .npz para rehacer).
# SOLO_IMG1=1 calcula sólo la imagen 1; PARALELO=n cambia los procesos simultáneos (4).
# CPU a propósito: las GPUs entrenan. ~10 min por par con 4 hilos, ~35 min con 4 en paralelo.
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${EIGEN_GPU:-}"   # vacío = CPU; EIGEN_GPU=0 usa esa tarjeta (sólo si está libre)
O=docs/presentacion/salida/eig
IMG1=datasets/bsds500/images/test/100007.jpg   # el oso, igual que docs/avance
IMG2=datasets/bsds500/images/test/100039.jpg   # textura, la "otra imagen" de docs/avance
for par in "$@"; do
  tag=${par%%=*}; ck=${par#*=}
  for k in $([[ -n "${SOLO_IMG1:-}" ]] && echo 1 || echo 1 2); do
    img=IMG$k; suf=$([[ $k == 2 ]] && echo _img2)
    [[ -f $O/eig_${tag}${suf}.npz ]] && continue
    OMP_NUM_THREADS=4 uv run python docs/avance/eigen.py $ck ${tag}${suf} ${!img} $O 1000 \
      > $O/log_${tag}${suf}.txt 2>&1 &
    while (( $(jobs -r | wc -l) >= ${PARALELO:-4} )); do wait -n; done
  done
done
wait
echo listo
