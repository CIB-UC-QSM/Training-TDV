#!/bin/bash
# Eigenfunciones (docs/avance/eigen.py) en CPU para los checkpoints de la presentación.
#   bash docs/presentacion/eigen_lote.sh                 # las que faltan
#   bash docs/presentacion/eigen_lote.sh final           # sólo las dos finales (rehacer mañana)
# CPU a propósito: las GPUs están entrenando. ~10 min por modelo e imagen con 4 hilos.
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=""
O=docs/presentacion/salida/eig
IMG1=datasets/bsds500/images/test/100007.jpg   # el oso, igual que docs/avance
IMG2=datasets/bsds500/images/test/100039.jpg   # textura, la "otra imagen" de docs/avance
declare -A CK=(
  [final_s0]=runs/t8delta-100k-s0/best.pth
  [final_s1]=runs/t8delta-100k-s1/best.pth
  [corte_s0]=runs/t8delta-100k-corte-s0/best.pth
  [tinit10d]=runs/tinit-10delta-10k/best.pth
)
SOLO=${1:-}
for tag in "${!CK[@]}"; do
  [[ -n "$SOLO" && "$tag" != ${SOLO}* ]] && continue
  for k in 1 2; do
    img=IMG$k; suf=$([[ $k == 2 ]] && echo _img2)
    [[ -z "$SOLO" && -f $O/eig_${tag}${suf}.npz ]] && continue
    OMP_NUM_THREADS=4 uv run python docs/avance/eigen.py ${CK[$tag]} ${tag}${suf} ${!img} $O 1000 \
      > $O/log_${tag}${suf}.txt 2>&1 &
    while (( $(jobs -r | wc -l) >= 4 )); do wait -n; done
  done
done
wait
echo listo
