#!/bin/bash
# Cola completa de eigenfunciones de la presentación, desacoplada de la sesión:
#   nohup setsid nice -n 10 bash docs/presentacion/eigen_cola.sh >> docs/presentacion/salida/eig/cola.log 2>&1 &
# Idempotente: salta los .npz que ya existen (borrar uno para rehacerlo). Los init quedan fuera:
# R recién inicializado es casi nulo (Λ ≈ −2·10⁻⁴) y cualquier imagen es «eigenfunción».
cd "$(dirname "$0")/../.."
E=docs/presentacion/eigen_extra.sh
I=docs/presentacion/salida/ckpt/inst/t8delta-100k-s1
P=${PARALELO:-8}
# 1) finales (las dos imágenes) y comparación
PARALELO=$P bash $E final_s1=runs/t8delta-100k-s1/best.pth final_s0=runs/t8delta-100k-s0/best.pth \
  estallada=runs/precision/rapido-seed10/last.pth ref100k=runs/referencia-100k/best.pth \
  viva4k=runs/precision/rapido-seed0/best.pth
# 2) trayectorias (imagen 1)
SOLO_IMG1=1 PARALELO=$P bash $E s1_18k=runs/t8delta-100k-corte-s1/last.pth \
  s0_16k=runs/t8delta-100k-corte-s0/best.pth ref_1k=runs/lambdainit/B-seed0/best.pth \
  ref_4k=runs/lambdainit/B-seed0/last.pth ref_100k=runs/referencia-100k/last.pth \
  dis_4k=runs/disyuntor-100k/best.pth dis_50k=runs/disyuntor-100k/last.pth \
  r10_3k=runs/precision/rapido-seed10/best.pth v4_100k=runs/replica-v4/last.pth \
  s1_64k=$I/it064000.pth s1_68k=$I/it068000.pth
echo "cola terminada $(date)"
