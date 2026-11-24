#!/bin/bash
# Weight-decay baseline for the symmetry-loss pilot: same cells as run_sym_pilot.sh (poly, N=200, 3%, k 0-2,
# seeds 0-2, learned nu from 0.05), but the extra term is g_wd * sum(c_k^2) instead of mu * L_sym.
# 3 wd values x 2 cases x 3 seeds = 18 tasks of 3 models = 54 models.
cd /home/minauser/research/symmetry-pinn
mkdir -p logs
for wd in 1e-6 1e-5 1e-4; do for name in g0 cubic; do for seed in 0 1 2; do
  echo "$wd $name $seed"
done; done; done | OMP_NUM_THREADS=1 xargs -P 6 -L1 sh -c 'python3 -m pinn.pinn_oracle 1 --name $1 --n 200 --sig 3 --k 0 1 2 --seed $2 --g-model poly --learn-nu 0.05 --g-wd $0 > logs/wd_poly_$1_N200_s3_wd$0_seed$2.log 2>&1'
