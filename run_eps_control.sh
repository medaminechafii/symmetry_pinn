#!/bin/bash
# False-positive control: stage 1 (learned g and nu, no symmetry loss) on near-symmetric data g = -u^3 + eps u,
# eps = 0.03 (e003) and 0.1 (e01). Same settings as run_lg_all.sh; N 200 and 1000, sigma 1 and 3%, both arms.
# The paired cubic runs already exist from stage 1.
# 2 eps x 2 N x 2 sigma x 2 arms x 3 seeds = 48 tasks of 3 models (k 0-2) = 144 models.
cd /home/minauser/research/symmetry-pinn
mkdir -p logs
python3 -m experiments.make_near_sym --eps e003 e01 --n 200 1000      # data (deterministic, a few seconds)
for N in 1000 200; do for sig in 1 3; do for arm in MLP poly; do for name in cubic_e003 cubic_e01; do for seed in 0 1 2; do
  echo "$arm $name $N $sig $seed"
done; done; done; done; done | OMP_NUM_THREADS=1 xargs -P 6 -L1 sh -c 'python3 -m pinn.pinn_oracle 1 --name $1 --n $2 --sig $3 --k 0 1 2 --seed $4 --g-model $0 --learn-nu 0.05 > logs/eps_$0_$1_N$2_s$3_seed$4.log 2>&1'
