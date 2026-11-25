#!/bin/bash
# Delta pilot (notes/symmetry_loop_design.md, section 2.6). Cell: cubic, poly, N 1000, held-out t > 1.5,
# sigma 1% AND noise-free (sigma 0). Per sigma:
# 1. baselines (stage 1 on t <= 1.5): k 0-2, seeds 0 and 1        -> 6 models (seed spread = chance fluctuation of E)
# 2. generator files from the seed-0 baselines: exact scaling (true for cubic), Galilean (false for cubic)
# 3. seed 0 with the general symmetry loss, mu = 1: scaling and Galilean, k 0-2   -> 6 models
# 24 trainings in all. Then: python -m experiments.delta_pilot
cd /home/minauser/research/symmetry-pinn
mkdir -p logs
COMMON="1 --name cubic --n 1000 --g-model poly --learn-nu 0.05 --holdout 1.5"
for sig in 1 0; do for k in 0 1 2; do for seed in 0 1; do echo "$sig $k $seed"; done; done; done \
  | OMP_NUM_THREADS=1 xargs -P 6 -L1 sh -c "python3 -m pinn.pinn_oracle $COMMON --sig \$0 --k \$1 --seed \$2 > logs/dpilot_base_s\$0_k\$1_seed\$2.log 2>&1"
for sig in 1 0; do for k in 0 1 2; do for gen in scaling Galilean; do
  python3 -m discovery.make_gen_file --name cubic --N 1000 --sig $sig --k $k --holdout 1.5 --gens $gen
done; done; done
G=results/oracle/gens
for sig in 1 0; do for k in 0 1 2; do for gen in scaling Galilean; do
  echo "$sig $k $G/oracle_cubic_N1000_s${sig}_lam1_k${k}_nu0.05_poly_ho1.5_seed0__${gen}.npz $gen"
done; done; done | OMP_NUM_THREADS=1 xargs -P 6 -L1 sh -c "python3 -m pinn.pinn_oracle $COMMON --sig \$0 --k \$1 --seed 0 --gen-file \$2 --mu 1 > logs/dpilot_\$3_s\$0_k\$1.log 2>&1"
