#!/bin/bash
# g0 "one symmetry at a time" (user's idea; notes/symmetry_loop_design.md section 8.2): TRUE generators added step by step
# with the general symmetry loss (mu = 1), poly, N 1000, held out t > 1.5, sigma 1% and noise-free.
# 1. baselines: k 0-2, seeds 0 and 1 (reference = the better of the two)                      -> 12 models
# 2. generator files from the seed-0 baselines
# 3. seed 0 with the symmetry loss. Cumulative steps first (the essential part), then each symmetry alone (optional):
#    scaling | scaling+Galilean | scaling+Galilean+projective | Galilean | projective            -> 30 models
# 42 trainings. Then: python -m experiments.g0_steps
cd /home/minauser/research/symmetry-pinn
mkdir -p logs
ARM=poly
COMMON="1 --name g0 --n 1000 --g-model $ARM --learn-nu 0.05 --holdout 1.5"
SETS="scaling scaling+Galilean scaling+Galilean+projective Galilean projective"
for sig in 1 0; do for k in 0 1 2; do for seed in 0 1; do echo "$sig $k $seed"; done; done; done \
  | OMP_NUM_THREADS=1 xargs -P 6 -L1 sh -c "python3 -m pinn.pinn_oracle $COMMON --sig \$0 --k \$1 --seed \$2 > logs/g0steps_base_s\$0_k\$1_seed\$2.log 2>&1"
for sig in 1 0; do for k in 0 1 2; do for set in $SETS; do
  python3 -m discovery.make_gen_file --name g0 --N 1000 --sig $sig --k $k --holdout 1.5 --arm $ARM --gens ${set//+/ }
done; done; done
G=results/oracle/gens
for set in $SETS; do for sig in 1 0; do for k in 0 1 2; do
  echo "$sig $k $G/oracle_g0_N1000_s${sig}_lam1_k${k}_nu0.05_${ARM}_ho1.5_seed0__${set}.npz $set"
done; done; done | OMP_NUM_THREADS=1 xargs -P 6 -L1 sh -c "python3 -m pinn.pinn_oracle $COMMON --sig \$0 --k \$1 --seed 0 --gen-file \$2 --mu 1 > logs/g0steps_\$3_s\$0_k\$1.log 2>&1"
