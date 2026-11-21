#!/bin/bash
cd /home/minauser/research/symmetry-pinn
mkdir -p logs
for N in 1000 200 50; do for sig in 1 3; do for arm in MLP poly; do for name in g0 cubic; do for seed in 0 1 2; do
  echo "$arm $name $N $sig $seed"
done; done; done; done; done | OMP_NUM_THREADS=1 xargs -P 6 -L1 sh -c 'python3 -m pinn.pinn_oracle 1 --name $1 --n $2 --sig $3 --k 0 1 2 --seed $4 --g-model $0 --learn-nu 0.05 > logs/lg_$0_$1_N$2_s$3_seed$4.log 2>&1'
