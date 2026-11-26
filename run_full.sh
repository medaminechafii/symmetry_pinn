#!/bin/bash
# Full evaluation of the symmetry loop with learnable candidate coefficients (design notes section 12):
# 4 data sets x noise 0/1/3% x N 1000/200/50 x k 0-2 x arms poly/MLP x preparations clean/sparse = 432 chains.
# Runs in PRIORITY order (tiers); results can be read at any time with  python -m experiments.loop_eval
#   tier 1  poly, N 1000, noise 0 and 1          (~8 h on 6 cores)
#   tier 2  MLP,  N 1000, noise 0 and 1
#   tier 3  poly + MLP, N 1000, noise 3
#   tier 4  N 200 (all noise levels, both arms)
#   tier 5  N 50
# Every tier first trains its baselines (2 seeds; existing ones are skipped), then the chains, P in parallel.
# Usage (venv active, from anywhere):  P=6 nohup setsid bash run_full.sh > logs/full.out 2>&1 &
cd "$(dirname "$0")"
P=${P:-6}
mkdir -p logs
NAMES="g0 cubic cubic_e01 cubic_e003"          # g0 first: longest chains start early

# data: the eps data for N = 50 (and anything else missing) -- deterministic
for N in 1000 200 50; do for e in e01 e003; do
  if [ ! -f "data/sparse_cubic_${e}_N${N}_s0.npz" ] || [ ! -f "data/sparse_cubic_${e}_N${N}_s1.npz" ] \
     || [ ! -f "data/sparse_cubic_${e}_N${N}_s3.npz" ]; then
    python3 -m experiments.make_near_sym --eps $e --n $N      # writes noise 0, 1, 3 % at once
  fi
done; done

tier () {   # $1 = label, $2 = arms, $3 = N values, $4 = noise levels
  echo "=== tier $1 start $(date)"
  for arm in $2; do for N in $3; do for s in $4; do for name in $NAMES; do for k in 0 1 2; do for seed in 0 1; do
    echo "$name $s $k $N $arm $seed"
  done; done; done; done; done; done | OMP_NUM_THREADS=1 xargs -P "$P" -L1 sh -c \
    'python3 -m pinn.pinn_oracle 1 --name $0 --n $3 --sig $1 --k $2 --seed $5 --g-model $4 --learn-nu 0.05 --holdout 1.5 --steps 4000 > logs/full_base_$0_N$3_s$1_k$2_$4_seed$5.log 2>&1'
  for arm in $2; do for N in $3; do for s in $4; do for name in $NAMES; do for k in 0 1 2; do for prep in clean sparse; do
    echo "$name $s $k $N $arm $prep"
  done; done; done; done; done; done | OMP_NUM_THREADS=1 xargs -P "$P" -L1 sh -c \
    'python3 -m discovery.sym_loop --name $0 --sig $1 --k $2 --N $3 --arm $4 --prep $5 --learn --quiet > logs/full_$0_N$3_s$1_k$2_$4_$5.log 2>&1'
  echo "=== tier $1 done $(date)"
}

tier 1 "poly" "1000" "0 1"
tier 2 "MLP"  "1000" "0 1"
tier 3 "poly MLP" "1000" "3"
tier 4 "poly MLP" "200" "0 1 3"
tier 5 "poly MLP" "50" "0 1 3"
echo "full run done $(date)"
