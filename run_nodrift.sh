#!/bin/bash
# Real test of the "no drift limit" rule (design notes, section 13): the offline re-evaluation (experiments/reeval_drift.py)
# only gives lower bounds where a pass was cut short. Same loop as run_full.sh tier 1 + 2, but --max-drift 180 (= off),
# clean preparation only: 4 data sets x noise 0/1 x k 0-2 x arms poly/MLP = 48 chains.
# Chain names end in _md180_lg, so the logged 10-degree chains are NOT overwritten.
# Reuses the baselines of run_full.sh (missing ones are trained by sym_loop itself).
# Results:  python -m experiments.loop_eval --N 1000      (cells "clean+learn md180" next to "clean+learn")
# Usage (venv active, from anywhere):  P=6 nohup bash run_nodrift.sh > logs/nodrift.out 2>&1 &
cd "$(dirname "$0")"
P=${P:-6}
mkdir -p logs
echo "=== nodrift start $(date)"
for arm in poly MLP; do for s in 0 1; do for name in g0 cubic cubic_e01 cubic_e003; do for k in 0 1 2; do
  echo "$name $s $k $arm"
done; done; done; done | OMP_NUM_THREADS=1 xargs -P "$P" -L1 sh -c \
  'python3 -m discovery.sym_loop --name $0 --sig $1 --k $2 --N 1000 --arm $3 --prep clean --learn --max-drift 180 --quiet > logs/nodrift_$0_s$1_k$2_$3.log 2>&1'
echo "=== nodrift done $(date)"
