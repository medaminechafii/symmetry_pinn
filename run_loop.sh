#!/bin/bash
# The symmetry loop (discovery/sym_loop.py; design notes/symmetry_loop_design.md section 9), poly arm, N 1000,
# held out t > 1.5, delta 0.1, mu 1, baseline seeds 0 1, max 2 rejections / 5 candidates / 2 passes (defaults).
# 4 data sets x 2 noise levels x 3 data draws = 24 chains, 6 in parallel; each chain runs its trainings one after another.
# cubic and g0 first (their baselines already exist from the pilots and are reused).
# Logs: logs/loop_<name>_s<sig>_k<k>.log ; decisions: results/oracle/loop/<chain>.jsonl ; summary: python -m experiments.loop_summary
cd /home/minauser/research/symmetry-pinn
mkdir -p logs
for name in cubic g0 cubic_e01 cubic_e003; do for sig in 1 0; do for k in 0 1 2; do
  echo "$name $sig $k"
done; done; done | OMP_NUM_THREADS=1 xargs -P 6 -L1 sh -c \
  'python3 -m discovery.sym_loop --name $0 --sig $1 --k $2 --quiet > logs/loop_$0_s$1_k$2.log 2>&1'
echo "loop done $(date)"
