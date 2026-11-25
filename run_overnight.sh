#!/bin/bash
# Back to back: MLP delta pilot (24 trainings), then the g0 one-symmetry-at-a-time experiment (42 trainings).
cd /home/minauser/research/symmetry-pinn
bash run_delta_pilot.sh MLP > logs/delta_pilot_MLP.out 2>&1
bash run_g0_steps.sh > logs/g0_steps.out 2>&1
echo "all done $(date)"
