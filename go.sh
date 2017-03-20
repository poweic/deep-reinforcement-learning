#!/bin/bash

# FIXME go code from gym-offroad-nav:master
./train.py \
  --game OffRoadNav-v0 \
  --base-dir /Data3/acer-post-icml2017/ \
  --max-global-steps 80000 \
  --estimator-type A3C \
  --log-file train.$(date +%s).log \
  --stats-file train.$(date +%s).stats.csv \
  --parallelism 4 \
  --track big_track \
  --save-every-n-minutes 30 \
  --policy-dist Beta \
  --t-max 40 \
  --drift False \
  --max-gradient 100.0 \
  --field-of-view 20 \
  --timestep 0.01 \
  --replay-ratio 16 \
  --avg-net-momentum 0.95 \
  --max-replay-buffer-size 5000 \
  --command-freq 5 \
  --discount-factor 0.99 \
  --n-agents-per-worker 8 \
  --vehicle-model-noise-level 5e-2 \
  --regenerate-size 10 \
  --l2-reg 1e-4 \
  $@

# --learning-rate 1e-3 \
# secret sauce is beta distribution with large learning rate
# ./train.py --game Humanoid-v1 --record-video True --exp exp/humanoid/trial-2 --policy-dist Beta
# ./train.py --game Humanoid-v1 --record-video True --exp exp/humanoid/trial-2 --policy-dist Gaussian
# ./train.py --game MountainCarContinuous-v0 --record-video True --exp exp/humanoid/trial-2 --policy-dist Gaussian --max-global-steps 1000 --render-every 1

# ./train.py \
#     --game Humanoid-v1 \
#     --eps-init 0 \
#     --max-gradient 1000 \
#     --save-every-n-minutes 30 \
#     --effective-timescale 1000 \
#     --max-global-steps 200000 \
#     --bi-directional False \
#     --max-steps 1000 \
#     --decay-steps 50000 \
#     --replay-ratio 8 \
#     --lr-vp-ratio 10 \
#     --avg-net-momentum 0.995 \
#     --exp exp/Beta-debug \
#     --policy-dist Beta \
#     --learning-rate 1e-4 \
#     --parallelism 4 \
#     --stats-file $(date +%s).stats.csv \
#     --log-file $(date +%s).log \
#    $@

# ./train.py \
#     --game Humanoid-v1 \
#     --eps-init 5e-2 \
#     --effective-timescale 1000 \
#     --max-global-steps 200000 \
#     --max-gradient 2000 \
#     --bi-directional True \
#     --decay-steps 40000 \
#     --replay-ratio 16 \
#     --avg-net-momentum 0.9 \
#     --exp exp/Gaussian-trial-3 \
#     --policy-dist Gaussian \
#     --learning-rate 5e-3 \
#     --parallelism 4 \
#     --stats-file $(date +%s).stats.csv \
#     --log-file $(date +%s).log \
#     $@

# ./train.py \
#     --game InvertedDoublePendulum-v1 \
#     --eps-init 1e-2 \
#     --max-gradient 100 \
#     --save-every-n-minutes 30 \
#     --effective-timescale 1000 \
#     --max-global-steps 40000 \
#     --max-steps 1000 \
#     --decay-steps 10000 \
#     --replay-ratio 8 \
#     --avg-net-momentum 0.99 \
#     --exp exp/Gaussian-trial-1 \
#     --policy-dist Gaussian \
#     --learning-rate 1e-3 \
#     --parallelism 4 \
#     --stats-file $(date +%s).stats.csv \
#     --log-file $(date +%s).log \
#    $@

# ./train.py \
#     --game InvertedDoublePendulum-v1 \
#     --eps-init 2e-2 \
#     --save-every-n-minutes 30 \
#     --effective-timescale 1000 \
#     --max-global-steps 40000 \
#     --max-steps 1000 \
#     --decay-steps 10000 \
#     --replay-ratio 8 \
#     --avg-net-momentum 0.99 \
#     --record-video True \
#     --exp exp/InvertedDoublePendulum-v1/trial-1 \
#     --policy-dist Beta \
#     --learning-rate 1e-3 \
#     --parallelism 16 \
#     --render-every 1 \
#     --stats-file $(date +%s).stats.csv \
#     --log-file $(date +%s).log \
#     $@

# ./train.py \
#     --game Humanoid-v1 \
#     --eps-init 2e-2 \
#     --effective-timescale 1000 \
#     --max-global-steps 100000 \
#     --decay-steps 20000 \
#     --replay-ratio 8 \
#     --avg-net-momentum 0.95 \
#     --record-video True \
#     --exp exp/humanoid/trial-4 \
#     --policy-dist Beta \
#     --learning-rate 1e-3 \
#     --parallelism 4 \
#     --render-every 1 \
#     --stats-file $(date +%s).stats.csv \
#     --log-file $(date +%s).log \
#     $@

# ./train.py \
#     --game MountainCarContinuous-v0 \
#     --replay-ratio 0.125 \
#     --avg-net-momentum 0.5 \
#     --record-video True \
#     --exp exp/humanoid/trial-2 \
#     --policy-dist Beta \
#     --max-global-steps 1000 \
#     --render-every 1 \
#     --stats-file log/$(date +%s).stats.csv \
#     --log-file log/$(date +%s).log

# ./mountain_car.py \
#   --render-every 1 \
#   --dist StudentT \
#   --learning-rate 0.1 \
#   --exp student-t \
#   --max-steps 10000 \
#   --logfile exp/$(date +%s).log \
#   $@
