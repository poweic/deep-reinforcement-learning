#!/bin/bash
# secret sauce is beta distribution with large learning rate
# ./train.py --game Humanoid-v1 --record-video True --exp exp/humanoid/trial-2 --policy-dist Beta
# ./train.py --game Humanoid-v1 --record-video True --exp exp/humanoid/trial-2 --policy-dist Gaussian
# ./train.py --game MountainCarContinuous-v0 --record-video True --exp exp/humanoid/trial-2 --policy-dist Gaussian --max-global-steps 1000 --render-every 1

./train.py \
    --game InvertedDoublePendulum-v1 \
    --eps-init 2e-2 \
    --save-every-n-minutes 30 \
    --effective-timescale 1000 \
    --max-global-steps 100000 \
    --decay-steps 20000 \
    --replay-ratio 4 \
    --avg-net-momentum 0.95 \
    --record-video True \
    --exp exp/InvertedDoublePendulum-v1/trial-1 \
    --policy-dist Beta \
    --learning-rate 1e-3 \
    --parallelism 4 \
    --render-every 1 \
    --stats-file $(date +%s).stats.csv \
    --log-file $(date +%s).log \
    $@

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
