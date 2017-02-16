#!/bin/bash

source /home/poweic/tensorflow/bin/activate
export MUJOCO_PY_MJKEY_PATH=/home/poweic/.mujoco/mjkey.txt 
export MUJOCO_PY_MJPRO_PATH=/home/poweic/.mujoco/mjpro131  

/home/poweic/Stochastic-Policy-Gradient-for-Continuous-DRL/train.py \
  --game Humanoid-v1 \
  --eps-init 1e-2 \
  --effective-timescale 1000 \
  --max-global-steps 100000 \
  --max-gradient 100 \
  --decay-steps 20000 \
  --replay-ratio 8 \
  --avg-net-momentum 0.95 \
  --record-video True \
  --exp exp/trial-2 \
  --policy-dist Beta \
  --learning-rate 1e-3 \
  --parallelism 10 \
  --render-every 1 \
  --stats-file $(date +%s).stats.csv \
  --log-file $(date +%s).log \
  $@

#  --game InvertedDoublePendulum-v1 \
#  --eps-init 1e-2 \
#  --save-every-n-minutes 30 \
#  --effective-timescale 1000 \
#  --max-global-steps 400000 \
#  --max-gradient 100 \
#  --max-steps 1000 \
#  --decay-steps 10000 \
#  --replay-ratio 8 \
#  --avg-net-momentum 0.995 \
#  --exp exp/Beta-trial-1 \
#  --policy-dist Beta \
#  --learning-rate 5e-3 \
#  --parallelism 16 \
#  --stats-file $(date +%s).stats.csv \
#  --log-file $(date +%s).log \
#  $@
