#!/bin/bash

# FIXME go code from gym-offroad-nav:master
./train.py \
  --game OffRoadNav-v0 \
  --base-dir /Data3/acer-post-icml2017/ \
  --max-global-steps 80000 \
  --max-steps 10000 \
  --n-steps 100 \
  --estimator-type ACER \
  --log-file train.$(date +%s).log \
  --stats-file train.$(date +%s).stats.csv \
  --parallelism 2 \
  --map-def map5 \
  --save-every-n-minutes 15 \
  --policy-dist Beta \
  --exp exp/debug \
  --t-max 40 \
  --drift False \
  --max-seq-length 512 \
  --max-gradient 1000.0 \
  --field-of-view 32 \
  --timestep 0.025 \
  --replay-ratio 4 \
  --off-policy-batch-size 1 \
  --lambda_ 0.97 \
  --num-sdn-samples 15 \
  --avg-net-momentum 0.995 \
  --max-replay-buffer-size 100 \
  --command-freq 5 \
  --discount-factor 0.99 \
  --n-agents-per-worker 16 \
  --vehicle-model-noise-level 2e-2 \
  --entropy-cost-mult 1e-2 \
  --learning-rate 2e-4 \
  --regenerate-size 10 \
  --min-mu-vf 0.5555555555555556 \
  --l2-reg 1e-4 \
  --summarize True \
  $@

#  --min-mu-vf -4.0 \
#  --max-mu-vf 4.0 \
