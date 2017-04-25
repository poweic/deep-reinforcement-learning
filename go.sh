#!/bin/bash

# FIXME go code from gym-offroad-nav:master
./train.py \
  --game OffRoadNav-v0 \
  --base-dir /Data3/acer-post-icml2017/ \
  --max-global-steps 8000000 \
  --max-steps 500 \
  --n-steps 256 \
  --estimator-type ACER \
  --log-file train.$(date +%s).log \
  --stats-file train.$(date +%s).stats.csv \
  --video-dir $(date +%s) \
  --parallelism 2 \
  --map-def map5 \
  --save-every-n-minutes 15 \
  --policy-dist Beta \
  --exp exp/debug2 \
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
  --max-replay-buffer-size 200 \
  --command-freq 5 \
  --discount-factor 0.99 \
  --n-agents-per-worker 16 \
  --vehicle-model-noise-level 1e-2 \
  --entropy-cost-mult 1e-2 \
  --learning-rate 1e-3 \
  --regenerate-size 10 \
  --l2-reg 1e-4 \
  --hidden-size 128 \
  --min-mu-vf -1 \
  --max-mu-vf 6 \
  --use-lstm False \
  --summarize True \
  $@

#  --min-mu-vf -4.0 \
#  --max-mu-vf 4.0 \
