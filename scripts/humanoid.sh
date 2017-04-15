#!/bin/bash

K=000
M=000000
./train.py \
  --game Humanoid-v1 \
  --base-dir /Data3/acer-post-icml2017/ \
  --estimator-type ACER \
  --max-gradient 10000 \
  --save-every-n-minutes 20 \
  --max-global-steps 8000$K \
  --bi-directional False  \
  --max-steps 1000 \
  --decay-steps 4000$K \
  --replay-ratio 8 \
  --off-policy-batch-size 1 \
  --n-steps 200 \
  --lr-vp-ratio 1 \
  --lambda_ 0.97 \
  --discount-factor 0.995 \
  --importance-weight-truncation-threshold 5 \
  --avg-net-momentum 0.995 \
  --max-replay-buffer-size 1000 \
  --num-sdn-samples 64 \
  --exp exp/debug \
  --policy-dist Beta \
  --staircase True \
  --prioritize-replay False \
  --learning-rate 1e-3 \
  --regenerate-size 500 \
  --entropy-cost-mult 1e-3 \
  --l2-reg 1e-4 \
  --parallelism 8 \
  --max-seq-length 256 \
  --stats-file $(date +%s).stats.csv \
  --log-file $(date +%s).log \
  --summarize True \
  $@
