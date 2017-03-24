#!/bin/bash

K=000
./train.py \
  --game Humanoid-v1 \
  --base-dir /Data3/acer-post-icml2017/ \
  --estimator-type ACER \
  --max-gradient 10000 \
  --save-every-n-minutes 20 \
  --max-global-steps 800$K \
  --bi-directional False  \
  --max-steps 1000 \
  --max-seq-length 1000 \
  --decay-steps 400$K \
  --replay-ratio 8 \
  --copy-params-every-nth 1 \
  --max-seq-length 128 \
  --lr-vp-ratio 1 \
  --discount-factor 0.99 \
  --importance-weight-truncation-threshold 5 \
  --avg-net-momentum 0.995 \
  --max-replay-buffer-size 10000 \
  --num-sdn-samples 31 \
  --exp exp/Beta-ff-only \
  --policy-dist Beta \
  --staircase False \
  --prioritize-replay False \
  --learning-rate 2e-4 \
  --regenerate-size 1000 \
  --entropy-cost-mult 1e-3 \
  --l2-reg 1e-4 \
  --parallelism 1 \
  --stats-file $(date +%s).stats.csv \
  --log-file $(date +%s).log \
  $@
