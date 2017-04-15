#!/bin/bash

K=000
M=000000
./train.py \
  --game InvertedDoublePendulum-v1 \
  --base-dir /Data3/acer-post-icml2017/ \
  --estimator-type ACER \
  --max-gradient 10000 \
  --save-every-n-minutes 20 \
  --max-global-steps 800$K \
  --bi-directional False  \
  --n-steps 1000 \
  --max-seq-length 1000 \
  --decay-steps 100$K \
  --replay-ratio 8 \
  --copy-params-every-nth 1 \
  --max-seq-length 1024 \
  --lr-vp-ratio 1 \
  --lambda_ 0.97 \
  --discount-factor 0.995 \
  --off-policy-batch-size 4 \
  --importance-weight-truncation-threshold 5 \
  --avg-net-momentum 0.995 \
  --max-replay-buffer-size 2000 \
  --num-sdn-samples 31 \
  --exp exp/test \
  --policy-dist Beta \
  --staircase False \
  --prioritize-replay False \
  --learning-rate 1e-3 \
  --regenerate-size 2000 \
  --entropy-cost-mult 1e-3 \
  --l2-reg 1e-5 \
  --parallelism 2 \
  --stats-file $(date +%s).stats.csv \
  --log-file $(date +%s).log \
  --hidden-size 64 \
  --use-lstm False \
  --share-network False \
  --summarize True \
  $@
