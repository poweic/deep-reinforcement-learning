#!/bin/bash
./train.py \
  --base-dir /Data3/acer-offroad-icml2017/ \
  --max-global-steps 8000 \
  --estimator-type ACER \
  --log-file train.$(date +%s).log \
  --stats-file train.$(date +%s).stats.csv \
  --parallelism 1 \
  --game s_shape \
  --save-every-n-minutes 10 \
  --t-max 20 \
  --max-gradient 100.0 \
  --field-of-view 20 \
  --timestep 0.01 \
  --replay-ratio 0.250 \
  --avg-net-momentum 0.95 \
  --max-replay-buffer-size 5000 \
  --command-freq 5 \
  --discount-factor 0.99 \
  --n-agents-per-worker 8 \
  --vehicle-model-noise-level 2e-2 \
  --l2-reg 1e-4 \
  $@

# --learning-rate 1e-3 \
