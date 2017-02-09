#!/bin/bash
./train.py \
  --model-dir /Data3/acer-offroad/ \
  --parallelism 2 \
  --game line \
  --save-every-n-minutes 60 \
  --t-max 4 \
  --max-gradient 100.0 \
  --field-of-view 40 \
  --timestep 0.001 \
  --replay-ratio 8 \
  --avg-net-momentum 0.995 \
  --max-replay-buffer-size 5000 \
  --command-freq 10 \
  --learning-rate 1e-3 \
  --discount-factor 0.99 \
  --n-agents-per-worker 8 \
  --vehicle-model-noise-level 1e-2 \
  --l2-reg 1e-4 \
  $@
