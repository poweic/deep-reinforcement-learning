#!/bin/bash
./train.py \
  --model-dir /Data3/acer-offroad-icml2017/ \
  --parallelism 2 \
  --game s_shape \
  --save-every-n-minutes 10 \
  --t-max 20 \
  --max-gradient 100.0 \
  --field-of-view 20 \
  --timestep 0.001 \
  --replay-ratio 0.125 \
  --avg-net-momentum 0.95 \
  --max-replay-buffer-size 1000 \
  --command-freq 5 \
  --learning-rate 1e-3 \
  --discount-factor 0.99 \
  --n-agents-per-worker 8 \
  --vehicle-model-noise-level 2e-2 \
  --l2-reg 1e-4 \
  $@
