#!/bin/bash
./train.py \
  --model_dir /Data3/acer-offroad/ \
  --parallelism 4 \
  --game maze4 \
  --save_every_n_minutes 5 \
  --t_max 20 \
  --learning_rate 2e-4 \
  --downsample 10 \
  --discount_factor 0.995 \
  --n_agents_per_worker 32 \
  --vehicle_model_noise_level 0.2 \
  --entropy_cost_mult 1e-2 \
  --l2_reg 5e-4 \
  $@
