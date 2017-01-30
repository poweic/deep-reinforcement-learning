#!/bin/bash
./train.py \
  --model_dir /Data3/acer-offroad/ \
  --parallelism 2 \
  --game line \
  --save_every_n_minutes 1 \
  --t_max 5 \
  --timestep 0.01 \
  --replay_ratio 4 \
  --avg_net_momentum 0.9 \
  --max_replay_buffer_size 5000 \
  --learning_rate 2e-4 \
  --downsample 10 \
  --discount_factor 0.995 \
  --n_agents_per_worker 8 \
  --vehicle_model_noise_level 0.2 \
  --entropy_cost_mult 1e-2 \
  --l2_reg 5e-4 \
  $@
