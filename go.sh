#!/bin/bash
./train.py \
  --model_dir /Data3/acer-offroad/ \
  --parallelism 8 \
  --game s_shape \
  --save_every_n_minutes 5 \
  --t_max 8 \
  --timestep 0.001 \
  --replay_ratio 4 \
  --avg_net_momentum 0.995 \
  --max_replay_buffer_size 5000 \
  --learning_rate 2.5e-4 \
  --discount_factor 0.995 \
  --n_agents_per_worker 4 \
  --vehicle_model_noise_level 0.01 \
  --entropy_cost_mult 5e-4 \
  --l2_reg 5e-4 \
  $@
