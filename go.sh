#!/bin/bash
./train.py \
  --model_dir /Data3/acer-offroad/ \
  --parallelism 4 \
  --game maze4 \
  --save_every_n_minutes 5 \
  --t_max 8 \
  --field_of_view 40 \
  --timestep 0.001 \
  --replay_ratio 4 \
  --avg_net_momentum 0.995 \
  --max_replay_buffer_size 5000 \
  --command_freq 20 \
  --learning_rate 2e-5 \
  --discount_factor 0.93 \
  --n_agents_per_worker 8 \
  --vehicle_model_noise_level 0.20 \
  --entropy_cost_mult 1e-3 \
  --l2_reg 5e-4 \
  $@
