#!/bin/bash
# secret sauce is beta distribution with large learning rate
./mountain_car.py \
  --render-every 1 \
  --dist Beta \
  --learning-rate 0.1 \
  --max-steps 10000 \
  --logfile exp/$(date +%s).log \
  $@
