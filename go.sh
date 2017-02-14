#!/bin/bash
# secret sauce is beta distribution with large learning rate
./mountain_car.py \
  --render-every 1 \
  --dist StudentT \
  --learning-rate 0.1 \
  --exp student-t \
  --max-steps 10000 \
  --logfile exp/$(date +%s).log \
  $@
