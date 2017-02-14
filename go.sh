#!/bin/bash
# secret sauce is beta distribution with large learning rate
# ./train.py --game Humanoid-v1 --record-video True --exp exp/humanoid/trial-2 --policy-dist Beta
# ./train.py --game Humanoid-v1 --record-video True --exp exp/humanoid/trial-2 --policy-dist Gaussian
./train.py --game MountainCarContinuous-v0 --record-video True --exp exp/humanoid/trial-2 --policy-dist Gaussian

"""
./mountain_car.py \
  --render-every 1 \
  --dist StudentT \
  --learning-rate 0.1 \
  --exp student-t \
  --max-steps 10000 \
  --logfile exp/$(date +%s).log \
  $@
"""
