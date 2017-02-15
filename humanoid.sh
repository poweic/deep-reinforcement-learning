#!/bin/bash
./go2.sh --exp exp/humanoid/trial-1/ --exp exp/humanoid/trial-1/train.log --dist Beta --learning-rate 1e-3 --render-every 100
./go2.sh --exp exp/humanoid/trial-2/ --exp exp/humanoid/trial-2/train.log --dist Beta --learning-rate 1e-3 --render-every 100
