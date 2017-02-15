#!/bin/bash

# Beta v.s Gaussian 1
./pendulum.py --exp beta-trial-1     --logfile log/$(date +%s).log --dist Beta
./pendulum.py --exp Gaussian-trial-1 --logfile log/$(date +%s).log --dist Gaussian

# Beta v.s Gaussian 2
./pendulum.py --exp beta-trial-2     --logfile log/$(date +%s).log --dist Beta
./pendulum.py --exp Gaussian-trial-2 --logfile log/$(date +%s).log --dist Gaussian

# Beta v.s Gaussian 3
./pendulum.py --exp beta-trial-3     --logfile log/$(date +%s).log --dist Beta
./pendulum.py --exp Gaussian-trial-3 --logfile log/$(date +%s).log --dist Gaussian

# Beta v.s Gaussian 4
./pendulum.py --exp beta-trial-4     --logfile log/$(date +%s).log --dist Beta
./pendulum.py --exp Gaussian-trial-4 --logfile log/$(date +%s).log --dist Gaussian

# Beta v.s Gaussian 5
./pendulum.py --exp beta-trial-5     --logfile log/$(date +%s).log --dist Beta
./pendulum.py --exp Gaussian-trial-5 --logfile log/$(date +%s).log --dist Gaussian

# Beta v.s Gaussian 6
./pendulum.py --exp beta-trial-6     --logfile log/$(date +%s).log --dist Beta
./pendulum.py --exp Gaussian-trial-6 --logfile log/$(date +%s).log --dist Gaussian

# Beta v.s Gaussian 7
./pendulum.py --exp beta-trial-7     --logfile log/$(date +%s).log --dist Beta
./pendulum.py --exp Gaussian-trial-7 --logfile log/$(date +%s).log --dist Gaussian
