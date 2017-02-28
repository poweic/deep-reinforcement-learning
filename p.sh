#!/bin/bash

# Beta v.s Gaussian 1
# ./pendulum.py --exp beta-trial-1     --logfile log/$(date +%s).log --dist Beta
echo "Gaussian trial 1"
./pendulum.py --exp Gaussian-trial-softclip-1 --logfile log/Pendulum/softclip-$(date +%s).log --dist Gaussian > log/Pendulum/softclip-$(date +%s).train.log 2>&1 &
sleep 2

# Beta v.s Gaussian 2
# ./pendulum.py --exp beta-trial-2     --logfile log/$(date +%s).log --dist Beta
echo "Gaussian trial 2"
./pendulum.py --exp Gaussian-trial-softclip-2 --logfile log/Pendulum/softclip-$(date +%s).log --dist Gaussian > log/Pendulum/softclip-$(date +%s).train.log 2>&1 &
sleep 2

# Beta v.s Gaussian 3
# ./pendulum.py --exp beta-trial-3     --logfile log/$(date +%s).log --dist Beta
echo "Gaussian trial 3"
./pendulum.py --exp Gaussian-trial-softclip-3 --logfile log/Pendulum/softclip-$(date +%s).log --dist Gaussian > log/Pendulum/softclip-$(date +%s).train.log 2>&1 &
sleep 2

# Beta v.s Gaussian 4
# ./pendulum.py --exp beta-trial-4     --logfile log/$(date +%s).log --dist Beta
echo "Gaussian trial 4"
./pendulum.py --exp Gaussian-trial-softclip-4 --logfile log/Pendulum/softclip-$(date +%s).log --dist Gaussian > log/Pendulum/softclip-$(date +%s).train.log 2>&1 &
sleep 2

# Beta v.s Gaussian 5
# ./pendulum.py --exp beta-trial-5     --logfile log/$(date +%s).log --dist Beta
echo "Gaussian trial 5"
./pendulum.py --exp Gaussian-trial-softclip-5 --logfile log/Pendulum/softclip-$(date +%s).log --dist Gaussian > log/Pendulum/softclip-$(date +%s).train.log 2>&1 &
sleep 2

# Beta v.s Gaussian 6
# ./pendulum.py --exp beta-trial-6     --logfile log/$(date +%s).log --dist Beta
echo "Gaussian trial 6"
./pendulum.py --exp Gaussian-trial-softclip-6 --logfile log/Pendulum/softclip-$(date +%s).log --dist Gaussian > log/Pendulum/softclip-$(date +%s).train.log 2>&1 &
sleep 2

# Beta v.s Gaussian 7
# ./pendulum.py --exp beta-trial-7     --logfile log/$(date +%s).log --dist Beta
echo "Gaussian trial 7"
./pendulum.py --exp Gaussian-trial-softclip-7 --logfile log/Pendulum/softclip-$(date +%s).log --dist Gaussian > log/Pendulum/softclip-$(date +%s).train.log 2>&1 &
sleep 2

# Beta v.s Gaussian 8
# ./pendulum.py --exp beta-trial-8     --logfile log/$(date +%s).log --dist Beta
echo "Gaussian trial 8"
./pendulum.py --exp Gaussian-trial-softclip-8 --logfile log/Pendulum/softclip-$(date +%s).log --dist Gaussian > log/Pendulum/softclip-$(date +%s).train.log 2>&1 &
sleep 2
