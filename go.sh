#!/bin/bash
./a3c/train.py --parallelism 1 --game maze3 --save_every_n_minutes 5 -t_max 50 $@
