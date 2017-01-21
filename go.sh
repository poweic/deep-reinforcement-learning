#!/bin/bash
./a3c/train.py --parallelism 1 --game maze2 --save_every_n_minutes 1 $@
