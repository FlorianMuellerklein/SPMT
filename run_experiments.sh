#!/bin/bash

python train.py --num_labeled 50000 --epochs 160 --batch_size 1024 --warmup 5

python train.py --num_labeled 4000 --epochs 750

python train.py --num_labeled 4000 --epochs 750 --mt

python train.py --num_labeled 4000 --epochs 750 --mt --spl