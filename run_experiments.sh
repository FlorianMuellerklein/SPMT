#!/bin/bash

python train.py --num_labeled 50000 --epochs 160

python train.py --num_labeled 4000 --epochs 350

python train.py --num_labeled 4000 --epochs 350 --mt

python train.py --num_labeled 4000 --epochs 350 --mt --spl

python train.py --num_labeled 4000 --epochs 350 --mt --spl --jsd