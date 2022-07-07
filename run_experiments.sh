#!/bin/bash

python train.py --num_labeled 50000 --epochs 160

python train.py --num_labeled 4000 --epochs 1600

python train.py --semi_supervised --num_labeled 4000 --epochs 1600 --tempens

python train.py --semi_supervised --num_labeled 4000 --epochs 1600 --tempens --spl