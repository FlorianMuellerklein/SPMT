#!/bin/bash

for ((i = 0 ; i < 10 ; i++)); do

    python train.py --num_labeled 50000 --epochs 160 --batch_size 1024 --lr 0.25 --wd 1e-4 --warmup 5 --lr 0.05 --model_name fully_supervised

    python train.py --num_labeled 4000 --epochs 100 --batch_size 1024 --model_name fully_supervised-4k

    #python train.py --num_labeled 4000 --mr --knn 3 --mr_lambda 75.

    #python train.py --num_labeled 4000 --mr --knn 3 --mr_lambda 75. --cpl

    python train.py --num_labeled 4000 --mr

    python train.py --num_labeled 4000 --mr --cpl

done
