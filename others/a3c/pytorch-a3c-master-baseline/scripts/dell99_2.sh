#!/bin/bash

env="Freeway-v0"
seeds=(111 222 333)


for seed in ${seeds[@]}
do
    echo "start training, seed:${seed}"
    date "+%Y-%m-%d %H:%M:%S"
    python -u ../main.py --env-name ${env} --num-processes 16 --lr 0.0001 --max-steps 50000000 --seed ${seed} > ../logs/${env}-baseline-50M-seed-${seed}.log
    date "+%Y-%m-%d %H:%M:%S"
    echo "finish"
    echo "----------------"
done
