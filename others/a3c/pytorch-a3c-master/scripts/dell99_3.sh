#!/bin/bash

envs=("DoubleDunk-v0" "Enduro-v0" "FishingDerby-v0")


for env in ${envs[@]} 
do
    echo "start train ${env}"
    python -u ../main.py --env-name ${env} --num-processes 16 --lr 0.0001 --max-steps 50000000 > ../logs/${env}-fuxian-50M.log
done
