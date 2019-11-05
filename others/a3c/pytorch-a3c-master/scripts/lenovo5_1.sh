#!/bin/bash

envs=("Freeway-v0" "Frostbite-v0" "Gopher-v0")


for env in ${envs[@]} 
do
    echo "start train ${env}"
    python -u ../main.py --env-name ${env} --num-processes 16 --lr 0.0001 --max-steps 50000000 > ../logs/${env}-fuxian-50M.log
done
