#!/bin/bash

envs=("Assault-v0" "Asterix-v0" "Asteriods-v0" "Atlantis-v0")


for env in ${envs[@]} 
do
    echo "start train ${env}"
    python main.py --env-name ${env} --num-processes 16 --lr 0.0001 --max-steps 50000000 > logs/${env}-fuxian-50M.log
done
