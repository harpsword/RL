#!/bin/bash

envs=("Asteroids-v0" "Atlantis-v0" "BankHeist-v0")


for env in ${envs[@]} 
do
    echo "start train ${env}"
    python -u ../main.py --env-name ${env} --num-processes 16 --lr 0.0001 --max-steps 50000000 > ../logs/${env}-fuxian-50M.log
done
