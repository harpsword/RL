#!/bin/bash

envs=("KungFuMaster-v0" "NameThisGame-v0")


for env in ${envs[@]} 
do
    echo "start train ${env}"
    date "+%Y-%m-%d %H:%M:%S"
    python -u ../main.py --env-name ${env} --num-processes 16 --lr 0.0001 --max-steps 50000000 > ../logs/${env}-fuxian-50M.log
    echo "finish "
    date "+%Y-%m-%d %H:%M:%S"
done
