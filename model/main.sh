#!/bin/bash

rm -rfv train-*
clear

dev=0
for model in tiny321 tiny311 tiny311 tiny311 tiny311 tiny311; do
    logdir=train-$model-dev$dev
    mkdir -p $logdir

    CUDA_VISIBLE_DEVICES=$dev nohup time python3 -BuW ignore \
            github/MetaGIN/model/main.py --gnn $model --save $logdir \
            >$logdir/stdout.log 2>$logdir/stderr.log &

    let dev=dev+1
done

sleep 4
tail -n 99 -f train-*-dev1/*.log

