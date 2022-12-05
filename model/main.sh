#!/bin/bash

rm -rfv train-*
clear

dev=`nvidia-smi -L | wc -l`
for model in tiny311 ; do
    for trial in 1 2 3 4 5 ; do
        let dev=dev-1
        logdir=train-$model-trial$trial
        mkdir -p $logdir

        CUDA_VISIBLE_DEVICES=$dev nohup time python3 -BuW ignore \
                github/MetaGIN/model/main.py --model $model --save $logdir \
                >$logdir/stdout.log 2>$logdir/stderr.log &
    done
done

sleep 4
tail -n 99 -f $logdir/*.log

