#!/bin/bash
# Example: launch five tiny311 runs on the last N GPUs from this directory.
# Paper: https://link.springer.com/article/10.1007/s11704-024-3784-y

set -euo pipefail

rm -rfv train-*
clear

dev=$(nvidia-smi -L | wc -l)
for model in tiny311 ; do
    for trial in 1 2 3 4 5 ; do
        dev=$((dev - 1))
        logdir=train-$model-trial$trial
        mkdir -p "$logdir"

        CUDA_VISIBLE_DEVICES=$dev nohup time python3 -BuW ignore \
                main.py --model "$model" --save "$logdir" \
                >"$logdir/stdout.log" 2>"$logdir/stderr.log" &
    done
done

sleep 4
tail -n 99 -f "$logdir"/*.log
