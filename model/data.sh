#!/bin/bash
# Preprocess PCQM4Mv2 into data/pcqm4m-metagin/ (run from model/ or adjust paths).
# Paper: https://link.springer.com/article/10.1007/s11704-024-3784-y

set -euo pipefail

mkdir -p data/pcqm4m-metagin
rm -rfv data/pcqm4m-metagin/process*
clear

nohup time python3 -BuW ignore data.py >data/pcqm4m-metagin/process.log &

sleep 4
tail -n 99 -f data/pcqm4m-metagin/process.log
