#!/bin/bash

rm -rfv data/pcqm4m-metagin/process*
clear

nohup time python3 -BuW ignore github/MetaGIN/model/data.py >data/pcqm4m-metagin/process.log &

sleep 4
tail -n 99 -f data/pcqm4m-metagin/process.log

