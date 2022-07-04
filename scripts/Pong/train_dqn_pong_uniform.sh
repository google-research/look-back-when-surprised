#!/bin/bash

source ../env/bin/activate
cd ..
python -m src.main --exp_name dqn_pong_uniform --algo dqn --env pong --train --seed $1 --snapshot_dir $2
