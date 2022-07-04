#!/bin/bash

source ../env/bin/activate
cd ..
python -m src.main --exp_name dqn_pong_prioritized --algo dqn --env pong --replay_buffer_sampler prioritized --train --seed $1 --snapshot_dir $2
