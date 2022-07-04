#!/bin/bash

source ../env/bin/activate
cd ..
python -m src.main --exp_name dqn_lunarlander_reverse --algo dqn --env lunarlander --train --replay_buffer_sampler reverse --seed $1 --snapshot_dir $2
