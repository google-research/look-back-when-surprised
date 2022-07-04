#!/bin/bash

source ../env/bin/activate
cd ..
python -W ignore -m src.main --exp_name dqn_lunarlander_prioritized --algo dqn --replay_buffer_sampler prioritized --env lunarlander --train --seed $1 --snapshot_dir $2