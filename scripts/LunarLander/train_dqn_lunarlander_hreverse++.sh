#!/bin/bash

source ../env/bin/activate
cd ..
python -m src.main --exp_name ddpg_lunarlander_hreverse++ --algo dqn --env lunarlander --train --replay_buffer_sampler hreverse++ --seed $1 --snapshot_dir $2
