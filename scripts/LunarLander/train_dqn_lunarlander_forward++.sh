#!/bin/bash

source ../env/bin/activate
cd ..
python -W ignore -m src.main --exp_name dqn_lunarlander_forward++ --algo dqn --replay_buffer_sampler forward++ --env lunarlander --train --seed $1 --snapshot_dir $2