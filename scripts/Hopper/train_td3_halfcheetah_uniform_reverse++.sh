#!/bin/bash

source ../env/bin/activate
cd ..
python -W ignore -m src.main --exp_name td3_hopper_uniform_reverse++ --algo td3 --env hopper --replay_buffer_sampler uniform_reverse++ --train --seed $1 --snapshot_dir $2
