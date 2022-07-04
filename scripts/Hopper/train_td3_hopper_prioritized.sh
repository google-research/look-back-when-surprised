#!/bin/bash

source ../env/bin/activate
cd ..
python -W ignore -m src.main --exp_name td3_hopper_prioritized --algo td3 --env hopper --replay_buffer_sampler prioritized --train --seed $1 --config $2
