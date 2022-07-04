#!/bin/bash

source ../env/bin/activate
cd ..
python -W ignore -m src.main --exp_name td3_hopper_forward++_inc_grad_steps --algo td3 --env hopper --replay_buffer_sampler forward++ --train --seed $1 --snapshot_dir $2
