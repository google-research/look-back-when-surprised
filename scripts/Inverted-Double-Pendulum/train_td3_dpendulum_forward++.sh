#!/bin/bash

source ../env/bin/activate
cd ..
python -W ignore -m src.main --exp_name td3_dpendulum_forward++_batch128 --algo td3 --env dpendulum --replay_buffer_sampler forward++ --train --seed $1 --snapshot_dir $2
