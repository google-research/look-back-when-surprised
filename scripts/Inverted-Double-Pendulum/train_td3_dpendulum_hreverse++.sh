#!/bin/bash

source ../env/bin/activate
cd ..
python -W ignore -m src.main --exp_name td3_dpendulum_hreverse++ --algo td3 --env dpendulum --replay_buffer_sampler hreverse++ --train --seed $1 --snapshot_dir $2
