#!/bin/bash

source ../env/bin/activate
cd ..
python -W ignore -m src.main --exp_name td3_hopper_hreverse++ --algo td3 --env hopper --replay_buffer_sampler hreverse++ --train --seed $1 --snapshot-dir $2
