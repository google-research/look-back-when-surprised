#!/bin/bash

source ../env/bin/activate
cd ..
python -W ignore -m src.main --exp_name td3_halfcheetah_optimistic --algo td3 --env halfcheetah --replay_buffer_sampler optimistic --train --seed $1 --config $2
