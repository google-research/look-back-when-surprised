#!/bin/bash

source ../env/bin/activate
cd ..
python -W ignore -m src.main --exp_name td3_halfcheetah_hreverse++_improved --algo td3 --env halfcheetah --replay_buffer_sampler hreverse++ --train --seed $1 --config $2
