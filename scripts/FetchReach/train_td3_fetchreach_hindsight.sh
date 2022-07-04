#!/bin/bash

source ../env/bin/activate
cd ..
python -W ignore -m src.main --exp_name td3_fetchreach_hindsight --replay_buffer_sampler hindsight --algo td3 --env fetchreach --train --seed $1 --snapshot_dir $2
