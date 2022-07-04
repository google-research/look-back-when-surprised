#!/bin/bash

source ../env/bin/activate
cd ..
python -W ignore0 -m src.main --exp_name td3_hopper_uniform --algo td3 --env hopper --train --seed $1 --snapshot_dir $2
