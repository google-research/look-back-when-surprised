#!/bin/bash

source ../env/bin/activate
cd ..
python -W ignore -m src.main --exp_name td3_halfcheetah_uniform --algo td3 --env halfcheetah --train --seed $1 --snapshot_dir $2
