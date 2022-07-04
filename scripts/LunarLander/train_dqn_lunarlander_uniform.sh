#!/bin/bash

source ../env/bin/activate
cd ..
python -m src.main --exp_name dqn_lunarlander_uniform --algo dqn --env lunarlander --train --seed $1 --snapshot_dir $2
