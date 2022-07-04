#!/bin/bash

git pull

gpu=`nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits | sort -nr | sed "s/^[0-9]+,[ \t]*//" -r | head -1`
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_HOME=/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ramnathk/.mujoco/mujoco200/bin

source ../venv/bin/activate

cd ..
python -W ignore -m src.main --exp_name dqn_enduro_uniform --algo dqn --env enduro --train --seed $1 >/dev/null 2>&1 &
