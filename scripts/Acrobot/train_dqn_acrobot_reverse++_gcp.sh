#!/bin/bash

git pull

gpu=`nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits | sort -nr | sed "s/^[0-9]+,[ \t]*//" -r | head -1`
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_HOME=/usr/local/cuda/bin
export MPLBACKEND=TKAgg

source ../venv/bin/activate

cd ..
echo $1

python -W ignore -m src.main --exp_name dqn_acrobot_reverse++_newest --algo dqn --replay_buffer_sampler reverse++ --env acrobot --train --seed $1 --snapshot_dir /mnt/disk1/ramnathk/ >/dev/null 2>&1 &
