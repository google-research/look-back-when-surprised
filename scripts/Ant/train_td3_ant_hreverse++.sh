#!/bin/bash
#SBATCH --job-name=hreverse++
#SBATCH --output=../logs/td3_ant_hreverse++_%a.out
#SBATCH --error=../logs/td3_ant_hreverse++_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --array=5

source ../env/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..

sleep $[ ( $RANDOM % 10 )  + 1 ]s
python -W ignore -m src.main --exp_name td3_ant_hreverse++_128 --algo td3 --env ant --replay_buffer_sampler hreverse++ --train --seed $SLURM_ARRAY_TASK_ID --config $SLURM_TMPDIR
