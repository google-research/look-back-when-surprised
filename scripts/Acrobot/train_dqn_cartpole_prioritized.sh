#!/bin/bash
#SBATCH --partition=main
#SBATCH --job-name=prioritized
#SBATCH --output=../logs/dqn_cartpole_prioritized_%a.out
#SBATCH --error=../logs/dqn_cartpole_prioritized_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --array=5

source ../env/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..
python -m src.main --exp_name dqn_cartpole_prioritized --replay_buffer_sampler prioritized --train --seed $SLURM_ARRAY_TASK_ID
