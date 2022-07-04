#!/bin/bash
#SBATCH --job-name=uniform
#SBATCH --output=../logs/dqn_cartpole_uniform_%a.out
#SBATCH --error=../logs/dqn_cartpole_uniform_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --array=4

source ../env/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..
python -m src.main --exp_name dqn_cartpole_uniform --train --seed $SLURM_ARRAY_TASK_ID
