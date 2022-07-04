#!/bin/bash
#SBATCH --partition=main
#SBATCH --job-name=uniform_reverse++
#SBATCH --output=../logs/dqn_acrobot_uniform_reverse++_%a.out
#SBATCH --error=../logs/dqn_acrobot_uniform_reverse++_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --array=2

source ../env/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..
sleep $[ ( $RANDOM % 10 )  + 1 ]s
python -m src.main --exp_name dqn_acrobot_uniform_reverse++ --replay_buffer_sampler uniform_reverse++ --env acrobot --train --seed $SLURM_ARRAY_TASK_ID --config $SLURM_TMPDIR
