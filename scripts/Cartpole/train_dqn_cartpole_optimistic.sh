#!/bin/bash
#SBATCH --partition=main
#SBATCH --job-name=optimistic
#SBATCH --output=../logs/dqn_cartpole_optimistic_%a.out
#SBATCH --error=../logs/dqn_cartpole_optimistic_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --array=2-5

source ../env/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..

sleep $[ ( $RANDOM % 10 )  + 1 ]s
python -m src.main --exp_name dqn_cartpole_optimistic --replay_buffer_sampler optimistic --train --seed $SLURM_ARRAY_TASK_ID --snapshot_dir $SLURM_TMPDIR
