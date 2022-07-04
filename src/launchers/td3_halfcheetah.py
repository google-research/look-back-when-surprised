# Copyright 2022 Google LLC
# Copyright (c) 2019 Reinforcement Learning Working Group

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

#!/usr/bin/env python3

"""An example to train TD3 algorithm on InvertedDoublePendulum PyTorch."""
import torch
from torch.nn import functional as F

# from garage.np.exploration_policies import AddGaussianNoise
from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.np.exploration_policies import AddGaussianNoise
from garage.np.policies import UniformRandomPolicy
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from src.algos import TD3
from garage.torch.policies import DeterministicMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer
from garage.torch import set_gpu_mode
from src.replay_buffer import *


@wrap_experiment(snapshot_mode='none')
def train(ctxt=None):
    """Train TD3 with InvertedDoublePendulum-v2 environment.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
        determinism.
    """
    set_seed(config.seed)

    n_epochs = 500
    steps_per_epoch = 20
    sampler_batch_size = 250
    num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size
    buffer_batch_size = 100
    grad_steps_per_env_step = 100
    trainer = Trainer(ctxt)
    env = normalize(GymEnv('HalfCheetah-v2'))

    policy = DeterministicMLPPolicy(env_spec=env.spec,
                                    hidden_sizes=[256, 256],
                                    hidden_nonlinearity=F.relu,
                                    output_nonlinearity=torch.tanh)

    exploration_policy = AddGaussianNoise(env.spec,
                                          policy,
                                          total_timesteps=num_timesteps,
                                          max_sigma=0.1,
                                          min_sigma=0.1)

    uniform_random_policy = UniformRandomPolicy(env.spec)

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)
    use_custom_sampling_pdist = False
    if config.replay_buffer_sampler == 'reverse':
        replay_buffer = ReversePathBuffer(capacity_in_transitions=int(1e6))
    elif config.replay_buffer_sampler == 'reverse++':
        buffer_batch_size = 150
        use_custom_sampling_pdist = True
        replay_buffer = ReversePPPathBuffer(capacity_in_transitions=int(1e6))
    elif config.replay_buffer_sampler == 'forward++':
        use_custom_sampling_pdist = True
        replay_buffer = ForwardPPPathBuffer(capacity_in_transitions=int(1e6))
    elif config.replay_buffer_sampler == 'hreverse++':
        buffer_batch_size = 150
        use_custom_sampling_pdist = True
        replay_buffer = HReversePPPathBuffer(capacity_in_transitions=int(1e6))
    elif config.replay_buffer_sampler == 'uniform_reverse++':
        use_custom_sampling_pdist = True
        replay_buffer = UniformReversePPPathBuffer(
            capacity_in_transitions=int(1e6))
    elif config.replay_buffer_sampler == 'optimistic':
        use_custom_sampling_pdist = True
        replay_buffer = OptimisticPathBuffer(capacity_in_transitions=int(1e6))
    elif config.replay_buffer_sampler == 'hindsight':
        replay_buffer = HERReplayBuffer(
            replay_k=4, reward_fn=env.compute_reward,
            capacity_in_transitions=int(1e6), env_spec=env.spec)
    elif config.replay_buffer_sampler == 'prioritized':
        replay_buffer = PrioritizedReplayBuffer(
            capacity_in_transitions=int(1e6))
    else:
        replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    sampler = LocalSampler(agents=exploration_policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           worker_class=FragmentWorker)

    td3 = TD3(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              replay_buffer=replay_buffer,
              sampler=sampler,
              policy_optimizer=torch.optim.Adam,
              qf_optimizer=torch.optim.Adam,
              exploration_policy=exploration_policy,
              uniform_random_policy=uniform_random_policy,
              target_update_tau=0.005,
              discount=0.99,
              policy_noise_clip=0.5,
              policy_noise=0.2,
              policy_lr=1e-3,
              qf_lr=1e-3,
              steps_per_epoch=40,
              start_steps=1000,
              grad_steps_per_env_step=grad_steps_per_env_step,
              min_buffer_size=int(1e3),
              buffer_batch_size=100,
              use_custom_sampling_pdist=use_custom_sampling_pdist)

    if torch.cuda.is_available():
        set_gpu_mode(True)
        td3.to()
    trainer.setup(algo=td3, env=env)
    trainer.train(n_epochs=500, batch_size=buffer_batch_size)


def train_td3_halfcheetah(args):
    global config
    config = args
    train({'log_dir': args.snapshot_dir,
           'use_existing_dir': True})
