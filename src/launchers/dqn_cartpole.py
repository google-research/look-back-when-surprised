# Copyright 2022 Google LLC
# Copyright (c) 2019 Reinforcement Learning Working Group

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

#!/usr/bin/env python3

"""An example to train a task with DQN algorithm.

Here it creates a gym environment CartPole, and trains a DQN with 50k steps.
"""

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.exploration_policies import EpsilonGreedyPolicy
from src.replay_buffer import *
from garage.sampler import FragmentWorker, LocalSampler
from src.algos import DQN
from garage.torch.policies import DiscreteQFArgmaxPolicy
from garage.torch.q_functions import DiscreteMLPQFunction
from garage.trainer import Trainer
from garage.torch import set_gpu_mode
import torch


@wrap_experiment
def train(ctxt=None):
    """Train DQN with CartPole-v0 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
    """
    set_seed(config.seed)
    runner = Trainer(ctxt)

    n_epochs = 100
    steps_per_epoch = 10
    use_custom_sampling_pdist = False
    sampler_batch_size = 512
    qf_lr = 5e-5
    min_buffer_size = int(1e4)
    buffer_batch_size = 64
    num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size
    env = GymEnv('CartPole-v0')
    if config.replay_buffer_sampler == 'reverse':
        replay_buffer = ReversePathBuffer(capacity_in_transitions=int(1e6))
    elif config.replay_buffer_sampler == 'reverse++':
        use_custom_sampling_pdist = True
        replay_buffer = ReversePPPathBuffer(capacity_in_transitions=int(1e6))
    elif config.replay_buffer_sampler == 'hreverse++':
        use_custom_sampling_pdist = True
        replay_buffer = HReversePPPathBuffer(capacity_in_transitions=int(1e6))
    elif config.replay_buffer_sampler == 'forward++':
        use_custom_sampling_pdist = True
        replay_buffer = ForwardPPPathBuffer(capacity_in_transitions=int(1e6))
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
    qf = DiscreteMLPQFunction(env_spec=env.spec, hidden_sizes=(8, 5))
    policy = DiscreteQFArgmaxPolicy(env_spec=env.spec, qf=qf)
    exploration_policy = EpsilonGreedyPolicy(env_spec=env.spec,
                                             policy=policy,
                                             total_timesteps=num_timesteps,
                                             max_epsilon=1.0,
                                             min_epsilon=0.01,
                                             decay_ratio=0.4)
    sampler = LocalSampler(agents=exploration_policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           worker_class=FragmentWorker)
    algo = DQN(env_spec=env.spec,
               policy=policy,
               qf=qf,
               exploration_policy=exploration_policy,
               replay_buffer=replay_buffer,
               sampler=sampler,
               steps_per_epoch=steps_per_epoch,
               use_custom_sampling_pdist=use_custom_sampling_pdist,
               qf_lr=qf_lr,
               discount=0.9,
               min_buffer_size=min_buffer_size,
               n_train_steps=500,
               target_update_freq=30,
               buffer_batch_size=buffer_batch_size)

    if torch.cuda.is_available():
        set_gpu_mode(True)
        algo.to()
    runner.setup(algo, env)
    runner.train(
        n_epochs=n_epochs, batch_size=sampler_batch_size)

    env.close()


def train_dqn_cartpole(args):
    global config
    config = args
    train({'log_dir': args.snapshot_dir,
           'use_existing_dir': True})
