#!/usr/bin/env python3
"""An example to train a task with DQN algorithm.

Here it creates a gym environment Breakout, and trains a DQN with 50k steps.
"""

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.exploration_policies import EpsilonGreedyPolicy
from src.replay_buffer import PathBuffer, ReversePathBuffer,\
    ForwardPathBuffer, ReversePPPathBuffer, OptimisticPathBuffer,\
    PessimisticPathBuffer, HERReplayBuffer, PrioritizedReplayBuffer,\
    HReversePPPathBuffer, ForwardPPPathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from src.algos import DQN
from garage.torch.policies import DiscreteQFArgmaxPolicy
from garage.torch.q_functions import DiscreteCNNQFunction
from garage.trainer import Trainer
from garage.envs.wrappers.noop import Noop
import gym
from garage.envs.wrappers.max_and_skip import MaxAndSkip
from garage.envs.wrappers.episodic_life import EpisodicLife
from garage.envs.wrappers.fire_reset import FireReset
from garage.envs.wrappers.grayscale import Grayscale
from garage.envs.wrappers.resize import Resize
from garage.envs.wrappers.clip_reward import ClipReward
from garage.envs.wrappers.stack_frames import StackFrames
import torch
import gc
import numpy as np
from garage.torch import set_gpu_mode
import psutil
import math


@wrap_experiment
def train(ctxt=None):
    """Train DQN with Breakout-v0 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
    """
    set_seed(config.seed)
    trainer = Trainer(ctxt)

    num_workers = psutil.cpu_count(logical=False)
    n_epochs = math.ceil(
        int(10e6) / (20 *
                     500))
    steps_per_epoch = 20
    use_custom_sampling_pdist = False
    sampler_batch_size = 500
    qf_lr = 1e-4
    min_buffer_size = int(1e4)
    buffer_batch_size = 32
    num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size
    env = gym.make('EnduroNoFrameskip-v4')
    env = Noop(env, noop_max=30)
    env = MaxAndSkip(env, skip=4)
    env = EpisodicLife(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireReset(env)
    env = Grayscale(env)
    env = Resize(env, 84, 84)
    env = ClipReward(env)
    env = StackFrames(env, 4, axis=0)
    env = GymEnv(env, is_image=True)

    if config.replay_buffer_sampler == 'reverse':
        replay_buffer = ReversePathBuffer(capacity_in_transitions=int(1e4))
    elif config.replay_buffer_sampler == 'reverse++':
        use_custom_sampling_pdist = True
        replay_buffer = ReversePPPathBuffer(capacity_in_transitions=int(1e4))
    elif config.replay_buffer_sampler == 'forward++':
        use_custom_sampling_pdist = True
        replay_buffer = ForwardPPPathBuffer(capacity_in_transitions=int(1e4))
    elif config.replay_buffer_sampler == 'hreverse++':
        use_custom_sampling_pdist = True
        replay_buffer = HReversePPPathBuffer(capacity_in_transitions=int(1e4))
    elif config.replay_buffer_sampler == 'optimistic':
        use_custom_sampling_pdist = True
        replay_buffer = OptimisticPathBuffer(capacity_in_transitions=int(1e4))
    elif config.replay_buffer_sampler == 'pessimistic':
        use_custom_sampling_pdist = True
        replay_buffer = PessimisticPathBuffer(capacity_in_transitions=int(1e4))
    elif config.replay_buffer_sampler == 'hindsight':
        replay_buffer = HERReplayBuffer(
            replay_k=4, reward_fn=env.compute_reward,
            capacity_in_transitions=int(1e4), env_spec=env.spec)
    elif config.replay_buffer_sampler == 'prioritized':
        replay_buffer = PrioritizedReplayBuffer(
            capacity_in_transitions=int(1e4))
    else:
        replay_buffer = PathBuffer(capacity_in_transitions=int(1e4))

    qf = DiscreteCNNQFunction(
        env_spec=env.spec,
        image_format='NCHW',
        hidden_channels=(32, 64, 64),
        kernel_sizes=(8, 4, 3),
        strides=(4, 2, 1),
        hidden_w_init=(
            lambda x: torch.nn.init.orthogonal_(x, gain=np.sqrt(2))),
        hidden_sizes=(512, ))

    policy = DiscreteQFArgmaxPolicy(env_spec=env.spec, qf=qf)
    exploration_policy = EpsilonGreedyPolicy(
        env_spec=env.spec,
        policy=policy,
        total_timesteps=num_timesteps,
        max_epsilon=1.0,
        min_epsilon=0.01,
        decay_ratio=0.1)

    n_workers = psutil.cpu_count(logical=False)
    sampler = LocalSampler(agents=exploration_policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           worker_class=FragmentWorker,
                           n_workers=n_workers)

    algo = DQN(env_spec=env.spec,
               policy=policy,
               qf=qf,
               exploration_policy=exploration_policy,
               replay_buffer=replay_buffer,
               sampler=sampler,
               steps_per_epoch=steps_per_epoch,
               qf_lr=1e-4,
               clip_gradient=10,
               discount=0.99,
               min_buffer_size=int(1e4),
               n_train_steps=125,
               target_update_freq=2,
               buffer_batch_size=32,
               use_custom_sampling_pdist=use_custom_sampling_pdist)

    set_gpu_mode(False)
    torch.set_num_threads(20)
    if torch.cuda.is_available():
        set_gpu_mode(True)
        algo.to()

    trainer.setup(algo, env)

    trainer.train(n_epochs=n_epochs, batch_size=sampler_batch_size)
    env.close()


def train_dqn_enduro(args):
    global config
    config = args
    torch.cuda.empty_cache()
    gc.collect()
    train({'log_dir': args.snapshot_dir,
           'use_existing_dir': True})
