# Copyright 2022 Google LLC
# Copyright (c) 2019 Reinforcement Learning Working Group

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from pyvirtualdisplay import Display
import wandb
import argparse
from src.trainer import Trainer, Tester
from glob import glob
import logging
import time
logging.basicConfig(level=logging.INFO)
Display().start()


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser('Experience Replay RL')
    # General
    parser.add_argument('--algo', type=str,
                        choices=['dqn', 'ddpg', 'td3'],
                        default='dqn',
                        help='Algorithm to be used (default: dqn).')
    parser.add_argument('--env', type=str,
                        choices=['cartpole', 'pong', 'fetchreach', 'lunarlander', 'halfcheetah',
                                 'enduro', 'dpendulum', 'acrobot', 'ant', 'hopper'],
                        default='cartpole',
                        help='Environment to be used (default: cartpole).')
    parser.add_argument('--env_sub_name', type=str,
                        choices=['Surround', 'Freeway', 'IceHockey'],
                        default='Surround',
                        help='Sub-Environment of ATARI to be used (default: Surround).')
    parser.add_argument('--train', action='store_true',
                        help='Train the model (default: False).')
    parser.add_argument('--vis', action='store_true',
                        help='Visualize policy (default: False).')
    parser.add_argument('--replay_buffer_sampler', type=str,
                        choices=['uniform', 'forward', 'reverse',
                                 'optimistic', 'pessimistic', 'hindsight',
                                 'prioritized', 'reverse++', 'hreverse++',
                                 'cumulative_reverse++', 'se_reverse++', 'uniform_reverse++',
                                 'incremental_reverse++', 'forward++', 'hforward++', 'freverse++',
                                 'incremental_freverse++'],
                        default='uniform',
                        help='Typle of sampling scheme in replay buffer to be used (default: uniform).')
    parser.add_argument('--epochs', type=int, default=600,
                        help='Number of epochs to run for (default: 600)')
    parser.add_argument('--batch_size_per_task', type=int, default=1024,
                        help='Batch size per task (default: 1024)')
    parser.add_argument('--snapshot_dir', type=str, default='config/',
                        help='Path to save the log and iteration snapshot.')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name'
                        '(default: None).')
    parser.add_argument('--policy_optimizer_lr', type=float, default=1e-3,
                        help='Learning rate of policy optimizer (default: 1e-3)')
    parser.add_argument('--encoder_optimizer_lr', type=float, default=1e-3,
                        help='Learning rate of encoder optimizer (default: 1e-3)')
    parser.add_argument('--inference_optimizer_lr', type=float, default=1e-3,
                        help='Learning rate of inference optimizer (default: 1e-3)')
    parser.add_argument('--policy_ent_coeff', type=float, default=1e-3,
                        help='Policy_ent_coeff (default: 1e-3)')
    parser.add_argument('--encoder_ent_coeff', type=float, default=1e-3,
                        help='Encoder_ent_coeff (default: 1e-3)')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--seed', type=int, default=1,
                      help='random seed (default: 1)')

    args, unknownargs = parser.parse_known_args()

    args.snapshot_dir = f'{args.snapshot_dir}{args.algo}/{args.env}/{args.replay_buffer_sampler}/{time.strftime("%Y-%m-%d-%H%M%S")}'
    return args


if __name__ == '__main__':
    args = parse_args()
    logging.info(args)
    wandb.init(project='Replay-Buffer-RL', entity='project', config=args,
                settings=wandb.Settings(start_method='spawn'),
                name=args.exp_name, reinit=True)

    if args.train:
        Trainer(args)
    wandb.finish()
