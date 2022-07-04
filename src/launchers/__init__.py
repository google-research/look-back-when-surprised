# Copyright 2022 Google LLC
# Copyright (c) 2019 Reinforcement Learning Working Group

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from src.launchers.dqn_cartpole import train_dqn_cartpole
from src.launchers.dqn_enduro import train_dqn_enduro
from src.launchers.td3_halfcheetah import train_td3_halfcheetah
from src.launchers.dqn_lunarlander import train_dqn_lunarlander
from src.launchers.td3_dpendulum import train_td3_dpendulum
from src.launchers.dqn_pong import train_dqn_pong
from src.launchers.td3_fetchreach import train_td3_fetchreach
from src.launchers.dqn_acrobot import train_dqn_acrobot
from src.launchers.td3_ant import train_td3_ant
from src.launchers.td3_hopper import train_td3_hopper

__all__ = ['train_dqn_cartpole', 'train_td3_halfcheetah',
           'train_dqn_lunarlander', 'train_dqn_enduro',
           'train_td3_dpendulum', 'train_dqn_pong',
           'train_td3_fetchreach', 'train_dqn_acrobot',
           'train_td3_ant', 'train_td3_hopper']
