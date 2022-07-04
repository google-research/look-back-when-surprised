from src.launchers.dqn_cartpole import train_dqn_cartpole
from src.launchers.dqn_enduro import train_dqn_enduro
from src.launchers.dqn_breakout import train_dqn_breakout
from src.launchers.td3_halfcheetah import train_td3_halfcheetah
from src.launchers.dqn_lunarlander import train_dqn_lunarlander
from src.launchers.dqn_minigrid import train_dqn_minigrid
from src.launchers.dqn_mountaincar import train_dqn_mountaincar
from src.launchers.td3_dpendulum import train_td3_dpendulum
from src.launchers.dqn_pong import train_dqn_pong
from src.launchers.ddpg_fetchpush import train_ddpg_fetchpush
from src.launchers.td3_fetchreach import train_td3_fetchreach
from src.launchers.td3_fetchpush import train_td3_fetchpush
from src.launchers.td3_fetchslide import train_td3_fetchslide
from src.launchers.td3_fetchpickplace import train_td3_fetchpickplace
from src.launchers.dqn_acrobot import train_dqn_acrobot
from src.launchers.td3_ant import train_td3_ant
from src.launchers.td3_hopper import train_td3_hopper
from src.launchers.td3_walker import train_td3_walker
from src.launchers.td3_swimmer import train_td3_swimmer
from src.launchers.dqn_pendulum import train_dqn_pendulum

__all__ = ['train_dqn_cartpole', 'train_dqn_breakout', 'train_td3_halfcheetah',
           'train_dqn_lunarlander', 'train_dqn_minigrid', 'train_dqn_enduro',
           'train_dqn_mountaincar', 'train_td3_dpendulum', 'train_dqn_pong',
           'train_ddpg_fetchpush', 'train_td3_fetchpush', 'train_td3_fetchreach',
           'train_td3_fetchslide', 'train_td3_fetchpickplace', 'train_dqn_acrobot',
           'train_td3_ant', 'train_td3_hopper', 'train_td3_walker', 'train_td3_swimmer', 'train_dqn_pendulum']
