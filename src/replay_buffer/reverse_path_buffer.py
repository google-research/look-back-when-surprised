# Copyright 2022 Google LLC
# Copyright (c) 2019 Reinforcement Learning Working Group

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""A replay buffer that efficiently stores and can sample whole paths."""
import collections

import numpy as np
from src.replay_buffer import PathBuffer
from garage import StepType
import logging
logging.basicConfig(level=logging.INFO)


class ReversePathBuffer(PathBuffer):
    """A replay buffer that stores and can sample whole episodes.

    This buffer only stores valid steps, and doesn't require paths to
    have a maximum length.

    Args:
        capacity_in_transitions (int): Total memory allocated for the buffer.
        env_spec (EnvSpec): Environment specification.

    """

    def __init__(self, capacity_in_transitions, env_spec=None):
        self._capacity = capacity_in_transitions
        self._env_spec = env_spec
        self._transitions_stored = 0
        self._first_idx_of_next_path = 0
        # Each path in the buffer has a tuple of two ranges in
        # self._path_segments. If the path is stored in a single contiguous
        # region of the buffer, the second range will be range(0, 0).
        # The "left" side of the deque contains the oldest episode.
        self._path_segments = collections.deque()
        self._buffer = {}
        self.offset = 0

    def add_episode_batch(self, episodes):
        """Add a EpisodeBatch to the buffer.

        Args:
            episodes (EpisodeBatch): Episodes to add.

        """
        if self._env_spec is None:
            self._env_spec = episodes.env_spec
        env_spec = episodes.env_spec
        obs_space = env_spec.observation_space
        for eps in episodes.split():
            terminals = np.array([
                step_type == StepType.TERMINAL for step_type in eps.step_types
            ],
                dtype=bool)
            path = {
                'observations': obs_space.flatten_n(eps.observations),
                'next_observations':
                obs_space.flatten_n(eps.next_observations),
                'actions': env_spec.action_space.flatten_n(eps.actions),
                'rewards': eps.rewards.reshape(-1, 1),
                'terminals': terminals.reshape(-1, 1),
            }
            self.add_path(path)
        self.offset = 0

    def sample_path(self):
        """Sample a single path from the buffer.

        Returns:
            path: A dict of arrays of shape (path_len, flat_dim).

        """
        path_idx = -1
        first_seg, second_seg = self._path_segments[path_idx]
        first_seg_indices = np.arange(first_seg.start, first_seg.stop)
        second_seg_indices = np.arange(second_seg.start, second_seg.stop)
        indices = np.concatenate([first_seg_indices, second_seg_indices])
        path = {key: buf_arr[indices] for key, buf_arr in self._buffer.items()}
        return path

    def sample_transitions(self, batch_size):
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            dict: A dict of arrays of shape (batch_size, flat_dim).

        """
        if self.n_transitions_stored == self._capacity:
            idx = np.array([(self.n_transitions_stored - self.offset - i)
                           for i in range(1, batch_size+1)])
        else:
            idx = np.array([(self.n_transitions_stored - self.offset - i)
                           for i in range(batch_size)])
        self.offset += batch_size
        return {key: buf_arr[idx] for key, buf_arr in self._buffer.items()}
