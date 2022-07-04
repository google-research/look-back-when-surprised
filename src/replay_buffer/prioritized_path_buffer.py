# Copyright 2022 Google LLC
# Copyright (c) 2019 Reinforcement Learning Working Group

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""A replay buffer that efficiently stores and can sample whole paths."""
import collections
import torch
import numpy as np
from src.replay_buffer import PathBuffer
from garage import StepType
import logging
from torch.autograd import Variable
import random
logging.basicConfig(level=logging.INFO)


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer(PathBuffer):
    """A replay buffer that stores and can sample whole episodes.

    This buffer only stores valid steps, and doesn't require paths to
    have a maximum length.

    Args:
        capacity_in_transitions (int): Total memory allocated for the buffer.
        env_spec (EnvSpec): Environment specification.

    """

    def __init__(self, capacity_in_transitions, env_spec=None,
                 e=0.01, a=0.6, beta=0.4, beta_increment_per_sampling=0.001):
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
        self.tree = SumTree(self._capacity)
        self.e = e
        self.a = a
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.idxs = []
        self.is_weight = None

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add_episode_batch(self, episodes, error=None):
        """Add a EpisodeBatch to the buffer.

        Args:
            episodes (EpisodeBatch): Episodes to add.

        """
        if self._env_spec is None:
            self._env_spec = episodes.env_spec
        env_spec = episodes.env_spec
        obs_space = env_spec.observation_space
        i = 0
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
            p = self._get_priority(error[i])
            self.tree.add(p, path)
            i += 1
            self.add_path(path)

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    @ property
    def _idxs(self):
        return self.idx

    @ property
    def _is_weight(self):
        return self.is_weight

    def sample_transitions(self, batch_size):
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            dict: A dict of arrays of shape (batch_size, flat_dim).

        """
        segment = self.tree.total() / batch_size
        priorities = []
        idx = []
        batch = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (id, p, data) = self.tree.get(s)
            batch.append(data)
            priorities.append(p)
            idx.append(id)
        self.idx = idx
        # logging.info(f"Sampled buffer: {idx}")
        sampling_probabilities = priorities / self.tree.total()
        self.is_weight = np.power(self.tree.n_entries * sampling_probabilities,
                                  -self.beta)
        self.is_weight /= self.is_weight.max()
        batch_dict = {}
        for key in batch[0].keys():
            batch_dict[key] = np.array([batch[i][key][0]
                                        for i in range(len(batch))])
        return batch_dict
