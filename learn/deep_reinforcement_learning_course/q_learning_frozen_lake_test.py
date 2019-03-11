"""Unit tests for :q_learning_frozen_lake.py."""

import gym

from labm8 import app
from labm8 import test
from learn.deep_reinforcement_learning_course import q_learning_frozen_lake

FLAGS = app.FLAGS


def test_Train():
  """End-to-end test."""
  q_table = q_learning_frozen_lake.Train(
      gym.make("FrozenLake-v0"),
      total_episodes=10,
      max_steps=2,
      learning_rate=0.8,
      gamma=0.95,
      init_epsilon=1.0,
      min_epsilon=0.01,
      decay_rate=0.01,
      seed=0)
  assert q_table.shape == (16, 4)


if __name__ == '__main__':
  test.Main()
