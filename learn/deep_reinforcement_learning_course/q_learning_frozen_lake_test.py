"""Unit tests for :q_learning_frozen_lake.py."""
import gym
import pytest
import sys
import typing
from absl import app
from absl import flags

from learn.deep_reinforcement_learning_course import q_learning_frozen_lake


FLAGS = flags.FLAGS


def test_Train():
  """End-to-end test."""
  q_table = q_learning_frozen_lake.Train(
      gym.make("FrozenLake-v0"), total_episodes=10, max_steps=2,
      learning_rate=0.8, gamma=0.95, init_epsilon=1.0, min_epsilon=0.01,
      decay_rate=0.01, seed=0)
  assert q_table.shape == (16, 4)


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
