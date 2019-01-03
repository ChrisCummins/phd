"""Unit tests for //experimental/compilers/random_opt/environments.py."""

import gym
import pytest

from experimental.compilers.random_opt import environments
from labm8 import test
from absl import flags
FLAGS = flags.FLAGS


@pytest.mark.parametrize('environment', environments.ENVIRONMENTS)
def test_environments(environment: str):
  """Simple black box test of environment."""
  env = gym.make(environment)
  env.seed(0)
  env.reset()
  # Take a random step. We're doing a full test, just checking that we can at
  # least take a step without error.
  env.render()
  env.step(env.action_space.sample())
  env.render()


if __name__ == '__main__':
  test.Main()
