"""Unit tests for //learn/deep_reinforcement_learning_course/deep_q_doom.py."""
import random

import pytest
from absl import flags

from labm8 import test
from learn.deep_reinforcement_learning_course import deep_q_doom

FLAGS = flags.FLAGS


@pytest.mark.xfail(reason='Vizdoom not working')
def test_environment():
  """Random action tests."""
  game, actions = deep_q_doom.CreateEnvironment()
  episodes = 10
  for i in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
      state = game.get_state()
      img = state.screen_buffer
      misc = state.game_variables
      action = random.choice(actions)
      reward = game.make_action(action)
      assert img
      assert misc
      assert reward
  game.close()


if __name__ == '__main__':
  test.Main()
