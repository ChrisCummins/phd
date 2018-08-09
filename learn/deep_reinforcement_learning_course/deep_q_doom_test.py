"""Unit tests for //learn/deep_reinforcement_learning_course/deep_q_doom.py."""
import pytest
import random
import sys
import typing
from absl import app
from absl import flags

from learn.deep_reinforcement_learning_course import deep_q_doom


FLAGS = flags.FLAGS


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


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
