"""Deep Q learning for the game Doom.

See: https://medium.freecodecamp.org/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8
"""
import pathlib
import typing

import skimage
# FIXME(cec): Vizdoom currently disabled.
# import vizdoom
from absl import app
from absl import flags
from absl import logging

from labm8 import bazelutil

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'doom_config',
    str(
        bazelutil.DataPath(
            'phd/learn/deep_reinforcement_learning_course/data/doom_config.cfg')
    ), 'Path to Doom config file.')

flags.DEFINE_string(
    'doom_scenario',
    str(
        bazelutil.DataPath(
            'phd/learn/deep_reinforcement_learning_course/data/doom_scenario.wad'
        )), 'Path to Doom scenario file.')


def CreateEnvironment(config_path: typing.Optional[pathlib.Path] = None,
                      scenario_path: typing.Optional[pathlib.Path] = None
                     ) -> typing.Tuple[None, typing.List[typing.List[int]]]:
  """Create the Doom game environment.

  Returns:
     A tuple of the environment and action space.
  """
  config_path = config_path or FLAGS.doom_config
  scenario_path = scenario_path or FLAGS.doom_scenario

  game = vizdoom.DoomGame()
  game.load_config(config_path)
  game.set_doom_scenario_path(scenario_path)
  game.init()

  left = [1, 0, 0]
  right = [0, 1, 0]
  shoot = [0, 0, 1]
  possible_actions = [left, right, shoot]

  return game, possible_actions


def PreprocessFrame(frame):
  # Crop the screen (remove the roof because it contains no information).
  cropped_frame = frame[30:-10, 30:-30]

  # Normalize Pixel Values.
  normalized_frame = cropped_frame / 255.0

  # Resize.
  preprocessed_frame = skimage.transform.resize(normalized_frame, [84, 84])

  return preprocessed_frame


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  logging.info('Done.')


if __name__ == '__main__':
  app.run(main)
