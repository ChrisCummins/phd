"""Main entry point for CLdrive script."""
from absl import app
from absl import flags
from absl import logging

from gpu.cldrive import env


FLAGS = flags.FLAGS
flags.DEFINE_boolean(
    'ls_env', False,
    'If set, list the names and details of available OpenCL environments, and '
    'exit.')


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    logging.warning("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  if FLAGS.ls_env:
    env.PrintOpenClEnvironments()


if __name__ == '__main__':
  app.run(main)
