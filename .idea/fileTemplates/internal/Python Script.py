"""TODO: Short summary of file.

TODO: Long description of file.
"""
from absl import app
from absl import flags


FLAGS = flags.FLAGS


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unknown flags "{}".'.format(', '.join(argv[1:])))


if __name__ == '__main__':
  app.run(main)
