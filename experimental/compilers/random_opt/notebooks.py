"""Launch a Jupyter notebook server. This process never terminates."""
from notebook import notebookapp

from labm8 import app

FLAGS = app.FLAGS


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  notebookapp.main()


if __name__ == '__main__':
  app.RunWithArgs(main)
