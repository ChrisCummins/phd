"""Launch a Jupyter notebook server. This process never terminates."""
from absl import app
from absl import flags
from notebook import notebookapp

FLAGS = flags.FLAGS


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  notebookapp.main()


if __name__ == '__main__':
  app.run(main)
