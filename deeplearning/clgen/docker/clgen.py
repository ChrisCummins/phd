"""Main entry point for docker binary.

The job of this file is to simply call into the main CLgen binary, having
set some default arg values.
"""
import sys

from deeplearning.clgen import clgen
from labm8 import app

FLAGS = app.FLAGS


def main():
  # Set default value for config flag. Set the default value before the true
  # args, so that if the user provides a --config value, it will override
  # the default.
  FLAGS([sys.argv[0], '--config=/clgen/config.pbtxt'] + sys.argv[1:])
  clgen.RunWithErrorHandling(clgen.DoFlagsAction)


if __name__ == '__main__':
  app.Run(main)
