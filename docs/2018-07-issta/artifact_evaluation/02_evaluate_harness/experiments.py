"""This file defines TODO:

TODO: Detailed explanation of this file.
"""
import pathlib

from absl import app
from absl import flags

from deeplearning.deepsmith import datastore

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    unknown_args = ', '.join(argv[1:])
    raise app.UsageError(f"Unknown arguments {unknown_args}")

  ds = datastore.DataStore.FromFile(pathlib.Path(
    './02_evaluate_harness/data/datastore.pbtxt'))


if __name__ == '__main__':
  app.run(main)
