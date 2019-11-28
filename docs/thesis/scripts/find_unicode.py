#!/usr/bin/env python
"""Remove redundant files in bibtex entries"""
import pathlib

from absl import app
from absl import flags


FLAGS = flags.FLAGS


def main(argv):
  """Main entry point."""
  del argv

  # Read the input.
  with open("thesis.tex") as f:
    string = f.read()
  if "\u200B" in string:
    print("found!")


if __name__ == "__main__":
  app.run(main)
