"""Logging interface.
"""
from __future__ import print_function

import json

from labm8 import system


def colourise(colour, *args):
  return "".join([colour] + list(args) + [Colours.RESET])


def printf(colour, *args, **kwargs):
  string = colourise(colour, *args)
  print(string, **kwargs)


def pprint(data, **kwargs):
  print(
      json.dumps(data, sort_keys=True, indent=2, separators=(",", ": ")),
      **kwargs)


def info(*args, **kwargs):
  print("[INFO  ]", *args, **kwargs)


def debug(*args, **kwargs):
  print("[DEBUG ]", *args, **kwargs)


def warn(*args, **kwargs):
  print("[WARN  ]", *args, **kwargs)


def error(*args, **kwargs):
  print("[ERROR ]", *args, **kwargs)


def fatal(*args, **kwargs):
  returncode = kwargs.pop("status", 1)
  error("fatal:", *args, **kwargs)
  system.exit(returncode)


def prof(*args, **kwargs):
  """
  Print a profiling message.

  Profiling messages are intended for printing runtime performance
  data. They are prefixed by the "PROF" title.

  Arguments:

      *args, **kwargs: Message payload.
  """
  print("[PROF  ]", *args, **kwargs)


class Colours:
  """
  Shell escape colour codes.
  """
  RESET = '\033[0m'
  GREEN = '\033[92m'
  YELLOW = '\033[93m'
  BLUE = '\033[94m'
  RED = '\033[91m'
