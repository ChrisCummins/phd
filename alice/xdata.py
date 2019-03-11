"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
import contextlib
import os
import pathlib

from labm8 import app

FLAGS = app.FLAGS


class NotInWonderland(EnvironmentError):
  pass


def Id() -> int:
  id = os.environ.get('ALICE_XDATA_ID')
  if not id:
    raise NotImplementedError
  return id


def CreateArtifactDirectory(name: str) -> pathlib.Path:
  pass


class JsonWriter(object):
  pass


@contextlib.contextmanager
def TemporaryInMemoryDirectory(name: str) -> pathlib.Path:
  pass


@contextlib.contextmanager
def TemporaryDirectory(name: str) -> pathlib.Path:
  pass
