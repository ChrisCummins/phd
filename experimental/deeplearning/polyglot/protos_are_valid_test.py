"""Test that //experimental/polyglot/baselines:protos are valid."""
import sys

import pytest

from datasets.github.scrape_repos.proto import scrape_repos_pb2
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import sampler_pb2
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import pbutil
from labm8.py import test

MODULE_UNDER_TEST = None  # No coverage.


def DirContainsProtos(data_path: str, proto_class) -> None:
  """Assert that contains protos of the given class."""
  for path in bazelutil.DataPath(data_path).iterdir():
    assert pbutil.ProtoIsReadable(
      bazelutil.DataPath(data_path) / path, proto_class()
    )


def test_clone_lists_are_valid():
  """Test that clone_lists are valid."""
  DirContainsProtos(
    "phd/experimental/deeplearning/polyglot/clone_lists",
    scrape_repos_pb2.LanguageCloneList,
  )


def test_corpuses_are_valid():
  """Test that corpuses are valid."""
  DirContainsProtos(
    "phd/experimental/deeplearning/polyglot/corpuses", corpus_pb2.Corpus
  )


def test_samplers_are_valid():
  """Test that samplers are valid."""
  DirContainsProtos(
    "phd/experimental/deeplearning/polyglot/samplers", sampler_pb2.Sampler
  )


if __name__ == "__main__":
  test.Main()
