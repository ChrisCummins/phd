# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""Unit tests for //deeplearning/clgen/sampler.py."""
import typing

import numpy as np
import pytest

from deeplearning.clgen import errors
from deeplearning.clgen import samplers
from deeplearning.clgen.proto import sampler_pb2
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


class AtomizerMock(object):
  """Mock class for atomizer."""

  @staticmethod
  def AtomizeString(string) -> np.ndarray:
    """Mock for string atomizer"""
    del string
    return np.array([1])

  @staticmethod
  def TokenizeString(string) -> typing.List[str]:
    """Mock for string tokenizer."""
    del string
    return ["a"]


# AssertConfigIsValid() tests.


def test_AssertConfigIsValid_no_start_text(clgen_cache_dir, abc_sampler_config):
  """Test that an error is thrown if start_text field is not set."""
  del clgen_cache_dir
  # Field not set.
  abc_sampler_config.ClearField("start_text")
  with test.Raises(errors.UserError) as e_info:
    samplers.Sampler(abc_sampler_config)
  assert "Sampler.start_text must be a string" == str(e_info.value)
  # Value is an empty string.
  abc_sampler_config.start_text = ""
  with test.Raises(errors.UserError) as e_info:
    samplers.Sampler(abc_sampler_config)
  assert "Sampler.start_text must be a string" == str(e_info.value)


def test_AssertConfigIsValid_invalid_batch_size(abc_sampler_config):
  """Test that an error is thrown if batch_size is < 1."""
  # Field not set.
  abc_sampler_config.ClearField("batch_size")
  with test.Raises(errors.UserError) as e_info:
    samplers.Sampler(abc_sampler_config)
  assert "Sampler.batch_size must be > 0" == str(e_info.value)
  # Value is zero.
  abc_sampler_config.batch_size = 0
  with test.Raises(errors.UserError) as e_info:
    samplers.Sampler(abc_sampler_config)
  assert "Sampler.batch_size must be > 0" == str(e_info.value)
  # Value is negative.
  abc_sampler_config.batch_size = -1
  with test.Raises(errors.UserError) as e_info:
    samplers.Sampler(abc_sampler_config)
  assert "Sampler.batch_size must be > 0" == str(e_info.value)


def test_AssertConfigIsValid_invalid_temperature_micros(abc_sampler_config):
  """Test that an error is thrown if temperature_micros is < 0."""
  # Field not set.
  abc_sampler_config.ClearField("temperature_micros")
  with test.Raises(errors.UserError) as e_info:
    samplers.Sampler(abc_sampler_config)
  assert "Sampler.temperature_micros must be > 0" == str(e_info.value)
  # Value is negative.
  abc_sampler_config.temperature_micros = -1
  with test.Raises(errors.UserError) as e_info:
    samplers.Sampler(abc_sampler_config)
  assert "Sampler.temperature_micros must be > 0" == str(e_info.value)


# MaxlenTerminationCriterion tests.


def test_MaxlenTerminationCriterion_invalid_maximum_tokens_in_sample():
  """Test that error is raised if maximum_tokens_in_sample is invalid."""
  config = sampler_pb2.MaxTokenLength()
  # Field is missing.
  with test.Raises(errors.UserError) as e_info:
    samplers.MaxlenTerminationCriterion(config)
  assert "MaxTokenLength.maximum_tokens_in_sample must be > 0" == str(
    e_info.value
  )
  # Value is zero.
  config.maximum_tokens_in_sample = 0
  with test.Raises(errors.UserError) as e_info:
    samplers.MaxlenTerminationCriterion(config)
  assert "MaxTokenLength.maximum_tokens_in_sample must be > 0" == str(
    e_info.value
  )


def test_MaxlenTerminationCriterion_SampleIsComplete():
  """Test SampleIsComplete() returns expected values."""
  t = samplers.MaxlenTerminationCriterion(
    sampler_pb2.MaxTokenLength(maximum_tokens_in_sample=3)
  )
  assert not t.SampleIsComplete([])
  assert not t.SampleIsComplete(["a"])
  assert not t.SampleIsComplete(["a", "b"])
  assert t.SampleIsComplete(["a", "b", "c"])
  assert t.SampleIsComplete(["a", "b", "c", "d"])
  assert t.SampleIsComplete(["a", "b", "c", "d", "e"])


# SymmetricalTokenDepthCriterion tests.


def test_SymmetricalTokenDepthCriterion_depth_increase_token():
  """Test that error is raised if depth_increase_token is invalid."""
  config = sampler_pb2.SymmetricalTokenDepth(depth_decrease_token="a")
  # Field is missing.
  with test.Raises(errors.UserError) as e_info:
    samplers.SymmetricalTokenDepthCriterion(config)
  assert "SymmetricalTokenDepth.depth_increase_token must be a string" == str(
    e_info.value
  )
  # Value is empty.
  config.depth_increase_token = ""
  with test.Raises(errors.UserError) as e_info:
    samplers.SymmetricalTokenDepthCriterion(config)
  assert "SymmetricalTokenDepth.depth_increase_token must be a string" == str(
    e_info.value
  )


def test_SymmetricalTokenDepthCriterion_depth_increase_token():
  """Test that error is raised if depth_increase_token is invalid."""
  config = sampler_pb2.SymmetricalTokenDepth(depth_increase_token="a")
  # Field is missing.
  with test.Raises(errors.UserError) as e_info:
    samplers.SymmetricalTokenDepthCriterion(config)
  assert "SymmetricalTokenDepth.depth_decrease_token must be a string" == str(
    e_info.value
  )
  # Value is empty.
  config.depth_decrease_token = ""
  with test.Raises(errors.UserError) as e_info:
    samplers.SymmetricalTokenDepthCriterion(config)
  assert "SymmetricalTokenDepth.depth_decrease_token must be a string" == str(
    e_info.value
  )


def test_SymmetricalTokenDepthCriterion_same_tokens():
  """test that error is raised if depth tokens are the same."""
  config = sampler_pb2.SymmetricalTokenDepth(
    depth_increase_token="a", depth_decrease_token="a"
  )
  with test.Raises(errors.UserError) as e_info:
    samplers.SymmetricalTokenDepthCriterion(config)
  assert "SymmetricalTokenDepth tokens must be different" == str(e_info.value)


def test_SymmetricalTokenDepthCriterion_SampleIsComplete():
  """Test SampleIsComplete() returns expected values."""
  t = samplers.SymmetricalTokenDepthCriterion(
    sampler_pb2.SymmetricalTokenDepth(
      depth_increase_token="+", depth_decrease_token="-"
    )
  )
  # Depth 0, incomplete.
  assert not t.SampleIsComplete([])
  # Depth 1, incomplete.
  assert not t.SampleIsComplete(["+"])
  # Depth -1, complete.
  assert t.SampleIsComplete(["-"])
  # Depth 0, complete.
  assert t.SampleIsComplete(["+", "-"])
  # Depth 1, incomplete.
  assert not t.SampleIsComplete(["a", "+", "b", "c"])
  # Depth 1, incomplete.
  assert not t.SampleIsComplete(["a", "+", "+", "b", "c", "-"])
  # Depth 0, complete.
  assert t.SampleIsComplete(["a", "+", "-", "+", "b", "c", "-"])


def test_SymmetrcalTokenDepthCriterion_SampleIsComplete_reverse_order():
  """Test that sample is not complete if right token appears before left."""
  t = samplers.SymmetricalTokenDepthCriterion(
    sampler_pb2.SymmetricalTokenDepth(
      depth_increase_token="+", depth_decrease_token="-"
    )
  )
  assert not t.SampleIsComplete(["-", "+"])
  assert not t.SampleIsComplete(["-", "a", "b", "c", "+"])
  assert t.SampleIsComplete(["-", "a", "b", "c", "+", "+", "-"])


# Sampler tests.


def test_Sampler_config_type_error():
  """Test that a TypeError is raised if config is not a Sampler proto."""
  with test.Raises(TypeError) as e_info:
    samplers.Sampler(1)
  assert "Config must be a Sampler proto. Received: 'int'" == str(e_info.value)


def test_Sampler_start_text(abc_sampler_config: sampler_pb2.Sampler):
  """Test that start_text is set from Sampler proto."""
  s = samplers.Sampler(abc_sampler_config)
  assert s.start_text == abc_sampler_config.start_text


def test_Sampler_temperature(abc_sampler_config: sampler_pb2.Sampler):
  """Test that temperature is set from Sampler proto."""
  abc_sampler_config.temperature_micros = 1000000
  s = samplers.Sampler(abc_sampler_config)
  assert pytest.approx(1.0) == s.temperature


def test_Sampler_batch_size(abc_sampler_config: sampler_pb2.Sampler):
  """Test that batch_size is set from Sampler proto."""
  abc_sampler_config.batch_size = 99
  s = samplers.Sampler(abc_sampler_config)
  assert 99 == s.batch_size


# Sampler.Specialize() tests.


def test_Sampler_Specialize_invalid_depth_tokens(
  abc_sampler_config: sampler_pb2.Sampler,
):
  """Test that InvalidSymtokTokens raised if depth tokens cannot be encoded."""
  t = abc_sampler_config.termination_criteria.add()
  t.symtok.depth_increase_token = "{"
  t.symtok.depth_decrease_token = "}"
  s = samplers.Sampler(abc_sampler_config)

  def MockAtomizeString(string):
    """AtomizeString() with a vocab error on depth tokens."""
    if string == "{" or string == "}":
      raise errors.VocabError()
    else:
      return np.ndarray([1])

  mock = AtomizerMock()
  mock.AtomizeString = MockAtomizeString
  with test.Raises(errors.InvalidSymtokTokens) as e_info:
    s.Specialize(mock)
  assert (
    "Sampler symmetrical depth tokens cannot be encoded using the "
    "corpus vocabulary"
  ) == str(e_info.value)


def test_Sampler_Specialize_multiple_tokens_per(
  abc_sampler_config: sampler_pb2.Sampler,
):
  """Test that InvalidSymtokTokens raised if depth tokens encode to mult."""
  t = abc_sampler_config.termination_criteria.add()
  t.symtok.depth_increase_token = "abc"
  t.symtok.depth_decrease_token = "cba"
  s = samplers.Sampler(abc_sampler_config)

  def MockAtomizeString(string):
    """AtomizeString() with a multi-token output."""
    del string
    return np.array([1, 2, 3])

  mock = AtomizerMock()
  mock.AtomizeString = MockAtomizeString
  with test.Raises(errors.InvalidSymtokTokens) as e_info:
    s.Specialize(mock)
  assert (
    "Sampler symmetrical depth tokens do not encode to a single "
    "token using the corpus vocabulary"
  )


def test_Sampler_Specialize_encoded_start_text(
  abc_sampler_config: sampler_pb2.Sampler,
):
  s = samplers.Sampler(abc_sampler_config)
  assert s.encoded_start_text is None
  s.Specialize(AtomizerMock())
  np.testing.assert_array_equal(np.array([1]), s.encoded_start_text)


if __name__ == "__main__":
  test.Main()
