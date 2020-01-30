# Copyright (c) 2016-2020 Chris Cummins.
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
"""Neural network backends for CLgen models."""
import typing

import numpy as np

from deeplearning.clgen import samplers
from deeplearning.clgen.corpuses import atomizers
from deeplearning.clgen.proto import model_pb2
from labm8.py import app
from labm8.py import cache

FLAGS = app.FLAGS


class BackendBase(object):
  """The base class for a language model backend.

  A language model backend encapsulates all of the neural network logic.
  """

  def __init__(
    self,
    config: model_pb2.Model,
    fs_cache: cache.FSCache,
    atomizer: atomizers.AtomizerBase,
  ):
    self.config = config
    self.cache = fs_cache
    self.atomizer = atomizer

  def Train(self, corpus: "Corpus", **extra_kwargs) -> None:
    """Train the backend."""
    raise NotImplementedError

  def InitSampling(
    self, sampler: samplers.Sampler, seed: typing.Optional[int] = None
  ) -> None:
    """Initialize backend for sampling."""
    raise NotImplementedError

  def InitSampleBatch(self, sampler: samplers.Sampler) -> None:
    """Begin a new sampling batch. Only called after InitSampling()."""
    raise NotImplementedError

  def SampleNextIndices(
    self, sampler: samplers.Sampler, done: np.ndarray
  ) -> np.ndarray:
    """Sample the next indices for the current sample batch.

    Returns:
      A numpy array of int32 values with shape (batch_size,).
    """
    raise NotImplementedError
