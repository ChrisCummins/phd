"""Neural network backends for CLgen models."""
import typing

import numpy as np
from absl import flags
from phd.lib.labm8 import cache

from deeplearning.clgen import samplers
from deeplearning.clgen.corpuses import atomizers
from deeplearning.clgen.proto import model_pb2


FLAGS = flags.FLAGS


class BackendBase(object):
  """The base class for a language model backend.

  A language model backend encapsulates all of the neural network logic.
  """

  def __init__(self, config: model_pb2.Model, fs_cache: cache.FSCache,
               atomizer: atomizers.AtomizerBase):
    self.config = config
    self.cache = fs_cache
    self.atomizer = atomizer

  def Train(self, corpus: 'Corpus') -> None:
    """Train the backend."""
    raise NotImplementedError

  def InitSampling(self, sampler: samplers.Sampler,
                   seed: typing.Optional[int] = None) -> int:
    """Initialize backend for sampling."""
    raise NotImplementedError

  def InitSampleBatch(self, sampler: samplers.Sampler, batch_size: int) -> None:
    """Begin a new sampling batch. Only called after InitSampling()."""
    raise NotImplementedError

  def SampleNextIndices(self, sampler: samplers.Sampler,
                        batch_size: int) -> np.ndarray:
    """Sample the next indices for the current sample batch.

    Returns:
      A numpy array of int32 values with shape (batch_size,).
    """
    raise NotImplementedError
