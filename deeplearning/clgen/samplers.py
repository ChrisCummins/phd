"""Samplers for CLgen language models."""
import pathlib

from deeplearning.clgen import cache
from deeplearning.clgen import errors
from deeplearning.clgen.proto import sampler_pb2
from lib.labm8 import crypto
from lib.labm8 import pbutil


class Sampler(object):
  """CLgen sampler for models.

  Please note sampler instances should be treated as immutable. Upon
  instantiation, a sampler's properties are used to determine its hash. If you
  modify a property after instantiation, the hash will be out of date, which
  can lead to bad things happening.
  """

  def __init__(self, config: sampler_pb2.Sampler):
    """Instantiate a sampler.

    Args:
      config: A Sampler message.
    """
    self.config = sampler_pb2.Sampler()
    self.config.CopyFrom(config)
    self.hash = self._ComputeHash(self.config)
    if not config.start_text:
      raise errors.UserError('Sampler.start_text not set')
    if config.batch_size < 1:
      raise errors.UserError('Sampler.batch_size must be >= 1')

  @staticmethod
  def _ComputeHash(config: sampler_pb2.Sampler) -> str:
    """Compute sampler hash.

    The hash is computed from the serialized representation of the config
    proto.
    """
    return crypto.sha1(config.SerializeToString())

  def _FlushMeta(self, cache_):
    pbutil.ToFile(self.meta, pathlib.Path(cache_.keypath('META.pbtxt')))

  @property
  def shorthash(self) -> str:
    return cache.ShortHash(self.hash, cache.cachepath('sampler'))

  @property
  def start_text(self) -> str:
    return self.config.start_text

  def __repr__(self) -> str:
    """String representation."""
    return f'sampler[{self.shorthash}]: "{self.config.start_text}"'

  def __eq__(self, rhs) -> bool:
    if not isinstance(rhs, Sampler):
      return False
    return rhs.hash == self.hash

  def __ne__(self, rhs) -> bool:
    return not self.__eq__(rhs)
