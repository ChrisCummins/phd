"""Samplers for CLgen language models."""
import pathlib
import typing

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

    Raises:
      TypeError: If the config argument is not a Sampler proto.
      UserError: If the config contains invalid values.
    """
    if not isinstance(config, sampler_pb2.Sampler):
      t = type(config).__name__
      raise TypeError(f"Config must be a Sampler proto. Received: '{t}'")
    if not config.start_text:
      raise errors.UserError('Sampler.start_text not set')
    if config.batch_size < 1:
      raise errors.UserError('Sampler.batch_size must be >= 1')

    self.config = sampler_pb2.Sampler()
    self.config.CopyFrom(config)
    self.hash = self._ComputeHash(self.config)

    # Determine the termination criteria.
    self.max_length = -1
    self.special_token_left = None
    self.special_token_right = None
    for criterion in self.config.termination_criteria:
      if criterion.HasField('maxlen'):
        self.max_length = criterion.maxlen.maximum_tokens_in_sample
        if not criterion.maxlen.include_start_text_in_maximum:
          self.max_length += len(self.config.start_text)
      elif criterion.HasField('symtok'):
        self.symmetrical_token_left = criterion.symtok.depth_increase_token
        self.symmetrical_token_right = criterion.symtok.depth_decrease_token
    self.has_max_length = self.max_length > 0
    self.has_symmetrical_tokens = (
        self.special_token_left and self.special_token_right)

  def SampleIsComplete(self, sample_in_progress: typing.List[str]) -> bool:
    """Determine whether to stop sampling.

    Args:
      sample_in_progress: A sample in progress, as a sequence of decoded tokens.

    Returns:
      True if the sample is "complete", else False to continue sampling.
    """
    if self.has_max_length:
      if len(sample_in_progress) >= self.max_length:
        return True
    if self.has_symmetrical_tokens:
      left_token_count = sample_in_progress.count(self.symmetrical_token_left)
      right_token_count = sample_in_progress.count(self.symmetrical_token_left)
      if left_token_count and (left_token_count - right_token_count == 0):
        return True
    return False

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
