"""A lightweight version of the CLgen module for inference only.
"""
import contextlib
import os
import pathlib
import typing

from absl import flags
from phd.lib.labm8 import pbutil

from deeplearning.clgen import errors
from deeplearning.clgen import samplers
from deeplearning.clgen.models import pretrained
from deeplearning.clgen.proto import clgen_pb2


FLAGS = flags.FLAGS


class Instance(object):
  """A CLgen instance."""

  def __init__(self, config: clgen_pb2.Instance):
    """Instantiate an instance.

    Args:
      config: An Instance proto.

    Raises:
      UserError: If the instance proto contains invalid values, is missing
        a model or sampler fields.
    """
    try:
      pbutil.AssertFieldIsSet(config, 'pretrained_model')
      pbutil.AssertFieldIsSet(config, 'sampler')
    except pbutil.ProtoValueError as e:
      raise errors.UserError(e)

    self.working_dir = None
    if config.HasField('working_dir'):
      self.working_dir: pathlib.Path = pathlib.Path(
          os.path.expandvars(config.working_dir)).expanduser().absolute()
    # Enter a session so that the cache paths are set relative to any requested
    # working directory.
    with self.Session():
      self.model: pretrained.PreTrainedModel = pretrained.PreTrainedModel(
          pathlib.Path(config.pretrained_model))
      self.sampler: samplers.Sampler = samplers.Sampler(config.sampler)

  @contextlib.contextmanager
  def Session(self) -> 'Instance':
    """Scoped $CLGEN_CACHE value."""
    old_working_dir = os.environ.get('CLGEN_CACHE', '')
    if self.working_dir:
      os.environ['CLGEN_CACHE'] = str(self.working_dir)
    yield self
    if self.working_dir:
      os.environ['CLGEN_CACHE'] = old_working_dir

  def Train(self, *args, **kwargs) -> None:
    with self.Session():
      self.model.Train(*args, **kwargs)

  def Sample(self, *args, **kwargs) -> typing.List['Sample']:
    with self.Session():
      return self.model.Sample(self.sampler, *args, **kwargs)

  @classmethod
  def FromFile(cls, path: pathlib.Path) -> 'Instance':
    return cls(pbutil.FromFile(path, clgen_pb2.Instance()))
