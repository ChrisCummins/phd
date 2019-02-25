"""Base class for OpenCL heterogeneous device mapping models."""
import pathlib
import typing

import pandas as pd
from absl import flags

from deeplearning.clgen.corpuses import atomizers

FLAGS = flags.FLAGS


class HeterogeneousMappingModel(object):
  """A model for predicting OpenCL heterogeneous device mappings.

  Attributes:
    __name__ (str): Model name.
  __basename__ (str): Shortened name, used for files
  """
  __name__ = None
  __basename__ = None

  def init(self, seed: int, atomizer: atomizers.AtomizerBase) -> None:
    """Initialize the model.

    Do whatever is required to setup a new heterogeneous model here.
    This method is called prior to training and predicting.
    This method may be omitted if no initial setup is required.

    Args:
      seed (int): The seed value used to reproducible results. May be 'None',
        indicating that no seed is to be used.
      atomizer: The atomizer used to tokenize training examples.
    """
    pass

  # TODO(cec): Switch to exclusively pathlib.Path for argument.
  def save(self, outpath: typing.Union[str, pathlib.Path]) -> None:
    """Save model state.

    This must capture all of the relevant state of the model. It is up
    to implementing classes to determine how best to save the model.

    Args:
      outpath (str): The path to save the model state to.
    """
    raise NotImplementedError

  # TODO(cec): Switch to exclusively pathlib.Path for argument.
  def restore(self, inpath: typing.Union[str, pathlib.Path]) -> None:
    """Load a trained model from file.

    This is called in place of init() if a saved model file exists. It
    must restore all of the required model state.

    Args:
      inpath (str): The path to load the model from. This is the same path as
        was passed to save() to create the file.
    """
    raise NotImplementedError

  def train(self, df: pd.DataFrame, platform_name: str,
            verbose: bool = False) -> None:
    """Train a model.

    Args:
      df: The dataframe of training data.
      platform_name: The name of the gpu being trained for
      verbose: Whether to print verbose status messages during training.
    """
    raise NotImplementedError

  def predict(self, df: pd.DataFrame, platform_name: str,
              verbose: bool = False) -> typing.Iterable[int]:
    """Make predictions for programs.

    Args:
      df: The dataframe of training data.
      platform_name: The name of the gpu being trained for
      verbose: Whether to print verbose status messages during training.

    Returns:
      A sequence of predicted 'y' values (optimal device mappings).
    """
    raise NotImplementedError
