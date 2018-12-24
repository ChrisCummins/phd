"""Models for predicting heterogeneous device mapping."""
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from absl import flags
from absl import logging
from keras import models as keras_models
from keras.layers import Dense, Embedding, Input, LSTM
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from sklearn import tree as sktree

from deeplearning.clgen.corpuses import atomizers


FLAGS = flags.FLAGS


class HeterogemeousMappingModel(object):
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

  def save(self, outpath: str) -> None:
    """Save model state.

    This must capture all of the relevant state of the model. It is up
    to implementing classes to determine how best to save the model.

    Args:
      outpath (str): The path to save the model state to.
    """
    raise NotImplementedError

  def restore(self, inpath: str) -> None:
    """Load a trained model from file.

    This is called in place of init() if a saved model file exists. It
    must restore all of the required model state.

    Args:
      inpath (str): The path to load the model from. This is the same path as
        was passed to save() to create the file.
    """
    raise NotImplementedError

  def train(self, df: pd.DataFrame, features: np.array, sequences: np.array,
            y: np.array, y_1hot: np.array, verbose: bool = False) -> None:
    """Train a model.

    Args:
      df: The platform dataframe.
      features: An array of feature vectors of shape (n,4).
      sequences: An array of encoded source code sequences of shape
        (n,seq_length).
      y: An array of optimal device mappings of shape (n,1).
      y_1hot: An array of optimal device mappings of shape (n,2), in 1-hot
        encoding.
      verbose: Whether to print verbose status messages during training.
    """
    raise NotImplementedError

  def predict(self, features: np.array, sequences: np.array, y: np.array,
              y_1hot: np.array, verbose: bool = False) -> np.array:
    """Make predictions for programs.

    Args:
      features: An array of feature vectors of shape (n,4).
      sequences: An array of encoded source code sequences of shape
        (n,seq_length).
      y: An array of optimal device mappings of shape (n,1).
      y_1hot: An array of optimal device mappings of shape (n,2), in 1-hot
        encoding.
      verbose: Whether to print verbose status messages.

    Returns:
      Predicted 'y' values (optimal device mappings) with shape (n,1).
    """
    raise NotImplementedError


class StaticMapping(HeterogemeousMappingModel):
  __name__ = "Static mapping"
  __basename__ = "static"

  def init(self, seed: int, atomizer: atomizers.AtomizerBase):
    return self

  def save(self, outpath):
    with open(outpath, "wb") as outfile:
      pickle.dump(self.model, outfile)

  def restore(self, inpath):
    with open(inpath, "rb") as infile:
      self.model = pickle.load(infile)

  def train(self, dataset=None, **train):
    platform_name = train.get('platform_name')
    if not platform_name:
      raise ValueError('platform_name kwarg not set!')

    # select the Zero-R device: the most frequently optimal device
    cpu_gpu_runtimes = dataset.df[[
      'runtime:intel_core_i7_3820',
      f'runtime:{platform_name}'
    ]].values
    oracles = np.array(
        ["GPU" if gpu < cpu else "CPU" for cpu, gpu in cpu_gpu_runtimes])

    self.model = Counter(oracles).most_common(1)[0][0]

  def predict(self, **test):
    logging.info("Predicting %d %s mappings for device %s",
                 len(test['features']), self.model, test['platform_name'])
    if self.model == "GPU":
      return np.ones(len(test['features']), dtype=np.int32)
    elif self.model == "CPU":
      return np.zeros(len(test['features']), dtype=np.int32)
    else:
      return LookupError


class Grewe(HeterogemeousMappingModel):
  """Grewe et al. predictive model for heterogeneous device mapping.

  The Grewe et al. predictive model uses decision trees and hand engineered
  features to predict optimal device mapping, described in publication:

    ﻿Grewe, D., Wang, Z., & O’Boyle, M. (2013). Portable Mapping of Data
    Parallel Programs to OpenCL for Heterogeneous Systems. In CGO. IEEE.
    https://doi.org/10.1109/CGO.2013.6494993
  """
  __name__ = "Grewe et al."
  __basename__ = "grewe"

  def init(self, seed: int, atomizer: atomizers.AtomizerBase):
    self.model = sktree.DecisionTreeClassifier(
        random_state=seed, splitter="best",
        criterion="entropy", max_depth=5,
        min_samples_leaf=5)
    return self

  def save(self, outpath):
    with open(outpath, "wb") as outfile:
      pickle.dump(self.model, outfile)

  def restore(self, inpath):
    with open(inpath, "rb") as infile:
      self.model = pickle.load(infile)

  def train(self, **train):
    self.model.fit(train["features"], train["y"])

  def predict(self, **test):
    return self.model.predict(test["features"])


class DeepTune(HeterogemeousMappingModel):
  """DeepTune predictive model for heterogeneous device mapping.

  DeepTune predicts optimal device mapping from raw source code inputs, and the
  two auxiliary inputs wgsize and dsize (which cannot be obtained statically
  from the program source).

  Described in:

      ﻿Cummins, C., Petoumenos, P., Wang, Z., & Leather, H. (2017). End-to-end
      Deep Learning of Optimization Heuristics. In PACT. IEEE.
  """
  __name__ = "DeepTune"
  __basename__ = "deeptune"

  def init(self, seed: int, atomizer: atomizers.AtomizerBase):
    np.random.seed(seed)

    # Language model. Takes as inputs source code sequences.
    code_in = Input(shape=(1024,), dtype="int32", name="code_in")
    x = Embedding(input_dim=atomizer.vocab_size + 1, input_length=1024,
                  output_dim=64, name="embedding")(code_in)
    x = LSTM(64, implementation=1, return_sequences=True, name="lstm_1")(x)
    x = LSTM(64, implementation=1, name="lstm_2")(x)
    langmodel_out = Dense(2, activation="sigmoid")(x)

    # Auxiliary inputs. wgsize and dsize.
    auxiliary_inputs = Input(shape=(2,))

    # Heuristic model. Takes as inputs the language model,
    #   outputs 1-hot encoded device mapping
    x = Concatenate()([auxiliary_inputs, x])
    x = BatchNormalization()(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(2, activation="sigmoid")(x)

    self.model = Model(inputs=[auxiliary_inputs, code_in],
                       outputs=[out, langmodel_out])
    self.model.compile(
        optimizer="adam", metrics=['accuracy'],
        loss=["categorical_crossentropy", "categorical_crossentropy"],
        loss_weights=[1., .2])

    return self

  def save(self, outpath):
    self.model.save(outpath)

  def restore(self, inpath):
    self.model = keras_models.load_model(inpath)

  def train(self, **train):
    self.model.fit([train["aux_in"], train["sequences"]],
                   [train["y_1hot"], train["y_1hot"]],
                   epochs=50, batch_size=64, verbose=train["verbose"],
                   shuffle=True)

  def predict(self, **test):
    p = np.array(self.model.predict(
        [test["aux_in"], test["sequences"]], batch_size=64,
        verbose=test["verbose"]))
    indices = [np.argmax(x) for x in p[0]]
    return indices
