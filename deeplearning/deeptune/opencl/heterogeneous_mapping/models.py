"""Models for predicting heterogeneous device mapping."""
import pathlib
import pickle
import tempfile
import tarfile
import typing
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
from keras.preprocessing import sequence as keras_sequence
from sklearn import tree as sktree

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
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
              verbose: bool = False) -> np.array:
    """Make predictions for programs.

    Args:
      df: The dataframe of training data.
      platform_name: The name of the gpu being trained for
      verbose: Whether to print verbose status messages during training.

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

  def train(self, df: pd.DataFrame, platform_name: str,
            verbose: bool = False):
    del verbose
    # select the Zero-R device: the most frequently optimal device
    cpu_gpu_runtimes = df[[
      'runtime:intel_core_i7_3820',
      f'runtime:{platform_name}'
    ]].values
    oracles = np.array(
        ["GPU" if gpu < cpu else "CPU" for cpu, gpu in cpu_gpu_runtimes])

    self.model = Counter(oracles).most_common(1)[0][0]

  def predict(self, df: pd.DataFrame, platform_name: str,
              verbose: bool = False):
    del verbose
    logging.info("Predicting %d %s mappings for device %s",
                 len(df), self.model, platform_name)
    if self.model == "GPU":
      return np.ones(len(df), dtype=np.int32)
    elif self.model == "CPU":
      return np.zeros(len(df), dtype=np.int32)
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

  def train(self, df: pd.DataFrame, platform_name: str,
            verbose: bool = False):
    del verbose
    features = opencl_device_mapping_dataset.ComputeGreweFeaturesForGpu(
        platform_name, df).values
    self.model.fit(features, df["y"])

  def predict(self, df: pd.DataFrame, platform_name: str,
              verbose: bool = False):
    del verbose
    features = opencl_device_mapping_dataset.ComputeGreweFeaturesForGpu(
        platform_name, df).values
    return self.model.predict(features)


def EncodeAndPadSources(atomizer: atomizers.AtomizerBase,
                        srcs: typing.List[str],
                        maxlen: int) -> np.array:
  """Encode and pad source code for learning."""
  seqs = [atomizer.AtomizeString(src) for src in srcs]
  pad_val = atomizer.vocab_size
  encoded = np.array(keras_sequence.pad_sequences(
      seqs, maxlen=maxlen, value=pad_val))
  return np.vstack([np.expand_dims(x, axis=0) for x in encoded])


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

  def __init__(self, lstm_layer_size: int = 64, dense_layer_size: int = 32,
               num_epochs: int = 50, batch_size: int = 64,
               max_sequence_length: int = 1024):
    """Constructor.

    Args:
      lstm_layer_size: The number of neurons in the LSTM layers.
      dense_layer_size: The number of neurons in the dense layers.
      num_epochs: The number of epochs to train for.
      batch_size: The training and inference batch sizes.
    """
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.lstm_layer_size = lstm_layer_size
    self.dense_layer_size = dense_layer_size
    self.max_sequence_length = max_sequence_length
    self._atomizer = None

  def init(self, seed: int, atomizer: atomizers.AtomizerBase):
    np.random.seed(seed)

    self._atomizer = atomizer

    # Language model. Takes as inputs source code sequences.
    code_in = Input(shape=(1024,), dtype="int32", name="code_in")
    x = Embedding(input_dim=atomizer.vocab_size + 1,
                  input_length=self.max_sequence_length,
                  output_dim=self.lstm_layer_size, name="embedding")(code_in)
    x = LSTM(self.lstm_layer_size, implementation=1, return_sequences=True,
             name="lstm_1")(x)
    x = LSTM(self.lstm_layer_size, implementation=1, name="lstm_2")(x)
    langmodel_out = Dense(2, activation="sigmoid")(x)

    # Auxiliary inputs. wgsize and dsize.
    auxiliary_inputs = Input(shape=(2,))

    # Heuristic model. Takes as inputs a concatenation of the language model
    # and auxiliary inputs, outputs 1-hot encoded device mapping.
    x = Concatenate()([auxiliary_inputs, x])
    x = BatchNormalization()(x)
    x = Dense(self.dense_layer_size, activation="relu")(x)
    out = Dense(2, activation="sigmoid")(x)

    self.model = Model(inputs=[auxiliary_inputs, code_in],
                       outputs=[out, langmodel_out])
    self.model.compile(
        optimizer="adam", metrics=['accuracy'],
        loss=["categorical_crossentropy", "categorical_crossentropy"],
        loss_weights=[1., .2])

    return self

  @property
  def atomizer(self):
    if self._atomizer is None:
      raise ValueError("Cannot acccess atomizer before init() called")
    return self._atomizer

  def save(self, outpath):
    # DeepTune models are stored as an uncompressed tarball with the following
    # contents:
    #     /keras_model.h5 - Full keras model.
    #     /atomizer.pkl - Pickled atomizer.
    with tempfile.TemporaryDirectory(prefix='phd_') as d:
      d = pathlib.Path(d)

      # Write the model files to a temporary directory.
      self.model.save(d / 'keras_model.h5')
      with open(d / 'atomizer.pkl', 'w') as outfile:
        pickle.dump(self._atomizer, outfile)

      # Package the files as an uncompressed tarball.
      with tarfile.open(outpath, mode='w') as tar:
        tar.add(d / 'keras_model.h5', arcname='keras_model.h5')
        tar.add(d / 'atomizer.pkl', arcname='atomizer.pkl')


  def restore(self, inpath):
    with tempfile.TemporaryDirectory(prefix='phd_') as d:
      d = pathlib.Path(d)

      # Unpack the tarball to a temporary directory.
      with tarfile.open(inpath) as tar:
        tar.extractall(d)

      # Restore model properties from files.
      self.model = keras_models.load_model(d / 'keras_model.h5')
      with open(d / 'atomizer.pkl', 'rb') as f:
        self._atomizer = pickle.load(f)

  def train(self, df: pd.DataFrame, platform_name: str,
            verbose: bool = False):
    self.model.fit(self.DataFrameToModelInputs(df, platform_name),
                   self.DataFrameToModelTargets(df),
                   epochs=self.num_epochs, batch_size=self.batch_size,
                   verbose=verbose, shuffle=True)

  def predict(self, df: pd.DataFrame, platform_name: str,
              verbose: bool = False):
    p = np.array(self.model.predict(
        self.DataFrameToModelInputs(df, platform_name),
        batch_size=self.batch_size, verbose=verbose))
    indices = [np.argmax(x) for x in p[0]]
    return indices

  def DataFrameToModelInputs(
      self, df: pd.DataFrame,
      gpu_name: str) -> typing.List[np.ndarray]:
    """Convert a pandas table to a list of model inputs."""
    sequences = EncodeAndPadSources(
        self.atomizer, df['program:opencl_src'], self.max_sequence_length)
    aux_in = np.array([
      df[f"feature:{gpu_name}:transfer"].values,
      df[f"param:{gpu_name}:wgsize"].values,
    ]).T
    return [aux_in, sequences]

  @staticmethod
  def DataFrameToModelTargets(
      df: pd.DataFrame) -> typing.List[np.ndarray]:
    """Convert a pandas table to a list of model targets."""
    return [np.vstack(df['y_1hot'].values), np.vstack(df['y_1hot'].values)]


class NCC(DeepTune):
  """Neural code comprehension predictive model for device mapping.

  Described in:

      ﻿Ben-Nun, T., Jakobovits, A. S., & Hoefler, T. (2018). Neural Code
      Comprehension: A Learnable Representation of Code Semantics. In NeurIPS.
      https://doi.org/arXiv:1806.07336v3
  """
  __name__ = "NCC"
  __basename__ = "ncc"

  def __init__(self, embedding_dim: int, **kwargs):
    super(NCC, self).__init__(**kwargs)
    self.embedding_dim = embedding_dim

  def init(self, seed: int, atomizer: atomizers.AtomizerBase):
    # This is the same as DeepTune init, except that both LSTM layers have
    # a number of neurons equal to the embedding dimensionality, rather than
    # 64 neurons per layer.
    np.random.seed(seed)

    # Keras model
    inp = Input(shape=(1024, self.embedding_dim,), dtype="float32",
                name="code_in")
    x = LSTM(self.embedding_dim, implementation=1, return_sequences=True,
             name="lstm_1")(inp)
    x = LSTM(self.embedding_dim, implementation=1, name="lstm_2")(x)
    langmodel_out = Dense(2, activation="sigmoid")(x)

    # Auxiliary inputs. wgsize and dsize.
    auxiliary_inputs = Input(shape=(2,))
    x = Concatenate()([auxiliary_inputs, x])
    x = BatchNormalization()(x)
    x = Dense(self.dense_layer_size, activation="relu")(x)
    out = Dense(2, activation="sigmoid")(x)

    self.model = Model(inputs=[auxiliary_inputs, inp],
                       outputs=[out, langmodel_out])
    self.model.compile(
        optimizer="adam",
        metrics=['accuracy'],
        loss=["categorical_crossentropy", "categorical_crossentropy"],
        loss_weights=[1., .2])

    return self
