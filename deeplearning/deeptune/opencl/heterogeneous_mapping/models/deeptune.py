# Copyright (c) 2017, 2018, 2019 Chris Cummins.
#
# DeepTune is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepTune is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepTune.  If not, see <https://www.gnu.org/licenses/>.
"""DeepTune model."""
import pathlib
import pickle
import tarfile
import tempfile
import typing

import numpy as np
import pandas as pd
from keras import Input
from keras import Model
from keras import models as keras_models
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.preprocessing import sequence as keras_sequence

from deeplearning.clgen.corpuses import atomizers
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import base
from labm8.py import app

FLAGS = app.FLAGS


def EncodeAndPadSources(
  atomizer: atomizers.AtomizerBase, srcs: typing.List[str], maxlen: int
) -> np.array:
  """Encode and pad source code for learning."""
  seqs = [atomizer.AtomizeString(src) for src in srcs]
  pad_val = atomizer.vocab_size
  encoded = np.array(
    keras_sequence.pad_sequences(seqs, maxlen=maxlen, value=pad_val)
  )
  return np.vstack([np.expand_dims(x, axis=0) for x in encoded])


class DeepTune(base.HeterogeneousMappingModel):
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

  def __init__(
    self,
    lstm_layer_size: int = 64,
    dense_layer_size: int = 32,
    num_epochs: int = 50,
    batch_size: int = 64,
    input_shape: typing.List[int] = (1024,),
    input_type: str = "int32",
    with_embedding_layer: bool = True,
  ):
    """Constructor.

    Args:
      lstm_layer_size: The number of neurons in the LSTM layers.
      dense_layer_size: The number of neurons in the dense layers.
      num_epochs: The number of epochs to train for.
      batch_size: The training and inference batch sizes.
      input_shape: The shape of sequence inputs, as a tuple. If
        with_embedding_layer is true, this should be a tuple (seqlen,), where
        seqlen is the length of input sequences. Else, it should be a tuple
        (seqlen,seqdim), where seqdim is the dimensionality of each vector in
        the sequence inputs.
      input_type: The type of sequence inputs.
      with_embedding_layer: If True, feed inputs into an embedding layer prior
        to input into the LSTMs. Else, the values returned by
        DataFrameToModelInputs() are fed directly into the LSTMs.
    """
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.lstm_layer_size = lstm_layer_size
    self.dense_layer_size = dense_layer_size
    self.input_shape = input_shape
    self.input_type = input_type
    self._atomizer = None
    self.with_embedding_layer = with_embedding_layer

  def init(self, seed: int, atomizer: atomizers.AtomizerBase):
    np.random.seed(seed)

    self._atomizer = atomizer

    # Language model. It begins with an optional embedding layer, then has two
    # layers of LSTM network, returning a single vector of size
    # self.lstm_layer_size.
    input_layer = Input(
      shape=self.input_shape, dtype=self.input_type, name="model_in"
    )
    lstm_input = input_layer

    if self.with_embedding_layer:
      embedding_dim = atomizer.vocab_size + 1
      lstm_input = Embedding(
        input_dim=embedding_dim,
        input_length=self.input_shape[0],
        output_dim=self.lstm_layer_size,
        name="embedding",
      )(input_layer)

    x = LSTM(
      self.lstm_layer_size,
      implementation=1,
      return_sequences=True,
      name="lstm_1",
    )(lstm_input)
    x = LSTM(self.lstm_layer_size, implementation=1, name="lstm_2")(x)
    langmodel_out = Dense(2, activation="sigmoid")(x)

    # Auxiliary inputs. wgsize and dsize.
    auxiliary_inputs = Input(shape=(2,), name="aux_in")

    # Heuristic model. Takes as inputs a concatenation of the language model
    # and auxiliary inputs, outputs 1-hot encoded device mapping.
    x = Concatenate()([auxiliary_inputs, x])
    x = BatchNormalization()(x)
    x = Dense(self.dense_layer_size, activation="relu")(x)
    out = Dense(2, activation="sigmoid")(x)

    self.model = Model(
      inputs=[auxiliary_inputs, input_layer], outputs=[out, langmodel_out]
    )
    self.model.compile(
      optimizer="adam",
      metrics=["accuracy"],
      loss=["categorical_crossentropy", "categorical_crossentropy"],
      loss_weights=[1.0, 0.2],
    )

    return self

  @property
  def atomizer(self):
    if self._atomizer is None:
      raise ValueError("Cannot access atomizer before init() called")
    return self._atomizer

  def save(self, outpath):
    # DeepTune models are stored as an uncompressed tarball with the following
    # contents:
    #     /keras_model.h5 - Full keras model.
    #     /atomizer.pkl - Pickled atomizer.
    with tempfile.TemporaryDirectory(prefix="phd_") as d:
      d = pathlib.Path(d)

      # Write the model files to a temporary directory.
      self.model.save(d / "keras_model.h5")
      with open(d / "atomizer.pkl", "wb") as outfile:
        pickle.dump(self._atomizer, outfile)

      # Package the files as an uncompressed tarball.
      with tarfile.open(outpath, mode="w") as tar:
        tar.add(d / "keras_model.h5", arcname="keras_model.h5")
        tar.add(d / "atomizer.pkl", arcname="atomizer.pkl")

  def restore(self, inpath):
    with tempfile.TemporaryDirectory(prefix="phd_") as d:
      d = pathlib.Path(d)

      # Unpack the tarball to a temporary directory.
      with tarfile.open(inpath) as tar:
        tar.extractall(d)

      # Restore model properties from files.
      self.model = keras_models.load_model(d / "keras_model.h5")
      with open(d / "atomizer.pkl", "rb") as f:
        self._atomizer = pickle.load(f)

  def train(self, df: pd.DataFrame, platform_name: str, verbose: bool = False):
    self.model.fit(
      self.DataFrameToModelInputs(df, platform_name),
      self.DataFrameToModelTargets(df),
      epochs=self.num_epochs,
      batch_size=self.batch_size,
      verbose=verbose,
      shuffle=True,
    )

  def predict(
    self, df: pd.DataFrame, platform_name: str, verbose: bool = False
  ):
    p = np.array(
      self.model.predict(
        self.DataFrameToModelInputs(df, platform_name),
        batch_size=self.batch_size,
        verbose=verbose,
      )
    )
    indices = [np.argmax(x) for x in p[0]]
    return indices

  def DataFrameToModelInputs(
    self, df: pd.DataFrame, gpu_name: str
  ) -> typing.List[np.ndarray]:
    """Convert a pandas table to a list of model inputs."""
    sequences = EncodeAndPadSources(
      self.atomizer, df["program:opencl_src"], self.input_shape[0]
    )
    aux_in = np.array(
      [
        df[f"feature:{gpu_name}:transfer"].values,
        df[f"param:{gpu_name}:wgsize"].values,
      ]
    ).T
    return [aux_in, sequences]

  @staticmethod
  def DataFrameToModelTargets(df: pd.DataFrame) -> typing.List[np.ndarray]:
    """Convert a pandas table to a list of model targets."""
    return [np.vstack(df["y_1hot"].values), np.vstack(df["y_1hot"].values)]
