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
"""Neural Code Comprehension.

An implementation of the inst2vec embeddings + RNN classifier described in:

  ﻿Ben-nun, T., Jakobovits, A. S., & Hoefler, T. (2018). Neural Code
  Comprehension: A Learnable Representation of Code Semantics. In NeurIPS.
"""
import multiprocessing
import pathlib
import typing

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing import sequence as keras_sequence
from labm8 import app
from labm8 import bazelutil

from compilers.llvm import clang
from deeplearning.clgen.preprocessors import opencl
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import deeptune
from deeplearning.ncc import task_utils as inst2vec_utils
from deeplearning.ncc import vocabulary as inst2vec_vocabulary

# The pre-trained embeddings used by default by DeepTuneInst2Vec models.
DEEPTUNE_INST2VEC_EMBEDDINGS = bazelutil.DataPath(
    'phd/deeplearning/ncc/published_results/emb.p')

# The vocabulary to use for encoding sequences used by DeepTuneInst2Vec models.
DEEPTUNE_INST2VEC_VOCAB_PATH = bazelutil.DataPath(
    'phd/deeplearning/ncc/published_results/vocabulary.zip')

# A zipfile containing the original OpenCL sources.
# TODO(cec): Add original OpenCL sources to dataframe, then this won't be
# necessary.
DEEPTUNE_INST2VEC_DATA_ARCHIVE = bazelutil.DataArchive(
    'phd/deeplearning/ncc/published_results/task_devmap_kernels.zip')


def ExtractLlvmByteCodeOrDie(src_file_path: pathlib.Path,
                             datafolder: pathlib.Path):
  """Read and compile to bytecode or die."""
  # Read the source file and strip any non-ascii characters.
  with open(src_file_path, 'rb') as f:
    src = f.read().decode('unicode_escape')
  src = src.encode('ascii', 'ignore').decode('ascii')

  # Compile src to bytecode.
  clang_args = opencl.GetClangArgs(use_shim=True) + [
      '-O0',
      '-S',
      '-emit-llvm',
      '-o',
      '-',
      '-i',
      '-',
      # No warnings, and fail immediately on error.
      '-Wno-everything',
      '-ferror-limit=1',
      # Kernels have headers.
      '-I',
      str(datafolder / 'kernels_cl'),
      # We don't need the full shim header, just the common constants:
      '-DCLGEN_OPENCL_SHIM_NO_COMMON_TYPES',
      '-DCLGEN_OPENCL_SHIM_NO_UNSUPPORTED_STORAGE_CLASSES_AND_QUALIFIERS',
  ]
  process = clang.Exec(clang_args, stdin=src, log=False)
  if process.returncode:
    app.Error("Failed to compile %s", src_file_path)
    app.Error("stderr: %s", process.stderr)
    app.Fatal(f"clang failed with returncode {process.returncode}")
  return process.stdout


def _EncodeSourceBatchOrDie(src_file_paths, datafolder):
  batch = []

  for src_file_path in src_file_paths:
    app.Log(2, 'Compiling %s', src_file_path.name)
    bytecode = ExtractLlvmByteCodeOrDie(src_file_path, datafolder)
    batch.append((src_file_path, bytecode))

  return batch


def EncodeAndPadSourcesWithInst2Vec(
    df: pd.DataFrame,
    vocab: inst2vec_vocabulary.VocabularyZipFile,
    datafolder: pathlib.Path,
    max_sequence_len: typing.Optional[int] = None
) -> typing.Tuple[np.array, int]:
  """Encode and pad source code using inst2vec translation."""
  sequence_lengths = []
  sequences = []

  # A map from source files to encoded sequences, as there can be multiple
  # entries in the dataframe using the same sequence.
  src_path_to_sequence = {}

  src_paths = list(
      set(
          DataFrameRowToKernelSrcPath(row, datafolder)
          for _, row in df.iterrows()))

  # Chunk the srcs and process in parallel.
  srcs_per_process = 16
  encode_args = [(src_paths[i:i + srcs_per_process], datafolder)
                 for i in range(0, len(src_paths), srcs_per_process)]
  batches = multiprocessing.Pool().starmap(_EncodeSourceBatchOrDie, encode_args)
  for batch in batches:
    for src_file_path, bytecode in batch:
      app.Log(2, 'Encoding %s', src_file_path.name)
      sequence = list(vocab.EncodeLlvmBytecode(bytecode).encoded)
      src_path_to_sequence[src_file_path] = sequence

  for _, row in df.iterrows():
    src_file_path = DataFrameRowToKernelSrcPath(row, datafolder)
    sequence = src_path_to_sequence[src_file_path]

    sequence_lengths.append(len(sequence))
    sequences.append(sequence)

  if max_sequence_len is None:
    max_sequence_len = max(sequence_lengths)
  app.Log(2, 'Sequence lengths: min=%d, avg=%.2f, max=%d',
          min(sequence_lengths), np.mean(sequence_lengths), max_sequence_len)

  encoded = np.array(
      keras_sequence.pad_sequences(sequences,
                                   maxlen=max_sequence_len,
                                   value=vocab.unknown_token_index))
  encoded = np.vstack([np.expand_dims(x, axis=0) for x in encoded])

  return encoded, max_sequence_len


class DeepTuneInst2Vec(deeptune.DeepTune):
  """DeepTune model using inst2vec neural code comprehension embeddings.

  The inst2vec embeddings are described in:

      ﻿Ben-Nun, T., Jakobovits, A. S., & Hoefler, T. (2018). Neural Code
      Comprehension: A Learnable Representation of Code Semantics. In NeurIPS.
      https://doi.org/arXiv:1806.07336v3
  """
  __name__ = "DeepTuneInst2Vec"
  __basename__ = "deeptune_inst2vec"

  def __init__(self,
               embedding_matrix: np.ndarray = None,
               input_shape: typing.List[int] = (4075,),
               vocabulary_file: typing.Optional[pathlib.Path] = None,
               **deeptune_opts):
    # input_shape, lstm_layer_size, and input_type args are ignored.

    # If no embedding matrix is provided, the default is used.
    if embedding_matrix is None:
      embedding_matrix = inst2vec_utils.ReadEmbeddingFile(
          DEEPTUNE_INST2VEC_EMBEDDINGS)
    if vocabulary_file is None:
      vocabulary_file = DEEPTUNE_INST2VEC_VOCAB_PATH

    self.embedding_matrix = embedding_matrix
    self.vocabulary_file = vocabulary_file

    # This model has the same architecture as DeepTune, except that both LSTM
    # layers have a number of neurons equal to the embedding dimensionality,
    # rather than 64 neurons per layer.

    # Append the embedding dimensionality to the input shape.
    _, embedding_dim = embedding_matrix.shape
    deeptune_opts['input_shape'] = (input_shape[0], embedding_dim)

    deeptune_opts['with_embedding_layer'] = False
    deeptune_opts['lstm_layer_size'] = embedding_dim

    # Embedding vectors are floats.
    deeptune_opts['input_type'] = 'float32'

    super(DeepTuneInst2Vec, self).__init__(**deeptune_opts)

  def EncodeAndPadSources(self,
                          df: pd.DataFrame,
                          maxlen: typing.Optional[int] = None
                         ) -> typing.Tuple[np.array, int]:
    """Encode and pad source sequences."""
    # TODO(cec): This is hardcoded to OpenClDeviceMappingsDataset, and is
    # mighty slow.
    with DEEPTUNE_INST2VEC_DATA_ARCHIVE as datafolder:
      with inst2vec_vocabulary.VocabularyZipFile(self.vocabulary_file) as vocab:
        return EncodeAndPadSourcesWithInst2Vec(df, vocab, datafolder, maxlen)

  def DataFrameToModelInputs(self, df: pd.DataFrame, gpu_name: str):
    """Convert a pandas table to a list of model inputs.

    This override of DeepTune.DataFrameToModelInputs() provides the inst2vec
    functionality, returning embeddings.
    """
    sequences, _ = self.EncodeAndPadSources(df, self.input_shape[0])

    # Translate encoded sequences into sequences of normalized embeddings.
    sequence_ph = tf.compat.v1.placeholder(dtype=tf.int32)
    normalized_embedding_matrix = tf.nn.l2_normalize(self.embedding_matrix,
                                                     axis=1)
    embedding_input_op = tf.nn.embedding_lookup(normalized_embedding_matrix,
                                                sequence_ph)

    with tf.compat.v1.Session() as sess:
      # Tensor of shape (len(df), sequence length, embedding dimension).
      embedding_input = sess.run(embedding_input_op,
                                 feed_dict={sequence_ph: sequences})

    # Get the auxiliary inputs.
    aux_in = np.array([
        df[f"feature:{gpu_name}:transfer"].values,
        df[f"param:{gpu_name}:wgsize"].values,
    ]).T

    return [aux_in, embedding_input]


def DataFrameRowToKernelSrcPath(row: typing.Dict[str, typing.Any],
                                datafolder: pathlib.Path) -> pathlib.Path:
  """Translate row into an OpenCL kernel path."""
  # TODO(cec): This won't be necessary once we add the original OpenCL srcs to
  # the dataframe.
  file_name_stub = '-'.join([
      row['program:benchmark_suite_name'], row['program:benchmark_name'],
      row['program:opencl_kernel_name']
  ])

  opencl_src_path = datafolder / 'kernels_cl' / (file_name_stub + '.cl')
  if opencl_src_path.is_file():
    return opencl_src_path

  # Some of the benchmark sources are dataset dependent. This is reflected by
  # the dataset name being concatenated to the path.
  opencl_src_path = (
      datafolder / 'kernels_cl' /
      (file_name_stub + '_' + str(row['data:dataset_name']) + '.cl'))
  if opencl_src_path.is_file():
    return opencl_src_path

  raise FileNotFoundError(f"File not found: '{opencl_src_path}'")
