"""Models for predicting heterogeneous device mapping.

Attributes:
  ALL_MODELS: A set of HeterogeneousMappingModel subclasses.
"""
import collections
import multiprocessing
import pathlib
import pickle
import tarfile
import tempfile
import typing

import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import flags
from absl import logging
from graph_nets.demos import models as gn_models
from keras import models as keras_models
from keras.layers import Dense, Embedding, Input, LSTM
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing import sequence as keras_sequence
from sklearn import tree as sktree

from compilers.llvm import clang
from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.clgen.corpuses import atomizers
from deeplearning.clgen.preprocessors import opencl
from deeplearning.ncc import task_utils as inst2vec_utils
from deeplearning.ncc import vocabulary as inst2vec_vocabulary
from experimental.compilers.reachability import llvm_util
from labm8 import bazelutil
from labm8 import labtypes


FLAGS = flags.FLAGS

# The pre-trained embeddings used by default by DeepTuneInst2Vec models.
DEEPTUNE_INST2VEC_EMBEDDINGS = bazelutil.DataPath(
    'phd/deeplearning/ncc/published_results/emb.p')

# The vocabulary to use for encoding sequences used by DeepTuneInst2Vec models.
DEEPTUNE_INST2VEC_VOCAB_PATH = bazelutil.DataPath(
    'phd/deeplearning/ncc/published_results/vocabulary.zip')

# TODO(cec): Add original OpenCL sources to dataframe, then this won't be
# necessary.
DEEPTUNE_INST2VEC_DATA_ARCHIVE = bazelutil.DataArchive(
    'phd/deeplearning/ncc/published_results/task_devmap_kernels.zip')


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


class StaticMapping(HeterogeneousMappingModel):
  __name__ = "Static mapping"
  __basename__ = "static"

  def __init__(self):
    self.model = None

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

    if np.mean(df['y']) >= 0.5:
      self.model = "GPU"
    else:
      self.model = "CPU"

  def predict(self, df: pd.DataFrame, platform_name: str,
              verbose: bool = False):
    del platform_name
    del verbose
    if self.model == "GPU":
      return np.ones(len(df), dtype=np.int32)
    elif self.model == "CPU":
      return np.zeros(len(df), dtype=np.int32)
    else:
      raise LookupError


class Grewe(HeterogeneousMappingModel):
  """Grewe et al. predictive model for heterogeneous device mapping.

  The Grewe et al. predictive model uses decision trees and hand engineered
  features to predict optimal device mapping, described in publication:

    ﻿Grewe, D., Wang, Z., & O’Boyle, M. (2013). Portable Mapping of Data
    Parallel Programs to OpenCL for Heterogeneous Systems. In CGO. IEEE.
    https://doi.org/10.1109/CGO.2013.6494993
  """
  __name__ = "Grewe et al."
  __basename__ = "grewe"

  def __init__(self):
    self.model = None

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


def _ExtractLlvmByteCodeOrDie(src_file_path: pathlib.Path,
                              datafolder: pathlib.Path):
  """Read and compile to bytecode or die."""
  # Read the source file and strip any non-ascii characters.
  with open(src_file_path, 'rb') as f:
    src = f.read().decode('unicode_escape')
  src = src.encode('ascii', 'ignore').decode('ascii')

  # Compile src to bytecode.
  clang_args = opencl.GetClangArgs(use_shim=True) + [
    '-O0', '-S', '-emit-llvm', '-o', '-', '-i', '-',
    # No warnings, and fail immediately on error.
    '-Wno-everything', '-ferror-limit=1',
    # Kernels have headers.
    '-I', str(datafolder / 'kernels_cl'),
    # We don't need the full shim header, just the common constants:
    '-DCLGEN_OPENCL_SHIM_NO_COMMON_TYPES',
    '-DCLGEN_OPENCL_SHIM_NO_UNSUPPORTED_STORAGE_CLASSES_AND_QUALIFIERS',
  ]
  process = clang.Exec(clang_args, stdin=src)
  if process.returncode:
    logging.error("Failed to compile %s", src_file_path)
    logging.error("stderr: %s", process.stderr)
    logging.fatal(f"clang failed with returncode {process.returncode}")
  return process.stdout


def _EncodeSourceBatch(src_file_paths, vocab, datafolder):
  batch = []

  for src_file_path in src_file_paths:
    logging.info('Compiling and encoding %s', src_file_path.name)
    bytecode = _ExtractLlvmByteCodeOrDie(src_file_path, datafolder)
    batch.append((src_file_path,
                  list(vocab.EncodeLlvmBytecode(bytecode).encoded)))

  return batch


def EncodeAndPadSourcesWithInst2Vec(
    df: pd.DataFrame, vocab: inst2vec_vocabulary.VocabularyZipFile,
    datafolder: pathlib.Path, max_sequence_len: typing.Optional[int] = None
) -> typing.Tuple[np.array, int]:
  """Encode and pad source code using inst2vec translation."""
  sequence_lengths = []
  sequences = []

  # A map from source files to encoded sequences, as there can be multiple
  # entries in the dataframe using the same sequence.
  src_path_to_sequence = {}

  src_paths = list(set(
      DataFrameRowToKernelSrcPath(row, datafolder) for _, row in df.iterrows()))

  # Chunk the srcs and process in parallel.
  srcs_per_process = 16
  encode_args = [
    (src_paths[i:i + srcs_per_process], vocab, datafolder)
    for i in range(0, len(src_paths), srcs_per_process)
  ]
  for batch in multiprocessing.Pool().starmap(_EncodeSourceBatch, encode_args):
    for src_file_path, sequence in batch:
      src_path_to_sequence[src_file_path] = sequence

  for _, row in df.iterrows():
    src_file_path = DataFrameRowToKernelSrcPath(row, datafolder)
    sequence = src_path_to_sequence[src_file_path]

    sequence_lengths.append(len(sequence))
    sequences.append(sequence)

  if max_sequence_len is None:
    max_sequence_len = max(sequence_lengths)
  logging.info('Sequence lengths: min=%d, avg=%.2f, max=%d',
               min(sequence_lengths), np.mean(sequence_lengths),
               max_sequence_len)

  encoded = np.array(keras_sequence.pad_sequences(
      sequences, maxlen=max_sequence_len, value=vocab.unknown_token_index))
  encoded = np.vstack([np.expand_dims(x, axis=0) for x in encoded])

  return encoded, max_sequence_len


class DeepTune(HeterogeneousMappingModel):
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
               input_shape: typing.List[int] = (1024,),
               input_type: str = 'int32',
               with_embedding_layer: bool = True):
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
        shape=self.input_shape, dtype=self.input_type, name="model_in")
    lstm_input = input_layer

    if self.with_embedding_layer:
      embedding_dim = atomizer.vocab_size + 1
      lstm_input = Embedding(
          input_dim=embedding_dim, input_length=self.input_shape[0],
          output_dim=self.lstm_layer_size, name="embedding")(input_layer)

    x = LSTM(self.lstm_layer_size, implementation=1, return_sequences=True,
             name="lstm_1")(lstm_input)
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

    self.model = Model(inputs=[auxiliary_inputs, input_layer],
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
      with open(d / 'atomizer.pkl', 'wb') as outfile:
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
        self.atomizer, df['program:opencl_src'], self.input_shape[0])
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


class DeepTuneInst2Vec(DeepTune):
  """DeepTune model using inst2vec neural code comprehension embeddings.

  The inst2vec embeddings are described in:

      ﻿Ben-Nun, T., Jakobovits, A. S., & Hoefler, T. (2018). Neural Code
      Comprehension: A Learnable Representation of Code Semantics. In NeurIPS.
      https://doi.org/arXiv:1806.07336v3
  """
  __name__ = "DeepTuneInst2Vec"
  __basename__ = "deeptune_inst2vec"

  def __init__(self, embedding_matrix: np.ndarray = None,
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

  def EncodeAndPadSources(
      self, df: pd.DataFrame,
      maxlen: typing.Optional[int] = None) -> typing.Tuple[np.array, int]:
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
    sequence_ph = tf.placeholder(dtype=tf.int32)
    normalized_embedding_matrix = tf.nn.l2_normalize(
        self.embedding_matrix, axis=1)
    embedding_input_op = tf.nn.embedding_lookup(
        normalized_embedding_matrix, sequence_ph)

    with tf.Session() as sess:
      # Tensor of shape (len(df), sequence length, embedding dimension).
      embedding_input = sess.run(
          embedding_input_op, feed_dict={sequence_ph: sequences})

    # Get the auxiliary inputs.
    aux_in = np.array([
      df[f"feature:{gpu_name}:transfer"].values,
      df[f"param:{gpu_name}:wgsize"].values,
    ]).T

    return [aux_in, embedding_input]


def _ExtractGraphBatch(src_file_paths: typing.List[pathlib.Path],
                       datafolder: pathlib.Path):
  batch = []

  for src_file_path in src_file_paths:
    logging.info('Compiling %s', src_file_path.name)
    bytecode = _ExtractLlvmByteCodeOrDie(src_file_path, datafolder)
    dot_strings = list(llvm_util.DotCfgsFromBytecode(bytecode))
    if len(dot_strings) != 1:
      logging.fatal('Found %d CFGs in %s', len(dot_strings), src_file_path.name)
    cfg = llvm_util.ControlFlowGraphFromDotSource(dot_strings[0])
    ffg = cfg.BuildFullFlowGraph()
    batch.append((src_file_path, ffg))

  return batch


# A model parameter which has different values for training and testing.
TrainTestParam = collections.namedtuple('TrainTestParam', ['train', 'test'])


class Lda(HeterogeneousMappingModel):
  """Work in progress."""
  __name__ = "lda"
  __basename__ = "lda"

  def __init__(self, embedding_matrix: np.ndarray = None,
               vocabulary_file: typing.Optional[pathlib.Path] = None,
               batch_size: TrainTestParam = TrainTestParam(32, 100)):

    # If no embedding matrix is provided, the default is used.
    if embedding_matrix is None:
      embedding_matrix = inst2vec_utils.ReadEmbeddingFile(
          DEEPTUNE_INST2VEC_EMBEDDINGS)
    if vocabulary_file is None:
      vocabulary_file = DEEPTUNE_INST2VEC_VOCAB_PATH

    self.embedding_matrix = embedding_matrix
    self.vocabulary_file = vocabulary_file
    self.batch_size = batch_size

  def init(self, seed: int, atomizer: atomizers.AtomizerBase):
    # tf.reset_default_graph()

    _, embedding_dim = self.embedding_matrix.shape
    self.model = gn_models.EncodeProcessDecode(global_output_size=2)
    # input_ph, target_ph = CreatePlaceholdersFromGraphs(specs_tr, self.batch_size.train)
    # num_processing_steps = GetNumberOfMessagePassingSteps(specs_tr)

    # A list of outputs, one per processing step.
    # output_ops_tr = model(input_ph, num_processing_steps_tr)
    # output_ops_ge = model(input_ph, num_processing_steps_ge)

    # Training loss.
    # loss_ops_tr = CreateLossOps(target_ph, output_ops_tr)
    # # Loss across processing steps.
    # loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr
    # # Test/generalization loss.
    # loss_ops_ge = CreateLossOps(target_ph, output_ops_ge)
    # loss_op_ge = loss_ops_ge[-1]  # Loss from final processing step.
    #
    # # Optimizer.
    # learning_rate = 1e-3
    # optimizer = tf.train.AdamOptimizer(learning_rate)
    # step_op = optimizer.minimize(loss_op_tr)
    #
    # # Lets an iterable of TF graphs be output from a session as NP graphs.
    # input_ph, target_ph = MakeRunnableInSession(input_ph, target_ph)

  def save(self, outpath: typing.Union[str, pathlib.Path]) -> None:
    """Save model state."""
    pass

  def restore(self, inpath: typing.Union[str, pathlib.Path]) -> None:
    """Load a trained model from file.

    This is called in place of init() if a saved model file exists. It
    must restore all of the required model state.

    Args:
      inpath (str): The path to load the model from. This is the same path as
        was passed to save() to create the file.
    """
    pass

  def train(self, df: pd.DataFrame, platform_name: str,
            verbose: bool = False) -> None:
    """Train a model."""
    feed_dict = self.InputAndTargetGraphsToFeedDict(
        self.DataFrameToInputAndTargetGraphs(df))

  def predict(self, df: pd.DataFrame, platform_name: str,
              verbose: bool = False) -> np.array:
    """Make predictions for programs."""
    return np.ones(len(df))

  def ExtractGraphs(
      self, df: pd.DataFrame, datafolder: pathlib.Path
  ) -> typing.List[llvm_util.LlvmControlFlowGraph]:
    """Extract LLVM full flow graphs for kernels in dataframe."""
    # A map from source files to graphs, as there can be multiple entries in the
    # dataframe using the source.
    src_path_to_graph = {}

    src_paths = list(set(
        DataFrameRowToKernelSrcPath(row, datafolder) for _, row in
        df.iterrows()))

    # Chunk the srcs and process in parallel.
    srcs_per_process = 16
    encode_args = [
      (src_paths[i:i + srcs_per_process], datafolder)
      for i in range(0, len(src_paths), srcs_per_process)
    ]
    for batch in multiprocessing.Pool().starmap(_ExtractGraphBatch,
                                                encode_args):
      for src_file_path, graph in batch:
        src_path_to_graph[src_file_path] = graph

    return [src_path_to_graph[DataFrameRowToKernelSrcPath(row, datafolder)]
            for _, row in df.iterrows()]

  def EncodeGraphsWithInst2Vec(
      self, graphs: typing.Iterable[llvm_util.LlvmControlFlowGraph]
  ) -> typing.Iterable[llvm_util.LlvmControlFlowGraph]:
    with inst2vec_vocabulary.VocabularyZipFile(self.vocabulary_file) as vocab:
      for graph in graphs:
        yield graph

  def DataFrameToInputAndTargetGraphs(self, df: pd.DataFrame):
    with DEEPTUNE_INST2VEC_DATA_ARCHIVE as datafolder:
      graphs = list(
          self.EncodeGraphsWithInst2Vec(self.ExtractGraphs(df, datafolder)))

    model_inputs_ands_targets = []
    for (_, row), graph in zip(df.iterrows(), graphs):
      model_inputs_ands_targets.append(self.GraphToInputTarget(row, graph))

    input_graphs, target_graphs = zip(model_inputs_ands_targets)
    return input_graphs, target_graphs

  def InputAndTargetGraphsToFeedDict(self, input_graphs, target_graphs):
    """Creates placeholders for the model training and evaluation.

    Args:
        graphs: A list of graphs that will be inspected for vector sizes.
            batch_size: Total number of graphs per batch.
        input_ph: The input graph's placeholders, as a graph namedtuple.
        target_ph: The target graph's placeholders, as a graph namedtuple.

    Returns:
        feed_dict: The feed `dict` of input and target placeholders and data.
        raw_graphs: The `dict` of raw networkx graphs.
    """
    # input_graphs = graph_net_utils_np.networkxs_to_graphs_tuple(input_graphs)
    # target_graphs = graph_net_utils_np.networkxs_to_graphs_tuple(target_graphs)
    # feed_dict = {input_ph: input_graphs, target_ph: target_graphs}
    # return feed_dict, graphs

  @staticmethod
  def GraphToInputTarget(
      row: typing.Dict[str, typing.Any],
      graph: llvm_util.LlvmControlFlowGraph
  ) -> typing.Tuple[nx.DiGraph, nx.DiGraph]:
    """Produce two graphs with input and target feature vectors for training.

    A 'features' attributes is added with to nodes, edges, and the global graph,
    which are numpy arrays. The shape of arrays is consistent across input and
    target nodes, edges, and graphs.
    """
    input_graph = graph.copy()
    target_graph = graph.copy()

    # Set node features.
    for _, data in input_graph.nodes(data=True):
      data['features'] = data['inst2vec']

    for _, data in target_graph.nodes(data=True):
      data['features'] = np.ones(1, dtype=float)

    # Set edge features.
    for _, data in input_graph.edges(data=True):
      data['features'] = np.ones(1, dtype=float)

    for _, data in target_graph.edges(data=True):
      data['features'] = np.ones(1, dtype=float)

    # Set global (graph) features.
    input_graph.graph['features'] = np.ones(1, dtype=float)

    target_graph.graph['features'] = row['y_1hot']

    return input_graph, target_graph

  @staticmethod
  def CreateLossOps(target_op, output_ops):
    # TODO(cec): Replace with graph features.
    return [
      tf.losses.softmax_cross_entropy(target_op.nodes, output_op.nodes) +
      tf.losses.softmax_cross_entropy(target_op.edges, output_op.edges)
      for output_op in output_ops
    ]


ALL_MODELS = labtypes.AllSubclassesOfClass(HeterogeneousMappingModel)
