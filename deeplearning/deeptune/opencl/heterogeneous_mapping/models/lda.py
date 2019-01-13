"""LDA model."""
import collections
import multiprocessing
import pathlib
import typing

import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import flags
from absl import logging
from graph_nets import utils_np as graph_net_utils_np
from graph_nets import utils_tf as graph_net_utils_tf
from graph_nets.demos import models as gn_models

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.deeptune.opencl.heterogeneous_mapping import utils
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import base
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import ncc
from deeplearning.ncc import inst2vec_pb2
from deeplearning.ncc import task_utils as inst2vec_utils
from deeplearning.ncc import vocabulary as inst2vec_vocabulary
from experimental.compilers.reachability import llvm_util


FLAGS = flags.FLAGS

# A value which has different values for training and testing.
TrainTestValue = collections.namedtuple('TrainTestValue', ['train', 'test'])

InputTargetValue = collections.namedtuple('InputTargetValue',
                                          ['input', 'target'])


def _ExtractGraphBatchOrDie(
    src_file_paths: typing.List[pathlib.Path], headers_dir: pathlib.Path
) -> typing.List[typing.Tuple[pathlib.Path, llvm_util.LlvmControlFlowGraph]]:
  """Process a patch of OpenCL sources to graphs.

  Args:
    src_file_paths: A list of source code paths.
    headers_dir: The directory containing header files.

  Returns:
    A list of <path,cfg> tuples.
  """
  batch = []

  for src_file_path in src_file_paths:
    logging.info('Compiling %s', src_file_path.name)
    bytecode = ncc.ExtractLlvmByteCodeOrDie(src_file_path, headers_dir)
    dot_strings = list(llvm_util.DotCfgsFromBytecode(bytecode))
    cfgs = [llvm_util.ControlFlowGraphFromDotSource(dot) for dot in dot_strings]
    if len(cfgs) != 1:
      logging.fatal('Found %d CFGs in %s: %s', len(dot_strings),
                    src_file_path.name, [c.graph['name'] for c in cfgs])
    ffg = cfgs[0].BuildFullFlowGraph()

    # Set the input bytecode as a graph property.
    ffg.graph['llvm_bytecode'] = bytecode

    batch.append((src_file_path, ffg))

  return batch


class Lda(base.HeterogeneousMappingModel):
  """Work in progress."""
  __name__ = "lda"
  __basename__ = "lda"

  def __init__(self, embedding_matrix: np.ndarray = None,
               vocabulary_file: typing.Optional[pathlib.Path] = None,
               batch_size: TrainTestValue = TrainTestValue(32, 100)):

    # If no embedding matrix is provided, the default is used.
    if embedding_matrix is None:
      embedding_matrix = inst2vec_utils.ReadEmbeddingFile(
          ncc.DEEPTUNE_INST2VEC_EMBEDDINGS)
    if vocabulary_file is None:
      vocabulary_file = ncc.DEEPTUNE_INST2VEC_VOCAB_PATH

    self.embedding_matrix = embedding_matrix
    self.vocabulary_file = vocabulary_file
    self.batch_size = batch_size

    # Initialized in init().
    self.model: typing.Optional[gn_models.EncodeProcessDecode] = None
    self.num_processing_steps: typing.Optional[TrainTestValue] = None
    self.loss_ops: typing.Optional[TrainTestValue] = None
    self.train_op: typing.Optional[tf.Operation] = None
    self.placeholders: typing.Optional[TrainTestValue] = None

  def init(self, seed: int, atomizer):
    # tf.reset_default_graph()
    self.model = gn_models.EncodeProcessDecode(global_output_size=2)

    # Extract input and target graphs from the full dataset.
    full_df = utils.AddClassificationTargetToDataFrame(
        opencl_device_mapping_dataset.OpenClDeviceMappingsDataset().df,
        "amd_tahiti_7970")
    input_graphs, target_graphs = zip(*self.GraphsToInputTargets(
        self.EncodeGraphs(
            self.ExtractGraphs(full_df))))

    # Create the placeholders.
    placeholders = self.CreatePlaceholdersFromGraphs(
        input_graphs, target_graphs)

    # TODO(cec): Consider whether this should be an __init__ arg.
    num_processing_steps = self.GetNumberOfMessagePassingSteps(
        input_graphs, target_graphs)
    self.num_processing_steps = TrainTestValue(
        num_processing_steps, num_processing_steps)

    # A list of outputs, one per processing step.
    output_ops = TrainTestValue(
        self.model(placeholders.input, self.num_processing_steps.train),
        self.model(placeholders.input, self.num_processing_steps.test))

    # Training loss.
    loss_ops_tr = self.CreateLossOps(placeholders.target, output_ops.train)
    self.loss_ops = TrainTestValue(
        # Training loss is across processing steps.
        sum(loss_ops_tr) / self.num_processing_steps.train,
        # Test loss is from the final processing step.
        self.CreateLossOps(placeholders.target, output_ops.test)[-1],
    )

    # Optimizer.
    learning_rate = 1e-3
    optimizer = tf.train.AdamOptimizer(learning_rate)
    self.train_op = optimizer.minimize(self.loss_ops.train)

    # Lets an iterable of TF graphs be output from a session as NP graphs.
    self.placeholders = InputTargetValue(
        *self.MakeRunnableInSession(*placeholders))

  @property
  def embedding_dim(self):
    """Return the embedding dimensionality."""
    _, embedding_dim = self.embedding_matrix.shape
    return embedding_dim

  def save(self, outpath: typing.Union[str, pathlib.Path]) -> None:
    """Save model state."""
    # TODO(cec): Implement.
    pass

  def restore(self, inpath: typing.Union[str, pathlib.Path]) -> None:
    """Load a trained model from file.

    This is called in place of init() if a saved model file exists. It
    must restore all of the required model state.

    Args:
      inpath (str): The path to load the model from. This is the same path as
        was passed to save() to create the file.
    """
    # TODO(cec): Implement.
    pass

  def train(self, df: pd.DataFrame, platform_name: str,
            verbose: bool = False) -> None:
    """Train a model."""
    graphs = InputTargetValue(
        zip(*self.GraphsToInputTargets(
            self.EncodeGraphs(
                self.ExtractGraphs(df)))))
    feed_dict = self.CreateFeedDict(graphs)

  def predict(self, df: pd.DataFrame, platform_name: str,
              verbose: bool = False) -> np.array:
    """Make predictions for programs."""
    return np.ones(len(df))

  def EncodeGraphs(
      self, data: typing.Iterable[typing.Tuple[typing.Dict[str, typing.Any],
                                               llvm_util.LlvmControlFlowGraph]]
  ) -> typing.Iterable[typing.Tuple[
    typing.Dict[str, typing.Any], llvm_util.LlvmControlFlowGraph]]:
    """Encode inst2vec attributes on graphs.

    Args:
      data: An iterator of <row,cfg> tuples.

    Returns:
      An iterator <row,cfg> tuples.
    """
    with inst2vec_vocabulary.VocabularyZipFile(self.vocabulary_file) as vocab:
      # Create embedding lookup op.
      embedding_lookup_input_ph = tf.placeholder(dtype=tf.int32)
      normalized_embedding_matrix = tf.nn.l2_normalize(
          self.embedding_matrix, axis=1)
      embedding_lookup_op = tf.nn.embedding_lookup(
          normalized_embedding_matrix, embedding_lookup_input_ph)

      with tf.Session() as session:
        for i, (row, graph) in enumerate(data):
          logging.info('Encoding graph %d %s', i, row['program:benchmark_name'])
          yield row, self.EncodeGraph(
              graph, vocab, session, embedding_lookup_op,
              embedding_lookup_input_ph)

  @staticmethod
  def EncodeGraph(graph: llvm_util.LlvmControlFlowGraph,
                  vocab: inst2vec_vocabulary.VocabularyZipFile,
                  session: tf.Session,
                  embedding_lookup_op,
                  embedding_lookup_input_ph) -> llvm_util.LlvmControlFlowGraph:
    """Encode inst2vec attributes on a graph.

    Args:
      graph: The graph to encode.
      vocab: The vocabulary to encode.
      embedding_matrix: The embedding matrix.

    Returns:
      The graph.
    """
    # Encode the entire file with debugging options set. We need to process
    # the entire file so that we can get the struct_dict, which we will need
    # when encoding individual nodes. This could be made faster by simply
    # calling `vocab.GetStructDict(graph.graph['llvm_bytecode'].split('\n'))`,
    # but the extra debug information is useful.
    result = vocab.EncodeLlvmBytecode(
        graph.graph['llvm_bytecode'],
        inst2vec_pb2.EncodeBytecodeOptions(
            set_bytecode_after_preprocessing=True,
            set_unknown_statements=True,
            set_struct_dict=True,
        ))

    # if len(result.encoded) != graph.number_of_nodes():
    #   raise ValueError(
    #       f"Encoded bytecode file contains {len(result.encoded)} statements, "
    #       f"but full flow graph contains {graph.number_of_nodes()} nodes. The "
    #       "two should be equal")

    struct_dict = dict(result.struct_dict)

    # Set debug info as global graph attributes.
    graph.graph['num_unknown_statements'] = len(result.unknown_statements)
    graph.graph['struct_dict'] = struct_dict
    graph.graph[
      'llvm_bytecode_preprocessed'] = result.bytecode_after_preprocessing

    for _, data in graph.nodes(data=True):
      bytecode = data['text']

      # Encode the node's bytecode using the struct dict we derived from the
      # entire file. Since this is a full-flow graph, each instruction's
      # bytecode is a single statement.
      encoded = vocab.EncodeLlvmBytecode(
          bytecode, struct_dict=struct_dict).encoded
      if len(encoded) != 1:
        raise ValueError(
            f"Encoded line `{bytecode}` to {len(encoded)} statements")
      data['inst2vec_encoded'] = encoded[0]

      # Lookup the encoded value in the embedding matrix.
      # TODO(cec): This is a very slow way of doing it. Better would be to
      # collect the encoded values into an array and perform the embedding
      # lookup once.
      sequences = np.array(encoded, dtype=np.int32).reshape((1, 1))
      embedding_vector = session.run(
          embedding_lookup_op,
          feed_dict={embedding_lookup_input_ph: sequences})
      data['inst2vec'] = embedding_vector[0][0]

    return graph

  @staticmethod
  def BuildSrcPathToGraphMap(
      df: pd.DataFrame, headers_dir: pathlib.Path
  ) -> typing.Dict[pathlib.Path, llvm_util.LlvmControlFlowGraph]:
    """Construct a map of OpenCL sources to control flow graphs.

    Args:
      df: The dataset table.
      headers_dir: Directory containing OpenCL headers.

    Returns:
      A dictionary of <path,cfg>s.
    """
    # A map from source files to graphs, as there can be multiple entries in the
    # dataframe using the source.
    src_path_to_graph = {}

    src_paths = list(set(
        ncc.DataFrameRowToKernelSrcPath(row, headers_dir) for _, row in
        df.iterrows()))

    # Chunk the srcs and process in parallel.
    srcs_per_process = 16
    encode_args = [
      (src_paths[i:i + srcs_per_process], headers_dir)
      for i in range(0, len(src_paths), srcs_per_process)
    ]
    batches = multiprocessing.Pool().starmap(
        _ExtractGraphBatchOrDie, encode_args)
    for batch in batches:
      for src_file_path, graph in batch:
        src_path_to_graph[src_file_path] = graph

    return src_path_to_graph

  @staticmethod
  def GetNumberOfMessagePassingSteps(
      input_graphs: typing.List[nx.DiGraph],
      target_graphs: typing.List[nx.DiGraph]) -> int:
    del target_graphs
    return max([g.number_of_edges() for g in input_graphs])

  @classmethod
  def ExtractGraphs(
      cls, df: pd.DataFrame
  ) -> typing.Iterable[typing.Tuple[typing.Dict[str, typing.Any],
                                    llvm_util.LlvmControlFlowGraph]]:
    """Extract control flow graphs from a dataframe.

    Args:
      df: The dataset table.

    Returns:
      An iterator of <row,cfg> tuples.
    """
    with ncc.DEEPTUNE_INST2VEC_DATA_ARCHIVE as headers_dir:
      # Make a list of source paths so that we can use it to index into the
      # src_path_to_graph map.
      src_paths = [
        ncc.DataFrameRowToKernelSrcPath(row, headers_dir)
        for _, row in df.iterrows()
      ]
      # Build a map of src paths to graphs.
      src_path_to_graph = cls.BuildSrcPathToGraphMap(df, headers_dir)

    for (_, row), src_path in zip(df.iterrows(), src_paths):
      graph = src_path_to_graph[src_path]
      yield row, graph

  @classmethod
  def GraphsToInputTargets(cls, data: typing.Iterable[
    typing.Tuple[typing.Dict[str, typing.Any],
                 llvm_util.LlvmControlFlowGraph]]):
    """Produce two graphs with input and target feature vectors for training.

    Args:
      An iterator of <row,cfg> pairs.
    """
    for row, graph in data:
      yield cls.GraphToInputTarget(row, graph)

  @staticmethod
  def GraphToInputTarget(
      row: typing.Dict[str, typing.Any],
      graph: llvm_util.LlvmControlFlowGraph
  ) -> typing.Tuple[nx.DiGraph, nx.DiGraph]:
    """Produce two graphs with input and target feature vectors for training.

    A 'features' attributes is added with to nodes, edges, and the global graph,
    which are numpy arrays. The shape of arrays is consistent across input and
    target nodes, edges, and graphs.

    Args:
      row: A dataframe row as a dictionary.
      graph: A control flow graph.

    Returns:
      A pair of graphs.
    """
    input_graph = graph.copy()
    target_graph = graph.copy()

    # Set node features.
    for _, data in input_graph.nodes(data=True):
      data['features'] = data['inst2vec'].astype(np.float64)

    for _, data in target_graph.nodes(data=True):
      data['features'] = np.ones(1, dtype=np.float64)

    # Set edge features.
    for _, _, data in input_graph.edges(data=True):
      data['features'] = np.ones(1, dtype=np.float64)

    for _, _, data in target_graph.edges(data=True):
      data['features'] = np.ones(1, dtype=np.float64)

    # Set global (graph) features.
    input_graph.graph['features'] = np.ones(1, dtype=np.float64)

    target_graph.graph['features'] = row['y_1hot'].astype(np.float64)

    return input_graph, target_graph

  @staticmethod
  def CreateLossOps(target_op, output_ops):
    return [
      tf.losses.softmax_cross_entropy(target_op.globals, output_op.globals)
      for output_op in output_ops
    ]

  @staticmethod
  def MakeRunnableInSession(*args):
    """Lets an iterable of TF graphs be output from a session as NP graphs."""
    return [graph_net_utils_tf.make_runnable_in_session(a) for a in args]

  @staticmethod
  def CreatePlaceholdersFromGraphs(input_graphs: typing.List[nx.DiGraph],
                                   target_graphs: typing.List[nx.DiGraph]):
    """Creates placeholders for the model training and evaluation.

    Args:
      input_graphs: A list of input graphs.
      target_graphs: A list of input graphs.

    Returns:
      A tuple of the input graph's and target graph's placeholders, as a
      graph namedtuple.
    """
    input_ph = graph_net_utils_tf.placeholders_from_networkxs(
        input_graphs, force_dynamic_num_graphs=True)
    target_ph = graph_net_utils_tf.placeholders_from_networkxs(
        target_graphs, force_dynamic_num_graphs=True)
    return InputTargetValue(input_ph, target_ph)

  def CreateFeedDict(self, graphs: InputTargetValue):
    """Creates placeholders for the model training and evaluation.

    Args:
        graphs: A list of graphs that will be inspected for vector sizes.

    Returns:
        The feed `dict` of input and target placeholders and data.
    """
    input_graphs = graph_net_utils_np.networkxs_to_graphs_tuple(graphs.input)
    target_graphs = graph_net_utils_np.networkxs_to_graphs_tuple(graphs.target)
    feed_dict = {
      self.placeholders.input: input_graphs,
      self.placeholders.target: target_graphs
    }
    return feed_dict
