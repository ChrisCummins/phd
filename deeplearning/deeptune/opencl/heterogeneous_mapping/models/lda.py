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
from graph_nets.demos import models as gn_models

from deeplearning.deeptune.opencl.heterogeneous_mapping.models import base
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import ncc
from deeplearning.ncc import task_utils as inst2vec_utils
from deeplearning.ncc import vocabulary as inst2vec_vocabulary
from experimental.compilers.reachability import llvm_util


FLAGS = flags.FLAGS

# A model parameter which has different values for training and testing.
TrainTestParam = collections.namedtuple('TrainTestParam', ['train', 'test'])


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
    if len(dot_strings) != 1:
      logging.fatal('Found %d CFGs in %s', len(dot_strings), src_file_path.name)
    cfg = llvm_util.ControlFlowGraphFromDotSource(dot_strings[0])
    ffg = cfg.BuildFullFlowGraph()

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
               batch_size: TrainTestParam = TrainTestParam(32, 100)):

    # If no embedding matrix is provided, the default is used.
    if embedding_matrix is None:
      embedding_matrix = inst2vec_utils.ReadEmbeddingFile(
          ncc.DEEPTUNE_INST2VEC_EMBEDDINGS)
    if vocabulary_file is None:
      vocabulary_file = ncc.DEEPTUNE_INST2VEC_VOCAB_PATH

    self.embedding_matrix = embedding_matrix
    self.vocabulary_file = vocabulary_file
    self.batch_size = batch_size

  def init(self, seed: int, atomizer):
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
    feed_dict = self.InputTargetsToFeedDict(
        self.GraphsToInputTargets(
            self.EncodeGraphs(
                self.ExtractGraphs(df))))

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
      for row, graph in data:
        yield row, self.EncodeGraph(graph, vocab, self.embedding_matrix)

  @staticmethod
  def EncodeGraph(graph: llvm_util.LlvmControlFlowGraph,
                  vocab: inst2vec_vocabulary.VocabularyZipFile,
                  embedding_matrix: np.array) -> llvm_util.LlvmControlFlowGraph:
    """Encode inst2vec attributes on a graph.
    
    Args:
      graph: The graph to encode. 
      vocab: The vocabulary to encode. 
      embedding_matrix: The embedding matrix. 
      
    Returns:
      The graph.
    """
    for _, data in graph.nodes(data=True):
      # TODO(cec): Pre-process the instruction, lookup in vocabulary,
      # lookup embedding.
      data['inst2vec'] = 'TODO'

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

  def InputTargetsToFeedDict(
      self, data: typing.Iterable[typing.Tuple[nx.DiGraph, nx.DiGraph]]):
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
    input_graphs, target_graphs = zip(*data)
    # input_graphs = graph_net_utils_np.networkxs_to_graphs_tuple(input_graphs)
    # target_graphs = graph_net_utils_np.networkxs_to_graphs_tuple(target_graphs)
    # feed_dict = {input_ph: input_graphs, target_ph: target_graphs}
    # return feed_dict, graphs

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
      data['features'] = data['inst2vec']

    for _, data in target_graph.nodes(data=True):
      data['features'] = np.ones(1, dtype=float)

    # Set edge features.
    for _, _, data in input_graph.edges(data=True):
      data['features'] = np.ones(1, dtype=float)

    for _, _, data in target_graph.edges(data=True):
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
