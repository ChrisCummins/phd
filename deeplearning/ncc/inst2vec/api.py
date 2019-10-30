"""My (Chris's) API for inst2vec codebase."""
import copy
import pickle
import typing

import networkx as nx
import numpy as np
from labm8 import app
from labm8 import bazelutil

from deeplearning.ncc import vocabulary
from deeplearning.ncc.inst2vec import inst2vec_preprocess as preprocess

FLAGS = app.FLAGS

# Type hints for inst2vec parameters and return values.
MultiEdges = typing.Dict[str, typing.List[str]]

INST2VEC_DICITONARY_PATH = bazelutil.DataPath(
    'phd/deeplearning/ncc/published_results/dic_pickle')


def PreprocessLlvmBytecode(bytecode: str) -> str:
  """Pre-process an LLVM bytecode for encoding."""
  bytecode_lines = bytecode.split('\n')
  preprocessed, functions_declared_in_files = preprocess.preprocess(
      [bytecode_lines])
  del functions_declared_in_files
  return '\n'.join(preprocessed[0])


def PretrainedEmbeddingIndicesDictionary() -> typing.Dict[str, int]:
  """Read and return the embeddings indices dictionary."""
  with open(INST2VEC_DICITONARY_PATH, 'rb') as f:
    return pickle.load(f)


def EncodeLlvmBytecode(bytecode: str,
                       vocab: vocabulary.VocabularyZipFile) -> typing.List[int]:
  """Encode an LLVM bytecode to an array of vocabulary indices."""
  raise NotImplementedError


def EmbedEncoded(encoded: typing.List[int], embedding_matrix) -> np.ndarray:
  """Embed an array of vocabulary indices."""
  raise NotImplementedError


def Inst2Vec(bytecode: str, vocab: vocabulary.VocabularyZipFile,
             embedding) -> np.ndarray:
  """Transform an LLVM bytecode to an array of embeddings.

  Args:
    bytecode: The input bytecode.
    vocab: The vocabulary.
    embedding: The embedding.

  Returns:
    An array of embeddings.
  """
  embed = lambda x: EmbedEncoded(x, embedding)
  encode = lambda x: EncodeLlvmBytecode(x, vocab)
  return embed(encode(PreprocessLlvmBytecode(bytecode)))


def LlvmBytecodeToContextualFlowGraph(bytecode: str) -> nx.DiGraph:
  """
  Given a file of source code, construct a context graph.

  Args:
    bytecode: The input bytecode.
    vocab: The vocabulary.

  Returns:
    A <digraph, multi_edge_list> tuple, where <digraph> is a directed graph in
    which nodes are identifiers or ad-hoc and edges are statements which is
    meant as a representation of both data and flow control of the code
    capturing the notion of context; and <multi_edge_list> is a dictionary of
    edges that have parallel edges.
  """
  # First preprocess the bytecode, with the side-effect of getting the list of
  # function declarations.
  bytecode_lines = bytecode.split('\n')
  preprocessed_bytecodes, functions_declared_in_files = preprocess.preprocess(
      [bytecode_lines])
  preprocessed_bytecode = preprocessed_bytecodes[0]
  functions_declared_in_file = functions_declared_in_files[0]

  # Then build the XFG from the preprocessed bytecode.
  #
  # File name is required by BuildContextualFlowGraph(), but is used only to
  # produce descriptive error messages, so any value will do.
  xfg, multi_edges = preprocess.BuildContextualFlowGraph(
      preprocessed_bytecode, functions_declared_in_file, filename='[input]')
  del multi_edges  # unused
  return xfg


def XfgToDot(xfg, dotpath):
  xfg = copy.deepcopy(xfg)
  for edge in xfg.edges:
    xfg.edges[edge]['label'] = xfg.edges[edge]['stmt'].encode(
        'ascii', 'ignore').decode('ascii').replace('\0', '0')
    del xfg.edges[edge]['stmt']
  nx.drawing.nx_pydot.write_dot(xfg, dotpath)
