"""My (Chris's) API for inst2vec codebase."""
import copy
import networkx as nx
import numpy as np
import typing

from deeplearning.ncc import vocabulary
from deeplearning.ncc.inst2vec import inst2vec_preprocess as preprocess
from labm8 import app


FLAGS = app.FLAGS


# Type hints for inst2vec parameters and return values.
MultiEdges = typing.Dict[str, typing.List[str]]


def PreprocessLlvmBytecode(bytecode: str) -> str:
  """Pre-process an LLVM bytecode for encoding."""
  bytecode_lines = bytecode.split('\n')
  preprocessed, functions_declared_in_files = preprocess.preprocess(
      [bytecode_lines])
  del functions_declared_in_files
  return '\n'.join(preprocessed[0])


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
  preprocessed_bytecode = PreprocessLlvmBytecode(bytecode)
  llvm_lines = preprocessed_bytecode.split('\n')
  functions_declared_in_file = preprocess.GetFunctionsDeclaredInFile(llvm_lines)
  # File name is required by BuildContextualFlowGraph(), but is used only to
  # produce descriptive error messages, so any value will do.
  filename = '[input]'
  xfg, multi_edges = preprocess.BuildContextualFlowGraph(
      llvm_lines, functions_declared_in_file, filename)
  del multi_edges  # unused
  return xfg


def XfgToDot(xfg, dotpath):
  xfg = copy.deepcopy(xfg)
  for edge in xfg.edges:
    xfg.edges[edge]['label'] = xfg.edges[edge]['stmt'].encode('ascii', 'ignore').decode('ascii').replace('\0', '0')
    del xfg.edges[edge]['stmt']
  nx.drawing.nx_pydot.write_dot(xfg, dotpath)
