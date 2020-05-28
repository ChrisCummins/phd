# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A module for encoding LLVM-IR program graphs using inst2vec."""
import pickle
from typing import List
from typing import Optional

import numpy as np

from deeplearning.ncc.inst2vec import inst2vec_preprocess
from labm8.py import bazelutil
from labm8.py import decorators
from programl.proto import node_pb2
from programl.proto import program_graph_pb2


DICTIONARY = bazelutil.DataPath(
  "phd/programl/ir/llvm/internal/inst2vec_augmented_dictionary.pickle"
)
AUGMENTED_INST2VEC_EMBEDDINGS = bazelutil.DataPath(
  "phd/programl/ir/llvm/internal/inst2vec_augmented_embeddings.pickle"
)


def NodeFullText(node: node_pb2.Node) -> str:
  """Get the full text of a node, or an empty string if not set."""
  if len(node.features.feature["full_text"].bytes_list.value):
    return (
      node.features.feature["full_text"].bytes_list.value[0].decode("utf-8")
    )
  return ""


class Inst2vecEncoder(object):
  """An encoder for LLVM program graphs using inst2vec."""

  def __init__(self):
    with open(str(DICTIONARY), "rb") as f:
      self.dictionary = pickle.load(f)

    # TODO(github.com/ChrisCummins/ProGraML/issues/12): In the future we may
    # want to add support for different numbers of embeddings tables, or
    # embeddings tables with different types. This is hardcoded to support only
    # two embeddings tables: our augmented inst2vec statement embeddings, and
    # a binary 'selector' table which can be used to select one or more nodes
    # of interests in graphs, e.g. for setting the starting point for computing
    # iterative data flow analyses.
    with open(str(AUGMENTED_INST2VEC_EMBEDDINGS), "rb") as f:
      self.node_text_embeddings = pickle.load(f)

  def Encode(
    self, proto: program_graph_pb2.ProgramGraph, ir: Optional[str] = None
  ) -> program_graph_pb2.ProgramGraph:
    """Pre-process the node text and set the text embedding index.

    For each node, this sets 'inst2vec_preprocessed' and 'inst2vec_embedding'
    features.

    Args:
      proto: The ProgramGraph to encode.
      ir: The LLVM IR that was used to construct the graph. This is required for
        struct inlining. If struct inlining is not required, this may be
        omitted.

    Returns:
      The input proto.
    """
    # Gather the instruction texts to pre-process.
    lines = [
      [NodeFullText(node)]
      for node in proto.node
      if node.type == node_pb2.Node.INSTRUCTION
    ]

    if ir:
      # NOTE(github.com/ChrisCummins/ProGraML/issues/57): Extract the struct
      # definitions from the IR and inline their definitions in place of the
      # struct names. This is brittle string substitutions, in the future we
      # should do this inlining in llvm2graph where we have a parsed
      # llvm::Module.
      try:
        structs = inst2vec_preprocess.GetStructTypes(ir)
        for line in lines:
          for struct, definition in structs.items():
            line[0] = line[0].replace(struct, definition)
      except ValueError:
        pass

    preprocessed_lines, _ = inst2vec_preprocess.preprocess(lines)
    preprocessed_texts = [
      inst2vec_preprocess.PreprocessStatement(x[0] if len(x) else "")
      for x in preprocessed_lines
    ]

    # Add the node features.
    var_embedding = self.dictionary["!IDENTIFIER"]
    const_embedding = self.dictionary["!IMMEDIATE"]

    text_index = 0
    for node in proto.node:
      if node.type == node_pb2.Node.INSTRUCTION:
        text = preprocessed_texts[text_index].encode("utf-8")
        text_index += 1
        embedding = self.dictionary.get(text, self.dictionary["!UNK"])
        node.features.feature["inst2vec_preprocessed"].bytes_list.value.append(
          text
        )
        node.features.feature["inst2vec_embedding"].int64_list.value.append(
          embedding
        )
      elif node.type == node_pb2.Node.VARIABLE:
        node.features.feature["inst2vec_embedding"].int64_list.value.append(
          var_embedding
        )
      elif node.type == node_pb2.Node.CONSTANT:
        node.features.feature["inst2vec_embedding"].int64_list.value.append(
          const_embedding
        )

    proto.features.feature["inst2vec_annotated"].int64_list.value.append(1)
    return proto

  @decorators.memoized_property
  def embeddings_tables(self) -> List[np.array]:
    """Return the embeddings tables."""
    node_selector = np.vstack([[1, 0], [0, 1],]).astype(np.float64)
    return [self.node_text_embeddings, node_selector]
