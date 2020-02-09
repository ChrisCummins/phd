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
"""Generate program graphs from XLA HLO modules."""
from typing import Optional

import networkx as nx
from tensorflow.compiler.xla.service import hlo_pb2

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.xla2graph.py import xla2graph_pybind
from labm8.py import app

FLAGS = app.FLAGS


def BuildProgramGraphProto(
  hlo_proto: hlo_pb2.HloProto,
  graph: Optional[programl_pb2.ProgramGraph] = None,
) -> programl_pb2.ProgramGraph:
  """Construct a program graph for the given LLVM IR.

  Args:
    hlo_proto: The LLVM IR string for a module.
    graph: An existing graph message to write the result to. If not provided,
      a new graph message is constructed.

  Returns:
    A ProgramGraph message instance.

  Raises:
    ValueError: If graph construction fails.
  """
  # This requires a round trip serialized to / from strings, since I can't
  # figure out a way to get pybind11 to auto-generate bindings for protocol
  # buffers.
  graph = graph or programl_pb2.ProgramGraph()
  serialized_graph = xla2graph_pybind.BuildProgramGraphProto(
    hlo_proto.SerializeToString()
  )
  graph.ParseFromString(serialized_graph)
  return graph


def BuildProgramGraphNetworkX(
  hlo_proto: hlo_pb2.HloProto,
  graph: Optional[programl_pb2.ProgramGraph] = None,
) -> nx.MultiDiGraph:
  """Construct a NetworkX program graph for the given LLVM IR.

  Args:
    hlo_proto: The LLVM IR string for a module.
    graph: An existing graph message to write the result to. If not provided,
      a new graph message is constructed.

  Returns:
    A NetworkX MultiDiGraph instance.

  Raises:
    ValueError: If graph construction fails.
  """
  # NOTE: A more direct approach to generating a networkx graph instance would
  # be to add a --stdout_fmt=json option to
  # //deeplearning/ml4pl/graphs/xla2graph which would produce output in the
  # format expected by networkx. See:
  # https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.convert.to_dict_of_dicts.html#networkx.convert.to_dict_of_dicts
  return programl.ProgramGraphToNetworkX(
    BuildProgramGraphProto(hlo_proto=hlo_proto, graph=graph)
  )
