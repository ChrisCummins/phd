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
"""Utility functions for working with program graphs.

When executed as a binary, this program reads a single program graph from
stdin, and writes a the same graph to stdout. Use --stdin_fmt and --stdout_fmt
to convert between different graph types.

Example usage:

  Convert a binary protocol buffer to a text version:

    $ bazel run //deeplearning/ml4pl/graphs:programl -- \
        --stdin_fmt=pb \
        --stdout_fmt=pbtxt \
        < /tmp/proto.pb > /tmp/proto.pbtxt
"""
import networkx as nx

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.py import graph_builder_pybind
from labm8.py import app

FLAGS = app.FLAGS


class GraphBuilder(graph_builder_pybind.GraphBuilder):
  """The format of a graph read from stdin."""

  @property
  def g(self) -> nx.MultiDiGraph:
    return programl.ProgramGraphProtoToNetworkX(self.proto)

  @property
  def proto(self) -> programl_pb2.ProgramGraphProto:
    proto = programl_pb2.ProgramGraphProto()
    proto.ParseFromString(self.GetSerializedGraphProto())
    return proto
