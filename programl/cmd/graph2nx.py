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
"""A module for encoding node embeddings.

When executed as a binary, this program reads a single program graph from
stdin, encodes it, and writes a graph to stdout. Use --stdin_fmt and
--stdout_fmt to convert between different graph types, and --ir to read the
IR file that the graph was constructed from, required for resolving struct
definitions.

Example usage:

  Encode a program graph binary proto and write the result as text format:

    $ bazel run //deeplearning/ml4pl/graphs/llvm2graph:node_encoder -- \
        --stdin_fmt=pb \
        --stdout_fmt=pbtxt \
        --ir=/tmp/source.ll \
        < /tmp/proto.pb > /tmp/proto.pbtxt
"""
import pickle
import sys

from labm8.py import pbutil
from programl.graph.format.py import nx_format
from programl.proto import program_graph_pb2


if __name__ == "__main__":
  proto = program_graph_pb2.ProgramGraph()
  pbutil.FromString(sys.stdin.buffer.read().decode("utf-8"), proto)
  pickle.dump(nx_format.ProgramGraphToNetworkX(proto), sys.stdout.buffer)
