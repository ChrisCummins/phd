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
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import fs
from labm8.py import humanize
from labm8.py import prof
from programl.graph.analysis.py import analysis
from programl.graph.py import graph_tuple_builder
from programl.ir.llvm import inst2vec_encoder
from programl.ir.llvm.py import llvm

FLAGS = app.FLAGS

LLVM_IR = bazelutil.DataPath("phd/programl/test/data/llvm_ir")


def Main():
  irs = [fs.Read(path) for path in LLVM_IR.iterdir()]
  ir_count = len(irs)

  with prof.ProfileToStdout(
    lambda t: (
      f"STAGE 1: Construct unlabelled graphs (llvm2graph)         "
      f"({humanize.Duration(t / ir_count)} / IR)"
    )
  ):
    graphs = [llvm.BuildProgramGraph(ir) for ir in irs]

  encoder = inst2vec_encoder.Inst2vecEncoder()
  with prof.ProfileToStdout(
    lambda t: (
      f"STAGE 2: Encode graphs (inst2vec)                         "
      f"({humanize.Duration(t / ir_count)} / IR)"
    )
  ):
    graphs = [encoder.Encode(graph, ir) for graph, ir in zip(graphs, irs)]

  features_count = 0
  features_lists = []
  with prof.ProfileToStdout(
    lambda t: (
      f"STAGE 3: Produce labelled graphs (reachability analysis)  "
      f"({humanize.Duration(t / features_count)} / graph)"
    )
  ):
    for graph in graphs:
      features_list = analysis.RunAnalysis("reachability", graph)
      features_count += len(features_list.graph)
      features_lists.append(features_list)

  builder = graph_tuple_builder.GraphTupleBuilder(
    features=graph_tuple_builder.Feature(
      "node", "data_flow_root_node", "int64"
    ),
    labels=graph_tuple_builder.Feature("node", "data_flow_value", "int64"),
  )

  max_node_size = 10000
  graph_tuples = []
  with prof.ProfileToStdout(
    lambda t: (
      f"STAGE 4: Construct graph tuples                           "
      f"({humanize.Duration(t / features_count)} / graph)"
    )
  ):
    node_size = 0
    for graph, features_list in zip(graphs, features_lists):
      for graph_features in features_list.graph:
        if node_size + len(graph.node) > max_node_size:
          graph_tuples.append(builder.Build())
          node_size = 0
        node_size += len(graph.node)
        builder.AddProgramGraphFeatures(graph, graph_features)

  print("=================================")
  print(f"Unlabelled graphs count: {ir_count}")
  print(f"  Labelled graphs count: {features_count}")
  print(f"     Graph tuples count: {len(graph_tuples)}")
  print(f"       Total node count: {sum(gt.node_count for gt in graph_tuples)}")
  print(f"       Total edge count: {sum(gt.edge_count for gt in graph_tuples)}")


if __name__ == "__main__":
  app.Run(Main)
