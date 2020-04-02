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
"""TODO."""
import pathlib
import random

from labm8.py import app
from labm8.py import humanize
from labm8.py import pbutil
from programl.graph.py import graph_tuple_builder
from programl.proto import program_graph_features_pb2
from programl.proto import program_graph_pb2

FLAGS = app.FLAGS

app.DEFINE_str(
  "path",
  str(pathlib.Path("~/programl/dataflow").expanduser()),
  "TODO.",
  is_dir=True,
)
app.DEFINE_string("analysis", "reachability", "TODO.")


def Main():
  path = pathlib.Path(FLAGS.path)
  analysis = FLAGS.analysis

  builder = graph_tuple_builder.GraphTupleBuilder(
    features=graph_tuple_builder.Feature(
      "node", "data_flow_root_node", "int64"
    ),
    labels=graph_tuple_builder.Feature("node", "data_flow_value", "int64"),
  )

  max_node_size = 10000
  node_size = 0
  graph_tuples = []

  graph_count = len(list((path / "train").iterdir()))

  files = list((path / "val").iterdir())
  random.shuffle(files)

  for i, graph_path in enumerate(files):
    stem = graph_path.name[: -len("ProgramGraph.pb")]
    name = f"{stem}ProgramGraphFeaturesList.pb"
    features_path = path / analysis / name
    if features_path.is_file():
      graph = pbutil.FromFile(graph_path, program_graph_pb2.ProgramGraph())
      features_list = pbutil.FromFile(
        features_path, program_graph_features_pb2.ProgramGraphFeaturesList()
      )
      for graph_features in features_list.graph:
        if node_size + len(graph.node) > max_node_size:
          graph_tuples.append(builder.Build())
          node_size = 0
          app.Log(
            1,
            "made a graph tuple (%s of %s graphs)",
            humanize.Commas(i + 1),
            humanize.Commas(graph_count),
          )
        node_size += len(graph.node)
        builder.AddProgramGraphFeatures(graph, graph_features)


if __name__ == "__main__":
  app.Run(Main)
