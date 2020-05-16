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
"""Train a GGNN to estimate solutions for classic data flow problems.

This script reads ProGraML graphs and uses a GGNN to predict binary
classification targets for data flow problems.
"""
import pathlib
from typing import Dict

from labm8.py import app
from programl.task.dataflow import dataflow

app.DEFINE_string(
  "path",
  str(pathlib.Path("~/programl/dataflow").expanduser()),
  "The path to read from",
)
app.DEFINE_string("analysis", "reachability", "The analysis type to use.")
app.DEFINE_integer(
  "val_graph_count", 10000, "The number of graphs to use in the validation set."
)
app.DEFINE_integer(
  "val_seed", 0xCC, "The seed value for randomly sampling validation graphs.",
)
app.DEFINE_integer(
  "batch_size",
  50000,
  "The number of nodes in a graph. "
  "On our system, we observed that a batch size of 50,000 nodes requires "
  "about 5.2GB of GPU VRAM.",
)
app.DEFINE_boolean(
  "limit_max_data_flow_steps",
  True,
  "If set, limit the size of dataflow-annotated graphs used to only those with "
  "data_flow_steps <= message_passing_step_count",
)
app.DEFINE_list(
  "train_graph_counts",
  [
    1000,
    2000,
    3000,
    4000,
    5000,
    10000,
    20000,
    30000,
    40000,
    50000,
    100000,
    200000,
    300000,
    400000,
    500000,
    1000000,
  ],
  "The list of cumulative training graph counts to evaluate at.",
)
app.DEFINE_boolean("test", True, "Whether to test the model after training.")
FLAGS = app.FLAGS


def LoadVocabulary(path: pathlib.Path) -> Dict[str, int]:
  with open(path) as f:
    vocab = f.readlines()
  return {v: i for i, v in enumerate(vocab)}


def Main():
  """Main entry point."""
  log_dir = dataflow.TrainDataflowGGNN(
    path=pathlib.Path(FLAGS.path),
    analysis=FLAGS.analysis,
    limit_max_data_flow_steps=FLAGS.limit_max_data_flow_steps,
    train_graph_counts=[int(x) for x in FLAGS.train_graph_counts],
    val_graph_count=FLAGS.val_graph_count,
    val_seed=FLAGS.val_seed,
    batch_size=FLAGS.batch_size,
  )

  if FLAGS.test:
    dataflow.TestDataflowGGNN(
      path=pathlib.Path(FLAGS.path),
      log_dir=log_dir,
      analysis=FLAGS.analysis,
      limit_max_data_flow_steps=FLAGS.limit_max_data_flow_steps,
      batch_size=FLAGS.batch_size,
    )


if __name__ == "__main__":
  app.Run(Main)
