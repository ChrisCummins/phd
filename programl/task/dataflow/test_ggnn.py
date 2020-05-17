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
"""Test a GGNN to estimate solutions for classic data flow problems.

This script reads ProGraML graphs and uses a GGNN to predict binary
classification targets for data flow problems.
"""
import pathlib

from labm8.py import app
from programl.task.dataflow import dataflow

app.DEFINE_string(
  "path",
  str(pathlib.Path("~/programl/dataflow").expanduser()),
  "The working directory for writing logs",
)
app.DEFINE_input_path(
  "model_to_test", None, "The working directory for writing logs", is_dir=True
)
app.DEFINE_string("analysis", "reachability", "The analysis type to use.")
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
  "If set, limit the size of dataflow-annotated graphs used to only those "
  "with data_flow_steps <= message_passing_step_count",
)
app.DEFINE_boolean(
  "cdfg",
  False,
  "If set, use the CDFG representation for programs. Defaults to ProGraML "
  "representations.",
)
FLAGS = app.FLAGS


def Main():
  """Main entry point."""
  dataflow.TestDataflowGGNN(
    path=pathlib.Path(FLAGS.path),
    log_dir=FLAGS.model_to_test,
    analysis=FLAGS.analysis,
    limit_max_data_flow_steps=FLAGS.limit_max_data_flow_steps,
    batch_size=FLAGS.batch_size,
    use_cdfg=FLAGS.cdfg,
  )


if __name__ == "__main__":
  app.Run(Main)
