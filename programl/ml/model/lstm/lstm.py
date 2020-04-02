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
"""LSTM sequential classifier models."""
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from labm8.py import app
from programl.ml.model import run
from programl.ml.model.lstm import graph_lstm
from programl.ml.model.lstm import node_lstm

FLAGS = app.FLAGS


def main():
  """Main entry point."""
  graph_db: graph_tuple_database.Database = FLAGS.graph_db()
  if graph_db.node_y_dimensionality:
    model_class = node_lstm.NodeLstm
  else:
    model_class = graph_lstm.GraphLstm

  run.Run(model_class, graph_db)


if __name__ == "__main__":
  app.Run(main)
