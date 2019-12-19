# Copyright 2019 the ProGraML authors.
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
"""Delete graphs with a single node."""
from deeplearning.ml4pl.graphs import graph_database
from labm8.py import app

FLAGS = app.FLAGS

app.DEFINE_database(
  "graph_db",
  graph_database.Database,
  None,
  "URL of database to modify.",
  must_exist=True,
)


def DeleteSingleNodeGraphs(graph_db: graph_database.Database) -> None:
  """Propagate the `group` column from one database to another."""
  with graph_db.Session() as session:
    query = session.query(graph_database.GraphMeta.id).filter(
      graph_database.GraphMeta.node_count == 1
    )
    ids_to_delete = [row.id for row in query]

  graph_db.DeleteGraphs(ids_to_delete)


def main():
  """Main entry point."""
  DeleteSingleNodeGraphs(FLAGS.graph_db())
  app.Log(1, "done")


if __name__ == "__main__":
  app.Run(main)
