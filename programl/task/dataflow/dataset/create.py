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
"""Export LLVM-IR from legacy database."""
import codecs
import multiprocessing
import os
import pathlib
import pickle
import random
import shutil
import subprocess

from MySQLdb import _mysql

from labm8.py import app
from labm8.py import bazelutil
from labm8.py import labtypes
from labm8.py import pbutil
from labm8.py import progress
from programl.ir.llvm.py import llvm
from programl.proto import ir_pb2

app.DEFINE_string("classifyapp", None, "Path of the classifyapp database.")
app.DEFINE_string("host", None, "The database to export from.")
app.DEFINE_string("user", None, "The database to export from.")
app.DEFINE_string("pwd", None, "The database to export from.")
app.DEFINE_string("db", None, "The database to export from.")
app.DEFINE_string("path", None, "The directory to export to.")
FLAGS = app.FLAGS

CREATE_LABELS = bazelutil.DataPath(
  "phd/programl/task/dataflow/dataset/create_labels"
)


def _ProcessRow(path, row, i) -> None:
  source, src_lang, ir_type, binary_ir = row

  # Decode database row.
  source = source.decode("utf-8")
  src_lang = {
    "C": "c",
    "CPP": "cc",
    "OPENCL": "cl",
    "SWIFT": "swift",
    "HASKELL": "hs",
    "FORTRAN": "f90",
  }[src_lang.decode("utf-8")]
  ir_type = ir_type.decode("utf-8")

  source = {
    "github.com/av-maramzin/SNU_NPB:NPB3.3-SER-C": "npb-3.3-ser-c",
    "pact17_opencl_devmap": "opencl",
  }.get(source, source)
  if source.startswith("sqlite:///"):
    source = "github"

  ir = pickle.loads(codecs.decode(binary_ir, "zlib"))
  compiler_version = {"LLVM_6_0": 600, "LLVM_3_5": 350,}[ir_type]

  ir = ir_pb2.Ir(
    type=ir_pb2.Ir.LLVM, compiler_version=compiler_version, text=ir
  )
  name = f"{source}.{i}.{src_lang}"
  pbutil.ToFile(ir, path / f"ir/{name}.Ir.pb")

  # Convert to ProgramGraph.
  try:
    graph = llvm.BuildProgramGraph(ir.text)
    pbutil.ToFile(graph, path / f"graphs/{name}.ProgramGraph.pb")

    # Put into train/val/test bin.
    r = random.random()
    if r < 0.6:
      dst = "train"
    elif r < 0.8:
      dst = "val"
    else:
      dst = "test"
    os.symlink(
      f"../graphs/{name}.ProgramGraph.pb",
      path / dst / f"{name}.ProgramGraph.pb",
    )
  except (ValueError, OSError, TimeoutError, AssertionError) as e:
    pass


def _ProcessRows(job) -> int:
  path, rows = job
  for (row, i) in rows:
    _ProcessRow(path, row, i)
  return len(rows)


class ExportIrDatabase(progress.Progress):
  """Export non-POJ104 IRs from MySQL database.

  The code which populated this MySQL database is the legacy
  //deeplearning/ml4pl codebase.
  """

  def __init__(self, path: pathlib.Path, db):
    self.path = path
    self.db = _mysql.connect(
      host=FLAGS.host, user=FLAGS.user, passwd=FLAGS.pwd, db=FLAGS.db
    )
    self.db.query(
      """
  SELECT COUNT(*) FROM intermediate_representation
  WHERE compilation_succeeded=1
  AND source NOT LIKE 'poj-104:%'
  """
    )
    n = int(self.db.store_result().fetch_row()[0][0].decode("utf-8"))
    super(ExportIrDatabase, self).__init__("ir db", i=0, n=n, unit="irs")

  def Run(self):
    with multiprocessing.Pool() as pool:
      # Run many smaller queries rather than one big query since MySQL
      # connections will die if hanging around for too long.
      for j in range(0, self.ctx.n, 512):
        self.db.query(
          f"""
SELECT
  source,
  source_language,
  type,
  binary_ir
FROM intermediate_representation
WHERE compilation_succeeded=1
AND source NOT LIKE 'poj-104:%'
LIMIT 1000
OFFSET {j}
"""
        )

        results = self.db.store_result()
        rows = [(item, i) for i, item in enumerate(results.fetch_row())]
        jobs = [(self.path, chunk) for chunk in labtypes.Chunkify(rows, 128)]

        for c in pool.imap_unordered(_ProcessRows, jobs):
          self.ctx.i += c

      self.ctx.i = self.ctx.n


def ExportClassifyAppGraphs(classifyapp: pathlib.Path, path: pathlib.Path):
  app.Log(1, "Copying POJ-104 graphs")
  for graph in (classifyapp / "graphs").iterdir():
    shutil.copy(graph, path / f"graphs/poj104.{graph.name}")
  app.Log(1, "Copying POJ-104 train")
  for graph in (classifyapp / "train").iterdir():
    os.symlink(
      f"../graphs/poj104.{graph.name}", path / f"train/poj104.{graph.name}"
    )
  app.Log(1, "Copying POJ-104 val")
  for graph in (classifyapp / "val").iterdir():
    os.symlink(
      f"../graphs/poj104.{graph.name}", path / f"val/poj104.{graph.name}"
    )
  app.Log(1, "Copying POJ-104 test")
  for graph in (classifyapp / "test").iterdir():
    os.symlink(
      f"../graphs/poj104.{graph.name}", path / f"test/poj104.{graph.name}"
    )


def Main():
  path = pathlib.Path(FLAGS.path)
  db = _mysql.connect(
    host=FLAGS.host, user=FLAGS.user, passwd=FLAGS.pwd, db=FLAGS.db
  )

  (path / "ir").mkdir(parents=True, exist_ok=True)
  (path / "graphs").mkdir(exist_ok=True)
  (path / "train").mkdir(exist_ok=True)
  (path / "val").mkdir(exist_ok=True)
  (path / "test").mkdir(exist_ok=True)

  export = ExportIrDatabase(path, db)
  progress.Run(export)

  ExportClassifyAppGraphs(pathlib.Path(FLAGS.classifyapp), path)

  app.Log(1, "Creating data flow analysis labels")
  subprocess.check_call([str(CREATE_LABELS), "--path", str(path)])


if __name__ == "__main__":
  app.Run(Main)