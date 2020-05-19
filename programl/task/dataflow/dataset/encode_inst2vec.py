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
"""Run inst2vec encoder on ProGraML graphs.

This runs each of the graphs in the graphs/ directory through inst2vec encoder.
"""
import multiprocessing
import pathlib
from typing import List
from typing import Tuple

from labm8.py import app
from labm8.py import labtypes
from labm8.py import pbutil
from labm8.py import progress
from programl.ir.llvm.inst2vec_encoder import Inst2vecEncoder
from programl.proto import program_graph_pb2
from programl.task.dataflow.dataset import pathflag

FLAGS = app.FLAGS


def _ProcessRows(job) -> int:
  encoder: Inst2vecEncoder = job[0]
  paths: List[Tuple[pathlib.Path, pathlib.Path]] = job[1]
  for graph_path, ir_path in paths:
    with open(ir_path) as f:
      ir = f.read()
    graph = pbutil.FromFile(graph_path, program_graph_pb2.ProgramGraph())
    encoder.Encode(graph, ir=ir)
    pbutil.ToFile(graph, graph_path)
  return len(paths)


class Inst2vecEncodeGraphs(progress.Progress):
  """Run inst2vec encoder on all graphs in the dataset."""

  def __init__(self, path: pathlib.Path):
    if not (path / "graphs").is_dir():
      raise FileNotFoundError(str(path / "graphs"))

    # Enumerate pairs of <program_graph, ir> paths.
    self.paths = [
      (
        graph_path,
        (path / "ir" / f"{graph_path.name[:-len('.ProgramGraph.pb')]}.ll"),
      )
      for graph_path in (path / "graphs").iterdir()
      if graph_path.name.endswith(".ProgramGraph.pb")
    ]
    super(Inst2vecEncodeGraphs, self).__init__(
      "inst2vec", i=0, n=len(self.paths), unit="graphs"
    )

    # Sanity check that the IR files exist.
    for _, ir_path in self.paths:
      if not ir_path.is_file():
        raise FileNotFoundError(str(ir_path))

  def Run(self):
    encoder = Inst2vecEncoder()
    jobs = [(encoder, chunk) for chunk in labtypes.Chunkify(self.paths, 256)]
    with multiprocessing.Pool() as pool:
      for processed_count in pool.imap_unordered(_ProcessRows, jobs):
        self.ctx.i += processed_count
    self.ctx.i = self.ctx.n


def Main():
  path = pathlib.Path(pathflag.path())
  progress.Run(Inst2vecEncodeGraphs(path))


if __name__ == "__main__":
  app.Run(Main)