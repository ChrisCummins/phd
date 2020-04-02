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
"""Download and create the POJ-104 dataset.

The POJ-104 dataset contains 52000 C++ programs implementing 104 different
algorithms with 500 examples of each.

The dataset is from:

  Mou, L., Li, G., Zhang, L., Wang, T., & Jin, Z. (2016). Convolutional Neural
  Networks over Tree Structures for Programming Language Processing. AAAI.

And is available at:

  https://sites.google.com/site/treebasedcnn/

This script creates the 52,000 source files, LLVM-IR files, and program graphs,
divided intro training, validation, and test data.
"""
import multiprocessing
import pathlib
import subprocess

from labm8.py import app
from labm8.py import bazelutil
from labm8.py import labtypes
from labm8.py import pbutil
from labm8.py import progress
from programl.ir.llvm import inst2vec_encoder
from programl.proto import program_graph_pb2

app.DEFINE_string(
  "url",
  "https://drive.google.com/u/0/uc?id=0B2i-vWnOu7MxVlJwQXN6eVNONUU&export=download",
  "The URL of the author-provided archive.tar.gz file.",
)
app.DEFINE_string(
  "path",
  str(pathlib.Path("~/programl/classifyapp").expanduser()),
  "The directory to export to.",
)
FLAGS = app.FLAGS

CREATE_HELPER = bazelutil.DataPath(
  "phd/programl/task/classifyapp/dataset/create_helper"
)


def _Encode(job):
  encoder, paths = job
  for path in paths:
    graph = pbutil.FromFile(path, program_graph_pb2.ProgramGraph())
    ir_name = f"{path.name[:-len('ProgramGraph.pb')]}ll"
    ir_path = path.parent.parent / "ir" / ir_name
    with open(ir_path) as f:
      ir = f.read()
    encoder.Encode(graph, ir=ir)
    pbutil.ToFile(graph, path)
  return len(paths)


class Inst2vecEncode(progress.Progress):
  def __init__(self, dir: pathlib.Path):
    self.encoder = inst2vec_encoder.Inst2vecEncoder()
    files = list(dir.iterdir())
    file_chunks = labtypes.Chunkify(files, 128)
    self.jobs = [(self.encoder, p) for p in file_chunks]
    super(Inst2vecEncode, self).__init__("inst2vec", i=0, n=len(files))

  def Run(self):
    with multiprocessing.Pool() as pool:
      for c in pool.imap_unordered(_Encode, self.jobs):
        self.ctx.i += c
    self.ctx.i = self.ctx.n


def Main():
  path = pathlib.Path(FLAGS.path)
  url = FLAGS.url

  subprocess.check_call(
    [str(CREATE_HELPER), "--path", str(FLAGS.path), "--url", url]
  )
  progress.Run(Inst2vecEncode(path / "graphs"))


if __name__ == "__main__":
  app.Run(Main)
