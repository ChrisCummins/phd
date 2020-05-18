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
"""Create a frequency table of node texts.

This reads the training ProgramGraphs and computes a frequency table of node
texts using inst2vec, ProGraML, and CDFG. The tables are then saved to file in
descending order of frequency. Each line is tab separated in the format:

  <cumulative textfreq> <cumulative nodefreq>  <count>  <node_text>

Where <cumulative text> is in the range [0, 1] and describes the propotion
of total node texts that are described by the current and prior lines.
<cumulative nodefreq> extends this to the proportion of total nodes, including
those without a text representation. <count> is the number of matching node
texts, and <node_text> is the unique text value.
"""
import collections
import pathlib
from typing import Dict

from labm8.py import app
from labm8.py import pbutil
from labm8.py import progress
from programl.proto import node_pb2
from programl.proto import program_graph_pb2

app.DEFINE_string(
  "path",
  str(pathlib.Path("~/programl/classifyapp").expanduser()),
  "The directory to read ProgramGraph.pb files from.",
)
FLAGS = app.FLAGS


def FrequencyTableToVocabulary(
  path: pathlib.Path, freq_counts: Dict[str, int], total_node_count: int
):
  total_text_count = sum(freq_counts.items())

  cumfreq = 0
  with open(path, "w") as f:
    for text, freq in sorted(freq_counts.items(), key=lambda x: -x[1]):
      cumfreq += freq
      print(
        f"{cumfreq / total_text_count:.5f}",
        f"{cumfreq / total_node_count:.5f}",
        freq,
        text,
        sep="\t",
        file=f,
      )


class CreateVocabularyFiles(progress.Progress):
  """Create a frequency table of node texts.

  After running this job with progress.Run(job), the frequency table is
  available as the job.texts property.
  """

  def __init__(self, path: pathlib.Path):
    self.path = path
    self.paths = [
      path
      for path in (path / "train").iterdir()
      if path.name.endswith(".ProgramGraph.pb")
    ]
    super(CreateVocabularyFiles, self).__init__("vocab", i=0, n=len(self.paths))

  def Run(self):
    (self.path / "vocab").mkdir(exist_ok=True)
    inst2vec_preprocessed_vocab = collections.defaultdict(int)
    inst2vec_vocab = collections.defaultdict(int)
    programl_vocab = collections.defaultdict(int)
    cdfg_vocab = collections.defaultdict(int)

    total_node_count = 0
    for self.ctx.i, path in enumerate(self.paths, start=1):
      graph = pbutil.FromFile(path, program_graph_pb2.ProgramGraph())
      total_node_count += len(graph.node)
      for node in graph.node:
        inst2vec_preprocessed = node.features.feature[
          "inst2vec_preprocessed"
        ].bytes_list.value
        if inst2vec_preprocessed:
          inst2vec_preprocessed_vocab[
            inst2vec_preprocessed[0].decode("utf-8")
          ] += 1

        inst2vec = node.features.feature["inst2vec_embedding"].int64_list.value
        if inst2vec:
          inst2vec_vocab[inst2vec[0]] += 1

        programl_vocab[node.text] += 1
        if node.type() == node_pb2.Node.INSTRUCTION:
          cdfg_vocab[node.text] += 1

    FrequencyTableToVocabulary(
      self.path / "vocab" / "inst2vec_preprocessed.csv",
      inst2vec_preprocessed_vocab,
      total_node_count,
    )
    FrequencyTableToVocabulary(
      self.path / "vocab" / "inst2vec.csv", inst2vec_vocab, total_node_count
    )
    FrequencyTableToVocabulary(
      self.path / "vocab" / "programl.csv", programl_vocab, total_node_count
    )
    FrequencyTableToVocabulary(
      self.path / "vocab" / "cdfg.csv", cdfg_vocab, total_node_count
    )

    self.ctx.i = self.ctx.n


def Main():
  path = pathlib.Path(FLAGS.path)
  progress.Run(CreateVocabularyFiles(path))


if __name__ == "__main__":
  app.Run(Main)
