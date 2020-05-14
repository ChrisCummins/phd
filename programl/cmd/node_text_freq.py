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

This reads ProgramGraph protocol buffers from a directory and computes a
frequency table of programl.Node.text fields. The table is then printed to
stdout in descending order of frequency. Each line is tab separated in the
format:

  <cumulative frequency>  <count>  <node_text>

Where <cumulative frequency> is in the range [0, 1] and describes the propotion
of total node texts that are described by the current and prior lines. <count>
is the number of matching node texts, and <node_text> is the unique text value.

Usage:

  $ node_text_freq --path=$HOME/programl/dataflow/graphs > freq_table.csv
"""
import collections
import pathlib
from typing import Dict

from labm8.py import app
from labm8.py import pbutil
from labm8.py import progress
from programl.proto import program_graph_pb2

app.DEFINE_string(
  "path", None, "The directory to read ProgramGraph.pb files from."
)
app.DEFINE_boolean("inst2vec", None, "The directory to export to.")
app.DEFINE_boolean("inst2vec_preprocessed", None, "The directory to export to.")
FLAGS = app.FLAGS


class Job(progress.Progress):
  """Create a frequency table of node texts.

  After running this job with progress.Run(job), the frequency table is
  available as the job.texts property.
  """

  def __init__(self, path: pathlib.Path):
    self.paths = [
      p for p in path.iterdir() if p.name.endswith(".ProgramGraph.pb")
    ]
    self.texts: Dict[str, int] = None
    super(Job, self).__init__("text freqs", i=0, n=len(self.paths))

  def Run(self):
    self.texts = collections.defaultdict(int)
    for self.ctx.i, path in enumerate(self.paths, start=1):
      graph = pbutil.FromFile(path, program_graph_pb2.ProgramGraph())
      for node in graph.node:
        if FLAGS.inst2vec_preprocessed:
          inst2vec_preprocessed = node.features.feature[
            "inst2vec_preprocessed"
          ].bytes_list.value
          if inst2vec_preprocessed:
            self.texts[inst2vec_preprocessed[0].decode("utf-8")] += 1
        elif FLAGS.inst2vec:
          inst2vec = node.features.feature[
            "inst2vec_embedding"
          ].int64_list.value
          if inst2vec:
            self.texts[inst2vec[0]] += 1
        else:
          self.texts[node.text] += 1


def Main():
  path = pathlib.Path(FLAGS.path)
  job = Job(path)
  progress.Run(job)
  freqtotal = sum(job.texts.values())
  cumfreq = 0
  for text, freq in sorted(job.texts.items(), key=lambda x: -x[1]):
    cumfreq += freq
    print(f"{cumfreq / freqtotal:.5f}", freq, text, sep="\t")


if __name__ == "__main__":
  app.Run(Main)
