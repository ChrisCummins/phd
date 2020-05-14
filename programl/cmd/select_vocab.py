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
"""Select and print a of unique vocabulary elements.

For example, to select the:

  $ select_vocab --target_coverage=.99 --max_size=1000 < freq_table.csv
"""
import csv
import sys

from labm8.py import app
from labm8.py import humanize

app.DEFINE_float("target_coverage", 1.0, "The directory to export to.")
app.DEFINE_integer("max_size", None, "The directory to export to.")
FLAGS = app.FLAGS


def Main():
  max_size = FLAGS.max_size
  target_coverage = FLAGS.target_coverage

  data = sys.stdin.readlines()
  for i, (cumfreq, _, text) in enumerate(csv.reader(data, delimiter="\t")):
    cumfreq = float(cumfreq)
    if cumfreq >= target_coverage:
      break
    if max_size and i >= max_size:
      break
    print(text)
  print(
    f"Selected {humanize.Commas(i)} node texts achieving {cumfreq:.2%} "
    "node text coverage",
    file=sys.stderr,
  )


if __name__ == "__main__":
  app.Run(Main)
