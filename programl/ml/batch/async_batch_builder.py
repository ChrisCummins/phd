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
"""This module defines a class for asynchronously reading a batch builder."""
import threading
import time

from labm8.py import app
from labm8.py import humanize
from programl.ml.batch.base_batch_builder import BaseBatchBuilder


class AsyncBatchBuilder(threading.Thread):
  """A class for running a batch builder in a separate thread.

  Usage:

      bb = AsyncBatchBuilder(MyBatchBuilder(my_graph_loader))
      bb.start()
      # ... do other stuff
      bb.join()
      for batch in bb.batches:
        # go nuts
  """

  def __init__(self, batch_builder: BaseBatchBuilder):
    super(AsyncBatchBuilder, self).__init__()
    self.batch_builder = batch_builder
    self.batches = []

  def run(self):
    start = time.time()
    self.batches = list(self.batch_builder)
    app.Log(
      2,
      "Async batch loader completed. %s batches loaded in %s",
      humanize.Commas(len(self.batches)),
      humanize.Duration(time.time() - start),
    )
