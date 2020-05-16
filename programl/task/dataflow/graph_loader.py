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
"""Train a GGNN to estimate solutions for classic data flow problems.

This script reads ProGraML graphs and uses a GGNN to predict binary
classification targets for data flow problems.
"""
import pathlib
import queue
import random
import threading
from typing import Iterable
from typing import Tuple

from labm8.py import app
from labm8.py import humanize
from labm8.py import pbutil
from programl.ml.batch import base_graph_loader
from programl.proto import epoch_pb2
from programl.proto import program_graph_features_pb2
from programl.proto import program_graph_pb2


class DataflowGraphLoader(base_graph_loader.BaseGraphLoader):
  def __init__(
    self,
    path: pathlib.Path,
    epoch_type: epoch_pb2.EpochType,
    analysis: str,
    seed: int = None,
    max_graph_count: int = None,
    data_flow_step_max: int = None,
    logfile=None,
  ):
    self._inq = queue.Queue(maxsize=1)
    self._outq = queue.Queue(maxsize=10)
    self._thread = self._Reader(
      path,
      epoch_type,
      analysis,
      self._inq,
      self._outq,
      seed=seed,
      max_graph_count=max_graph_count,
      data_flow_step_max=data_flow_step_max,
      logfile=logfile,
    )
    self._thread.start()
    self._stopped = False

  def Stop(self):
    if self._stopped:
      return
    self._stopped = True
    self._inq.put(self._EndOfIterator())
    # Read whatever's left in the queue.
    while self._thread.is_alive():
      if isinstance(self._outq.get(block=True), self._EndOfIterator):
        break
    self._thread.join()

  def __iter__(
    self,
  ) -> Iterable[
    Tuple[
      program_graph_pb2.ProgramGraph,
      program_graph_features_pb2.ProgramGraphFeatures,
    ]
  ]:
    value = self._outq.get(block=True)
    while not isinstance(value, self._EndOfIterator):
      yield value
      value = self._outq.get(block=True)
    self._thread.join()

  class _EndOfIterator(object):
    """Tombstone marker object for iterators."""

    pass

  class _Reader(threading.Thread):
    """Private graph reader."""

    def __init__(
      self,
      path,
      epoch_type: epoch_pb2.EpochType,
      analysis: str,
      inq,
      outq,
      seed: int = None,
      max_graph_count: int = None,
      data_flow_step_max: int = None,
      logfile=None,
    ):
      self.inq = inq
      self.outq = outq
      self.max_graph_count = max_graph_count
      self.data_flow_step_max = data_flow_step_max
      self.seed = seed
      # The number of skipped graphs.
      self.skip_count = 0
      self.logfile = logfile
      super(DataflowGraphLoader._Reader, self).__init__()

      self.graph_path = path / epoch_pb2.EpochType.Name(epoch_type).lower()
      if not self.graph_path.is_dir():
        raise FileNotFoundError(str(self.graph_path))

      self.labels_path = path / analysis
      if not self.labels_path.is_dir():
        raise FileNotFoundError(str(self.labels_path))

    def run(self):
      files = list(self.graph_path.iterdir())
      if self.seed:
        # If we are setting a reproducible seed, first sort the list of files
        # since iterdir() order is undefined, then seed the RNG for the shuffle.
        files = sorted(files)
        random.seed(self.seed)
      random.shuffle(files)

      i = 0

      for path in files:
        try:
          self.inq.get(block=False)
          break
        except queue.Empty:
          pass
        stem = path.name[: -len("ProgramGraph.pb")]
        name = f"{stem}ProgramGraphFeaturesList.pb"
        features_path = self.labels_path / name
        if features_path.is_file():
          app.Log(3, "Read %s", features_path)
          graph = pbutil.FromFile(path, program_graph_pb2.ProgramGraph())
          features_list = pbutil.FromFile(
            features_path, program_graph_features_pb2.ProgramGraphFeaturesList()
          )
          for j, features in enumerate(features_list.graph):
            step_count = features.features.feature[
              "data_flow_step_count"
            ].int64_list.value[0]
            if self.data_flow_step_max and step_count > self.data_flow_step_max:
              self.skip_count += 1
              app.Log(
                3,
                "Skipped graph with data_flow_step_count %d > %d "
                "(skipped %d / %d, %.2f%%)",
                step_count,
                self.data_flow_step_max,
                self.skip_count,
                (i + self.skip_count),
                (self.skip_count / (i + self.skip_count)) * 100,
              )
              continue
            i += 1
            if self.logfile:
              self.logfile.write(f"{features_path} {j}\n")
            self.outq.put((graph, features), block=True)
            if self.max_graph_count and i >= self.max_graph_count:
              app.Log(2, "Stopping after reading %d graphs", i)
              self._Done(i)
              return

      self._Done(i)

    def _Done(self, i: int) -> None:
      app.Log(
        2,
        "Skipped %s of %s graphs (%.2f%%)",
        humanize.Commas(self.skip_count),
        humanize.Commas(i + self.skip_count),
        (self.skip_count / (i + self.skip_count)) * 100,
      )
      self.outq.put(DataflowGraphLoader._EndOfIterator(), block=True)
      self.logfile.close()