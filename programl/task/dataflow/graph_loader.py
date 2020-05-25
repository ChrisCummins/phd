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
import random
import threading
from queue import Empty
from queue import Queue
from typing import Any
from typing import Iterable
from typing import Tuple

from labm8.py import app
from labm8.py import humanize
from labm8.py import pbutil
from programl.ml.batch import base_graph_loader
from programl.proto import epoch_pb2
from programl.proto import node_pb2
from programl.proto import program_graph_features_pb2
from programl.proto import program_graph_pb2


class DataflowGraphLoader(base_graph_loader.BaseGraphLoader):
  """A graph loader for dataflow graphs and features."""

  def __init__(
    self,
    path: pathlib.Path,
    epoch_type: epoch_pb2.EpochType,
    analysis: str,
    seed: int = None,
    min_graph_count: int = 1,
    max_graph_count: int = None,
    data_flow_step_max: int = None,
    logfile=None,
    use_cdfg: bool = False,
    require_inst2vec: bool = False,
  ):
    self._inq = Queue(maxsize=1)
    self._outq = Queue(maxsize=50)

    reader_opts = {
      "path": path,
      "epoch_type": epoch_type,
      "analysis": analysis,
      "inq": self._inq,
      "outq": self._outq,
      "seed": seed,
      "min_graph_count": min_graph_count,
      "max_graph_count": max_graph_count,
      "data_flow_step_max": data_flow_step_max,
      "logfile": logfile,
    }

    if use_cdfg:
      self._thread = self._CDFGReader(**reader_opts)
    else:
      reader_opts["require_inst2vec"] = require_inst2vec
      self._thread = self._Reader(**reader_opts)
    self._thread.start()
    self._stopped = False

  def IterableType(self) -> Any:
    return self._thread.IterableType()

  def Stop(self):
    if self._stopped:
      return
    self._stopped = True
    self._inq.put(self._EndOfIterator())
    # Read whatever's left in the
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
      inq: Queue,
      outq: Queue,
      seed: int = None,
      min_graph_count: int = None,
      max_graph_count: int = None,
      data_flow_step_max: int = None,
      logfile=None,
      require_inst2vec: bool = False,
    ):
      self.inq = inq
      self.outq = outq
      self.min_graph_count = min_graph_count
      self.max_graph_count = max_graph_count
      self.data_flow_step_max = data_flow_step_max
      self.seed = seed
      self.require_inst2vec = require_inst2vec
      # The number of skipped graphs.
      self.skip_count = 0
      self.logfile = logfile
      super(DataflowGraphLoader._Reader, self).__init__()

      self.graph_path = path / epoch_pb2.EpochType.Name(epoch_type).lower()
      if not self.graph_path.is_dir():
        raise FileNotFoundError(str(self.graph_path))

      self.labels_path = path / "labels" / analysis
      if not self.labels_path.is_dir():
        raise FileNotFoundError(str(self.labels_path))

    def IterableType(self) -> Any:
      return (
        program_graph_pb2.ProgramGraph,
        program_graph_features_pb2.ProgramGraphFeatures,
      )

    def run(self):
      files = list(self.graph_path.iterdir())
      app.Log(
        2, "Enumerated %s graph files to load", humanize.Commas(len(files))
      )

      graph_count = 0
      while graph_count < self.min_graph_count:
        if self.seed:
          # If we are setting a reproducible seed, first sort the list of files
          # since iterdir() order is undefined, then seed the RNG for the shuffle.
          files = sorted(files, key=lambda x: x.name)
          # Change the seed so that on the next execution of this loop we will
          # chose a different random ordering.
          self.seed += 1
        random.Random(self.seed).shuffle(files)

        for path in files:
          try:
            self.inq.get(block=False)
            break
          except Empty:
            pass
          stem = path.name[: -len("ProgramGraph.pb")]
          name = f"{stem}ProgramGraphFeaturesList.pb"
          features_path = self.labels_path / name
          # There is no guarantee that we have generated features for this program
          # graph, so we check for its existence. As a *very* defennsive measure,
          # we also check for the existence of the graph file that we enumearted
          # at the start of this function. This check can be removed later, it is
          # only useful during development when you might be modifying the dataset
          # at the same time as having test jobs running.
          if not path.is_file() or not features_path.is_file():
            self.skip_count += 1
            continue

          app.Log(3, "Read %s", features_path)
          graph = pbutil.FromFile(path, program_graph_pb2.ProgramGraph())
          # Skip empty graphs.
          if not len(graph.node):
            app.Log(2, "empty graph!")
            continue

          # Skip a graph without inst2vec
          if self.require_inst2vec and not len(
            graph.features.feature["inst2vec_annotated"].int64_list.value
          ):
            app.Log(2, "Skipping graph without inst2vec annotations")
            continue

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
                (graph_count + self.skip_count),
                (self.skip_count / (graph_count + self.skip_count)) * 100,
              )
              continue
            graph_count += 1
            if self.logfile:
              self.logfile.write(f"{features_path} {j}\n")
            self.outq.put((graph, features), block=True)
            if self.max_graph_count and graph_count >= self.max_graph_count:
              app.Log(2, "Stopping after reading %d graphs", graph_count)
              self._Done(graph_count)
              return

      self._Done(graph_count)

    def _Done(self, graph_count: int) -> None:
      app.Log(
        2,
        "Skipped %s of %s graphs (%.2f%%)",
        humanize.Commas(self.skip_count),
        humanize.Commas(graph_count + self.skip_count),
        (self.skip_count / max(graph_count + self.skip_count, 1)) * 100,
      )
      self.outq.put(DataflowGraphLoader._EndOfIterator(), block=True)
      if self.logfile:
        self.logfile.close()

  class _CDFGReader(threading.Thread):
    """CDFG graph reader."""

    def __init__(
      self,
      path,
      epoch_type: epoch_pb2.EpochType,
      analysis: str,
      inq: Queue,
      outq: Queue,
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
      super(DataflowGraphLoader._CDFGReader, self).__init__()

      self.graph_path = path / epoch_pb2.EpochType.Name(epoch_type).lower()
      if not self.graph_path.is_dir():
        raise FileNotFoundError(str(self.graph_path))

      self.labels_path = path / "labels" / analysis
      if not self.labels_path.is_dir():
        raise FileNotFoundError(str(self.labels_path))

      self.cdfg_path = path / "cdfg"
      if not self.cdfg_path.is_dir():
        raise FileNotFoundError(str(self.cdfg_path))

    def IterableType(self) -> Any:
      return (
        program_graph_pb2.ProgramGraph,
        program_graph_features_pb2.ProgramGraphFeatures,
        node_pb2.NodeIndexList,
      )

    def run(self):
      files = list(self.graph_path.iterdir())
      app.Log(2, "Enumerated %s files", humanize.Commas(len(files)))
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
        except Empty:
          pass
        stem = path.name[: -len("ProgramGraph.pb")]
        name = f"{stem}ProgramGraphFeaturesList.pb"
        features_path = self.labels_path / name
        cdfg_path = self.cdfg_path / path.name
        node_index_path = self.cdfg_path / f"{stem}NodeIndexList.pb"

        if cdfg_path.exists() and features_path.exists():
          app.Log(3, "Read %s", features_path)
          graph = pbutil.FromFile(cdfg_path, program_graph_pb2.ProgramGraph())
          node_list = pbutil.FromFile(node_index_path, node_pb2.NodeIndexList())
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

            self.outq.put((graph, features, node_list.node), block=True)
            if self.max_graph_count and i >= self.max_graph_count:
              app.Log(2, "Stopping after reading %d graphs", i)
              self._Done(i)
              return

      self._Done(i)

    def _Done(self, graph_count: int) -> None:
      app.Log(
        2,
        "Skipped %s of %s graphs (%.2f%%)",
        humanize.Commas(self.skip_count),
        humanize.Commas(graph_count + self.skip_count),
        (self.skip_count / max(graph_count + self.skip_count, 1)) * 100,
      )
      self.outq.put(DataflowGraphLoader._EndOfIterator(), block=True)
      if self.logfile:
        self.logfile.close()
