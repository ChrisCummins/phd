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
"""This module defines the interface for graph loaders."""
from typing import Iterable
from typing import Tuple

from programl.proto import program_graph_features_pb2
from programl.proto import program_graph_pb2


class BaseGraphLoader(object):
  """Base class for loading graphs from some dataset.

  This class behaves like an iterator over <ProgramGraph, ProgramGraphFeatures>
  tuples, with addition of a Stop() method to signal that no further tuples
  will be consumed.

  Example usage:

      graph_loader = MyGraphLoader(...)
      for graph, features in graph_loader:
        # ... do something with graphs
        if done:
          self.graph_loader.Stop()
  """

  def Stop(self) -> None:
    raise NotImplementedError("abstract class")

  def __iter__(
    self,
  ) -> Iterable[
    Tuple[
      program_graph_pb2.ProgramGraph,
      program_graph_features_pb2.ProgramGraphFeatures,
    ]
  ]:
    raise NotImplementedError("abstract class")
