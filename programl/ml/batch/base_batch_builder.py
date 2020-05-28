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
"""This module defines the base class for batch builders."""
from typing import Iterable

from programl.ml.batch.base_graph_loader import BaseGraphLoader
from programl.ml.batch.batch_data import BatchData


class BaseBatchBuilder(object):
  """Base class for building batches.

  A batch builder is a class which accepts as input a graph loader and produces
  an iterable sequence of batches.
  """

  def __init__(self, graph_loader: BaseGraphLoader):
    self.graph_loader = graph_loader

  def __iter__(self) -> Iterable[BatchData]:
    raise NotImplementedError("abstract class")
