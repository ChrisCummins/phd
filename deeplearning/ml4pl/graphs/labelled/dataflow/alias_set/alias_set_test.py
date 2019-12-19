# Copyright 2019 the ProGraML authors.
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
"""Unit tests for //deeplearning/ml4pl/graphs/labelled/dataflow/alias_set."""
import typing

import networkx as nx
import numpy as np

from compilers.llvm import clang
from compilers.llvm import opt
from deeplearning.ml4pl.graphs.labelled.dataflow.alias_set import alias_set
from deeplearning.ml4pl.graphs.unlabelled.llvm2graph import graph_builder
from labm8.py import app
from labm8.py import test


FLAGS = app.FLAGS


class InputPair(typing.NamedTuple):
  # A <graph, bytecode> tuple for inputting to alias_set.MakeAliasSetGraphs().
  graph: nx.MultiDiGraph
  bytecode: str


def CSourceToBytecode(source: str) -> str:
  """Build LLVM bytecode for the given C code."""
  process = clang.Exec(
    ["-xc", "-O0", "-S", "-emit-llvm", "-", "-o", "-"], stdin=source
  )
  assert not process.returncode
  return process.stdout


def CSourceToInputPair(source: str) -> InputPair:
  """Create a graph and bytecode for the given C source string.
  This is a convenience method for generating test inputs. If this method fails,
  it is because graph construction or clang is broken.
  """
  bytecode = CSourceToBytecode(source)
  builder = graph_builder.ProGraMLGraphBuilder()
  graph = builder.Build(bytecode)
  return InputPair(graph=graph, bytecode=bytecode)


def test_MakeAliasSetGraphs_invalid_bytecode():
  graph = nx.MultiDiGraph()
  bytecode = "invalid bytecode!"
  with test.Raises(opt.OptException):
    list(alias_set.MakeAliasSetGraphs(graph, bytecode))


@test.XFail(reason="TODO(github.com/ChrisCummins/ProGraML/issues/22)")
def test_MakeAliasSetGraphs_may_alias_set():
  # https://llvm.org/docs/AliasAnalysis.html
  input_pair = CSourceToInputPair(
    """
void A() {
  char C[2];
  char A[10];
  for (int i = 0; i != 10; ++i) {
    C[0] = A[i];          /* One byte store */
    C[1] = A[9-i];        /* One byte store */
  }
}
"""
  )
  graphs = list(alias_set.MakeAliasSetGraphs(*input_pair))
  assert len(graphs) == 1
  # Computed manually.
  identifiers_in_alias_set = {"A_%16_operand", "A_%10_operand"}
  for node, data in graphs[0].nodes(data=True):
    # Test the 'selector' node.
    assert data["x"][1] in {0, 1}
    # Test the labels.
    if node in identifiers_in_alias_set:
      assert np.array_equal(data["y"], [0, 1, 0])
    else:
      assert np.array_equal(data["y"], [1, 0, 0])


@test.XFail(reason="TODO(github.com/ChrisCummins/ProGraML/issues/22)")
def test_MakeAliasSetGraphs_multiple_functions():
  """Test alias """
  # https://llvm.org/docs/AliasAnalysis.html
  input_pair = CSourceToInputPair(
    """
void A() {
  char C[2];
  char A[10];
  for (int i = 0; i != 10; ++i) {
    C[0] = A[i];          /* One byte store */
    C[1] = A[9-i];        /* One byte store */
  }
}

void B() {
  char C[2];
  char A[10];
  for (int i = 0; i != 10; ++i) {
    C[0] = A[i];          /* One byte store */
    C[1] = A[9-i];        /* One byte store */
  }
}
"""
  )
  graphs = list(alias_set.MakeAliasSetGraphs(*input_pair))
  identifiers_in_alias_sets = [
    {"A_%16_operand", "A_%10_operand"},
    {"B_%16_operand", "B_%10_operand"},
  ]
  assert len(graphs) == 2
  # Computed manually.
  for identifiers_in_alias_set, graph in zip(identifiers_in_alias_sets, graphs):
    for node, data in graph.nodes(data=True):
      # Test the 'selector' node.
      assert data["x"][1] in {0, 1}
      # Test the labels.
      if node in identifiers_in_alias_set:
        assert np.array_equal(data["y"], [0, 1, 0])
      else:
        assert np.array_equal(data["y"], [1, 0, 0])


if __name__ == "__main__":
  test.Main()
