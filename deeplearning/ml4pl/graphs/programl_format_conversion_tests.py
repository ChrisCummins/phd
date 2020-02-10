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
"""Tests for //deeplearning/ml4pl/graphs:programl binary.

This file tests conversion different combinations of input and output graph
formats using the command line tool.
"""
import pickle
import subprocess

import networkx as nx

from deeplearning.ml4pl.graphs import programl_pb2
from labm8.py import bazelutil
from labm8.py import test

pytest_plugins = ["deeplearning.ml4pl.testing.fixtures.llvm_program_graph"]

BINARY = bazelutil.DataPath("phd/deeplearning/ml4pl/graphs/programl")

FLAGS = test.FLAGS


@test.Parametrize("stdout_fmt", ("pb", "pbtxt", "nx"))
def test_pb_conversion(
  llvm_program_graph: programl_pb2.ProgramGraphProto, stdout_fmt: str
):
  """Test format conversion from text protocol buffer."""
  assert subprocess.check_output(
    [str(BINARY), "--stdin_fmt=pb", f"--stdout_fmt={stdout_fmt}"],
    input=llvm_program_graph.SerializeToString(),
  )


@test.Parametrize("stdout_fmt", ("pb", "pbtxt", "nx"))
def test_pbtxt_conversion(
  llvm_program_graph: programl_pb2.ProgramGraphProto, stdout_fmt: str
):
  """Test format conversion from text protocol buffer."""
  assert subprocess.check_output(
    [str(BINARY), "--stdin_fmt=pbtxt", f"--stdout_fmt={stdout_fmt}"],
    input=str(llvm_program_graph).encode("utf-8"),
  )


@test.Parametrize("stdout_fmt", ("pb", "pbtxt", "nx"))
def test_nx_conversion(llvm_program_graph_nx: nx.MultiDiGraph, stdout_fmt: str):
  """Test format conversion from networkx graph."""
  assert subprocess.check_output(
    [str(BINARY), "--stdin_fmt=nx", f"--stdout_fmt={stdout_fmt}"],
    input=pickle.dumps(llvm_program_graph_nx),
  )


if __name__ == "__main__":
  test.Main()
