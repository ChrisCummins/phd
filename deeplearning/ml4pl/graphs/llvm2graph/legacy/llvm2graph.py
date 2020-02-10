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
"""Construct ProGraML graph representation from LLVM IR.

This program reads an LLVM bitcode file from stdin and prints the program graph
protobuf representation to stdout.

Example Usage
=============

Generate a C source file that can be used to derive an intermediate
representation file:

    $ echo "int main() { return 5; }" > /tmp/foo.c

Create an LLVM intermediate representation:

    $ bazel run //compilers/llvm:clang -- -- \\
        /tmp/foo.c -emit-llvm -S -o /tmp/foo.ll

Generate a program graph proto from this IR:

    $ bazel run //deeplearning/ml4pl/graphs/llvm2graph/legacy -- \\
        < /tmp/foo.ll
"""
import sys

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs.llvm2graph.legacy import graph_builder
from labm8.py import app

FLAGS = app.FLAGS

app.DEFINE_string(
  "opt",
  None,
  "The path of the LLVM opt binary to use. If not provided, the default "
  "project binary will be used.",
)


def Main():
  """Main entry point."""
  bytecode = sys.stdin.read()
  builder = graph_builder.ProGraMLGraphBuilder()
  g = builder.Build(bytecode, FLAGS.opt)
  print(programl.NetworkXToProgramGraphProto(g))


if __name__ == "__main__":
  app.Run(Main)
