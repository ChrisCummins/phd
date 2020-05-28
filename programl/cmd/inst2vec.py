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
"""Encode node embeddings using inst2vec.

This program reads a single program graph from file, encodes it, and writes
the modified graph to stdout.

Example usage:

  Encode a program graph proto and write the result to file:

    $ inst2vec --ir=/tmp/source.ll < program.pbtxt > inst2vec.pbtxt
"""
import sys

from labm8.py import app
from labm8.py import fs
from labm8.py import pbutil
from programl.ir.llvm import inst2vec_encoder
from programl.proto import program_graph_pb2

FLAGS = app.FLAGS
app.DEFINE_string(
  "stdin_fmt", "pbtxt", "The format for stdin. One of {pb,pbtxt}"
)
app.DEFINE_string(
  "stdout_fmt", "pbtxt", "The format for stdout. One of {pb,pbtxt}"
)
app.DEFINE_output_path(
  "ir",
  None,
  "The path of the IR file that was used to construct the graph. This is "
  "required to inline struct definitions. This argument may be omitted when "
  "struct definitions do not need to be inlined.",
)


def Main():
  proto = program_graph_pb2.ProgramGraph()
  if FLAGS.stdin_fmt == "pb":
    proto.ParseFromString(sys.stdin.buffer.read())
  elif FLAGS.stdin_fmt == "pbtxt":
    pbutil.FromString(sys.stdin.buffer.read().decode("utf-8"), proto)
  else:
    raise app.UsageError(
      f"Unknown --stdin_fmt={FLAGS.stdin_fmt}. Expected one of {{pb,pbtxt}}"
    )

  ir = fs.Read(FLAGS.ir) if FLAGS.ir else None
  encoder = inst2vec_encoder.Inst2vecEncoder()
  encoder.Encode(proto, ir)

  if FLAGS.stdout_fmt == "pb":
    sys.stdout.buffer.write(proto.SerializeToString())
  elif FLAGS.stdout_fmt == "pbtxt":
    print(proto)
  else:
    raise app.UsageError(
      f"Unknown --stdout_fmt={FLAGS.stdout_fmt}. Expected one of {{pb,pbtxt}}"
    )


if __name__ == "__main__":
  app.Run(Main)
