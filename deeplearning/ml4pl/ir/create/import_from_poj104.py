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
"""Import bytecodes from POJ-104 algorithm classification dataset.

The dataset can be downloaded from:

  https://polybox.ethz.ch/index.php/s/JOBjrfmAjOeWCyl/download

Usage:

  bazel run //deeplearning/ml4pl/bytecode/create:import_from_poj104
      -- --db='file:///path/to/db.mysql?reachability?charset=utf8'
      --dataset=/path/to/root/of/dataset

**NOTE ON MYSQL BACKEND** When using MySQL as the `--db` backend, I found that
I was occasionally getting errors about write size. To fix this, set a larger
buffer and packet values:

  $ mysql
  mysql> set global net_buffer_length=1000000;
  mysql> set global max_allowed_packet=1000000000;
"""
import os
import pathlib
import typing

from deeplearning.ml4pl import ml4pl_pb2
from deeplearning.ml4pl.bytecode import bytecode_database as database
from labm8.py import app
from labm8.py import fs
from labm8.py import labtypes


FLAGS = app.FLAGS

app.DEFINE_database(
  "db", database.Database, None, "Path of database to populate."
)
app.DEFINE_input_path(
  "dataset", None, "Path of dataset to import.", is_dir=True
)


def LlvmBytecodeIterator(
  base_path: pathlib.Path, source_name: str
) -> typing.Iterable[ml4pl_pb2.LlvmBytecode]:
  """Extract LLVM bytecodes from contentfiles.

  Args:
    base_path: The root directory containing IR codes.
    source_name: The name of the source which is attributed to bytecodes.

  Returns:
    An iterator of LlvmBytecode protos.
  """
  for entry in base_path.iterdir():
    if entry.is_dir() and not entry.name.endswith("_preprocessed"):
      for path in entry.iterdir():
        if path.name.endswith(".ll"):
          relpath = os.path.relpath(path, base_path)
          app.Log(1, "Read %s:%s", source_name, relpath)
          yield ml4pl_pb2.LlvmBytecode(
            source_name=source_name,
            relpath=relpath,
            lang="cpp",
            cflags="",
            bytecode=fs.Read(path),
            clang_returncode=0,
            error_message="",
          )


def ImportProtos(
  db: database.Database,
  bytecode_protos: typing.Iterable[ml4pl_pb2.LlvmBytecode],
) -> None:
  """Import bytecode protobufs to the database."""
  for chunk in labtypes.Chunkify(bytecode_protos, 256):
    with db.Session(commit=True) as s:
      bytecodes = [
        database.LlvmBytecode(**database.LlvmBytecode.FromProto(proto))
        for proto in chunk
      ]
      s.add_all(bytecodes)


def PopulateBytecodeTable(db: database.Database, dataset: pathlib.Path) -> None:
  """Import files to bytecode table."""
  ImportProtos(db, LlvmBytecodeIterator(dataset / "ir_train", "poj-104:train"))
  ImportProtos(db, LlvmBytecodeIterator(dataset / "ir_test", "poj-104:test"))
  ImportProtos(db, LlvmBytecodeIterator(dataset / "ir_val", "poj-104:val"))


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(" ".join(argv[1:])))

  PopulateBytecodeTable(FLAGS.db(), FLAGS.dataset)


if __name__ == "__main__":
  app.RunWithArgs(main)
