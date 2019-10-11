"""Import POJ-104 algorithm classification dataset.

The dataset can be downloaded from:

  https://polybox.ethz.ch/index.php/s/JOBjrfmAjOeWCyl/download

Usage:

  bazel run //experimental/compilers/reachability/datasets:import_from_poj104
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

from experimental.compilers.reachability import database
from experimental.compilers.reachability import reachability_pb2
from labm8 import app
from labm8 import fs


FLAGS = app.FLAGS

app.DEFINE_database('db', database.Database, None, 'Path of database to populate.')
app.DEFINE_input_path('dataset', None, 'Path of dataset to import.', is_dir=True)


def LlvmBytecodeIterator(
    base_path: pathlib.Path, source_name: str
) -> typing.Iterable[reachability_pb2.LlvmBytecode]:
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
          yield reachability_pb2.LlvmBytecode(
              source_name=source_name,
              relpath=os.path.relpath(path, base_path),
              lang='cpp',
              cflags='',
              bytecode=fs.Read(path),
              clang_returncode=0,
              error_message='',
          )


def ImportProtos(db: database.Database, bytecode_protos: typing.Iterable[reachability_pb2.LlvmBytecode]) -> None:
  """Import bytecode protobufs to the database."""
  with db.Session(commit=True) as s:
    for proto in bytecode_protos:
      s.GetOrAdd(database.LlvmBytecode, **database.LlvmBytecode.FromProto(proto))


def PopulateBytecodeTable(db: database.Database,
                          dataset: pathlib.Path) -> None:
  """Import files to bytecode table."""
  ImportProtos(db, LlvmBytecodeIterator(dataset / 'ir_train', 'poj-104:train'))
  ImportProtos(db, LlvmBytecodeIterator(dataset / 'ir_test', 'poj-104:test'))
  ImportProtos(db, LlvmBytecodeIterator(dataset / 'ir_val', 'poj-104:val'))


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  PopulateBytecodeTable(FLAGS.db(), FLAGS.dataset)


if __name__ == '__main__':
  app.RunWithArgs(main)