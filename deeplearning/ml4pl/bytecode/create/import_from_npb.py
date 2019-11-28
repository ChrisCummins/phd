"""Import bytecodes from a serial C implementation of NPB benchmarks.

This uses ﻿Aleksandr Maramzin's source tree:

  https://github.com/av-maramzin/SNU_NPB

Checkout the NPB sources and cmake build repo:
    $ cd /tmp
    $ git clone --recursive https://github.com/compor/nauseous.git
    $ git clone https://github.com/av-maramzin/SNU_NPB.git

Symlink the sources into the build repo:
    $ cd nauseous/utils/scripts/source_tree
    $ ./create-symlink-bmk-subdir.sh -c /tmp/nauseous/config/suite_all.txt \
        -s /tmp/SNU_NPB/NPB3.3-SER-C -t /tmp/nauseous -l src

Make a cmake build directory:
    $ mkdir /tmp/install && cd /tmp/install

Build, using clang's -save-temps=obj flag to save bitcode files:
    $ export CFLAGS="${C_FLAGS} -O0 -g -save-temps=obj -mcmodel=medium"
    $ /tmp/nauseous/utils/scripts/source_tree/build-llvm.sh && ninja

Pass --cmake_build_root=/tmp/install to this script.
"""
import pathlib
import subprocess
import tempfile
import typing

from compilers.llvm import llvm_dis
from deeplearning.ml4pl.bytecode import bytecode_database
from labm8.py import app
from labm8.py import fs
from labm8.py import sqlutil

FLAGS = app.FLAGS

app.DEFINE_database('bytecode_db', bytecode_database.Database, None,
                    'Path of database to populate.')
app.DEFINE_input_path(
    'cmake_build_root',
    None, 'The path to the root of the CMake build directory. See --help for '
    'instructions on generating this directory.',
    is_dir=True)
app.DEFINE_string('cflags', '-O0 -g',
                  'The C_FLAGS used to build the bytecodes.')


def AbsPathToRelpath(path: pathlib.Path):
  # CMakeFiles
  relpath = str(path)[len(str(FLAGS.cmake_build_root)) + 1:]
  benchmark_name = relpath.split('/')[0]
  file_name = f'{path.stem}.c'
  return f'{benchmark_name}/{file_name}'


def ProcessBitcode(path: pathlib.Path) -> bytecode_database.LlvmBytecode:
  """Process a bitecode file and return the database bytecode representation."""
  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    bytecode_path = pathlib.Path(d) / 'bytecode.ll'
    p = llvm_dis.Exec([str(path), '-o', str(bytecode_path)])
    if p.returncode or not bytecode_path.is_file():
      raise OSError(f"llvm-dis '{path}' failed")

    bytecode = fs.Read(bytecode_path)

  return bytecode_database.LlvmBytecode(
      source_name="github.com/av-maramzin/SNU_NPB:NPB3.3-SER-C",
      relpath=AbsPathToRelpath(path),
      language='c',
      cflags=FLAGS.cflags,
      charcount=len(bytecode),
      linecount=len(bytecode.split('\n')),
      bytecode=bytecode,
      clang_returncode=0,
      error_message='',
  )


def FindBitcodesToImport(
    cmake_build_root: pathlib.Path) -> typing.List[pathlib.Path]:
  """Identify the bitcode files to process."""
  results = subprocess.check_output(
      ['find', str(cmake_build_root), '-name', '*.bc'], universal_newlines=True)
  return [pathlib.Path(result) for result in results.split('\n') if result]


def ImportFromNpb(db: bytecode_database.Database,
                  cmake_build_root: pathlib.Path) -> int:
  """Import the cmake files from the given build root."""
  bytecodes_to_process = FindBitcodesToImport(cmake_build_root)
  i = 0
  with sqlutil.BufferedDatabaseWriter(db, max_queue=10).Session() as writer:
    for i, bytecode in enumerate(
        [ProcessBitcode(b) for b in (bytecodes_to_process)]):
      app.Log(1, '%s:%s', bytecode.source_name, bytecode.relpath)
      writer.AddOne(bytecode)
  return i


def main():
  db = FLAGS.bytecode_db()
  cmake_build_root = FLAGS.cmake_build_root
  if not cmake_build_root or not cmake_build_root.is_dir():
    raise app.UsageError("--cmake_build_root is not a directory")
  export_count = ImportFromNpb(db, cmake_build_root)
  app.Log(1, 'Imported %s bytecodes', export_count)


if __name__ == '__main__':
  app.Run(main)
