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
"""Import bytecodes from a directory."""
import os
import pathlib
import subprocess

from deeplearning.ml4pl.bytecode import bytecode_database
from labm8.py import app
from labm8.py import fs
from labm8.py import sqlutil


FLAGS = app.FLAGS

app.DEFINE_database(
  "bytecode_db",
  bytecode_database.Database,
  None,
  "Path of database to populate.",
)
app.DEFINE_input_path(
  "directory", None, "The directory to import bytecodes from.", is_dir=True
)
app.DEFINE_string("language", "", "The source language.")
app.DEFINE_string("source", "", "The source of the file.")
app.DEFINE_string("cflags", "", "The C_FLAGS used to build the bytecodes.")


def main():
  # Find the paths to import.
  paths = subprocess.check_output(
    ["find", str(FLAGS.directory), "-name", "*.ll"], universal_newlines=True
  )
  paths = [pathlib.Path(result) for result in paths.split("\n") if result]

  i = 0
  with sqlutil.BufferedDatabaseWriter(FLAGS.bytecode_db()) as writer:
    for i, path in enumerate(paths):
      bytecode = fs.Read(path)
      relpath = os.path.relpath(path, FLAGS.directory)
      app.Log(1, "%s:%s", FLAGS.source, relpath)
      writer.AddOne(
        bytecode_database.LlvmBytecode(
          source_name=FLAGS.source,
          relpath=relpath,
          language=FLAGS.language,
          cflags=FLAGS.cflags,
          charcount=len(bytecode),
          linecount=len(bytecode.split("\n")),
          bytecode=bytecode,
          clang_returncode=0,
          error_message="",
        )
      )

  app.Log(1, "Imported %s bytecodes", i)


if __name__ == "__main__":
  app.Run(main)
