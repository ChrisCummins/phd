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
  with sqlutil.BufferedDatabaseWriter(FLAGS.bytecode_db()).Session() as writer:
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
