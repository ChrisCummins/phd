"""Import bytecodes from a directory."""
import os
import pathlib
import subprocess

from labm8 import app
from labm8 import fs
from labm8 import sqlutil

from deeplearning.ml4pl.bytecode import bytecode_database

FLAGS = app.FLAGS

app.DEFINE_database('bytecode_db', bytecode_database.Database, None,
                    'Path of database to populate.')
app.DEFINE_input_path('directory',
                      None,
                      'The directory to import bytecodes from.',
                      is_dir=True)
app.DEFINE_string('language', '', 'The source language.')
app.DEFINE_string('source', '', 'The source of the file.')
app.DEFINE_string('cflags', '', 'The C_FLAGS used to build the bytecodes.')


def main():
  # Find the paths to import.
  paths = subprocess.check_output(
      ['find', str(FLAGS.directory), '-name', '*.ll'])
  paths = [pathlib.Path(result) for result in paths.split('\n') if result]

  with sqlutil.BufferedDatabaseWriter(FLAGS.db()).Session() as writer:
    for path in paths:
      bytecode = fs.Read(path)
      relpath = os.path.relpath(path, FLAGS.path)
      writer.AddOne(
          bytecode_database.LlvmBytecode(
              source_name=FLAGS.source,
              relpath=relpath,
              language=FLAGS.language,
              cflags=FLAGS.cflags,
              charcount=len(bytecode),
              linecount=len(bytecode.split('\n')),
              bytecode=bytecode,
              clang_returncode=0,
              error_message='',
          ))


if __name__ == '__main__':
  app.Run(main)
