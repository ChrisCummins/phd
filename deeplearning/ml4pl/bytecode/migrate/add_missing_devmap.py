"""Add the missing devmap bytecodes."""
from labm8 import app
from labm8 import crypto
from labm8 import fs

from deeplearning.ml4pl.bytecode import bytecode_database

FLAGS = app.FLAGS

app.DEFINE_database('bytecode_db',
                    bytecode_database.Database,
                    None,
                    'URL of database to read bytecodes from.',
                    must_exist=True)

app.DEFINE_input_path('bytecode',
                      None,
                      'The directory to import from.',
                      is_dir=True)


def main():
  """Main entry point."""
  bytecode_db = FLAGS.bytecode_db()

  with bytecode_db.Session() as session:
    query = session.query(bytecode_database.LlvmBytecode.relpath) \
        .filter(bytecode_database.LlvmBytecode.source_name == 'pact17_opencl_devmap') \
        .filter(bytecode_database.LlvmBytecode.clang_returncode != 0)
    to_import = [row.relpath for row in query]

  files = sorted([path.name for path in FLAGS.bytecode.iterdir()])

  for relpath in to_import:
    to_find = '-'.join(relpath.split(':'))
    for path in files:
      if path.startswith(to_find):
        app.Log(1, "%s", to_find)
        src = fs.Read(FLAGS.bytecode / path)
        with bytecode_db.Session(commit=True) as s:
          s.add(
              bytecode_database.LlvmBytecode(
                  source_name='pact17_opencl_devmap',
                  relpath=relpath,
                  language='opencl',
                  cflags='',
                  charcount=len(src),
                  linecount=len(src.split('\n')),
                  bytecode=src,
                  clang_returncode=0,
                  error_message='',
                  bytecode_sha1=crypto.sha1_str(src),
              ))
        break
    else:
      app.Error("no match for %s", to_find)

  app.Log(1, "%s to import", len(to_import))


if __name__ == '__main__':
  app.Run(main)
