"""Import code from GitHub repos mined by //datasets/github/scrape_repos.

Usage:

  bazel run //deeplearning/ml4pl/bytecode/create:import_from_github
      -- --cf=sqlite:////path/to/db.db
      --db='file:///path/to/db.mysql?reachability?charset=utf8'
      --lang=opencl

**NOTE ON MYSQL BACKEND** When using MySQL as the `--db` backend, I found that
I was occasionally getting errors about write size. To fix this, set a larger
buffer and packet values:

  $ mysql
  mysql> set global net_buffer_length=1000000;
  mysql> set global max_allowed_packet=1000000000;
"""
import multiprocessing
import subprocess
import tempfile
import typing

import sqlalchemy as sql

from compilers.llvm import clang
from datasets.github.scrape_repos import contentfiles
from deeplearning.clgen.preprocessors import opencl
from deeplearning.ml4pl import ml4pl_pb2
from deeplearning.ml4pl.bytecode import bytecode_database
from labm8 import app
from labm8 import fs
from labm8 import humanize
from labm8 import lockfile
from labm8 import sqlutil

FLAGS = app.FLAGS

# A dictionary mapping language name to a list of language-specific arguments
# to pass to clang.
LANGUAGE_TO_CLANG_ARGS = {
    'c': [
        '-xc',
        '-O0',
        '-ferror-limit=1',
        '-Wno-everything',  # No warnings please.
    ],
    'opencl': opencl.GetClangArgs(use_shim=True),
    # These languages are handled separately without using clang, but they still
    # need entries in this table:
    'swift': [],
    'haskell': [],
}

app.DEFINE_string('db', None, 'Path of database to populate.')
app.DEFINE_string('cf', None, 'Path of contentfiles database.')
app.DEFINE_string(
    'lang', None, 'Name of the language to process. One of: '
    f'{set(LANGUAGE_TO_CLANG_ARGS.keys())}.')
app.DEFINE_integer('batch_size', 32, 'The size of batches to process.')


def GetSwiftBytecodesFromContentFiles(
    source_name: str, content_files: typing.List[typing.Tuple[int, str]]
) -> typing.List[ml4pl_pb2.LlvmBytecode]:
  """Extract LLVM bytecodes from swift contentfiles.

  The process is swift -> LLVM bitcode, clang -> LLVM bytecode.

  This requires that the `swift` binary is in the system path.
  """
  protos = []

  with tempfile.TemporaryDirectory(prefix='phd_import_swift_') as d:
    with fs.chdir(d) as d:
      for content_file_id, text in content_files:
        swift_file = d / 'file.swift'
        bc_file = d / 'file.bc'
        fs.Write(swift_file, text.encode('utf-8'))
        swift = subprocess.Popen(
            ['swift', '-Xfrontend', '-emit-bc', swift_file.name],
            stderr=subprocess.DEVNULL)
        swift.communicate()
        if swift.returncode:
          continue
        if not bc_file.is_file():
          continue

        process = clang.Exec(['-S', '-emit-llvm', str(bc_file), '-o', '-'])
        if process.returncode:
          continue

        protos.append(
            ml4pl_pb2.LlvmBytecode(
                source_name=source_name,
                relpath=str(content_file_id),
                lang='swift',
                cflags='',
                bytecode=process.stdout,
                clang_returncode=0,
                error_message='',
            ))

  return protos


def GetHaskellBytecodesFromContentFiles(
    source_name: str, content_files: typing.List[typing.Tuple[int, str]]
) -> typing.List[ml4pl_pb2.LlvmBytecode]:
  """Extract LLVM bytecodes from haskell contentfiles.

  The process is haskell -> LLVM bytecode.

  This requires the glasgow haskell compiler and LLVM backend, install them on
  Ubuntu 16.04 using:

    $ sudo apt-get install ghc llvm-3.5
  """
  protos = []

  with tempfile.TemporaryDirectory(prefix='phd_import_haskell_') as d:
    with fs.chdir(d) as d:
      for content_file_id, text in content_files:
        haskell_file = d / 'file.hs'
        ll_file = d / 'file.ll'
        fs.Write(haskell_file, text.encode('utf-8'))
        ghc = subprocess.Popen([
            'ghc', '-fllvm', '-keep-llvm-files', '-fforce-recomp',
            haskell_file.name
        ],
                               stderr=subprocess.DEVNULL)
        ghc.communicate()
        if ghc.returncode:
          continue
        if not ll_file.is_file():
          continue

        protos.append(
            ml4pl_pb2.LlvmBytecode(
                source_name=source_name,
                relpath=str(content_file_id),
                lang='haskell',
                cflags='ghc -fllm -keep-llvm-files',
                bytecode=fs.Read(ll_file),
                clang_returncode=0,
                error_message='',
            ))

  return protos


def GetBytecodesFromContentFiles(
    source_name: str, language: str,
    content_files: typing.List[typing.Tuple[int, str]]
) -> typing.List[ml4pl_pb2.LlvmBytecode]:
  """Extract LLVM bytecodes from contentfiles.

  Args:
    source_name: The name of the content file database. This is the same across
      all content files.
    language: The source code language. This is the same across all content
      files.
    content_files: A list of <id,text> tuples, where each tuple is the ID and
      text of a row in the content file database.

  Returns:
    A list of zero LlvmBytecode protos, one for each contentfile which was
    successfully processed.
  """
  if language == 'swift':
    return GetSwiftBytecodesFromContentFiles(source_name, content_files)
  elif language == 'haskell':
    return GetHaskellBytecodesFromContentFiles(source_name, content_files)

  protos = []
  clang_args = LANGUAGE_TO_CLANG_ARGS[language] + [
      '-S', '-emit-llvm', '-', '-o', '-'
  ]

  for content_file_id, text in content_files:
    process = clang.Exec(clang_args, stdin=text)
    if process.returncode:
      continue

    protos.append(
        ml4pl_pb2.LlvmBytecode(
            source_name=source_name,
            relpath=str(content_file_id),
            lang=language,
            cflags=' '.join(clang_args),
            bytecode=process.stdout,
            clang_returncode=0,
            error_message='',
        ))

  return protos


def PopulateBytecodeTable(cf: contentfiles.ContentFiles,
                          language: str,
                          db: bytecode_database.Database,
                          pool: typing.Optional[multiprocessing.Pool] = None):
  writer = sqlutil.BufferedDatabaseWriter(db, max_queue=10)

  # Only one process at a time can run this method.
  mutex = lockfile.AutoLockFile(granularity='function')

  # We use the database URL as the name of the source.
  source_name = cf.url

  # Read source files from the contenfiles database, process them into
  # bytecodes, and, if successful, write them into the database. We process
  # files sorted by their numeric ID in the contentfiles database, so that if
  with db.Session() as s:
    # Get the ID of the last-processed bytecode file to resume from.
    resume_from = int((
        s.query(bytecode_database.LlvmBytecode.relpath).filter(
            bytecode_database.LlvmBytecode.source_name == cf.url).filter(
                bytecode_database.LlvmBytecode.language == language)
        # Note the cast to integer: relpath is a string column, sorting by it
        # in its native type would sort the string (e.g. '9' > '10'.
        .order_by(
            sql.cast(bytecode_database.LlvmBytecode.relpath,
                     sql.Integer).desc()).limit(1).first() or (0,))[0])

  with mutex, cf.Session() as cf_s:
    # Get the ID of the last contentfile to process.
    n = (cf_s.query(contentfiles.ContentFile.id).join(
        contentfiles.GitHubRepository).filter(
            contentfiles.GitHubRepository.language == language).order_by(
                contentfiles.ContentFile.id.desc()).limit(1).one_or_none() or
         (0,))[0]
    app.Log(1, 'Starting at row %s / %s', humanize.Commas(resume_from),
            humanize.Commas(n))

    # A query to return the <id,text> tuples of files to process.
    q = (cf_s.query(contentfiles.ContentFile.id, contentfiles.ContentFile.text).
         filter(contentfiles.ContentFile.id > resume_from).join(
             contentfiles.GitHubRepository).filter(
                 contentfiles.GitHubRepository.language == language).order_by(
                     contentfiles.ContentFile.id))

    row_batches = sqlutil.OffsetLimitBatchedQuery(
        q, batch_size=FLAGS.batch_size)

    with writer.Session() as writer:
      for i, batch in zip(range(resume_from, n + 1), row_batches):
        app.Log(
            1,
            'Processing batch of %d contentfiles -> bytecodes, %s / %s (%.1f%%)',
            FLAGS.batch_size, humanize.Commas(i), humanize.Commas(n),
            (i / n) * 100)
        protos = GetBytecodesFromContentFiles(source_name, language, batch.rows)
        writer.AddMany([
            bytecode_database.LlvmBytecode(
                **bytecode_database.LlvmBytecode.FromProto(proto))
            for proto in protos
        ])


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  language = FLAGS.lang
  if not language in LANGUAGE_TO_CLANG_ARGS:
    raise app.UsageError(f'Language `{language}` not supported. '
                         f'Must be one of: {LANGUAGE_TO_CLANG_ARGS.keys()}')

  db = bytecode_database.Database(FLAGS.db)
  cf = contentfiles.ContentFiles(FLAGS.cf)
  PopulateBytecodeTable(cf, language, db)


if __name__ == '__main__':
  app.RunWithArgs(main)
