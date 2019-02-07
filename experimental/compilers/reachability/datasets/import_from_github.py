"""Dataset of GitHub C repos mined by //datasets/github/scrape_repos.

Usage:

  bazel run //experimental/compilers/reachability/datasets:github
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
import humanize
import multiprocessing
import typing

import sqlalchemy as sql
from absl import app
from absl import flags
from absl import logging

from compilers.llvm import clang
from datasets.github.scrape_repos import contentfiles
from deeplearning.clgen.preprocessors import opencl
from experimental.compilers.reachability import database
from experimental.compilers.reachability import reachability_pb2
from labm8 import lockfile
from labm8 import ppar


FLAGS = flags.FLAGS

flags.DEFINE_string('db', None, 'Path of database to populate.')
flags.DEFINE_string('cf', None, 'Path of contentfiles database.')
flags.DEFINE_string('lang', None,
                    'Name of the language to process. One of: {c,opencl}.')

# A dictionary mapping language name to a list of language-specific arguments
# to pass to clang.
LANGUAGE_TO_CLANG_ARGS = {
  'c': [
    '-xc', '-O0',
    '-ferror-limit=1', '-Wno-everything',  # No warnings please.
  ],
  'opencl': opencl.GetClangArgs(use_shim=True),
}


def GetBytecodesFromContentFiles(
    source_name: str, language: str,
    content_files: typing.List[typing.Tuple[int, str]]
) -> typing.List[reachability_pb2.LlvmBytecode]:
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
  protos = []
  clang_args = LANGUAGE_TO_CLANG_ARGS[language] + [
    '-S', '-emit-llvm', '-', '-o', '-'
  ]

  for content_file_id, text in content_files:
    process = clang.Exec(clang_args, stdin=text)
    if process.returncode:
      continue

    protos.append(reachability_pb2.LlvmBytecode(
        source_name=source_name,
        relpath=str(content_file_id),
        lang=language,
        cflags=' '.join(clang_args),
        bytecode=process.stdout,
        clang_returncode=0,
        error_message='',
    ))

  return protos


def PopulateBytecodeTable(
    cf: contentfiles.ContentFiles,
    language: str,
    db: database.Database,
    pool: typing.Optional[multiprocessing.Pool] = None):
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
        s.query(database.LlvmBytecode.relpath)
        .filter(database.LlvmBytecode.source_name == cf.url)
        .filter(database.LlvmBytecode.language == language)
        # Note the cast to integer: relpath is a string column, sorting by it
        # in its native type would sort the string (e.g. '9' > '10'.
        .order_by(sql.cast(database.LlvmBytecode.relpath, sql.Integer).desc())
        .limit(1).first() or (0,))[0])

  with mutex, cf.Session() as cf_s:

    # Get the ID of the last contentfile to process.
    n = (cf_s.query(contentfiles.ContentFile.id)
         .join(contentfiles.GitHubRepository)
         .filter(contentfiles.GitHubRepository.language == language)
         .order_by(contentfiles.ContentFile.id.desc())
         .limit(1).one_or_none() or (0,))[0]
    logging.info('Starting at row %s / %s',
                 humanize.intcomma(resume_from), humanize.intcomma(n))

    # A query to return the <id,text> tuples of files to process.
    q = (cf_s.query(contentfiles.ContentFile.id, contentfiles.ContentFile.text)
         .filter(contentfiles.ContentFile.id > resume_from)
         .join(contentfiles.GitHubRepository)
         .filter(contentfiles.GitHubRepository.language == language)
         .order_by(contentfiles.ContentFile.id))

    batch_size = 256

    def _AddProtosToDatabase(
        protos: typing.List[reachability_pb2.LlvmBytecode]) -> None:
      bytecodes = [
        database.LlvmBytecode(**database.LlvmBytecode.FromProto(proto))
        for proto in protos
      ]
      with db.Session(commit=True) as s:
        s.add_all(bytecodes)

    def _StartBatch(i: int):
      logging.info(
        'Processing batch of %d contentfiles -> bytecodes, %s / %s (%.1f%%)',
        batch_size, humanize.intcomma((i + resume_from)), humanize.intcomma(n),
        ((i + resume_from) / n) * 100)

    ppar.MapDatabaseRowBatchProcessor(
        GetBytecodesFromContentFiles, q,
        generate_work_unit_args=lambda rows: (source_name, language, rows),
        work_unit_result_callback=_AddProtosToDatabase,
        start_of_batch_callback=_StartBatch,
        batch_size=batch_size,
        rows_per_work_unit=5,
        pool=pool,
    )


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  language = FLAGS.lang
  if not language in LANGUAGE_TO_CLANG_ARGS:
    raise app.UsageError(
        f'Language `{language}` not supported. '
        f'Must be one of: {LANGUAGE_TO_CLANG_ARGS.keys()}')

  db = database.Database(FLAGS.db)
  cf = contentfiles.ContentFiles(FLAGS.cf)
  PopulateBytecodeTable(cf, language, db)


if __name__ == '__main__':
  app.run(main)
