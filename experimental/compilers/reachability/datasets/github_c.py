"""Dataset of GitHub C repos mined by //datasets/github/scrape_repos"""
import humanize
import multiprocessing
import typing

import sqlalchemy as sql
from absl import app
from absl import flags
from absl import logging

from compilers.llvm import clang
from datasets.github.scrape_repos import contentfiles
from experimental.compilers.reachability import database
from experimental.compilers.reachability import reachability_pb2
from labm8 import lockfile
from labm8 import ppar


FLAGS = flags.FLAGS

flags.DEFINE_string('db', None, 'Path of database to populate.')
flags.DEFINE_string('cf', None, 'Path of contentfiles database.')


def GetBytecodesFromContentFiles(
    source_name: str,
    content_files: typing.List[typing.Tuple[int, str]]
) -> typing.List[reachability_pb2.LlvmBytecode]:
  """Extract LLVM bytecode from a contentfile, or return None.

  Args:
    source_name: The name of the content file database.
    content_files: A list of <id,text> tuples, where each tuple is the ID and
      text of a row in the content file database.

  Returns:
    A list of zero LlvmBytecode protos, one for each contentfile which was
    successfully processed.
  """
  protos = []

  clang_args = [
    '-xc', '-S', '-emit-llvm', '-', '-o', '-', '-O0',
    '-ferror-limit=1', '-Wno-everything',  # No warnings please.
  ]

  for content_file_id, text in content_files:
    process = clang.Exec(clang_args, stdin=text)
    if process.returncode:
      continue

    protos.append(reachability_pb2.LlvmBytecode(
        source_name=source_name,
        relpath=str(content_file_id),
        lang='C',
        cflags=' '.join(clang_args),
        bytecode=process.stdout,
        clang_returncode=0,
        error_message='',
    ))

  return protos


def PopulateBytecodeTable(
    cf: contentfiles.ContentFiles,
    db: database.Database,
    pool: typing.Optional[multiprocessing.Pool] = None):
  # Only one process at a time can run this method.
  lockfile.AutoLockFile().acquire()

  # Read source files from the contenfiles database, process them into
  # bytecodes, and, if successful, write them into the database. We process
  # files sorted by their numeric ID in the contentfiles database, so that if
  with db.Session() as s:
    # Get the ID of the last-processed bytecode file to resume from.
    resume_from = int((
        s.query(database.LlvmBytecode.relpath)
        .filter(database.LlvmBytecode.source_name == cf.url)
        # Note the cast to integer: relpath is a string column, sorting by it
        # in its native type would sort the string (e.g. '9' > '10'.
        .order_by(sql.cast(database.LlvmBytecode.relpath, sql.Integer).desc())
        .limit(1).first() or (0,))[0])

  with cf.Session() as cf_s:

    # Get the ID of the last contentfile to process.
    n = (cf_s.query(contentfiles.ContentFile.id)
         .order_by(contentfiles.ContentFile.id.desc())
         .limit(1).one_or_none() or (0,))[0]
    logging.info('Starting at row %s / %s',
                 humanize.intcomma(resume_from), humanize.intcomma(n))

    # A query to return the IDs and sources of files to process.
    q = cf_s.query(
        contentfiles.ContentFile.id, contentfiles.ContentFile.text) \
      .filter(contentfiles.ContentFile.id > resume_from) \
      .order_by(contentfiles.ContentFile.id)

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
        'Processing batch of contentfiles -> bytecodes, %s / %s (%.1f%%)',
        humanize.intcomma((i + resume_from)), humanize.intcomma(n),
        ((i + resume_from) / n) * 100)

    ppar.MapDatabaseRowBatchProcessor(
        GetBytecodesFromContentFiles, q,
        generate_work_unit_args=lambda rows: (cf.url, rows),
        work_unit_result_callback=_AddProtosToDatabase,
        start_of_batch_callback=_StartBatch,
        batch_size=256,
        rows_per_work_unit=5,
        pool=pool,
    )


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  db = database.Database(FLAGS.db)
  cf = contentfiles.ContentFiles(FLAGS.cf)
  PopulateBytecodeTable(cf, db)


if __name__ == '__main__':
  app.run(main)
