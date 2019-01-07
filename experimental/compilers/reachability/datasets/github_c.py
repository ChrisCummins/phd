"""Dataset of GitHub C repos mined by //datasets/github/scrape_repos"""
import multiprocessing
import typing

from absl import app
from absl import flags
from absl import logging

from compilers.llvm import clang
from datasets.github.scrape_repos import contentfiles
from experimental.compilers.reachability import database
from experimental.compilers.reachability import reachability_pb2
from labm8 import lockfile


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


def BatchedQuery(query, start_at: int = 0, yield_per: int = 1000):
  i = start_at
  batch = []
  while True:
    batch = query.offset(i).limit(yield_per).all()
    if batch:
      yield batch
      i += len(batch)
    else:
      break


def PopulateBytecodeTable(
    cf: contentfiles.ContentFiles,
    db: database.Database,
    pool: typing.Optional[multiprocessing.Pool] = None):
  pool = pool or multiprocessing.Pool()

  # Only one process at a time can run this method.
  mutex = lockfile.AutoLockFile()

  # Read source files from the contenfiles database, process them into
  # bytecodes, and, if successful, write them into the database. We process
  # files sorted by their numeric ID in the contentfiles database, so that if
  with mutex, cf.Session() as cf_s, db.Session(commit=True) as s:
    # Get the ID of the last-processed bytecode file to resume from.
    resume_from = (s.query(database.LlvmBytecode.relpath)
                   .filter(database.LlvmBytecode.source_name == cf.url)
                   .order_by(database.LlvmBytecode.relpath.desc())
                   .limit(1).first() or (0,))[0]

    # Get the ID of the last contentfile to process.
    n = (cf_s.query(contentfiles.ContentFile.id)
         .order_by(contentfiles.ContentFile.id.desc())
         .limit(1).one_or_none() or (0,))[0]

    logging.info('Starting at ID %d / %d', resume_from)

    # A query to return the IDs and sources of files to process.
    q = cf_s.query(
        contentfiles.ContentFile.id, contentfiles.ContentFile.text) \
      .filter(contentfiles.ContentFile.id > resume_from) \
      .order_by(contentfiles.ContentFile.id)

    # Process the source files in batches. For each batch: convert to bytecode
    # and store in the database. Files in batches are processed in parallel.
    # Database commits are per-batch.
    batch_size = 256
    for batch in BatchedQuery(q, yield_per=batch_size):
      i = batch[0][0]  # The ID of the first content file in the batch.
      logging.info('Processing contentfiles -> bytecode %d / %d', i, n)
      content_files_per_process = 5
      process_args = [
        (cf.url, batch[i:i + content_files_per_process])
        for i in range(0, len(batch), content_files_per_process)
      ]
      for protos in pool.starmap(GetBytecodesFromContentFiles, process_args):
        for proto in protos:
          s.add(database.LlvmBytecode(
              **database.LlvmBytecode.FromProto(proto)))
      s.commit()


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  db = database.Database(FLAGS.db)
  cf = contentfiles.ContentFiles(FLAGS.cf)
  PopulateBytecodeTable(cf, db)


if __name__ == '__main__':
  app.run(main)
