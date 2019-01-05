"""Dataset mined by //datasets/github/scrape_repos"""
import multiprocessing

from absl import app
from absl import flags

from compilers.llvm import clang
from datasets.github.scrape_repos import contentfiles
from experimental.compilers.reachability import database
from experimental.compilers.reachability import reachability_pb2


FLAGS = flags.FLAGS

flags.DEFINE_string('db', None, 'Path of database to populate.')
flags.DEFINE_string('cf', None, 'Path of contentfiles database.')


def BytecodeFromSrc(src: str) -> str:
  """Create bytecode from a C source file.

  Args:
    src: The source string.

  Returns:
    The bytecode as a string.

  Raises:
    ClangException: If compiling to bytecode fails.
  """
  clang_args = [
    '-S', '-emit-llvm', '-', '-o', '-', '-O0',
    '-Wno-everything',  # No warnings please.
  ]
  process = clang.Exec(clang_args, stdin=src)
  if process.returncode:
    raise clang.ClangException(
        returncode=process.returncode, stderr=process.stderr,
        command=clang_args)
  return process.stdout, clang_args


def ProcessContentFile(cf_url: str, cf_id: int,
                       text: str) -> reachability_pb2.LlvmBytecode:
  try:
    bytecode, cflags = BytecodeFromSrc(text)
    clang_returncode = 0
    error_message = ''
  except clang.ClangException as e:
    bytecode = ''
    cflags = e.command
    clang_returncode = e.returncode
    error_message = e.stderr

  return reachability_pb2.LlvmBytecode(
      source_name=cf_url,
      relpath=str(cf_id),
      lang='C',
      cflags=' '.join(cflags),
      bytecode=bytecode,
      clang_returncode=clang_returncode,
      error_message=error_message,
  )


def PopulateBytecodeTable(
    cf: contentfiles.ContentFiles,
    db: database.Database, commit_every: int = 1000):
  # Process each row of the table in parallel.
  pool = multiprocessing.Pool()

  with cf.Session() as cf_s:
    q = cf_s.query(cf.ContentFile.id, cf.ContentFile.text).limit(100)

    process_args = [(cf.url, cf_id, text) for cf_id, text in q]
    with db.Session(commit=True) as s:
      for i, proto in enumerate(pool.starmap(ProcessContentFile, process_args)):
        s.add(database.LlvmBytecode(**database.LlvmBytecode.FromProto(proto)))
        if not (i % commit_every):
          s.commit()


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  db = database.Database(FLAGS.db)
  cf = contentfiles.ContentFiles(FLAGS.cf)
  PopulateBytecodeTable(cf, db)
