"""Make the Very Big Compilers Dataset."""
import datetime
import multiprocessing
import pathlib
import tempfile
import time
import typing

import progressbar
import pyparsing

from compilers.llvm import opt
from datasets.github import api as github_api
from datasets.github.scrape_repos import cloner
from datasets.github.scrape_repos import contentfiles
from datasets.github.scrape_repos import importer
from datasets.github.scrape_repos import scraper
from datasets.github.scrape_repos.preprocessors import preprocessors
from datasets.github.scrape_repos.proto import scrape_repos_pb2
from experimental.compilers.reachability import control_flow_graph as cfg
from experimental.compilers.reachability import database
from experimental.compilers.reachability import llvm_util
from experimental.compilers.reachability import reachability_pb2
from experimental.compilers.reachability.datasets import import_from_github
from experimental.compilers.reachability.datasets import linux
from experimental.compilers.reachability.datasets import opencl
from labm8 import app
from labm8 import humanize
from labm8 import pbutil

FLAGS = app.FLAGS

app.DEFINE_string(
    'vbcd',
    'sqlite:////tmp/phd/experimental/compilers/reachability/datasets/vbcd.db',
    'Path of database to populate.')
app.DEFINE_integer(
    'vbcd_process_count', multiprocessing.cpu_count(),
    'The number of parallel processes to use when creating '
    'the dataset.')


def CreateControlFlowGraphsFromLlvmBytecode(
    bytecode_tuple: typing.Tuple[int, str]
) -> typing.List[reachability_pb2.ControlFlowGraphFromLlvmBytecode]:
  """Parallel worker process for extracting CFGs from bytecode."""
  # Expand the input tuple.
  bytecode_id, bytecode = bytecode_tuple
  protos = []

  # Extract the dot sources from the bytecode.
  dot_generator = llvm_util.DotCfgsFromBytecode(bytecode)
  cfg_id = 0
  # We use a while loop here rather than iterating over the dot_generator
  # directly because the dot_generator can raise an exception, and we don't
  # want to lose all of the dot files if only one of them would raise an
  # exception.
  while True:
    try:
      dot = next(dot_generator)
      # Instantiate a CFG from the dot source.
      graph = llvm_util.ControlFlowGraphFromDotSource(dot)
      graph.ValidateControlFlowGraph(strict=False)
      protos.append(
          reachability_pb2.ControlFlowGraphFromLlvmBytecode(
              bytecode_id=bytecode_id,
              cfg_id=cfg_id,
              control_flow_graph=graph.ToProto(),
              status=0,
              error_message='',
              block_count=graph.number_of_nodes(),
              edge_count=graph.number_of_edges(),
              is_strict_valid=graph.IsValidControlFlowGraph(strict=True)))
    except (UnicodeDecodeError, cfg.MalformedControlFlowGraphError, ValueError,
            opt.OptException, pyparsing.ParseException) as e:
      protos.append(
          reachability_pb2.ControlFlowGraphFromLlvmBytecode(
              bytecode_id=bytecode_id,
              cfg_id=cfg_id,
              control_flow_graph=reachability_pb2.ControlFlowGraph(),
              status=1,
              error_message=str(e),
              block_count=0,
              edge_count=0,
              is_strict_valid=False,
          ))
    except StopIteration:
      # Stop once the dot generator has nothing more to give.
      break

    cfg_id += 1

  return protos


def PopulateControlFlowGraphTable(db: database.Database,
                                  n: int = 10000) -> bool:
  """Populate the control flow graph table from LLVM bytecodes.

  For each row in the LlvmBytecode table, we ectract the CFGs and add them
  to the ControlFlowGraphProto table.
  """
  with db.Session(commit=True) as s:
    # We only process bytecodes which are not already in the CFG table.
    already_done_ids = s.query(database.ControlFlowGraphProto.bytecode_id)
    # Query that returns (id,bytecode) tuples for all bytecode files that were
    # successfully generated and have not been entered into the CFG table yet.
    todo = s.query(database.LlvmBytecode.id, database.LlvmBytecode.bytecode) \
      .filter(database.LlvmBytecode.clang_returncode == 0) \
      .filter(~database.LlvmBytecode.id.in_(already_done_ids)) \
      .limit(n)

    count = todo.count()
    if not count:
      return False

    bar = progressbar.ProgressBar()
    bar.max_value = count

    # Process bytecodes in parallel.
    pool = multiprocessing.Pool(FLAGS.vbcd_process_count)
    last_commit_time = time.time()
    for i, protos in enumerate(
        pool.imap_unordered(CreateControlFlowGraphsFromLlvmBytecode, todo)):
      the_time = time.time()
      bar.update(i)

      # Each bytecode file can produce multiple CFGs. Construct table rows from
      # the protos returned.
      rows = [
          database.ControlFlowGraphProto(
              **database.ControlFlowGraphProto.FromProto(proto))
          for proto in protos
      ]

      # Add the rows in a batch.
      s.add_all(rows)

      # Commit every 10 seconds.
      if the_time - last_commit_time > 10:
        s.commit()
        last_commit_time = the_time

  return True


def CreateFullFlowGraphFromCfg(
    cfg_tuple: typing.Tuple[int, int, str]
) -> typing.List[reachability_pb2.ControlFlowGraphFromLlvmBytecode]:
  """Parallel worker process for extracting CFGs from bytecode."""
  # Expand the input tuple.
  bytecode_id, cfg_id, proto_txt = cfg_tuple

  proto = pbutil.FromString(proto_txt, reachability_pb2.ControlFlowGraph())
  try:
    original_graph = llvm_util.LlvmControlFlowGraph.FromProto(proto)

    graph = original_graph.BuildFullFlowGraph()

    return reachability_pb2.ControlFlowGraphFromLlvmBytecode(
        bytecode_id=bytecode_id,
        cfg_id=cfg_id,
        control_flow_graph=graph.ToProto(),
        status=0,
        error_message='',
        block_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        is_strict_valid=graph.IsValidControlFlowGraph(strict=True))
  except Exception as e:
    return reachability_pb2.ControlFlowGraphFromLlvmBytecode(
        bytecode_id=bytecode_id,
        cfg_id=cfg_id,
        control_flow_graph=reachability_pb2.ControlFlowGraph(),
        status=1,
        error_message=str(e),
        block_count=0,
        edge_count=0,
        is_strict_valid=False,
    )


def PopulateFullFlowGraphTable(db: database.Database, n: int = 10000) -> bool:
  """Populate the full flow graph table from CFGs."""
  with db.Session(commit=True) as s:
    # Query that returns (bytecode_id,cfg_id) tuples for all CFGs.
    # TODO(cec): Exclude values already in the FFG table.
    todo = s.query(database.ControlFlowGraphProto.bytecode_id,
                   database.ControlFlowGraphProto.cfg_id,
                   database.ControlFlowGraphProto.proto) \
      .filter(database.ControlFlowGraphProto.status == 0) \
      .limit(n)

    count = todo.count()
    if not count:
      return False

    bar = progressbar.ProgressBar()
    bar.max_value = count

    # Process CFGs in parallel.
    pool = multiprocessing.Pool(FLAGS.vbcd_process_count)
    last_commit_time = time.time()
    for i, proto in enumerate(
        pool.imap_unordered(CreateFullFlowGraphFromCfg, todo)):
      the_time = time.time()
      bar.update(i)

      s.add(
          database.FullFlowGraphProto(
              **database.FullFlowGraphProto.FromProto(proto)))

      # Commit every 10 seconds.
      if the_time - last_commit_time > 10:
        s.commit()
        last_commit_time = the_time

  return True


def PopulateBytecodeTableFromGithubCSources(db: database.Database,
                                            tempdir: pathlib.Path):
  language_to_clone = scrape_repos_pb2.LanguageToClone(
      language='c',
      query=[
          scrape_repos_pb2.GitHubRepositoryQuery(
              string="language:c sort:stars fork:false", max_results=100),
      ],
      destination_directory=str(tempdir),
      importer=[
          scrape_repos_pb2.ContentFilesImporterConfig(
              source_code_pattern=".*\\.c",
              preprocessor=[
                  "datasets.github.scrape_repos.preprocessors.inliners:CxxHeadersDiscardUnknown",
              ]),
          scrape_repos_pb2.ContentFilesImporterConfig(
              source_code_pattern=".*\\.cpp",
              preprocessor=[
                  "datasets.github.scrape_repos.preprocessors.inliners:CxxHeadersDiscardUnknown",
              ]),
          scrape_repos_pb2.ContentFilesImporterConfig(
              source_code_pattern=".*\\.cc",
              preprocessor=[
                  "datasets.github.scrape_repos.preprocessors.inliners:CxxHeadersDiscardUnknown",
              ]),
          scrape_repos_pb2.ContentFilesImporterConfig(
              source_code_pattern=".*\\.cxx",
              preprocessor=[
                  "datasets.github.scrape_repos.preprocessors.inliners:CxxHeadersDiscardUnknown",
              ]),
      ])

  app.Log(1, "Scraping repos ...")
  connection = github_api.GetGithubConectionFromFlagsOrDie()
  for query in language_to_clone.query:
    scraper.RunQuery(scraper.QueryScraper(language_to_clone, query, connection))

  # Clone repos.
  directory = pathlib.Path(language_to_clone.destination_directory)
  meta_files = [
      pathlib.Path(directory / f)
      for f in directory.iterdir()
      if cloner.IsRepoMetaFile(f)
  ]
  worker = cloner.AsyncWorker(meta_files)
  app.Log(1, 'Cloning %s repos from GitHub ...', humanize.Commas(worker.max))
  bar = progressbar.ProgressBar(max_value=worker.max, redirect_stderr=True)
  worker.start()
  while worker.is_alive():
    bar.update(worker.i)
    worker.join(.5)
  bar.update(worker.i)

  app.Log(1, "Importing repos to contentfiles database ...")
  for imp in language_to_clone.importer:
    [preprocessors.GetPreprocessorFunction(p) for p in imp.preprocessor]

  pool = multiprocessing.Pool(FLAGS.processes)
  d = pathlib.Path(language_to_clone.destination_directory)
  d = d.parent / (str(d.name) + '.db')
  contentfiles_db = contentfiles.ContentFiles(f'sqlite:///{d}')
  if pathlib.Path(language_to_clone.destination_directory).is_dir():
    importer.ImportFromLanguage(contentfiles_db, language_to_clone, pool)

  app.Log(1, "Populating bytecode table ...")
  import_from_github.PopulateBytecodeTable(contentfiles_db, language_to_clone,
                                           db)


def NowString() -> str:
  return str(datetime.datetime.now())


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  db = database.Database(FLAGS.vbcd)

  # with db.Session() as session:
  #   opencl_dataset_imported = session.query(database.Meta) \
  #     .filter(database.Meta.key == 'opencl_dataset_imported').first()
  # if not opencl_dataset_imported:
  #   app.Log(1, "Importing OpenCL dataset ...")
  #   opencl.OpenClDeviceMappingsDataset().PopulateBytecodeTable(db)
  #   with db.Session(commit=True) as session:
  #     session.add(
  #         database.Meta(key='opencl_dataset_imported', value=NowString()))

  # with db.Session() as session:
  #   linux_sources_imported = session.query(database.Meta) \
  #     .filter(database.Meta.key == 'linux_sources_imported').first()
  # if not linux_sources_imported:
  #   app.Log(1, "Processing Linux dataset ...")
  #   linux.LinuxSourcesDataset().PopulateBytecodeTable(db)
  #   with db.Session(commit=True) as session:
  #     session.add(
  #         database.Meta(key='linux_sources_imported', value=NowString()))

  # with db.Session() as session:
  #   github_c_sources_imported = session.query(database.Meta) \
  #     .filter(database.Meta.key == 'github_c_sources_imported').first()
  # if not github_c_sources_imported:
  #   app.Log(1, 'Processing GitHub C sources ...')
  #   with tempfile.TemporaryDirectory(prefix='phd_') as d:
  #     PopulateBytecodeTableFromGithubCSources(db, pathlib.Path(d))
  #   with db.Session(commit=True) as session:
  #     session.add(
  #         database.Meta(key='github_c_sources_imported', value=NowString()))

  app.Log(1, "Processing CFGs ...")
  while PopulateControlFlowGraphTable(db):
    print()

  app.Log(1, "Processing full flow graphs ...")
  while PopulateFullFlowGraphTable(db):
    print()


if __name__ == '__main__':
  app.RunWithArgs(main)
