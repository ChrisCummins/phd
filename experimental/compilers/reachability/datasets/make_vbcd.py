"""Make the Very Big Compilers Dataset."""
import datetime
import multiprocessing
import pathlib
import tempfile
import time
import typing

import humanize
import pandas as pd
import progressbar
import pyparsing
from absl import app
from absl import flags
from absl import logging

from compilers.llvm import clang
from compilers.llvm import opt
from datasets.github.scrape_repos import cloner
from datasets.github.scrape_repos import contentfiles
from datasets.github.scrape_repos import importer
from datasets.github.scrape_repos import scraper
from datasets.github.scrape_repos.preprocessors import preprocessors
from datasets.github.scrape_repos.proto import scrape_repos_pb2
from datasets.linux import linux
from datasets.opencl.device_mapping import \
  opencl_device_mapping_dataset as ocl_dataset
from deeplearning.clgen.preprocessors import opencl
from experimental.compilers.reachability import control_flow_graph as cfg
from experimental.compilers.reachability import database
from experimental.compilers.reachability import llvm_util
from experimental.compilers.reachability import reachability_pb2
from experimental.compilers.reachability.datasets import import_from_github
from labm8 import decorators
from labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'vbcd',
    'sqlite:////tmp/phd/experimental/compilers/reachability/datasets/vbcd.db',
    'Path of database to populate.')

flags.DEFINE_integer('vbcd_process_count', multiprocessing.cpu_count(),
                     'The number of parallel processes to use when creating '
                     'the dataset.')


def BytecodeFromOpenClString(opencl_string: str,
                             optimization_level: str) -> str:
  """Create bytecode from OpenCL source string.

  Args:
    opencl_string: A string of OpenCL code.
    optimization_level: The optimization level to use, one of
        {-O0,-O1,-O2,-O3,-Ofast,-Os,-Oz}.

  Returns:
    A tuple of the arguments to clang, and the bytecode as a string.

  Raises:
    ClangException: If compiling to bytecode fails.
  """
  # Use -O3 to reduce CFGs.
  clang_args = opencl.GetClangArgs(use_shim=False) + [
    clang.ValidateOptimizationLevel(optimization_level),
    '-S', '-emit-llvm', '-o', '-', '-i', '-',
    '-Wno-everything',  # No warnings please.
  ]
  process = clang.Exec(clang_args, stdin=opencl_string)
  if process.returncode:
    raise clang.ClangException(
        "clang failed", returncode=process.returncode, stderr=process.stderr,
        command=clang_args)
  return process.stdout, clang_args


def CreateControlFlowGraphFromOpenClKernel(
    kernel_name: str,
    opencl_kernel: str) -> typing.Optional[cfg.ControlFlowGraph]:
  """Try to create a CFG proto from an opencl kernel.

  Args:
    kernel_name: The name of the OpenCL kernel defined in opencl_kernel.
    opencl_kernel: A string of OpenCL. This should contain a single kernel
      definition.

  Returns:
    A ControlFlowGraph instance, or None if compilation to bytecode fails.

  Raises:
    ClangException: If compiling to bytecode fails.
    ValueError: If opencl_kernel contains multiple functions.
  """
  bytecode, _ = BytecodeFromOpenClString(opencl_kernel, '-O0')

  # Extract a single dot source from the bytecode.
  dot_generator = llvm_util.DotCfgsFromBytecode(bytecode)
  dot = next(dot_generator)
  try:
    next(dot_generator)
    raise ValueError("Bytecode produced more than one dot source!")
  except StopIteration:
    pass

  # Instantiate a CFG from the dot source.
  graph = llvm_util.ControlFlowGraphFromDotSource(dot)

  # Set the name of the graph to the kernel name. This is because the src code
  # has been preprocessed, so that each kernel is named 'A'.
  graph.graph['name'] = kernel_name

  return graph


def ProcessProgramDfIterItem(
    row: typing.Dict[str, str]
) -> typing.Optional[typing.Dict[str, typing.Any]]:
  benchmark_suite_name = row['program:benchmark_suite_name']
  benchmark_name = row['program:benchmark_name']
  kernel_name = row['program:opencl_kernel_name']
  src = row['program:opencl_src']

  try:
    graph = CreateControlFlowGraphFromOpenClKernel(
        kernel_name, src).ValidateControlFlowGraph(strict=False)
  except (clang.ClangException, cfg.MalformedControlFlowGraphError):
    return None

  row = CfgDfRowFromControlFlowGraph(graph)
  row.update({
    'program:benchmark_suite_name': benchmark_suite_name,
    'program:benchmark_name': benchmark_name,
    'program:opencl_kernel_name': kernel_name,
  })
  return row


def CfgDfRowFromControlFlowGraph(
    graph: cfg.ControlFlowGraph) -> typing.Dict[str, typing.Any]:
  return {
    'cfg:graph': graph,
    'cfg:block_count': graph.number_of_nodes(),
    'cfg:edge_count': graph.number_of_edges(),
    'cfg:edge_density': graph.edge_density,
    'cfg:is_valid': graph.IsValidControlFlowGraph(strict=False),
    'cfg:is_strict_valid': graph.IsValidControlFlowGraph(strict=True),
  }


def ProcessOpenClProgramDfBytecode(
    row: typing.Dict[str, str]
) -> reachability_pb2.LlvmBytecode:
  benchmark_suite_name = row['program:benchmark_suite_name']
  benchmark_name = row['program:benchmark_name']
  kernel_name = row['program:opencl_kernel_name']
  src = row['program:opencl_src']

  try:
    bytecode, cflags = BytecodeFromOpenClString(src, '-O0')
    clang_returncode = 0
    error_message = ''
  except clang.ClangException as e:
    bytecode = ''
    cflags = e.command
    clang_returncode = e.returncode
    error_message = e.stderr

  return reachability_pb2.LlvmBytecode(
      source_name=f'{benchmark_suite_name}:{benchmark_name}',
      relpath=kernel_name,
      lang='OpenCL',
      cflags=' '.join(cflags),
      bytecode=bytecode,
      clang_returncode=clang_returncode,
      error_message=error_message,
  )


class OpenClDeviceMappingsDataset(ocl_dataset.OpenClDeviceMappingsDataset):
  """An extension of the OpenCL device mapping dataset for control flow graphs.

  The returned DataFrame has the following schema:

    program:benchmark_suite_name (str): The name of the benchmark suite.
    program:benchmark_name (str): The name of the benchmark program.
    program:opencl_kernel_name (str): The name of the OpenCL kernel.
    cfg:graph (ControlFlowGraph): A control flow graph instance.
    cfg:block_count (int): The number of basic blocks in the CFG.
    cfg:edge_count (int): The number of edges in the CFG.
    cfg:edge_density (float): Number of edges / possible edges, in range [0,1].
    cfg:is_valid (bool): Whether the CFG is valid.
    cfg:is_strict_valid (bool): Whether the CFG is valid when strict.
  """

  def PopulateBytecodeTable(
      self, db: database.Database, commit_every: int = 1000):
    programs_df = self.programs_df.reset_index()
    bar = progressbar.ProgressBar()
    bar.max_value = len(programs_df)

    # Process each row of the table in parallel.
    pool = multiprocessing.Pool(FLAGS.vbcd_process_count)
    with db.Session(commit=True) as s:
      for i, proto in enumerate(pool.imap_unordered(
          ProcessOpenClProgramDfBytecode,
          [d for _, d in programs_df.iterrows()])):
        bar.update(i)
        s.add(database.LlvmBytecode(**database.LlvmBytecode.FromProto(proto)))
        if not (i % commit_every):
          s.commit()

  @decorators.memoized_property
  def cfgs_df(self) -> pd.DataFrame:
    programs_df = self.programs_df.reset_index()

    # Process each row of the table in parallel.
    pool = multiprocessing.Pool(FLAGS.vbcd_process_count)
    rows = []
    for row in pool.imap_unordered(
        ProcessProgramDfIterItem, [d for i, d in programs_df.iterrows()]):
      if row:
        rows.append(row)

    # Create the output table.
    df = pd.DataFrame(rows, columns=[
      'program:benchmark_suite_name',
      'program:benchmark_name',
      'program:opencl_kernel_name',
      'cfg:graph',
      'cfg:block_count',
      'cfg:edge_count',
      'cfg:edge_density',
      'cfg:is_valid',
      'cfg:is_strict_valid',
    ])

    df.set_index([
      'program:benchmark_suite_name',
      'program:benchmark_name',
      'program:opencl_kernel_name',
    ], inplace=True)
    df.sort_index(inplace=True)
    return df


def BytecodeFromLinuxSrc(path: pathlib.Path, optimization_level: str) -> str:
  """Create bytecode from a Linux source file.

  Args:
    path: The path of the source file.
    optimization_level: The clang optimization level to use, one of
        {-O0,-O1,-O2,-O3,-Ofast,-Os,-Oz}.

  Returns:
    The bytecode as a string.

  Raises:
    ClangException: If compiling to bytecode fails.
  """
  root = linux.LinuxSourcesDataset().src_tree_root
  genroot = linux.LinuxSourcesDataset().generated_hdrs_root
  # A subset of the arguments found by running `make V=1` in the linux
  # build and grabbing a random C compile target.
  # The build I took this from: Wp,-MD,arch/x86/kernel/.asm-offsets.s.d  -nostdinc -isystem /usr/lib/gcc/x86_64-linux-gnu/5/include -I./arch/x86/include -I./arch/x86/include/generated  -I./include -I./arch/x86/include/uapi -I./arch/x86/include/generated/uapi -I./include/uapi -I./include/generated/uapi -include ./include/linux/kconfig.h -include ./include/linux/compiler_types.h -D__KERNEL__ -Wall -Wundef -Wstrict-prototypes -Wno-trigraphs -fno-strict-aliasing -fno-common -fshort-wchar -Werror-implicit-function-declaration -Wno-format-security -std=gnu89 -fno-PIE -DCC_HAVE_ASM_GOTO -mno-sse -mno-mmx -mno-sse2 -mno-3dnow -mno-avx -m64 -falign-jumps=1 -falign-loops=1 -mno-80387 -mno-fp-ret-in-387 -mpreferred-stack-boundary=3 -mskip-rax-setup -mtune=generic -mno-red-zone -mcmodel=kernel -funit-at-a-time -DCONFIG_X86_X32_ABI -DCONFIG_AS_CFI=1 -DCONFIG_AS_CFI_SIGNAL_FRAME=1 -DCONFIG_AS_CFI_SECTIONS=1 -DCONFIG_AS_FXSAVEQ=1 -DCONFIG_AS_SSSE3=1 -DCONFIG_AS_CRC32=1 -DCONFIG_AS_AVX=1 -DCONFIG_AS_AVX2=1 -DCONFIG_AS_AVX512=1 -DCONFIG_AS_SHA1_NI=1 -DCONFIG_AS_SHA256_NI=1 -pipe -Wno-sign-compare -fno-asynchronous-unwind-tables -mindirect-branch=thunk-extern -mindirect-branch-register -DRETPOLINE -fno-delete-null-pointer-checks -O2 --param=allow-store-data-races=0 -Wframe-larger-than=1024 -fstack-protector-strong -Wno-unused-but-set-variable -fno-var-tracking-assignments -g -gdwarf-4 -pg -mrecord-mcount -mfentry -DCC_USING_FENTRY -Wdeclaration-after-statement -Wno-pointer-sign -fno-strict-overflow -fno-merge-all-constants -fmerge-constants -fno-stack-check -fconserve-stack -Werror=implicit-int -Werror=strict-prototypes -Werror=date-time -Werror=incompatible-pointer-types -Werror=designated-init    -DKBUILD_BASENAME='"asm_offsets"' -DKBUILD_MODNAME='"asm_offsets"'  -fverbose-asm -S -o arch/x86/kernel/asm-offsets.s arch/x86/kernel/asm-offsets.c
  clang_args = [
    '-S', '-emit-llvm', '-o', '-',
    clang.ValidateOptimizationLevel(optimization_level),
    '-Wno-everything',  # No warnings please.
    '-I', str(root / 'arch/x86/include'),
    '-I', str(genroot / 'arch/x86/include/generated'),
    '-I', str(root / 'include'),
    '-I', str(root / 'arch/x86/include/uapi'),
    '-I', str(genroot / 'arch/x86/include/generated/uapi'),
    '-I', str(root / 'include/uapi'),
    '-I', str(genroot / 'include/generated/uapi'),
    '-I', str(genroot / 'arch/x86/include'),
    '-I', str(genroot / 'arch/x86/include/generated'),
    '-I', str(genroot / 'arch/x86/include/generated/uapi'),
    '-I', str(genroot / 'include'),
    '-I', str(genroot / 'include/generated'),
    '-include', str(genroot / 'include/linux/kconfig.h'),
    '-include', str(genroot / 'include/linux/compiler_types.h'),
    '-D__KERNEL__',
    '-m64',
    '-DCONFIG_X86_X32_ABI',
    '-DCONFIG_AS_CFI=1',
    '-DCONFIG_AS_CFI_SIGNAL_FRAME=1',
    '-DCONFIG_AS_CFI_SECTIONS=1',
    '-DCONFIG_AS_FXSAVEQ=1',
    '-DCONFIG_AS_SSSE3=1',
    '-DCONFIG_AS_CRC32=1',
    '-DCONFIG_AS_AVX=1',
    '-DCONFIG_AS_AVX2=1',
    '-DCONFIG_AS_AVX512=1',
    '-DCONFIG_AS_SHA1_NI=1',
    '-DCONFIG_AS_SHA256_NI=1',
    '-pipe',
    '-DRETPOLINE',
    '-DCC_USING_FENTRY',
    "-DKBUILD_BASENAME='\"asm_offsets\"'",
    "-DKBUILD_MODNAME='\"asm_offsets\"'",
    str(path),
  ]
  process = clang.Exec(clang_args)
  if process.returncode:
    raise clang.ClangException(returncode=process.returncode,
                               stderr=process.stderr, command=clang_args)
  return process.stdout, clang_args


def TryToCreateControlFlowGraphsFromLinuxSrc(
    path: pathlib.Path) -> typing.List[cfg.ControlFlowGraph]:
  """Try to create CFGs from a Linux C source file.

  On failure, an empty list is returned.

  Args:
    path: The path of the source file.

  Returns:
    A list of ControlFlowGraph instances.

  Raises:
    ClangException: If compiling to bytecode fails.
  """
  graphs = []

  try:
    bytecode, _ = BytecodeFromLinuxSrc(path, '-O0')
  except clang.ClangException:
    return graphs

  # Extract a dot sources from the bytecode.
  dot_generator = llvm_util.DotCfgsFromBytecode(bytecode)
  while True:
    try:
      dot = next(dot_generator)
      # Instantiate a CFG from the dot source.
      graph = llvm_util.ControlFlowGraphFromDotSource(dot)
      graph.ValidateControlFlowGraph(strict=False)
      graphs.append(graph)
    except (UnicodeDecodeError, cfg.MalformedControlFlowGraphError,
            ValueError, opt.OptException, pyparsing.ParseException):
      pass
    except StopIteration:
      break

  return graphs


def ProcessLinuxSrc(
    path: pathlib.Path) -> typing.Optional[typing.Dict[str, typing.Any]]:
  graphs = TryToCreateControlFlowGraphsFromLinuxSrc(path)

  src_root = LinuxSourcesDataset().src_tree_root

  rows = []
  for graph in graphs:
    row = CfgDfRowFromControlFlowGraph(graph)
    row.update({
      'program:src_relpath': str(path)[len(str(src_root)) + 1:],
    })
    rows.append(row)

  return rows


def ProcessLinuxSrcToBytecode(
    path: pathlib.Path) -> reachability_pb2.LlvmBytecode:
  src_root = LinuxSourcesDataset().src_tree_root
  version = LinuxSourcesDataset().version

  try:
    bytecode, cflags = BytecodeFromLinuxSrc(path, '-O0')
    clang_returncode = 0
    error_message = ''
  except clang.ClangException as e:
    bytecode = ''
    cflags = e.command
    clang_returncode = e.returncode
    error_message = e.stderr

  return reachability_pb2.LlvmBytecode(
      source_name=f'linux-{version}',
      relpath=str(path)[len(str(src_root)) + 1:],
      lang='C',
      cflags=' '.join(cflags),
      bytecode=bytecode,
      clang_returncode=clang_returncode,
      error_message=error_message,
  )


class LinuxSourcesDataset(linux.LinuxSourcesDataset):
  """Control flow graphs from a subset of the Linux source tree.

  The returned DataFrame has the following schema:

    program:src_relpath (str): The path of the source file within the linux
      source tree.
    cfg:graph (ControlFlowGraph): A control flow graph instance.
    cfg:block_count (int): The number of basic blocks in the CFG.
    cfg:edge_count (int): The number of edges in the CFG.
    cfg:edge_density (float): Number of edges / possible edges, in range [0,1].
    cfg:is_valid (bool): Whether the CFG is valid.
    cfg:is_strict_valid (bool): Whether the CFG is valid when strict.
  """

  def PopulateBytecodeTable(
      self, db: database.Database, commit_every: int = 1000):
    bar = progressbar.ProgressBar()
    bar.max_value = len(self.all_srcs)

    # Process each row of the table in parallel.
    pool = multiprocessing.Pool(FLAGS.vbcd_process_count)
    with db.Session(commit=True) as s:
      for i, proto in enumerate(
          pool.imap_unordered(ProcessLinuxSrcToBytecode, self.all_srcs)):
        bar.update(i)
        s.add(database.LlvmBytecode(**database.LlvmBytecode.FromProto(proto)))
        if not (i % commit_every):
          s.commit()

  @decorators.memoized_property
  def cfgs_df(self) -> pd.DataFrame:
    # Process each row of the table in parallel.
    pool = multiprocessing.Pool(FLAGS.vbcd_process_count)
    rows = []
    for row_batch in pool.imap_unordered(ProcessLinuxSrc, self.kernel_srcs):
      if row_batch:
        rows += row_batch

    # Create the output table.
    df = pd.DataFrame(rows, columns=[
      'program:src_relpath',
      'cfg:graph',
      'cfg:block_count',
      'cfg:edge_count',
      'cfg:edge_density',
      'cfg:is_valid',
      'cfg:is_strict_valid',
    ])

    df.set_index([
      'program:src_relpath',
    ], inplace=True)
    df.sort_index(inplace=True)
    return df


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
      protos.append(reachability_pb2.ControlFlowGraphFromLlvmBytecode(
          bytecode_id=bytecode_id,
          cfg_id=cfg_id,
          control_flow_graph=graph.ToProto(),
          status=0,
          error_message='',
          block_count=graph.number_of_nodes(),
          edge_count=graph.number_of_edges(),
          is_strict_valid=graph.IsValidControlFlowGraph(strict=True)
      ))
    except (UnicodeDecodeError, cfg.MalformedControlFlowGraphError,
            ValueError, opt.OptException, pyparsing.ParseException) as e:
      protos.append(reachability_pb2.ControlFlowGraphFromLlvmBytecode(
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


def PopulateControlFlowGraphTable(db: database.Database):
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
      .filter(~database.LlvmBytecode.id.in_(already_done_ids))

    bar = progressbar.ProgressBar()
    bar.max_value = todo.count()

    # Process bytecodes in parallel.
    pool = multiprocessing.Pool(FLAGS.vbcd_process_count)
    last_commit_time = time.time()
    for i, protos in enumerate(pool.imap_unordered(
        CreateControlFlowGraphsFromLlvmBytecode, todo)):
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
        is_strict_valid=graph.IsValidControlFlowGraph(strict=True)
    )
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


def PopulateFullFlowGraphTable(db: database.Database):
  """Populate the full flow graph table from CFGs."""
  with db.Session(commit=True) as s:
    # Query that returns (bytecode_id,cfg_id) tuples for all CFGs.
    # TODO(cec): Exclude values already in the FFG table.
    todo = s.query(database.ControlFlowGraphProto.bytecode_id,
                   database.ControlFlowGraphProto.cfg_id,
                   database.ControlFlowGraphProto.proto) \
      .filter(database.ControlFlowGraphProto.status == 0)

    bar = progressbar.ProgressBar()
    bar.max_value = todo.count()

    # Process CFGs in parallel.
    pool = multiprocessing.Pool(FLAGS.vbcd_process_count)
    last_commit_time = time.time()
    for i, proto in enumerate(pool.imap_unordered(
        CreateFullFlowGraphFromCfg, todo)):
      the_time = time.time()
      bar.update(i)

      s.add(database.FullFlowGraphProto(
          **database.FullFlowGraphProto.FromProto(proto)))

      # Commit every 10 seconds.
      if the_time - last_commit_time > 10:
        s.commit()
        last_commit_time = the_time


def PopulateBytecodeTableFromGithubCSources(db: database.Database,
                                            tempdir: pathlib.Path):
  language_to_clone = scrape_repos_pb2.LanguageToClone(
      language='c',
      query=[
        scrape_repos_pb2.GitHubRepositoryQuery(
            string="language:c sort:stars fork:false",
            max_results=100),
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

  logging.info("Scraping repos ...")
  for query in language_to_clone.query:
    scraper.RunQuery(scraper.QueryScraper(language_to_clone, query))

  # Clone repos.
  directory = pathlib.Path(language_to_clone.destination_directory)
  meta_files = [pathlib.Path(directory / f) for f in directory.iterdir() if
                cloner.IsRepoMetaFile(f)]
  worker = cloner.AsyncWorker(meta_files)
  logging.info('Cloning %s repos from GitHub ...',
               humanize.intcomma(worker.max))
  bar = progressbar.ProgressBar(max_value=worker.max, redirect_stderr=True)
  worker.start()
  while worker.is_alive():
    bar.update(worker.i)
    worker.join(.5)
  bar.update(worker.i)

  logging.info("Importing repos to contentfiles database ...")
  for imp in language_to_clone.importer:
    [preprocessors.GetPreprocessorFunction(p) for p in imp.preprocessor]

  pool = multiprocessing.Pool(FLAGS.processes)
  d = pathlib.Path(language_to_clone.destination_directory)
  d = d.parent / (str(d.name) + '.db')
  contentfiles_db = contentfiles.ContentFiles(f'sqlite:///{d}')
  if pathlib.Path(language_to_clone.destination_directory).is_dir():
    importer.ImportFromLanguage(contentfiles_db, language_to_clone, pool)

  logging.info("Populating bytecode table ...")
  import_from_github.PopulateBytecodeTable(
      contentfiles_db, language_to_clone, db)


def NowString() -> str:
  return str(datetime.datetime.now())


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  db = database.Database(FLAGS.vbcd)

  with db.Session() as session:
    opencl_dataset_imported = session.query(database.Meta) \
      .filter(database.Meta.key == 'opencl_dataset_imported').first()
  if not opencl_dataset_imported:
    logging.info("Importing OpenCL dataset ...")
    OpenClDeviceMappingsDataset().PopulateBytecodeTable(db)
    with db.Session(commit=True) as session:
      session.add(
          database.Meta(key='opencl_dataset_imported', value=NowString()))

  with db.Session() as session:
    linux_sources_imported = session.query(database.Meta) \
      .filter(database.Meta.key == 'linux_sources_imported').first()
  if not linux_sources_imported:
    logging.info("Processing Linux dataset ...")
    LinuxSourcesDataset().PopulateBytecodeTable(db)
    with db.Session(commit=True) as session:
      session.add(
          database.Meta(key='linux_sources_imported', value=NowString()))

  with db.Session() as session:
    github_c_sources_imported = session.query(database.Meta) \
      .filter(database.Meta.key == 'github_c_sources_imported').first()
  if not github_c_sources_imported:
    logging.info('Processing GitHub C sources ...')
    with tempfile.TemporaryDirectory(prefix='phd_') as d:
      PopulateBytecodeTableFromGithubCSources(db, pathlib.Path(d))
    with db.Session(commit=True) as session:
      session.add(
          database.Meta(key='github_c_sources_imported', value=NowString()))

  # logging.info("Processing CFGs ...")
  # PopulateControlFlowGraphTable(db)
  #
  # logging.info("Processing full flow graphs ...")
  # PopulateFullFlowGraphTable(db)


if __name__ == '__main__':
  app.run(main)
