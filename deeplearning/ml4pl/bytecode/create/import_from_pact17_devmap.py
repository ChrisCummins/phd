"""Import bytecodes from OpenCL benchmarks used in heterogeneous device mapping
experiments of this paper:

    ﻿Cummins, C., Petoumenos, P., Wang, Z., & Leather, H. (2017). End-to-end
    Deep Learning of Optimization Heuristics. In PACT. IEEE.
"""
import multiprocessing
import typing

import pandas as pd
import progressbar
from labm8 import app
from labm8 import decorators

from compilers.llvm import clang
from compilers.llvm import opt_util
from datasets.opencl.device_mapping import \
  opencl_device_mapping_dataset as ocl_dataset
from deeplearning.clgen.preprocessors import opencl
from deeplearning.ml4pl import ml4pl_pb2
from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs.unlabelled.cfg import control_flow_graph as cfg
from deeplearning.ml4pl.graphs.unlabelled.cfg import llvm_util

FLAGS = app.FLAGS

app.DEFINE_database('bytecode_db', bytecode_database.Database, None,
                    'Path of database to populate.')


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
      '-S',
      '-emit-llvm',
      '-o',
      '-',
      '-i',
      '-',
      '-Wno-everything',  # No warnings please.
  ]
  process = clang.Exec(clang_args, stdin=opencl_string)
  if process.returncode:
    raise clang.ClangException("clang failed",
                               returncode=process.returncode,
                               stderr=process.stderr,
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
  dot_generator = opt_util.DotControlFlowGraphsFromBytecode(bytecode)
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


def ProcessProgramDfIterItem(row: typing.Dict[str, str]
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
    row: typing.Dict[str, str]) -> ml4pl_pb2.LlvmBytecode:
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

  return ml4pl_pb2.LlvmBytecode(
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

  def PopulateBytecodeTable(self,
                            db: bytecode_database.Database,
                            commit_every: int = 1000):
    programs_df = self.programs_df.reset_index()
    bar = progressbar.ProgressBar()
    bar.max_value = len(programs_df)

    # Process each row of the table in parallel.
    pool = multiprocessing.Pool()
    with db.Session(commit=True) as s:
      for i, proto in enumerate(
          pool.imap_unordered(ProcessOpenClProgramDfBytecode,
                              [d for _, d in programs_df.iterrows()])):
        bar.update(i)
        s.add(
            bytecode_database.LlvmBytecode(
                **bytecode_database.LlvmBytecode.FromProto(proto)))
        if not (i % commit_every):
          s.commit()

  @decorators.memoized_property
  def cfgs_df(self) -> pd.DataFrame:
    programs_df = self.programs_df.reset_index()

    # Process each row of the table in parallel.
    pool = multiprocessing.Pool()
    rows = []
    for row in pool.imap_unordered(ProcessProgramDfIterItem,
                                   [d for i, d in programs_df.iterrows()]):
      if row:
        rows.append(row)

    # Create the output table.
    df = pd.DataFrame(rows,
                      columns=[
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
    ],
                 inplace=True)
    df.sort_index(inplace=True)
    return df


def main():
  db = FLAGS.bytecode_db()
  dataset = OpenClDeviceMappingsDataset()
  dataset.PopulateBytecodeTable(db)


if __name__ == '__main__':
  app.Run(main)
