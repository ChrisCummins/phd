"""Reachability analysis datasets."""
import multiprocessing
import typing

import pandas as pd
from absl import app
from absl import flags

from compilers.llvm import clang
from datasets.opencl.device_mapping import \
  opencl_device_mapping_dataset as ocl_dataset
from deeplearning.clgen.preprocessors import opencl
from experimental.compilers.reachability import llvm_util
from experimental.compilers.reachability import reachability_pb2
from labm8 import decorators


FLAGS = flags.FLAGS


def BytecodeFromOpenClString(opencl_string: str) -> str:
  """Create bytecode from OpenCL source string.

  Args:
    opencl_string: A string of OpenCL code.

  Returns:
    The bytecode as a string.

  Raises:
    ClangException: If compiling to bytecode fails.
  """
  clang_args = opencl.GetClangArgs(use_shim=False) + [
    '-O0', '-S', '-emit-llvm', '-o', '-', '-i', '-']
  process = clang.Exec(clang_args, stdin=opencl_string)
  if process.returncode:
    raise clang.ClangException("clang failed with returncode "
                               f"{process.returncode}:\n{process.stderr}")
  return process.stdout


def CreateControlFlowGraphProtoFromOpenClKernel(
    kernel_name: str, opencl_kernel: str) -> typing.Optional[
  reachability_pb2.ControlFlowGraph]:
  """Try to create a CFG proto from an opencl kernel.

  Args:
    kernel_name: The name of the OpenCL kernel defined in opencl_kernel.
    opencl_kernel: A string of OpenCL. This should contain a single kernel
      definition.

  Returns:
    A ControlFlowGraph proto, or None if compilation to bytecode fails.

  Raises:
    ClangException: If compiling to bytecode fails.
    ValueError: If opencl_kernel contains multiple functions.
  """
  bytecode = BytecodeFromOpenClString(opencl_kernel)

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

  # Return the graph as a proto.
  return graph.ToProto()


def ProcessProgramDfIterItem(
    row: typing.Dict[str, str]) -> typing.Optional[typing.Dict[str, str]]:
  benchmark_suite_name = row['program:benchmark_suite_name']
  benchmark_name = row['program:benchmark_name']
  kernel_name = row['program:opencl_kernel_name']
  src = row['program:opencl_src']

  try:
    graph_proto = CreateControlFlowGraphProtoFromOpenClKernel(
        kernel_name, src)
  except clang.ClangException:
    return None

  return {
    'program:benchmark_suite_name': benchmark_suite_name,
    'program:benchmark_name': benchmark_name,
    'program:opencl_kernel_name': kernel_name,
    'program:cfg_proto': graph_proto.SerializeToString(),
  }


class OpenClDeviceMappingsDataset(ocl_dataset.OpenClDeviceMappingsDataset):
  """An extension of the OpenCL device mapping dataset for control flow graphs.

  The returned DataFrame has the following schema:

    program:benchmark_suite_name (str): The name of the benchmark suite.
    program:benchmark_name (str): The name of the benchmark program.
    program:opencl_kernel_name (str): The name of the OpenCL kernel.
    program:cfg_proto (bytes): A serialized ControlFlowGraph proto.
  """

  @decorators.memoized_property
  def cfgs_df(self) -> pd.DataFrame:
    programs_df = self.programs_df.reset_index()

    # Process each row of the table in parallel.
    pool = multiprocessing.Pool()
    rows = []
    for row in pool.map_async(ProcessProgramDfIterItem, programs_df.iterrows()):
      if row:
        rows.append(row)

    # Create the output table.
    df = pd.DataFrame(rows, columns=[
      'program:benchmark_suite_name',
      'program:benchmark_name',
      'program:opencl_kernel_name',
      'program:cfg_proto',
    ])

    df.set_index([
      'program:benchmark_suite_name',
      'program:benchmark_name',
      'program:opencl_kernel_name',
    ], inplace=True)
    df.sort_index(inplace=True)
    return df


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))


if __name__ == '__main__':
  app.run(main)
