"""Reachability analysis datasets."""
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


class OpenClDeviceMappingsDataset(ocl_dataset.OpenClDeviceMappingsDataset):
  """An extension of the OpenCL device mapping dataset for control flow graphs.
  """

  @staticmethod
  def BytecodeFromOpenClKernel(opencl_kernel: str) -> str:
    clang_args = opencl.GetClangArgs(use_shim=False) + [
      '-O0', '-S', '-emit-llvm', '-o', '-', '-i', '-']
    process = clang.Exec(clang_args, stdin=opencl_kernel)
    if process.returncode:
      raise clang.ClangException("clang failed with returncode "
                                 f"{process.returncode}:\n{process.stderr}")
    return process.stdout

  @classmethod
  def CreateControlFlowGraphSetFromOpenClKernel(
      cls, opencl_kernel: str) -> reachability_pb2.ControlFlowGraphSet:
    cfg_set = reachability_pb2.ControlFlowGraphSet()
    bytecode = cls.BytecodeFromOpenClKernel(opencl_kernel)
    for dot in llvm_util.DotCfgsFromBytecode(bytecode):
      graph = llvm_util.ControlFlowGraphFromDotSource(dot)
      graph_proto = cfg_set.graph.add()
      graph.SetProto(graph_proto)
    return cfg_set

  @decorators.memoized_property
  def cfgs_df(self) -> pd.DataFrame:
    # TODO(cec): Create one row per CFG, with CFG name and proto columns, not
    # multiple CFGs per benchmark.
    df = self.programs_df.copy()
    df['program:cfg_set_proto'] = [
      self.CreateControlFlowGraphSetFromOpenClKernel(x)
      for x in df['program:opencl_src']]
    return df


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))


if __name__ == '__main__':
  app.run(main)
