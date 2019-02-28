"""Public API for cldrive."""

from absl import flags

from gpu.cldrive.legacy import env as _env
from gpu.cldrive.proto import cldrive_pb2
from gpu.oclgrind import oclgrind
from labm8 import bazelutil
from labm8 import pbutil
import pandas as pd

FLAGS = flags.FLAGS

_NATIVE_DRIVER = bazelutil.DataPath('phd/gpu/cldrive/native_driver')


def Drive(instances: cldrive_pb2.CldriveInstances,
          timeout_seconds: int = 360) -> cldrive_pb2.CldriveInstances:
  if (len(instances.instance) and
      instances.instance[0].device.name == _env.OclgrindOpenCLEnvironment().name
     ):
    command = [str(oclgrind.OCLGRIND_PATH), str(_NATIVE_DRIVER)]
  else:
    command = [str(_NATIVE_DRIVER)]

  pbutil.RunProcessMessageInPlace(
      command, instances, timeout_seconds=timeout_seconds)
  return instances


def InstancesToDataFrame(
    instances: cldrive_pb2.CldriveInstances) -> pd.DataFrame:
  """Convert result of CldriveInstances message from Drive() to dataframe.

  Cldrive uses a deeply nested protocol buffer schema:

    Instances -> Instance
      Instance -> Kernel
        Kernel -> Run
          Run -> Log

  This method flattens the protocol buffers to a pandas dataframe.

  The default (ascending integer) index is used.

  Args:
    instances: A CldriveInstances message, as returned by Drive().

  Returns:
    A Pandas DataFrame with the following columns:
      device (str): From CldriveInstance.device.name field.
      opencl_src (str): From CldriveInstance.opencl_src field.
      build_opts (str): From CldriveInstance.build_opts field.
      kernel (str): The OpenCL kernel name, from CldriveKernelInstance.name
        field. If CldriveInstance.outcome != PASS, this will be empty.
      work_item_local_mem_size (int): From
        CldriveKernelInstance.work_item_local_mem_size_in_bytes field. If
        CldriveInstance.outcome != PASS, this will be empty.
      work_item_private_mem_size (int): From
        CldriveKernelInstance.work_item_private_mem_size_in_bytes field. If
        CldriveInstance.outcome != PASS, this will be empty.
      global_size (int): From CldriveInstance.dynamic_params.global_size_x
        field. If CldriveInstance.outcome != PASS, this will be empty.
      local_size (int): From CldriveInstance.dynamic_params.local_size_x field.
        If CldriveInstance.outcome != PASS, this will be empty.
      outcome (str): A stringified enum value. Either CldriveInstance.outcome if
        CldriveInstance.outcome != PASS, else CldriveKernelInstance.outcome if
        CldriveKernelInstance.outcome != PASS, else CldriveKernelRun.outcome.
      runtime_ms (float): From CldriveKernelRun.log.runtime_ms. If outcome !=
        PASS, this will be empty.
      transferred_bytes (int): From CldriveKernelRun.log.transferred_bytes. If
        outcome != PASS, this will be empty.
  """
  raise NotImplementedError
