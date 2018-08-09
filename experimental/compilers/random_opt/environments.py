"""Gym environments for the LLVM optimizer, to be used with gym.make().

To add a new environment, add a new call to registration.register(), and add the
name to the ENVIRONMENTS list. The environment class is defined in:
    //experimental/compilers/random_opt/implementation.py
"""
from absl import flags
from gym.envs import registration

from compilers.llvm import opt
from datasets.benchmarks import bzip2
from datasets.benchmarks import llvm_test_suite
from experimental.compilers.random_opt.proto import random_opt_pb2


FLAGS = flags.FLAGS

# A list of all environments registered in this file.
ENVIRONMENTS = [
  'LLVM-bzip2-512K-v0',
  'LLVM-bzip2-1M-v0',
  'LLVM-queens-8x8-v0',
  'LLVM-queens-14x14-v0',
]

# A default environment name, registered below.
DEFAULT_ENV_ID = 'LLVM-queens-14x14-v0'


# Environment generator functions.

def Bzip2Environment(dataset_size: str):
  return {
    'entry_point': ('phd.experimental.compilers.random_opt'
                    '.implementation:LlvmOptEnv'),
    'kwargs': {
      'config': random_opt_pb2.Environment(
          input_src=bzip2.Bzip2.srcs,
          setup_cmd=f'head -c {dataset_size} </dev/urandom > @D/input.dat',
          exec_cmd=('$@ -z < @D/input.dat > @D/input.dat.bz2 && '
                    '$@ -d < @D/input.dat.bz2 > @D/output.dat'),
          eval_cmd='cmp --silent @D/input.dat @D/output.dat',
          candidate_pass=list(opt.ALL_PASSES),
      )
    }
  }


def QueensEnvironment(n: int):
  return {
    'entry_point': ('phd.experimental.compilers.random_opt'
                    '.implementation:LlvmOptEnv'),
    'kwargs': {
      'config': random_opt_pb2.Environment(
          input_src=llvm_test_suite.SingleSource.Benchmarks.McGill.queens.srcs,
          # Generate a gold standard using the binary. The assumes that the base
          # build (before any opt passes have been run) is correct.
          setup_cmd=f'$@ {n} > @D/gold_standard_output.txt',
          exec_cmd=f'$@ {n} > @D/output.txt',
          eval_cmd='cmp --silent @D/gold_standard_output.txt @D/output.txt',
          candidate_pass=list(opt.ALL_PASSES),
      )
    }
  }


# Register the environments.

registration.register(
    id='LLVM-bzip2-512K-v0',
    **Bzip2Environment('512K'),
)

registration.register(
    id='LLVM-bzip2-1M-v0',
    **Bzip2Environment('1M'),
)

registration.register(
    id='LLVM-queens-8x8-v0',
    **QueensEnvironment(8)
)

registration.register(
    id='LLVM-queens-10x10-v0',
    **QueensEnvironment(10)
)

registration.register(
    id='LLVM-queens-12x12-v0',
    **QueensEnvironment(12)
)

registration.register(
    id='LLVM-queens-14x14-v0',
    **QueensEnvironment(14)
)
