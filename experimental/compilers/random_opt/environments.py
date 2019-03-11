"""Gym environments for the LLVM optimizer, to be used with gym.make().

To add a new environment, add a new call to registration.register(), and add the
name to the ENVIRONMENTS list. The environment class is defined in:
    //experimental/compilers/random_opt/implementation.py
"""
from gym.envs import registration

from compilers.llvm import util as llvm_util
from datasets.benchmarks import bzip2
from datasets.benchmarks import llvm_test_suite
from experimental.compilers.random_opt.proto import random_opt_pb2
from labm8 import app
from labm8 import labtypes

FLAGS = app.FLAGS

app.DEFINE_integer(
    'runtime_num_runs', 10,
    'The number of times to execute a binary to get its runtime.')

# A list of all environments registered in this file.
ENVIRONMENTS = [
    'LLVM-bzip2-512K-v0',
    'LLVM-bzip2-1M-v0',
    'LLVM-queens-8x8-v0',
    'LLVM-queens-14x14-v0',
    'LLVM-delayed-reward-bzip2-512K-v0',
    'LLVM-delayed-reward-bzip2-1M-v0',
    'LLVM-delayed-reward-queens-8x8-v0',
    'LLVM-delayed-reward-queens-14x14-v0',
]

# A default environment name, registered below.
DEFAULT_ENV_ID = 'LLVM-delayed-reward-queens-14x14-v0'

# The list of opt passes which defines the action space.
DEFAULT_PASS_LIST = list(
    sorted(set(labtypes.flatten(llvm_util.GetOptArgs(['-O3'])))))

# Environment generator functions.


def _GetEntryPoint(delayed_reward: bool) -> str:
  if delayed_reward:
    return ('phd.experimental.compilers.random_opt.implementation:'
            'LlvmOptDelayedRewardEnv')
  else:
    return 'phd.experimental.compilers.random_opt.implementation:LlvmOptEnv'


def _GetBzip2EnvironmentArgs(dataset_size: str, delayed_reward: bool):
  return {
      'entry_point': _GetEntryPoint(delayed_reward),
      'kwargs': {
          'config':
          random_opt_pb2.Environment(
              input_src=bzip2.Bzip2.srcs,
              # Create random data for bzip to compress.
              setup_cmd=f'head -c {dataset_size} </dev/urandom > @D/input.dat',
              # Compress and deflate the input data.
              exec_cmd=('$@ -z < @D/input.dat > @D/input.dat.bz2 && '
                        '$@ -d < @D/input.dat.bz2 > @D/output.dat'),
              eval_cmd='cmp --silent @D/input.dat @D/output.dat',
              candidate_pass=DEFAULT_PASS_LIST,
          )
      }
  }


def _GetQueensEnvironmentArgs(n: int, delayed_reward: bool):
  return {
      'entry_point': _GetEntryPoint(delayed_reward),
      'kwargs': {
          'config':
          random_opt_pb2.Environment(
              input_src=llvm_test_suite.SingleSource.Benchmarks.McGill.queens.
              srcs,
              # Generate a gold standard using the binary. The assumes that the base
              # build (before any opt passes have been run) is correct.
              setup_cmd=f'$@ {n} > @D/gold_standard_output.txt',
              exec_cmd=f'$@ {n} > @D/output.txt',
              eval_cmd='cmp --silent @D/gold_standard_output.txt @D/output.txt',
              candidate_pass=DEFAULT_PASS_LIST,
          )
      }
  }


# Register the environments.

registration.register(
    id='LLVM-bzip2-512K-v0',
    **_GetBzip2EnvironmentArgs('512K', False),
)

registration.register(
    id='LLVM-bzip2-1M-v0',
    **_GetBzip2EnvironmentArgs('1M', False),
)

registration.register(
    id='LLVM-queens-8x8-v0',
    **_GetQueensEnvironmentArgs(8, False),
)

registration.register(
    id='LLVM-queens-10x10-v0',
    **_GetQueensEnvironmentArgs(10, False),
)

registration.register(
    id='LLVM-queens-12x12-v0',
    **_GetQueensEnvironmentArgs(12, False),
)

registration.register(
    id='LLVM-queens-14x14-v0',
    **_GetQueensEnvironmentArgs(14, False),
)

registration.register(
    id='LLVM-delayed-reward-bzip2-512K-v0',
    **_GetBzip2EnvironmentArgs('512K', True),
)

registration.register(
    id='LLVM-delayed-reward-bzip2-1M-v0',
    **_GetBzip2EnvironmentArgs('1M', True),
)

registration.register(
    id='LLVM-delayed-reward-queens-8x8-v0',
    **_GetQueensEnvironmentArgs(8, True),
)

registration.register(
    id='LLVM-delayed-reward-queens-10x10-v0',
    **_GetQueensEnvironmentArgs(10, True),
)

registration.register(
    id='LLVM-delayed-reward-queens-12x12-v0',
    **_GetQueensEnvironmentArgs(12, True),
)

registration.register(
    id='LLVM-delayed-reward-queens-14x14-v0',
    **_GetQueensEnvironmentArgs(14, True),
)
