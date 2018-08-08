"""Gym environments for the LLVM optimizer, to be used with gym.make().

To add a new environment, add a new call to registration.register(), and add the
name to the ENVIRONMENTS list.
"""
from absl import flags
from gym.envs import registration

from compilers.llvm import opt
from datasets.benchmarks import bzip2
from datasets.benchmarks import llvm_test_suite
from experimental.compilers.random_opt.proto import random_opt_pb2


FLAGS = flags.FLAGS

# A default environment name, registered below.
DEFAULT_ENV_ID = 'LLVM-bzip2-512K-v0'

# A list of all environments registered in this file.
ENVIRONMENTS = [
  'LLVM-bzip2-512K-v0',
  'LLVM-queens-14x14-v0',
]

registration.register(
    id='LLVM-bzip2-512K-v0',
    entry_point='phd.experimental.compilers.random_opt.env:LlvmOptEnv',
    kwargs={
      'config': random_opt_pb2.Environment(
          input_src=[str(x) for x in bzip2.BZIP2_SRCS],
          setup_cmd='head -c 512K </dev/urandom > @D/input.dat',
          exec_cmd=('$@ -z < @D/input.dat > @D/input.dat.bz2 && '
                    '$@ -d < @D/input.dat.bz2 > @D/output.dat'),
          eval_cmd=f'cmp --silent @D/input.dat @D/output.dat',
          candidate_pass=list(opt.ALL_PASSES),
      )
    },
)

registration.register(
    id='LLVM-queens-14x14-v0',
    entry_point='phd.experimental.compilers.random_opt.env:LlvmOptEnv',
    kwargs={
      'config': random_opt_pb2.Environment(
          input_src=
          llvm_test_suite.BENCHMARKS['SingleSource']['McGill']['queens'][
            'srcs'],
          setup_cmd='$@ 14 > @D/gold_standard_output.txt',
          exec_cmd='$@ 14 > @D/output.txt',
          eval_cmd='cmp --silent @D/gold_standard_output.txt @D/output.txt',
          candidate_pass=list(opt.ALL_PASSES),
      )
    }
)
