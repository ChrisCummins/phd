"""
Export dsmith_04 CSV files to Protos.
"""
import os
import pathlib
import typing

import numpy as np
import pandas as pd

from absl import app
from absl import flags
from absl import logging

from deeplearning.deepsmith.proto import deepsmith_pb2

FLAGS = flags.FLAGS

flags.DEFINE_string('csv_dir', None, 'Directory to import CSVs from.')
flags.DEFINE_string('proto_dir', None, 'Directory to export protos to.')


OPENCL_CSVS = {
    'assertions.csv',
    'classifications.csv',
    'clsmith_program_metas.csv',
    'clsmith_testcase_metas.csv',
    'dsmith_program_metas.csv',
    'dsmith_testcase_metas.csv',
    'majorities.csv',
    'platforms.csv',
    'programs.csv',
    'reductions.csv',
    'results.csv',
    'results_metas.csv',
    'stackdumps.csv',
    'stderrs.csv',
    'stdouts.csv',
    'testbeds.csv',
    'testcases.csv',
    'threads.csv',
    'unreachables.csv',
}


def _ReadOpenCLCsv(path: str) -> pd.DataFrame:
  df = pd.read_csv(path, names=['id', 'platform_id', 'optimizations'],
                   dtype={'id': np.int32, 'platform_id': np.int32,
                          'optimizations': np.bool}, index_col='id')
  return df.replace({np.nan: ''})


def _ReadOpenCLPlatformsCsv(path: str) -> pd.DataFrame:
  df = pd.read_csv(path, names=['id', 'platform', 'device', 'driver',
                                'opencl', 'devtype', 'host'],
                   dtype={'id': np.int32, 'device': str, 'opencl': str},
                   index_col='id')
  return df.replace({np.nan: ''})


PLATFORM_MAP = {
    'ComputeAorta': 'computeaorta',
    'Intel Gen OCL Driver': 'intel_gen',
    'Intel(R) OpenCL': 'intel',
    'NVIDIA CUDA': 'nvidia',
    'Oclgrind': 'oclgrind',
    'Portable Computing Language': 'pocl',
}


DEVTYPE_MAP = {
    '3': 'CPU',
}


HOSTS_MAP = {
    'openSUSE  13.1 64bit': 'openSUSE 13.1 64bit',
}


DEVICE_MAP = {
    'pthread-Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz': 'pocl_cpu_e5-2620',
    'Oclgrind 16.10': 'oclgrind_cpu',
    'GeForce GTX 780': 'nvidia_gpu_gtx780',
    'GeForce GTX 1080': 'nvidia_gpu_gtx1080',
    'GeForce GTX 1080': 'nvidia_gpu_gtx1080',
    'Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz': 'intel_cpu_e5-2620v4',
    'Intel(R) Many Integrated Core Acceleration Card': 'intel_xeon_phi',
    'Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz': 'intel_cpu_i5-4570',
    '      Intel(R) Xeon(R) CPU E5-2650 v2 @ 2.60GHz': 'intel_cpu_e5-2650v2',
    'Intel(R) HD Graphics Haswell GT2 Desktop': 'intel_gpu_gt2',
    'Codeplay Software Ltd. - host CPU': 'codeplay_cpu',
    'Oclgrind Simulator': 'oclgrind_cpu',
    '': 'clang',
}


def _SetIf(out, key, value, setvalue = None):
  if value:
    out[key] = setvalue or value


def _ParsePlatform(platform: typing.Dict) -> deepsmith_pb2.Testbed:
  name = DEVICE_MAP[platform['device']]
  opts = {}
  _SetIf(opts, 'opencl_device', platform['device'].strip())
  _SetIf(opts, 'opencl_version', platform['opencl'].strip())
  _SetIf(opts, 'host', HOSTS_MAP.get(platform['host'], platform['host']))
  if name == "clang":
    _SetIf(opts, 'llvm_version', platform['driver'].strip())
  else:
    _SetIf(opts, 'driver_version', platform['driver'].strip())
    _SetIf(opts, 'opencl_devtype', DEVTYPE_MAP.get(platform['devtype'], platform['devtype']))
    _SetIf(opts, 'opencl_platform', platform['platform'].strip())
  return deepsmith_pb2.Testbed(
      toolchain='opencl',
      name=name,
      opts=opts,
  )


def _ParseTestbed(
    testbed: typing.Dict,
    platforms: typing.Dict[int, deepsmith_pb2.Testbed]) -> deepsmith_pb2.Testbed:
  t = deepsmith_pb2.Testbed()
  t.CopyFrom(platforms[testbed['platform_id']])
  if t.name != "clang":
    t.opts['opencl_opt'] = 'enabled' if testbed['optimizations'] else 'disabled'
  return t



def _OpenClCsvsToProtos(csv_dir: str, proto_dir) -> None:
  logging.info('Converting OpenCL CSVs in %s to protos in %s',
               csv_dir, proto_dir)
  assert(all(os.path.isfile(csv_dir / x) for x in os.listdir(csv_dir)))

  testbeds_csv = _ReadOpenCLCsv(csv_dir / 'testbeds.csv')
  platforms_csv = _ReadOpenCLPlatformsCsv(csv_dir / 'platforms.csv')
  platforms = {i: _ParsePlatform(p)
               for i, p in platforms_csv.iterrows()}
  testbeds = {i: _ParseTestbed(t, platforms)
              for i, t in testbeds_csv.iterrows()}
  for _, t in testbeds.items():
    print(t)
    print()


def _CsvsToProtos() -> None:
  csv_dir = pathlib.Path(FLAGS.csv_dir)
  proto_dir = pathlib.Path(FLAGS.proto_dir)

  assert csv_dir and proto_dir

  _OpenClCsvsToProtos(csv_dir / 'dsmith_04_opencl', proto_dir / 'opencl')


def main(argv):
  del argv
  _CsvsToProtos()


if __name__ == '__main__':
  app.run(main)
