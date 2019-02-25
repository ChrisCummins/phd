"""
Export dsmith_04 databases to Protos.
"""
import configparser
import os
import pathlib
import typing

import MySQLdb
from absl import app
from absl import flags
from absl import logging

from deeplearning.deepsmith.proto import deepsmith_pb2

FLAGS = flags.FLAGS

flags.DEFINE_string('proto_dir', None, 'Directory to export protos to.')

OPENCL_DEVTYPE_MAP = {
    '3': 'CPU',
}

HOSTS_MAP = {
    'openSUSE  13.1 64bit': 'openSUSE 13.1 64bit',
}

OPENCL_DEVICE_MAP = {
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


def _SetIf(out: typing.Dict[str, typing.Any],
           key: typing.Any,
           value: typing.Any,
           setvalue: typing.Any = None) -> typing.Dict[str, typing.Any]:
  if value:
    out[key] = setvalue or value
  return out


def _GetOpenCLGenerator(generator_id) -> deepsmith_pb2.Generator:
  if generator_id == 0:
    return deepsmith_pb2.Generator(
        name="clsmith",
        opts={
            "git_commit": "b637b31c31e0f90ef199ca492af05172400df050",
            "git_remote": "https://github.com/ChrisCummins/CLSmith.git",
        })
  elif generator_id == 1:
    return deepsmith_pb2.Generator(
        name="clgen",
        opts={
            "git_commit": "9556e7112ba2bd6f79ee59eef74f0a2304efa007",
            "git_remote": "https://github.com/ChrisCummins/clgen.git",
            "version": "0.4.0.dev0",
        })
  elif generator_id == 2:
    return deepsmith_pb2.Generator(
        name="randchar",
        opts={
            "url":
            "https://github.com/ChrisCummins/dsmith/blob/fd986a36a23b2a398f33d5b5852d930b462401b1/dsmith/opencl/generators.py#L175",
        })
  else:
    raise LookupError


def _GetOpenCLHarness(harness_id, timeout) -> deepsmith_pb2.Harness:
  if harness_id == -1:
    return deepsmith_pb2.Harness(
        name="clang",
        opts={
            "timeout_seconds":
            str(int(timeout)),
            "url":
            "https://github.com/ChrisCummins/dsmith/blob/fd986a36a23b2a398f33d5b5852d930b462401b1/dsmith/opencl/harnesses.py#L292",
        })
  elif harness_id == 0:
    return deepsmith_pb2.Harness(
        name="cl_launcher",
        opts={
            "timeout_seconds": str(int(timeout)),
            "git_commit": "b637b31c31e0f90ef199ca492af05172400df050",
            "git_remote": "https://github.com/ChrisCummins/CLSmith.git",
        })
  elif harness_id == 1:
    return deepsmith_pb2.Harness(
        name="cldrive",
        opts={
            "timeout_seconds": str(int(timeout)),
            "git_commit": "9556e7112ba2bd6f79ee59eef74f0a2304efa007",
            "git_remote": "https://github.com/ChrisCummins/clgen.git",
            "version": "0.4.0.dev0",
        })
  else:
    raise LookupError


def _ExportOpenCLTestcases(cursor, start_id, proto_dir):
  batch_size = 1000
  testcase_id = start_id
  while True:
    cursor.execute(
        """
SELECT
  testcases.id,
  programs.generator,
  programs.date,
  programs.generation_time,
  programs.src,
  testcases.harness,
  testcases.timeout,
  threads.gsize_x,
  threads.gsize_y,
  threads.gsize_z,
  threads.lsize_x,
  threads.lsize_y,
  threads.lsize_z
FROM testcases
LEFT JOIN programs on testcases.program_id = programs.id
LEFT JOIN threads on testcases.threads_id = threads.id
WHERE testcases.id >= %s
AND testcases.id NOT IN (
  SELECT testcase_id FROM
    results
)
ORDER BY testcases.id
LIMIT %s
""", (testcase_id, batch_size))
    i = 0
    for row in cursor:
      i += 1
      (testcase_id, generator_id, program_date, program_generation_time,
       program_src, harness_id, harness_timeout, gsize_x, gsize_y, gsize_z,
       lsize_x, lsize_y, lsize_z) = row
      inputs = {
          "src": program_src,
      }
      if harness_id != -1:
        inputs["gsize"] = f"{gsize_x},{gsize_y},{gsize_z}"
        inputs["lsize"] = f"{lsize_x},{lsize_y},{lsize_z}"
      proto = deepsmith_pb2.Testcase(
          toolchain="opencl",
          generator=_GetOpenCLGenerator(generator_id),
          harness=_GetOpenCLHarness(harness_id, harness_timeout),
          inputs=inputs,
          invariant_opts={},
          profiling_events=[
              deepsmith_pb2.ProfilingEvent(
                  client="cc1",
                  type="generation",
                  duration_seconds=program_generation_time,
                  date_epoch_seconds=int(program_date.strftime('%s')),
              ),
          ])
      with open(proto_dir / 'opencl' / 'testcases' / str(testcase_id),
                'wb') as f:
        f.write(proto.SerializeToString())
    if i < batch_size:
      return


def _ExportOpenCLResults(cursor, start_id, proto_dir):
  batch_size = 1000
  result_id = start_id
  while True:
    cursor.execute(
        """
SELECT
  results.id,
  platforms.platform,
  platforms.device,
  platforms.driver,
  platforms.opencl,
  platforms.devtype,
  platforms.host,
  testbeds.optimizations,
  programs.generator,
  programs.date,
  programs.generation_time,
  programs.src,
  testcases.harness,
  testcases.timeout,
  results.date,
  results.returncode,
  results.runtime,
  stdouts.stdout,
  stderrs.stderr,
  stderrs.truncated,
  threads.gsize_x,
  threads.gsize_y,
  threads.gsize_z,
  threads.lsize_x,
  threads.lsize_y,
  threads.lsize_z
FROM results
LEFT JOIN testbeds ON results.testbed_id = testbeds.id
LEFT JOIN platforms ON testbeds.platform_id = platforms.id
LEFT JOIN testcases on results.testcase_id = testcases.id
LEFT JOIN programs on testcases.program_id = programs.id
LEFT JOIN threads on testcases.threads_id = threads.id
LEFT JOIN stdouts on results.stdout_id = stdouts.id
LEFT JOIN stderrs on results.stderr_id = stderrs.id
WHERE results.id >= %s
ORDER BY results.id
LIMIT %s
""", (result_id, batch_size))

    i = 0
    for row in cursor:
      i += 1
      (result_id, platform_name, device_name, driver_version, opencl_version,
       devtype, host_os, cl_opt, generator_id, program_date,
       program_generation_time, program_src, harness_id, harness_timeout,
       result_date, returncode, runtime, stdout, stderr, truncated_stderr,
       gsize_x, gsize_y, gsize_z, lsize_x, lsize_y, lsize_z) = row
      inputs = {
          "src": program_src,
      }
      if harness_id != -1:
        inputs["gsize"] = f"{gsize_x},{gsize_y},{gsize_z}"
        inputs["lsize"] = f"{lsize_x},{lsize_y},{lsize_z}"
      testbed_name = OPENCL_DEVICE_MAP[device_name]
      testbed_opts = {}
      _SetIf(testbed_opts, 'opencl_device', device_name.strip())
      _SetIf(testbed_opts, 'opencl_version', opencl_version.strip())
      _SetIf(testbed_opts, 'host', HOSTS_MAP.get(host_os, host_os))
      if testbed_name == "clang":
        _SetIf(testbed_opts, 'llvm_version', driver_version.strip())
      else:
        _SetIf(testbed_opts, 'driver_version', driver_version.strip())
        _SetIf(testbed_opts, 'opencl_devtype',
               OPENCL_DEVTYPE_MAP.get(devtype, devtype))
        _SetIf(testbed_opts, 'opencl_platform', platform_name.strip())
        _SetIf(testbed_opts, 'opencl_opt', 'enabled' if cl_opt else 'disabled')
      testbed = deepsmith_pb2.Testbed(
          toolchain='opencl',
          name=testbed_name,
          opts=testbed_opts,
      )

      proto = deepsmith_pb2.Result(
          testcase=deepsmith_pb2.Testcase(
              toolchain="opencl",
              generator=_GetOpenCLGenerator(generator_id),
              harness=_GetOpenCLHarness(harness_id, harness_timeout),
              inputs=inputs,
              invariant_opts={},
              profiling_events=[
                  deepsmith_pb2.ProfilingEvent(
                      client="cc1",
                      type="generation",
                      duration_seconds=program_generation_time,
                      date_epoch_seconds=int(program_date.strftime('%s')),
                  ),
              ]),
          testbed=testbed,
          returncode=returncode,
          outputs={
              "stdout": stdout,
              "stderr": stderr,
          },
          profiling_events=[
              deepsmith_pb2.ProfilingEvent(
                  client={
                      'Ubuntu 16.04 64bit': 'cc1',
                      'CentOS Linux 7.1.1503 64bit': 'fuji',
                      'openSUSE  13.1 64bit': 'kobol',
                  }[host_os],
                  type="runtime",
                  duration_seconds=runtime,
                  date_epoch_seconds=int(result_date.strftime('%s')),
              ),
          ],
      )
      with open(proto_dir / 'opencl' / 'results' / str(result_id), 'wb') as f:
        f.write(proto.SerializeToString())
    if i < batch_size:
      return


def _ExportSolidityTestcases(cursor, start_id, proto_dir):
  batch_size = 1000
  testcase_id = start_id
  while True:
    cursor.execute(
        """
SELECT
  testcases.id,
  programs.generator,
  programs.date,
  programs.generation_time,
  programs.src,
  testcases.harness,
  testcases.timeout
FROM testcases
LEFT JOIN programs on testcases.program_id = programs.id
WHERE testcases.id >= %s
AND testcases.id NOT IN (
  SELECT testcase_id FROM
    results
)
ORDER BY testcases.id
LIMIT %s
""", (testcase_id, batch_size))
    i = 0
    for row in cursor:
      i += 1
      (testcase_id, generator_id, program_date, program_generation_time,
       program_src, harness_id, harness_timeout) = row
      proto = deepsmith_pb2.Testcase(
          toolchain='solidity',
          generator=_GetSolidityGenerator(generator_id),
          harness=deepsmith_pb2.Harness(
              name='solc',
              opts={
                  'timeout_seconds':
                  str(int(harness_timeout)),
                  'url':
                  'https://github.com/ChrisCummins/dsmith/blob/5181c7c95575d428b5144a25549e5a5a55a3da31/dsmith/sol/harnesses.py#L117',
              },
          ),
          inputs={
              "src": program_src,
          },
          invariant_opts={},
          profiling_events=[
              deepsmith_pb2.ProfilingEvent(
                  client="cc1",
                  type="generation",
                  duration_seconds=program_generation_time,
                  date_epoch_seconds=int(program_date.strftime('%s')),
              ),
          ])
      with open(proto_dir / 'sol' / 'testcases' / str(testcase_id), 'wb') as f:
        f.write(proto.SerializeToString())
    if i < batch_size:
      return


def _GetSolidityGenerator(generator_id) -> deepsmith_pb2.Generator:
  if generator_id == -1:
    return deepsmith_pb2.Generator(name="github", opts={})
  elif generator_id == 1:
    return deepsmith_pb2.Generator(
        name="clgen",
        opts={
            "git_commit": "9556e7112ba2bd6f79ee59eef74f0a2304efa007",
            "git_remote": "https://github.com/ChrisCummins/clgen.git",
            "version": "0.4.0.dev0",
        })
  elif generator_id == 2:
    return deepsmith_pb2.Generator(
        name="randchar",
        opts={
            "url":
            "https://github.com/ChrisCummins/dsmith/blob/5181c7c95575d428b5144a25549e5a5a55a3da31/dsmith/sol/generators.py#L203",
        })
  else:
    raise LookupError


def _ExportSolidityResults(cursor, start_id, proto_dir):
  batch_size = 1000
  result_id = start_id
  while True:
    cursor.execute(
        """
SELECT
  results.id,
  platforms.platform,
  platforms.version,
  platforms.host,
  testbeds.optimizations,
  programs.generator,
  programs.date,
  programs.generation_time,
  programs.src,
  testcases.harness,
  testcases.timeout,
  results.date,
  results.returncode,
  results.runtime,
  stdouts.stdout,
  stderrs.stderr
FROM results
LEFT JOIN testbeds ON results.testbed_id = testbeds.id
LEFT JOIN platforms ON testbeds.platform_id = platforms.id
LEFT JOIN testcases on results.testcase_id = testcases.id
LEFT JOIN programs on testcases.program_id = programs.id
LEFT JOIN stdouts on results.stdout_id = stdouts.id
LEFT JOIN stderrs on results.stderr_id = stderrs.id
WHERE results.id >= %s
ORDER BY results.id
LIMIT %s
""", (result_id, batch_size))
    i = 0
    for row in cursor:
      i += 1
      (result_id, platform_name, platform_version, host_os, optimizations,
       generator_id, program_date, program_generation_time, program_src,
       harness_id, harness_timeout, result_date, returncode, runtime, stdout,
       stderr) = row
      assert harness_id == 2
      proto = deepsmith_pb2.Result(
          testcase=deepsmith_pb2.Testcase(
              toolchain='solidity',
              generator=_GetSolidityGenerator(generator_id),
              harness=deepsmith_pb2.Harness(
                  name='solc',
                  opts={
                      'timeout_seconds':
                      str(int(harness_timeout)),
                      'url':
                      'https://github.com/ChrisCummins/dsmith/blob/5181c7c95575d428b5144a25549e5a5a55a3da31/dsmith/sol/harnesses.py#L117',
                  },
              ),
              inputs={
                  "src": program_src,
              },
              invariant_opts={},
              profiling_events=[
                  deepsmith_pb2.ProfilingEvent(
                      client="cc1",
                      type="generation",
                      duration_seconds=program_generation_time,
                      date_epoch_seconds=int(program_date.strftime('%s')),
                  ),
              ]),
          testbed=deepsmith_pb2.Testbed(
              toolchain='solidity',
              name=platform_name,
              opts={
                  'version': platform_version,
                  'optimizations': 'enabled' if optimizations else 'disabled',
              },
          ),
          returncode=returncode,
          outputs={
              "stdout": stdout,
              "stderr": stderr,
          },
          profiling_events=[
              deepsmith_pb2.ProfilingEvent(
                  client='cc1',
                  type="runtime",
                  duration_seconds=runtime,
                  date_epoch_seconds=int(result_date.strftime('%s')),
              ),
          ],
      )
      with open(proto_dir / 'sol' / 'results' / str(result_id), 'wb') as f:
        f.write(proto.SerializeToString())
    if i < batch_size:
      return


def _GetMySqlCredentials():
  cfg = configparser.ConfigParser()
  cfg.read(os.path.expanduser('~/.my.cnf'))
  return cfg['mysql']['user'], cfg['mysql']['password']


def _ExportProtos() -> None:
  proto_dir = pathlib.Path(FLAGS.proto_dir)

  assert proto_dir

  credentials = _GetMySqlCredentials()
  cnx = MySQLdb.connect(
      database='dsmith_04_opencl', user=credentials[0], password=credentials[1])
  cursor = cnx.cursor()

  # Get the last exported result.
  (proto_dir / 'opencl' / 'results').mkdir(parents=True, exist_ok=True)
  ids = sorted([int(x) for x in os.listdir(proto_dir / 'opencl' / 'results')])
  last_result_id = int(ids[-1]) if ids else 0
  _ExportOpenCLResults(cursor, last_result_id, proto_dir)
  logging.info("Exported OpenCL results")

  # Get the last exported testcase.
  (proto_dir / 'opencl' / 'testcases').mkdir(parents=True, exist_ok=True)
  ids = sorted([int(x) for x in os.listdir(proto_dir / 'opencl' / 'testcases')])
  last_testcase_id = int(ids[-1]) if ids else 0
  _ExportOpenCLTestcases(cursor, last_testcase_id, proto_dir)
  logging.info("Exported OpenCL testcases")

  cursor.close()
  cnx.close()

  cnx = MySQLdb.connect(
      database='dsmith_04_sol', user=credentials[0], password=credentials[1])
  cursor = cnx.cursor()

  # Get the last exported result.
  (proto_dir / 'sol' / 'results').mkdir(parents=True, exist_ok=True)
  ids = sorted([int(x) for x in os.listdir(proto_dir / 'sol' / 'results')])
  last_result_id = int(ids[-1]) if ids else 0
  _ExportSolidityResults(cursor, last_result_id, proto_dir)
  logging.info("Exported Solidity results")

  # Get the last exported testcase.
  (proto_dir / 'sol' / 'testcases').mkdir(parents=True, exist_ok=True)
  ids = sorted([int(x) for x in os.listdir(proto_dir / 'sol' / 'testcases')])
  last_testcase_id = int(ids[-1]) if ids else 0
  _ExportSolidityTestcases(cursor, last_testcase_id, proto_dir)
  logging.info("Exported Solidity testcases")

  cursor.close()
  cnx.close()


def main(argv):
  del argv
  _ExportProtos()


if __name__ == '__main__':
  app.run(main)
