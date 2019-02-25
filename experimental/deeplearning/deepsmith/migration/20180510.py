"""Export testcases and results for specific dsmith_04 programs to protos."""
import configparser
import os
import pathlib
import typing

import MySQLdb
from absl import app
from absl import flags
from absl import logging

from deeplearning.deepsmith.proto import deepsmith_pb2
from labm8 import fs
from labm8 import labdate
from labm8 import pbutil

FLAGS = flags.FLAGS

flags.DEFINE_string('proto_dir', None, 'Directory to export protos to.')
flags.DEFINE_list('program_ids', None, 'IDs of OpenCL programs to export.')

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
            "https://github.com/ChrisCummins/dsmith/blob"
            "/fd986a36a23b2a398f33d5b5852d930b462401b1/dsmith/opencl/generators.py#L175",
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
            "https://github.com/ChrisCummins/dsmith/blob"
            "/fd986a36a23b2a398f33d5b5852d930b462401b1/dsmith/opencl/harnesses.py#L292",
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


def _ExportOpenCLResults(cursor, program_id, proto_dir):
  cursor.execute(
      """
SELECT
  results.id,
  programs.id,
  testcases.id,
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
  threads.lsize_z,
  clsmith_testcase_metas.oclverified,
  dsmith_testcase_metas.gpuverified,
  dsmith_testcase_metas.oclverified,
  dsmith_program_metas.contains_floats,
  dsmith_program_metas.vector_inputs,
  dsmith_program_metas.compiler_warnings
FROM results
LEFT JOIN testbeds ON results.testbed_id = testbeds.id
LEFT JOIN platforms ON testbeds.platform_id = platforms.id
LEFT JOIN testcases ON results.testcase_id = testcases.id
LEFT JOIN programs ON testcases.program_id = programs.id
LEFT JOIN threads ON testcases.threads_id = threads.id
LEFT JOIN stdouts ON results.stdout_id = stdouts.id
LEFT JOIN stderrs ON results.stderr_id = stderrs.id
LEFT JOIN clsmith_testcase_metas ON testcases.id=clsmith_testcase_metas.id
LEFT JOIN dsmith_testcase_metas ON testcases.id=dsmith_testcase_metas.id
LEFT JOIN dsmith_program_metas ON programs.id=dsmith_program_metas.id
WHERE programs.id = %s AND platforms.platform <> 'clang'
""", (program_id,))

  i = 0
  for row in cursor:
    i += 1
    (result_id, programs_id, testcase_id, platform_name, device_name,
     driver_version, opencl_version, devtype, host_os, cl_opt, generator_id,
     program_date, program_generation_time, program_src, harness_id,
     harness_timeout, result_date, returncode, runtime, stdout, stderr,
     truncated_stderr, gsize_x, gsize_y, gsize_z, lsize_x, lsize_y, lsize_z,
     clsmith_oclverified, dsmith_gpuverified, dsmith_oclverified,
     dsmith_program_contains_floats, dsmith_program_vector_inputs,
     dsmith_program_compiler_warnings) = row
    inputs = {
        'src': program_src,
    }
    if harness_id != -1:
      inputs['gsize'] = f'{gsize_x},{gsize_y},{gsize_z}'
      inputs['lsize'] = f'{lsize_x},{lsize_y},{lsize_z}'
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
    invariant_opts = {}
    if clsmith_oclverified == 0:
      invariant_opts['oclverify'] = 'fail'
    elif clsmith_oclverified == 1:
      invariant_opts['oclverify'] = 'pass'
    elif dsmith_oclverified == 0:
      invariant_opts['oclverify'] = 'fail'
    elif dsmith_oclverified == 1:
      invariant_opts['oclverify'] = 'pass'
    if dsmith_gpuverified == 0:
      invariant_opts['gpuverify'] = 'fail'
    elif dsmith_gpuverified == 1:
      invariant_opts['gpuverify'] = 'pass'
    if dsmith_program_contains_floats == 0:
      invariant_opts['kernel_uses_floats'] = 'false'
    elif dsmith_program_contains_floats == 1:
      invariant_opts['kernel_uses_floats'] = 'true'
    if dsmith_program_vector_inputs == 0:
      invariant_opts['kernel_has_vector_inputs'] = 'false'
    elif dsmith_program_vector_inputs == 1:
      invariant_opts['kernel_has_vector_inputs'] = 'true'
    if dsmith_program_compiler_warnings == 0:
      invariant_opts['kernel_throws_compiler_warning'] = 'false'
    elif dsmith_program_compiler_warnings == 1:
      invariant_opts['kernel_throws_compiler_warning'] = 'true'
    testbed = deepsmith_pb2.Testbed(
        toolchain='opencl',
        name=testbed_name,
        opts=testbed_opts,
    )

    testcase = deepsmith_pb2.Testcase(
        toolchain="opencl",
        generator=_GetOpenCLGenerator(generator_id),
        harness=_GetOpenCLHarness(harness_id, harness_timeout),
        inputs=inputs,
        invariant_opts=invariant_opts,
        profiling_events=[
            deepsmith_pb2.ProfilingEvent(
                client="cc1",
                type="generation",
                duration_ms=int(program_generation_time * 1000),
                event_start_epoch_ms=labdate.MillisecondsTimestamp(
                    program_date),
            ),
        ])
    result = deepsmith_pb2.Result(
        testcase=testcase,
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
                duration_ms=int(runtime * 1000),
                event_start_epoch_ms=labdate.MillisecondsTimestamp(result_date),
            ),
        ],
    )
    # Write the testcase to file.
    outpath = proto_dir / 'testcases' / (str(testcase_id) + '.pbtxt')
    pbutil.ToFile(testcase, outpath)
    # Write the results to file.
    outpath = proto_dir / 'results' / (str(result_id) + '.pbtxt')
    pbutil.ToFile(result, outpath)


def _GetMySqlCredentials():
  cfg = configparser.ConfigParser()
  cfg.read(os.path.expanduser('~/.my.cnf'))
  return cfg['mysql']['user'], cfg['mysql']['password']


def _ExportProtos() -> None:
  proto_dir = pathlib.Path(FLAGS.proto_dir)

  assert proto_dir

  credentials = _GetMySqlCredentials()
  cnx = MySQLdb.connect(
      database='dsmith_04_opencl',
      host='cc1',
      user=credentials[0],
      password=credentials[1])
  cursor = cnx.cursor()

  (proto_dir / 'testcases').mkdir(parents=True, exist_ok=True)
  (proto_dir / 'results').mkdir(parents=True, exist_ok=True)
  for program_id in FLAGS.program_ids:
    logging.info("Exporting OpenCL program %s", program_id)
    _ExportOpenCLResults(cursor, program_id, proto_dir)

  cursor.close()
  cnx.close()

  logging.info('Exported %d testcases and %d results',
               len(fs.ls(proto_dir / 'testcases')),
               len(fs.ls(proto_dir / 'results')))


def main(argv):
  del argv
  _ExportProtos()


if __name__ == '__main__':
  app.run(main)
