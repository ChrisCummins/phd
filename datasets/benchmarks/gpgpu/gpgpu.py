"""A collection of seven GPGPU benchmark suites.

These seven benchmark suites were responsible for 92% of GPU results published
in 25 top tier conference papers. For more details, see:

  ﻿Cummins, C., Petoumenos, P., Zang, W., & Leather, H. (2017). Synthesizing
  Benchmarks for Predictive Modeling. In CGO. IEEE.
"""
import contextlib
import functools
import multiprocessing
import os
import pathlib
import subprocess
import tempfile
import typing

from absl import flags
from absl import logging

from gpu.oclgrind import oclgrind
from labm8 import bazelutil
from labm8 import fs
from labm8 import labdate
from labm8 import system


FLAGS = flags.FLAGS

# The list of all GPGPU benchmark suites.
_BENCHMARK_SUITE_NAMES = [
  'amd-app-sdk-3.0',
  'npb-3.3',
  'nvidia-4.2',
  'parboil-0.2',
  'polybench-gpu-1.0',
  'rodinia-3.1',
  'shoc-1.1.5',
  'dummy_just_for_testing',
]

flags.DEFINE_list('gpgpu_benchmark_suites', _BENCHMARK_SUITE_NAMES,
                  'The names of benchmark suites to run. Defaults to all '
                  'benchmark suites.')
flags.DEFINE_list('gpgpu_device_types', ['oclgrind'],
                  'The device types to execute benchmark suites on. One or '
                  'more of {cpu,gpu,oclgrind}.')
flags.DEFINE_string('gpgpu_outdir', '/tmp/datasets/benchmarks/gpgpu',
                    'The directory to write log files to.')

# The path of libcecl directory, containing the libcecl header, library, and
# run script.
_LIBCECL = bazelutil.DataPath('phd/gpu/libcecl/libcecl.so')
_LIBCECL_HEADER = bazelutil.DataPath('phd/gpu/libcecl/libcecl.h')

# Path to OpenCL headers and library.
_OPENCL_HEADERS_DIR = bazelutil.DataPath('opencl_120_headers')
if system.is_linux():
  _LIBOPENCL_DIR = bazelutil.DataPath('libopencl')

_DUMMY_BENCHMARK = bazelutil.DataPath(
    'phd/datasets/benchmarks/gpgpu/dummy_just_for_testing/dummy_benchmark')


def CheckCall(command: typing.Union[str, typing.List[str]],
              shell: bool = False, env: typing.Dict[str, str] = None):
  """Wrapper around subprocess.check_call() to log executed commands."""
  if shell:
    logging.debug('$ %s', command)
    subprocess.check_call(command, shell=True, env=env)
  else:
    command = [str(x) for x in command]
    logging.debug('$ %s', ' '.join(command))
    subprocess.check_call(command, env=env)


def RewriteClDeviceType(device_type: str, path: pathlib.Path):
  """Rewrite all instances of CL_DEVICE_TYPE_XXX in the given path."""
  cl_device_type = {
    'cpu': 'CL_DEVICE_TYPE_CPU',
    'gpu': 'CL_DEVICE_TYPE_GPU',
    'oclgrind': 'CL_DEVICE_TYPE_CPU',
  }[device_type]
  CheckCall(f"""\
for f in $(find '{path}' -type f); do
  grep CL_DEVICE_TYPE_ $f &>/dev/null && {{
    sed -E -i 's/CL_DEVICE_TYPE_[A-Z]+/{cl_device_type}/g' $f
    echo Set {cl_device_type} in $f
  }}
done""", shell=True)


@functools.lru_cache(maxsize=1)
def OpenClCompileAndLinkFlags() -> typing.Tuple[str, str]:
  """Get device-specific OpenCL compile and link flags."""
  if system.is_linux():
    return (f'-isystem {_OPENCL_HEADERS_DIR}',
            f'-L{_LIBOPENCL_DIR} -Wl,-rpath,{_LIBOPENCL_DIR} -lOpenCL')
  else:
    return f'-isystem {_OPENCL_HEADERS_DIR}', '-framework OpenCL'


@contextlib.contextmanager
def MakeEnv(path: pathlib.Path) -> typing.Dict[str, str]:
  """Return a build environment for GPGPU benchmarks."""
  cflags, ldflags = OpenClCompileAndLinkFlags()

  with fs.chdir(path):
    with tempfile.TemporaryDirectory(prefix='phd_gpu_libcecl_header_') as d:
      fs.cp(_LIBCECL_HEADER, pathlib.Path(d) / 'cecl.h')

      env = os.environ.copy()
      env['CFLAGS'] = f'-isystem {d} {cflags}'
      env['LDFLAGS'] = f'-lcecl -L{_LIBCECL.parent} {ldflags}'
      yield env


def Make(target: str, path: pathlib.Path) -> None:
  """Run make target in the given path."""
  # Build relative to the path, rather than using `make -c <path>`. This is
  # because some of the source codes have hard-coded relative paths.
  with MakeEnv(path) as env:
      CheckCall(['make', '-j', multiprocessing.cpu_count(), target], env=env)


def FindExecutableInDir(path: pathlib.Path) -> pathlib.Path:
  """Find an executable file in a directory."""
  exes = [f for f in path.iterdir() if f.is_file() and os.access(f, os.X_OK)]
  if len(exes) != 1:
    raise EnvironmentError(f"Expected a single executable, found {len(exes)}")
  return exes[0]


@contextlib.contextmanager
def RunEnv(path: pathlib.Path) -> typing.Dict[str, str]:
  """Return an execution environment for a GPGPU benchmark."""
  with fs.chdir(path):
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = _LIBCECL.parent
    env['DYLD_LIBRARY_PATH'] = _LIBCECL.parent
    yield env


def RunCeclToLogFile(executable: pathlib.Path, logdir: pathlib.Path,
                     device_type: str, benchmark_name: str):
  """Run executable using runcecl script and log output."""
  logging.info('Executing benchmark %s', executable.name)
  logdir.mkdir(exist_ok=True, parents=True)

  # Create the name of the logfile now, so that is timestamped to the start of
  # execution.
  log_name = '.'.join([
    benchmark_name,
    device_type,
    system.HOSTNAME,
    str(labdate.MillisecondsTimestamp()),
    'txt'
  ])

  with RunEnv(executable.parent) as env:
    if device_type == 'oclgrind':
      command = [str(oclgrind.OCLGRIND_PATH), executable]
    else:
      command = [executable]
    process = subprocess.Popen(command, env=env, stderr=subprocess.PIPE,
                               universal_newlines=True)
    _, stderr = process.communicate()

    if process.returncode:
      raise OSError(f'Process failed with returncode {process.returncode} and '
                    f'stderr: `{stderr}`')

    with open(logdir / log_name, 'w') as f:
      for line in stderr.split('\n'):
        if line.startswith('[CECL] '):
          print(line[len('[CECL] '):], file=f)


class _BenchmarkSuite(object):
  """Abstract base class for a GPGPU benchmark suite.

  A benchmark suite provides two methods: ForceDeviceType(), which forces all
  of the benchmarks within the suite to execute on a given device type (CPU or
  GPU), and Run(), which executes the benchmarks and logs output to a directory.
  Example usage:

    with SomeBenchmarkSuite() as bs:
      bs.ForceDeviceType('gpu')
      bs.Run('/tmp/logs/gpu')
      bs.ForceDeviceType('cpu')
      bs.Run('/tmp/logs/cpu/1')
      bs.Run('/tmp/logs/cpu/2')
  """

  def __init__(self, name: str):
    if name not in _BENCHMARK_SUITE_NAMES:
      raise ValueError(f"Unknown benchmark suite: {name}")

    self._name = name
    self._device_type = None
    self._input_files = bazelutil.DataPath(
        f'phd/datasets/benchmarks/gpgpu/{name}')
    self._mutable_location = None

  def __enter__(self) -> pathlib.Path:
    prefix = f'phd_datasets_benchmarks_gpgpu_{self._name}'
    self._mutable_location = pathlib.Path(tempfile.mkdtemp(prefix=prefix))
    fs.cp(self._input_files, self._mutable_location)
    return self

  def __exit__(self, *args):
    fs.rm(self._mutable_location)
    self._mutable_location = None

  @property
  def path(self):
    """Return the path of the mutable copy of the benchmark sources."""
    if self._mutable_location is None:
      raise TypeError("Must be used as a context manager")
    return self._mutable_location

  def ForceDeviceType(self, device_type: str) -> None:
    """Force benchmarks to execute with the given device type."""
    if device_type not in {'cpu', 'gpu', 'oclgrind'}:
      raise ValueError(f"Unknown device type: {device_type}")
    self._device_type = device_type
    return self._ForceDeviceType(device_type)

  @property
  def device_type(self) -> str:
    return self._device_type

  def Run(self, logdir: pathlib.Path) -> None:
    """Run benchmarks and log results to directory."""
    logdir.mkdir(parents=True, exist_ok=True)
    if self.device_type is None:
      raise TypeError("Must call ForceDeviceType() before Run()")
    return self._Run(logdir)

  # Abstract attributes that must be provided by subclasses.

  @property
  def benchmarks(self) -> typing.List[str]:
    """Return a list of all benchmark names."""
    raise NotImplementedError("abstract property")

  def _ForceDeviceType(self, device_type: str) -> None:
    """Set the given device type."""
    raise NotImplementedError("abstract method")

  def _Run(self, logdir: pathlib.Path) -> None:
    """Run the benchmarks and produce output log files."""
    raise NotImplementedError("abstract method")


class DummyJustForTesting(_BenchmarkSuite):
  """A dummy benchmark suite for testing purposes.

  It sill behaves like a real benchmark suite, but without running any expensive
  binaries.
  """

  @property
  def name(self) -> str:
    return "dummy_just_for_testing"

  @property
  def benchmarks(self) -> typing.List[str]:
    return ["dummy_benchmark"]

  def _ForceDeviceType(self, device_type: str):
    logging.info("Dummy benchmarks switching to %s", device_type)

  def _Run(self):
    logging.info("Executing dummy benchmarks!")
    self._ExecToLogFile(_DUMMY_BENCHMARK, 'dummy_benchmark')


class AmdAppSdkBenchmarkSuite(_BenchmarkSuite):
  """The AMD App SDK benchmarks."""

  def __init__(self):
    super(AmdAppSdkBenchmarkSuite, self).__init__('amd-app-sdk-3.0')


class PolybenchGpuBenchmarkSuite(_BenchmarkSuite):

  def __init__(self):
    super(PolybenchGpuBenchmarkSuite, self).__init__('polybench-gpu-1.0')

  @property
  def benchmarks(self) -> typing.List[str]:
    return [
      '2DCONV',
      '2MM',
      '3DCONV',
      '3MM',
      'ATAX',
      'BICG',
      'CORR',
      'COVAR',
      # Bad: 'FDTD-2D',
      'GEMM',
      'GESUMMV',
      'GRAMSCHM',
      'MVT',
      'SYR2K',
      'SYRK',
    ]

  def _ForceDeviceType(self, device_type: str):
    RewriteClDeviceType(device_type, self.path / 'OpenCL')
    for benchmark in self.benchmarks:
      logging.info('Building benchmark %s', benchmark)
      Make('clean', self.path / 'OpenCL' / benchmark)
      Make('all', self.path / 'OpenCL' / benchmark)

  def _Run(self, logdir: pathlib.Path):
    for benchmark in self.benchmarks:
      executable = FindExecutableInDir(self.path / 'OpenCL' / benchmark)
      RunCeclToLogFile(executable, logdir, self.device_type, benchmark)


# A map of benchmark suite names to classes.
BENCHMARK_SUITES = {
  bs().name: bs for bs in labtypes.AllSubclassesOfClass(_BenchmarkSuite)
}


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  # Run the requested benchmark suites on the requested devices.
  outdir = pathlib.Path(FLAGS.gpgpu_outdir)
  for benchmark_suite_name in FLAGS.gpgpu_benchmark_suites:
    benchmark_suite_class = BENCHMARK_SUITES.get(benchmark_suite_name)
    if not benchmark_suite_class:
      logging.fatal(
          f'Unknown benchmark suite. Legal values: {BENCHMARK_SUITES.keys()}')

    with benchmark_suite_class() as benchmark_suite:
      for device_type in FLAGS.gpgpu_device_types:
        logging.info('Building and running %s on %s', benchmark_suite.name,
                     device_type)
        benchmark_suite.ForceDeviceType(device_type)
        benchmark_suite.Run(outdir)


if __name__ == '__main__':
  app.run(main)
