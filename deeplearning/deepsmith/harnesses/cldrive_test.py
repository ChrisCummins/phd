"""Unit tests for //deeplearning/deepsmith/services/cldrive.py."""
import pathlib
import subprocess
import tempfile

import pytest
from absl import flags

import gpu.cldrive.env
from deeplearning.deepsmith.harnesses import cldrive
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import harness_pb2
from deeplearning.deepsmith.proto import service_pb2
from gpu.oclgrind import oclgrind
from labm8 import test


FLAGS = flags.FLAGS


# Test fixtures.

@pytest.fixture(scope='function')
def abc_testcase() -> deepsmith_pb2.Testcase():
  """A test fixture which returns a very simple test case."""
  return deepsmith_pb2.Testcase(
      toolchain='opencl',
      harness=deepsmith_pb2.Harness(name='cldrive'),
      inputs={
        'src': 'kernel void A(global int* a) {a[get_global_id(0)] = 10;}',
        'gsize': '1,1,1',
        'lsize': '1,1,1',
      }
  )


@pytest.fixture(scope='function')
def abc_harness_config() -> harness_pb2.CldriveHarness:
  """A test fixture which returns an oclgrind harness config."""
  config = harness_pb2.CldriveHarness()
  config.opencl_env.extend([gpu.cldrive.env.OclgrindOpenCLEnvironment().name])
  config.opencl_opt.extend([True])
  return config


@pytest.fixture(scope='function')
def abc_harness(abc_harness_config) -> cldrive.CldriveHarness:
  """A test fixture which returns an oclgrind harness."""
  return cldrive.CldriveHarness(abc_harness_config)


@pytest.fixture(scope='function')
def abc_run_testcases_request(
    abc_testcase, abc_harness) -> harness_pb2.RunTestcasesRequest:
  """A test fixture which returns a RunTestcasesRequest for the abc_testcase."""
  return harness_pb2.RunTestcasesRequest(
      testbed=abc_harness.testbeds[0], testcases=[abc_testcase])


# Unit tests.


# CompileDriver() tests.


def test_CompileDriver_returned_path():
  """Test that output path is returned."""
  with tempfile.TemporaryDirectory() as d:
    p = cldrive.CompileDriver("int main() {}", pathlib.Path(d) / 'exe',
                              0, 0, timeout_seconds=60)
    assert p == pathlib.Path(d) / 'exe'


def test_CompileDriver_null_c():
  """Test compile a C program which does nothing."""
  with tempfile.TemporaryDirectory() as d:
    p = cldrive.CompileDriver("int main() {return 0;}", pathlib.Path(d) / 'exe',
                              0, 0, timeout_seconds=60)
    assert p.is_file()


def test_CompileDriver_hello_world_c():
  """Test compile a C program which prints "Hello, world!"."""
  with tempfile.TemporaryDirectory() as d:
    p = cldrive.CompileDriver("""
#include <stdio.h>

int main() {
  printf("Hello, world!\\n");
  return 0;
}
""", pathlib.Path(d) / 'exe', 0, 0, timeout_seconds=60)
    assert p.is_file()
    output = subprocess.check_output([p], universal_newlines=True)
    assert output == "Hello, world!\n"


def test_CompileDriver_opencl_header():
  """Test compile a C program which includes the OpenCL headers."""
  with tempfile.TemporaryDirectory() as d:
    p = cldrive.CompileDriver("""
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
int main() {}
""", pathlib.Path(d) / 'exe', 0, 0, timeout_seconds=60)
    assert p.is_file()


def test_CompileDriver_DriverCompilationError_syntax_error():
  """Test that DriverCompilationError is raised if code does not compile."""
  with tempfile.TemporaryDirectory() as d:
    with pytest.raises(cldrive.DriverCompilationError):
      cldrive.CompileDriver("ina39lid s#yntax!", pathlib.Path(d) / 'exe',
                            0, 0, timeout_seconds=60)
    assert not (pathlib.Path(d) / 'exe').is_file()


def test_CompileDriver_invalid_cflags():
  """Test that DriverCompilationError is raised if cflags are invalid."""
  with tempfile.TemporaryDirectory() as d:
    with pytest.raises(cldrive.DriverCompilationError):
      cldrive.CompileDriver('int main() {}', pathlib.Path(d) / 'exe',
                            0, 0, cflags=['--not_a_real_flag'])


def test_CompileDriver_valid_cflags():
  """Test that additional cflags are passed to build."""
  with tempfile.TemporaryDirectory() as d:
    cldrive.CompileDriver('MY_TYPE main() {}', pathlib.Path(d) / 'exe',
                          0, 0, cflags=['-DMY_TYPE=int'])
    assert (pathlib.Path(d) / 'exe').is_file()


# MakeDriver() tests.


def test_MakeDriver_ValueError_no_gsize():
  """Test that ValueError is raised if gsize input not set."""
  testcase = deepsmith_pb2.Testcase(inputs={
    'lsize': "1,1,1",
    'src': "kernel void A() {}"
  })
  with pytest.raises(ValueError) as e_ctx:
    cldrive.MakeDriver(testcase, True)
  assert "Field not set: 'Testcase.inputs[\"gsize\"]'" == str(e_ctx.value)


def test_MakeDriver_ValueError_no_lsize():
  """Test that ValueError is raised if lsize input not set."""
  testcase = deepsmith_pb2.Testcase(inputs={
    'gsize': "1,1,1",
    'src': "kernel void A() {}"
  })
  with pytest.raises(ValueError) as e_ctx:
    cldrive.MakeDriver(testcase, True)
  assert "Field not set: 'Testcase.inputs[\"lsize\"]'" == str(e_ctx.value)


def test_MakeDriver_ValueError_no_src():
  """Test that ValueError is raised if src input not set."""
  testcase = deepsmith_pb2.Testcase(inputs={
    'lsize': "1,1,1",
    'gsize': "1,1,1",
  })
  with pytest.raises(ValueError) as e_ctx:
    cldrive.MakeDriver(testcase, True)
  assert "Field not set: 'Testcase.inputs[\"src\"]'" == str(e_ctx.value)


def test_MakeDriver_ValueError_invalid_lsize():
  """Test that ValueError is raised if gsize is not an NDRange."""
  testcase = deepsmith_pb2.Testcase(inputs={
    'lsize': "abc",
    'gsize': "1,1,1",
    'src': 'kernel void A() {}'
  })
  with pytest.raises(ValueError) as e_ctx:
    cldrive.MakeDriver(testcase, True)
  assert "invalid literal for int() with base 10: 'abc'" == str(e_ctx.value)


def test_MakeDriver_ValueError_invalid_gsize():
  """Test that ValueError is raised if gsize is not an NDRange."""
  testcase = deepsmith_pb2.Testcase(inputs={
    'lsize': "1,1,1",
    'gsize': "abc",
    'src': 'kernel void A() {}'
  })
  with pytest.raises(ValueError) as e_ctx:
    cldrive.MakeDriver(testcase, True)
  assert "invalid literal for int() with base 10: 'abc'" == str(e_ctx.value)


def test_MakeDriver_CompileDriver_hello_world():
  """And end-to-end test."""
  testcase = deepsmith_pb2.Testcase(inputs={
    'lsize': '1,1,1',
    'gsize': '1,1,1',
    'src': 'kernel void A(global int* a) {a[get_global_id(0)] += 10;}'
  })
  driver = cldrive.MakeDriver(testcase, True)
  with tempfile.TemporaryDirectory() as d:
    binary = cldrive.CompileDriver(
        driver, pathlib.Path(d) / 'exe', 0, 0, timeout_seconds=60)
    proc = oclgrind.Exec([str(binary)])
  assert '[cldrive] Platform:' in proc.stderr
  assert '[cldrive] Device:' in proc.stderr
  assert '[cldrive] OpenCL optimizations: on\n' in proc.stderr
  assert '[cldrive] Kernel: "A"\n' in proc.stderr
  assert 'done.\n' in proc.stderr
  assert proc.stdout.split('\n')[-2] == (
    'global int * a: 10 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 '
    '22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 '
    '46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 '
    '70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 '
    '94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 '
    '114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 '
    '132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 '
    '150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 '
    '168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 '
    '186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 '
    '204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 '
    '222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 '
    '240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255'
  )


def test_MakeDriver_optimizations_on():
  """Test that OpenCL optimizations are enabled when requested."""
  testcase = deepsmith_pb2.Testcase(inputs={
    'lsize': "1,1,1",
    'gsize': "1,1,1",
    'src': 'kernel void A() {}'
  })
  src = cldrive.MakeDriver(testcase, True)
  assert '[cldrive] OpenCL optimizations: on' in src
  assert 'clBuildProgram(program, 0, NULL, NULL, NULL, NULL);' in src


def test_MakeDriver_optimizations_off():
  """Test that OpenCL optimizations are disabled when requested."""
  testcase = deepsmith_pb2.Testcase(inputs={
    'lsize': "1,1,1",
    'gsize': "1,1,1",
    'src': 'kernel void A() {}'
  })
  src = cldrive.MakeDriver(testcase, False)
  print(src)
  assert '[cldrive] OpenCL optimizations: off' in src
  assert (
      'clBuildProgram(program, 0, NULL, "-cl-opt-disable", NULL, NULL);' in src)


# CldriveHarness() tests.

def test_CldriveHarness_oclgrind_testbed_uneven_name_and_opt():
  """Error is raised if number of opt_opt != number of opencl_env."""
  oclgrind_env_name = gpu.cldrive.env.OclgrindOpenCLEnvironment().name

  config = harness_pb2.CldriveHarness()
  config.opencl_env.extend([oclgrind_env_name, oclgrind_env_name])
  config.opencl_opt.extend([True])

  with pytest.raises(ValueError) as e_ctx:
    cldrive.CldriveHarness(config, default_to_all_environments=False)
  assert ('CldriveHarness.opencl_env and CldriveHarness.opencl_opt lists are '
          'not the same length') in str(e_ctx.value)


def test_CldriveHarness_oclgrind_testbed_count_one():
  """Test that correct number of testbeds are instantiated."""
  oclgrind_env_name = gpu.cldrive.env.OclgrindOpenCLEnvironment().name

  config = harness_pb2.CldriveHarness()
  config.opencl_env.extend([oclgrind_env_name])
  config.opencl_opt.extend([True])

  harness = cldrive.CldriveHarness(config, default_to_all_environments=False)
  assert len(harness.testbeds) == 1


def test_CldriveHarness_oclgrind_testbed_count_two():
  """Test that correct number of testbeds are instantiated."""
  oclgrind_env_name = gpu.cldrive.env.OclgrindOpenCLEnvironment().name

  config = harness_pb2.CldriveHarness()
  config.opencl_env.extend([oclgrind_env_name, oclgrind_env_name])
  config.opencl_opt.extend([True, False])

  harness = cldrive.CldriveHarness(config)
  assert len(harness.testbeds) == 2


def test_CldriveHarness_oclgrind_testbed_names():
  """Test that correct names set on testbeds."""
  oclgrind_env_name = gpu.cldrive.env.OclgrindOpenCLEnvironment().name

  config = harness_pb2.CldriveHarness()
  config.opencl_env.extend([oclgrind_env_name, oclgrind_env_name])
  config.opencl_opt.extend([True, False])

  harness = cldrive.CldriveHarness(config)
  assert harness.testbeds[0].name == oclgrind_env_name
  assert harness.testbeds[1].name == oclgrind_env_name


def test_CldriveHarness_oclgrind_testbed_opts():
  """Test that opencl_opt option set on testbeds."""
  oclgrind_env_name = gpu.cldrive.env.OclgrindOpenCLEnvironment().name

  config = harness_pb2.CldriveHarness()
  config.opencl_env.extend([oclgrind_env_name, oclgrind_env_name])
  config.opencl_opt.extend([True, False])

  harness = cldrive.CldriveHarness(config)
  assert harness.testbeds[0].opts['opencl_opt'] == 'enabled'
  assert harness.testbeds[1].opts['opencl_opt'] == 'disabled'


def test_CldriveHarness_RunTestcases_no_testbed():
  """Test that invalid request params returned if no testbed requested."""
  config = harness_pb2.CldriveHarness()
  harness = cldrive.CldriveHarness(config)
  req = harness_pb2.RunTestcasesRequest(testbed=None, testcases=[])
  res = harness.RunTestcases(req, None)
  assert (res.status.returncode ==
          service_pb2.ServiceStatus.INVALID_REQUEST_PARAMETERS)
  assert res.status.error_message == 'Requested testbed not found.'


def test_CldriveHarness_RunTestcases_no_testcases():
  """Test that empty results returned if no testcase requested."""
  config = harness_pb2.CldriveHarness()
  harness = cldrive.CldriveHarness(config)
  assert len(harness.testbeds)
  req = harness_pb2.RunTestcasesRequest(
      testbed=harness.testbeds[0], testcases=[])
  res = harness.RunTestcases(req, None)
  assert res.status.returncode == service_pb2.ServiceStatus.SUCCESS
  assert not res.results


def test_CldriveHarness_RunTestcases_oclgrind_abc_testcase(
    abc_harness, abc_run_testcases_request):
  """And end-to-end test of the abc_testcase on oclgrind."""
  res = abc_harness.RunTestcases(abc_run_testcases_request, None)
  assert res.status.returncode == service_pb2.ServiceStatus.SUCCESS
  assert len(res.results) == 1

  # Check that the driver_type invariant opt has been added to the testcase.
  result = res.results[0]
  assert len(result.testcase.invariant_opts) == 1
  assert result.testcase.invariant_opts['driver_type'] == 'compile_and_run'

  # The returned testcase is identical to the input testcase.
  assert result.testcase == abc_run_testcases_request.testcases[0]

  # Check the result properties.
  assert result.outcome == deepsmith_pb2.Result.PASS
  assert '[cldrive] Platform: Oclgrind' in result.outputs['stderr']
  assert '[cldrive] Device: Oclgrind Simulator' in result.outputs['stderr']
  assert '[cldrive] OpenCL optimizations: on' in result.outputs['stderr']
  assert '[cldrive] Kernel: "A"' in result.outputs['stderr']
  assert result.outputs['stdout'] == (
    'global int * a: 10 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 '
    '22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 '
    '46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 '
    '70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 '
    '94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 '
    '114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 '
    '132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 '
    '150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 '
    '168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 '
    '186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 '
    '204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 '
    '222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 '
    '240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255\n')


def test_CldriveHarness_RunTestcases_driver_cflags(
    abc_harness_config, abc_run_testcases_request):
  """Test that valid driver cflags do not break the build."""
  abc_harness_config.driver_cflag.extend(['-O3', '-g'])
  harness = cldrive.CldriveHarness(abc_harness_config)
  res = harness.RunTestcases(abc_run_testcases_request, None)
  assert res.status.returncode == service_pb2.ServiceStatus.SUCCESS
  assert len(res.results) == 1
  result = res.results[0]
  # Nothing interesting to see here.
  assert result.outcome == deepsmith_pb2.Result.PASS


def test_CldriveHarness_RunTestcases_invalid_driver_cflags(
    abc_harness_config, abc_run_testcases_request):
  """Test that invalid driver cflags cause driver to fail to build."""
  abc_harness_config.driver_cflag.extend(['--not_a_real_flag'])
  harness = cldrive.CldriveHarness(abc_harness_config)
  res = harness.RunTestcases(abc_run_testcases_request, None)
  assert res.status.returncode == service_pb2.ServiceStatus.SUCCESS
  assert len(res.results) == 1
  result = res.results[0]
  # A driver compilation error is an unknown outcome.
  assert result.outcome == deepsmith_pb2.Result.UNKNOWN


if __name__ == '__main__':
  test.Main()
