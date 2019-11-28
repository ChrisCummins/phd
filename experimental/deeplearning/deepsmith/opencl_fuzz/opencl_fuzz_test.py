"""Unit tests for //experimental/deeplearning/deepsmith/opencl_fuzz/opencl_fuzz.py."""
import pathlib
import tempfile
import typing

import pytest

from deeplearning.deepsmith.difftests import difftests
from deeplearning.deepsmith.harnesses import cl_launcher
from deeplearning.deepsmith.harnesses import cldrive
from deeplearning.deepsmith.harnesses import harness
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import harness_pb2
from experimental.deeplearning.deepsmith.opencl_fuzz import opencl_fuzz
from gpu.cldrive.legacy import env
from labm8.py import app
from labm8.py import pbutil
from labm8.py import test

FLAGS = app.FLAGS

# Test fixtures.


@pytest.fixture(scope="function")
def cldrive_harness_config() -> harness_pb2.CldriveHarness:
  """Test fixture to return an Cldrive test harness config."""
  config = harness_pb2.CldriveHarness()
  config.opencl_env.extend([env.OclgrindOpenCLEnvironment().name])
  config.opencl_opt.extend([True])
  return config


@pytest.fixture(scope="function")
def cldrive_harness(
  cldrive_harness_config: harness_pb2.CldriveHarness,
) -> cldrive.CldriveHarness:
  """Test fixture to return an Cldrive test harness."""
  return cldrive.CldriveHarness(cldrive_harness_config)


@pytest.fixture(scope="function")
def cl_launcher_harness_config() -> harness_pb2.ClLauncherHarness:
  """Test fixture to return a cl_launcher test harness."""
  config = harness_pb2.ClLauncherHarness()
  config.opencl_env.extend([env.OclgrindOpenCLEnvironment().name])
  config.opencl_opt.extend([True])
  return config


@pytest.fixture(scope="function")
def cl_launcher_harness(
  cl_launcher_harness_config: harness_pb2.ClLauncherHarness,
) -> harness_pb2.ClLauncherHarness:
  """Test fixture to return a cl_launcher test harness."""
  return cldrive.CldriveHarness(cl_launcher_harness_config)


@pytest.fixture(scope="function")
def dummy_result() -> deepsmith_pb2.Result:
  """A test fixture which returns a dummy result."""
  return deepsmith_pb2.Result(
    testcase=deepsmith_pb2.Testcase(
      harness=deepsmith_pb2.Harness(name="name"),
      inputs={"src": "Kernel source.", "gsize": "1,1,1", "lsize": "2,2,2",},
    ),
    outputs={"stdout": "Standard output.", "stderr": "Standard error.",},
  )


@pytest.fixture(scope="function")
def clsmith_result(dummy_result: deepsmith_pb2.Result) -> pathlib.Path:
  """A test fixture which returns a dummy CLSmith result."""
  dummy_result.testcase.harness.name = "cl_launcher"
  with tempfile.TemporaryDirectory(prefix="phd_") as d:
    pbutil.ToFile(dummy_result, pathlib.Path(d) / "result.pbtxt")
    yield pathlib.Path(d) / "result.pbtxt"


# Mock classes.


class MockFilters(difftests.FiltersBase):
  """A mock class for simple filters."""

  def __init__(self, return_val: bool = True):
    super(MockFilters, self).__init__()
    self.return_val = return_val
    self.PreExec_call_args = []
    self.PostExec_call_args = []
    self.PreDifftest_call_args = []
    self.PostDifftest_call_args = []

  def PreExec(
    self, testcase: deepsmith_pb2.Testcase
  ) -> typing.Optional[deepsmith_pb2.Testcase]:
    self.PreExec_call_args.append(testcase)
    return testcase if self.return_val else None

  def PostExec(
    self, result: deepsmith_pb2.Result
  ) -> typing.Optional[deepsmith_pb2.Result]:
    self.PostExec_call_args.append(result)
    return result if self.return_val else None

  def PreDifftest(
    self, difftest: deepsmith_pb2.DifferentialTest
  ) -> typing.Optional[deepsmith_pb2.DifferentialTest]:
    self.PreDifftest_call_args.append(difftest)
    return difftest if self.return_val else None

  def PostDifftest(
    self, difftest: deepsmith_pb2.DifferentialTest
  ) -> typing.Optional[deepsmith_pb2.DifferentialTest]:
    self.PostDifftest_call_args.append(difftest)
    return difftest if self.return_val else None


class MockUnaryTester(difftests.UnaryTester):
  """A mock unary tester."""

  def __init__(self, return_val: typing.List[int] = None):
    super(MockUnaryTester, self).__init__()
    self.call_args = []
    self.return_val = return_val

  def __call__(
    self, results: typing.List[deepsmith_pb2.Result]
  ) -> typing.List[int]:
    self.call_args.append(results)
    return self.return_val


class MockGoldStandardDiffTester(difftests.GoldStandardDiffTester):
  """A mock gold standard difftester."""

  def __init__(self, return_val: typing.List[int] = None):
    super(MockGoldStandardDiffTester, self).__init__(
      difftests.OutputsEqualityTest()
    )
    self.call_args = []
    self.return_val = return_val

  def __call__(
    self, results: typing.List[deepsmith_pb2.Result]
  ) -> typing.List[int]:
    self.call_args.append(results)
    return self.return_val


class MockHarness(harness.HarnessBase):
  """A mock harness."""

  def __init__(self, return_val: harness_pb2.RunTestcasesResponse = None):
    super(MockHarness, self).__init__(None)
    self.return_val = return_val
    self.RunTestcases_call_requests = []

  def RunTestcases(
    self, request: harness_pb2.RunTestcasesRequest, context
  ) -> harness_pb2.RunTestcasesResponse:
    """Mock method which returns return_val."""
    del context
    self.RunTestcases_call_requests.append(request)
    return self.return_val


# RunTestcases() tests.


@pytest.mark.parametrize("opencl_opt", [True, False])
def test_RunTestcases_cldrive_pass(
  cldrive_harness_config: harness_pb2.CldriveHarness, opencl_opt: bool
):
  """Test execution of a simple test case."""
  cldrive_harness_config.opencl_opt[0] = opencl_opt
  harness = cldrive.CldriveHarness(cldrive_harness_config)
  testcases = [
    deepsmith_pb2.Testcase(
      toolchain="opencl",
      harness=deepsmith_pb2.Harness(name="cldrive"),
      inputs={
        "src": "kernel void A(global int* a) {a[get_global_id(0)] = 10;}",
        "gsize": "1,1,1",
        "lsize": "1,1,1",
        "timeout_seconds": "60",
      },
    )
  ]
  results = opencl_fuzz.RunTestcases(harness, testcases)
  assert len(results) == 1
  # Testcase.invariant_opts.driver_type field is set by cldrive harness.
  testcases[0].invariant_opts["driver_type"] = "compile_and_run"
  assert testcases[0] == results[0].testcase
  assert results[0].testbed == cldrive.OpenClEnvironmentToTestbed(
    harness.envs[0]
  )
  assert results[0].outcome == deepsmith_pb2.Result.PASS
  assert results[0].outputs["stdout"] == (
    "global int * a: 10 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 "
    "22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 "
    "46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 "
    "70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 "
    "94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 "
    "114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 "
    "132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 "
    "150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 "
    "168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 "
    "186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 "
    "204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 "
    "222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 "
    "240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255\n"
  )
  opt_str = "on" if opencl_opt else "off"
  assert (
    results[0].outputs["stderr"]
    == f"""\
[cldrive] Platform: Oclgrind
[cldrive] Device: Oclgrind Simulator
[cldrive] OpenCL optimizations: {opt_str}
[cldrive] Kernel: "A"
done.
"""
  )


@pytest.mark.parametrize("opencl_opt", [True, False])
def test_RunTestcases_cldrive_syntax_error(
  cldrive_harness_config: harness_pb2.CldriveHarness, opencl_opt: bool
):
  """Test execution of a test case with invalid syntax."""
  cldrive_harness_config.opencl_opt[0] = opencl_opt
  harness = cldrive.CldriveHarness(cldrive_harness_config)
  testcases = [
    deepsmith_pb2.Testcase(
      toolchain="opencl",
      harness=deepsmith_pb2.Harness(name="cldrive"),
      inputs={
        "src": "kernel void A(global int* a) {\n!11@invalid syntax!",
        "gsize": "1,1,1",
        "lsize": "1,1,1",
        "timeout_seconds": "60",
      },
    )
  ]
  results = opencl_fuzz.RunTestcases(harness, testcases)
  assert len(results) == 1
  # Testcase.invariant_opts.driver_type field is set by cldrive harness.
  testcases[0].invariant_opts["driver_type"] = "compile_only"
  assert testcases[0] == results[0].testcase
  assert results[0].testbed == cldrive.OpenClEnvironmentToTestbed(
    harness.envs[0]
  )
  assert results[0].outcome == deepsmith_pb2.Result.BUILD_FAILURE
  assert results[0].outputs["stdout"] == ""
  print(results[0].outputs["stderr"])
  opt_str = "on" if opencl_opt else "off"
  assert (
    results[0].outputs["stderr"]
    == f"""\
[cldrive] Platform: Oclgrind
[cldrive] Device: Oclgrind Simulator
[cldrive] OpenCL optimizations: {opt_str}
1 warning and 3 errors generated.
input.cl:1:34: error: expected ';' after expression
kernel void A(global int* a) {{!11@invalid syntax!
                                 ^
                                 ;
input.cl:1:34: error: expected expression
input.cl:1:50: error: expected '}}'
kernel void A(global int* a) {{!11@invalid syntax!
                                                 ^
input.cl:1:30: note: to match this '{{'
kernel void A(global int* a) {{!11@invalid syntax!
                             ^
input.cl:1:31: warning: expression result unused
kernel void A(global int* a) {{!11@invalid syntax!
                              ^~~
clBuildProgram CL_BUILD_PROGRAM_FAILURE
"""
  )


@pytest.mark.parametrize("opencl_opt", [True, False])
def test_RunTestcases_cl_launcher_pass(
  cl_launcher_harness_config: harness_pb2.ClLauncherHarness, opencl_opt: bool
):
  """Test execution of a simple test case."""
  cl_launcher_harness_config.opencl_opt[0] = opencl_opt
  harness = cl_launcher.ClLauncherHarness(cl_launcher_harness_config)
  testcases = [
    deepsmith_pb2.Testcase(
      toolchain="opencl",
      harness=deepsmith_pb2.Harness(name="cl_launcher"),
      inputs={
        "src": """\
// -g 1,1,1 -l 1,1,1
#define int64_t long
#define uint64_t ulong
#define int_least64_t long
#define uint_least64_t ulong
#define int_fast64_t long
#define uint_fast64_t ulong
#define intmax_t long
#define uintmax_t ulong
#define int32_t int
#define uint32_t uint
#define int16_t short
#define uint16_t ushort
#define int8_t char
#define uint8_t uchar

#define INT64_MIN LONG_MIN
#define INT64_MAX LONG_MAX
#define INT32_MIN INT_MIN
#define INT32_MAX INT_MAX
#define INT16_MIN SHRT_MIN
#define INT16_MAX SHRT_MAX
#define INT8_MIN CHAR_MIN
#define INT8_MAX CHAR_MAX
#define UINT64_MIN ULONG_MIN
#define UINT64_MAX ULONG_MAX
#define UINT32_MIN UINT_MIN
#define UINT32_MAX UINT_MAX
#define UINT16_MIN USHRT_MIN
#define UINT16_MAX USHRT_MAX
#define UINT8_MIN UCHAR_MIN
#define UINT8_MAX UCHAR_MAX

#define transparent_crc(X, Y, Z) transparent_crc_(&crc64_context, X, Y, Z)

#define VECTOR(X , Y) VECTOR_(X, Y)
#define VECTOR_(X, Y) X##Y

#ifndef NO_GROUP_DIVERGENCE
#define GROUP_DIVERGE(x, y) get_group_id(x)
#else
#define GROUP_DIVERGE(x, y) (y)
#endif

#ifndef NO_FAKE_DIVERGENCE
#define FAKE_DIVERGE(x, y, z) (x - y)
#else
#define FAKE_DIVERGE(x, y, z) (z)
#endif

#include "CLSmith.h"

__kernel void entry(__global ulong *result) {
    uint64_t crc64_context = 0xFFFFFFFFFFFFFFFFUL;
    result[get_linear_global_id()] = crc64_context ^ 0xFFFFFFFFFFFFFFFFUL;
}""",
        "gsize": "1,1,1",
        "lsize": "1,1,1",
        "timeout_seconds": "60",
      },
    )
  ]
  results = opencl_fuzz.RunTestcases(harness, testcases)
  assert len(results) == 1
  print(results[0].outputs["stderr"])
  assert testcases[0] == results[0].testcase
  assert results[0].testbed == cldrive.OpenClEnvironmentToTestbed(
    harness.envs[0]
  )
  assert results[0].outcome == deepsmith_pb2.Result.PASS
  assert results[0].outputs["stdout"] == "0,"
  opt_str = "on" if opencl_opt else "off"
  assert (
    results[0].outputs["stderr"]
    == f"""\
3-D global size 1 = [1, 1, 1]
3-D local size 1 = [1, 1, 1]
OpenCL optimizations: {opt_str}
Platform: Oclgrind
Device: Oclgrind Simulator
Compilation terminated successfully...
"""
  )


@pytest.mark.parametrize("opencl_opt", [True, False])
def test_RunTestcases_cl_launcher_syntax_error(
  cl_launcher_harness_config: harness_pb2.ClLauncherHarness, opencl_opt: bool
):
  """Test execution of a simple test case."""
  cl_launcher_harness_config.opencl_opt[0] = opencl_opt
  harness = cl_launcher.ClLauncherHarness(cl_launcher_harness_config)
  testcases = [
    deepsmith_pb2.Testcase(
      toolchain="opencl",
      harness=deepsmith_pb2.Harness(name="cl_launcher"),
      inputs={
        "src": "__kernel void entry(\n!11@invalid syntax!",
        "gsize": "1,1,1",
        "lsize": "1,1,1",
        "timeout_seconds": "60",
      },
    )
  ]
  results = opencl_fuzz.RunTestcases(harness, testcases)
  assert len(results) == 1
  print(results[0].outputs["stderr"])
  assert testcases[0] == results[0].testcase
  assert results[0].testbed == cldrive.OpenClEnvironmentToTestbed(
    harness.envs[0]
  )
  assert results[0].outcome == deepsmith_pb2.Result.BUILD_FAILURE
  assert results[0].outputs["stdout"] == ""
  opt_str = "on" if opencl_opt else "off"
  assert (
    results[0].outputs["stderr"]
    == f"""\
3-D global size 1 = [1, 1, 1]
3-D local size 1 = [1, 1, 1]
OpenCL optimizations: {opt_str}
Platform: Oclgrind
Device: Oclgrind Simulator
3 errors generated.
Error found (callback):

Oclgrind - OpenCL runtime error detected
\tFunction: clBuildProgram
\tError:    CL_BUILD_PROGRAM_FAILURE

Error building program: -11
input.cl:2:1: error: expected parameter declarator
!11@invalid syntax!
^
input.cl:2:1: error: expected ')'
input.cl:1:20: note: to match this '('
__kernel void entry(
                   ^
input.cl:2:20: error: expected function body after function declarator
!11@invalid syntax!
                   ^
"""
  )


# ResultIsInteresting() tests.


def test_ResultIsInteresting_unknown():
  """An unknown outcome is not interesting."""
  gs_harness = MockHarness()
  filters = MockFilters()
  result = opencl_fuzz.ResultIsInteresting(
    deepsmith_pb2.Result(outcome=deepsmith_pb2.Result.UNKNOWN),
    difftests.UnaryTester(),
    difftests.GoldStandardDiffTester(difftests.NamedOutputIsEqual("stdout")),
    gs_harness,
    filters,
  )
  assert not result
  # Only the unary tester was called, no differential test was required.
  assert not gs_harness.RunTestcases_call_requests
  assert len(filters.PreDifftest_call_args) == 0


def test_ResultIsInteresting_build_crash():
  """A build crash is interesting."""
  gs_harness = MockHarness()
  filters = MockFilters()
  result = opencl_fuzz.ResultIsInteresting(
    deepsmith_pb2.Result(outcome=deepsmith_pb2.Result.BUILD_CRASH),
    difftests.UnaryTester(),
    difftests.GoldStandardDiffTester(difftests.NamedOutputIsEqual("stdout")),
    gs_harness,
    filters,
  )
  assert result
  assert result.outputs["difftest_outcome"] == "ANOMALOUS_BUILD_FAILURE"
  # Only the unary tester was called, no differential test was required.
  assert not gs_harness.RunTestcases_call_requests
  assert len(filters.PreDifftest_call_args) == 0


def test_ResultIsInteresting_build_timeout():
  """A build timeout is interesting."""
  gs_harness = MockHarness()
  filters = MockFilters()
  result = opencl_fuzz.ResultIsInteresting(
    deepsmith_pb2.Result(outcome=deepsmith_pb2.Result.BUILD_TIMEOUT),
    difftests.UnaryTester(),
    difftests.GoldStandardDiffTester(difftests.NamedOutputIsEqual("stdout")),
    gs_harness,
    filters,
  )
  assert result
  assert result.outputs["difftest_outcome"] == "ANOMALOUS_BUILD_FAILURE"
  # Only the unary tester was called, no differential test was required.
  assert not gs_harness.RunTestcases_call_requests
  assert len(filters.PreDifftest_call_args) == 0


# UnpackResult() tests.


@pytest.fixture(scope="function")
def clgen_result(dummy_result: deepsmith_pb2.Result) -> pathlib.Path:
  """A test fixture which returns a dummy CLgen result."""
  dummy_result.testcase.harness.name = "cldrive"
  with tempfile.TemporaryDirectory(prefix="phd_") as d:
    pbutil.ToFile(dummy_result, pathlib.Path(d) / "result.pbtxt")
    yield pathlib.Path(d) / "result.pbtxt"


def test_UnpackResult_no_result():
  """An error raised if no result to unpack is provided."""
  with pytest.raises(app.UsageError):
    opencl_fuzz.UnpackResult(None)
  with pytest.raises(app.UsageError):
    opencl_fuzz.UnpackResult("")


def test_UnpackResult_clsmith_result(clsmith_result: deepsmith_pb2.Result):
  """Unpacking a CLSmith result creates expected files."""
  result_dir = clsmith_result.parent
  opencl_fuzz.UnpackResult(str(clsmith_result))
  assert (result_dir / "stdout.txt").is_file()
  assert (result_dir / "stderr.txt").is_file()
  assert (result_dir / "kernel.cl").is_file()
  assert (result_dir / "driver.c").is_file()


def test_UnpackResult_clgen_result(clgen_result: deepsmith_pb2.Result):
  """Unpacking a CLgen result creates expected files."""
  result_dir = clgen_result.parent
  opencl_fuzz.UnpackResult(str(clgen_result))
  assert (result_dir / "stdout.txt").is_file()
  assert (result_dir / "stderr.txt").is_file()
  assert (result_dir / "kernel.cl").is_file()
  assert (result_dir / "driver.c").is_file()


if __name__ == "__main__":
  test.Main()
