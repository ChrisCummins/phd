"""Unit tests for //deeplearning/deepsmith/services/cldrive.py."""
import pathlib
import pytest
import subprocess
import sys
import tempfile
from absl import app
from absl import flags
from absl import logging


FLAGS = flags.FLAGS

from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.services import cldrive


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


def test_CompileDriver_DriverCompilationError_syntax_error():
  """Test that DriverCompilationError is raised if code does not compile."""
  with tempfile.TemporaryDirectory() as d:
    with pytest.raises(cldrive.DriverCompilationError):
      cldrive.CompileDriver("ina39lid s#yntax!", pathlib.Path(d) / 'exe',
                            0, 0, timeout_seconds=60)
    assert not (pathlib.Path(d) / 'exe').is_file()


def test_MakeDriver_ValueError_no_gsize():
  """Test that ValueError is raised if gsize input not set."""
  testcase = deepsmith_pb2.Testcase(inputs={
    'lsize': "1,1,1",
    'src': "kernel void A() {}"
  })
  with pytest.raises(ValueError) as e_ctx:
    cldrive.MakeDriver(testcase)
  assert "Field not set: 'Testcase.inputs[\"gsize\"]'" == str(e_ctx.value)


def test_MakeDriver_ValueError_no_lsize():
  """Test that ValueError is raised if lsize input not set."""
  testcase = deepsmith_pb2.Testcase(inputs={
    'gsize': "1,1,1",
    'src': "kernel void A() {}"
  })
  with pytest.raises(ValueError) as e_ctx:
    cldrive.MakeDriver(testcase)
  assert "Field not set: 'Testcase.inputs[\"lsize\"]'" == str(e_ctx.value)


def test_MakeDriver_ValueError_no_src():
  """Test that ValueError is raised if src input not set."""
  testcase = deepsmith_pb2.Testcase(inputs={
    'lsize': "1,1,1",
    'gsize': "1,1,1",
  })
  with pytest.raises(ValueError) as e_ctx:
    cldrive.MakeDriver(testcase)
  assert "Field not set: 'Testcase.inputs[\"src\"]'" == str(e_ctx.value)


def test_MakeDriver_ValueError_invalid_lsize():
  """Test that ValueError is raised if gsize is not an NDRange."""
  testcase = deepsmith_pb2.Testcase(inputs={
    'lsize': "abc",
    'gsize': "1,1,1",
    'src': 'kernel void A() {}'
  })
  with pytest.raises(ValueError) as e_ctx:
    cldrive.MakeDriver(testcase)
  assert "invalid literal for int() with base 10: 'abc'" == str(e_ctx.value)


def test_MakeDriver_ValueError_invalid_gsize():
  """Test that ValueError is raised if gsize is not an NDRange."""
  testcase = deepsmith_pb2.Testcase(inputs={
    'lsize': "1,1,1",
    'gsize': "abc",
    'src': 'kernel void A() {}'
  })
  with pytest.raises(ValueError) as e_ctx:
    cldrive.MakeDriver(testcase)
  assert "invalid literal for int() with base 10: 'abc'" == str(e_ctx.value)


def test_MakeDriver_CompileDriver_hello_world():
  """And end-to-end test."""
  testcase = deepsmith_pb2.Testcase(inputs={
    'lsize': '1,1,1',
    'gsize': '1,1,1',
    'src': 'kernel void A(global int* a) {a[get_global_id(0)] += 10;}'
  })
  driver = cldrive.MakeDriver(testcase)
  with tempfile.TemporaryDirectory() as d:
    binary = cldrive.CompileDriver(
        driver, pathlib.Path(d) / 'exe', 0, 0, timeout_seconds=60)
    output = subprocess.check_output([binary], universal_newlines=True,
                                     stderr=subprocess.STDOUT)
  assert '[cldrive] Platform:' in output
  assert '[cldrive] Device:' in output
  assert '[cldrive] OpenCL optimizations: on\n' in output
  assert '[cldrive] Kernel: "A"\n' in output
  assert 'done.\n' in output
  assert output.split('\n')[-2] == (
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


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    logging.warning("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
