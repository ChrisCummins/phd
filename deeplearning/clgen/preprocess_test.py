#
# Copyright 2016, 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
import os
import sys

import pytest
from absl import app

from deeplearning.clgen import errors
from deeplearning.clgen import preprocess
from deeplearning.clgen.tests import testlib as tests


# Invoke tests with UPDATE_GS_FILES set to update the gold standard
# tests. E.g.:
#
#   $ UPDATE_GS_FILES=1 python3 ./setup.py test
#
UPDATE_GS_FILES = True if 'UPDATE_GS_FILES' in os.environ else False


def preprocess_pair(basename, preprocessor=preprocess.preprocess):
  gs_path = tests.data_path(os.path.join('cl', str(basename) + '.gs'),
                            exists=not UPDATE_GS_FILES)
  tin_path = tests.data_path(os.path.join('cl', str(basename) + '.cl'))

  # Run preprocess
  tin = tests.data_str(tin_path)
  tout = preprocessor(tin)

  if UPDATE_GS_FILES:
    gs = tout
    with open(gs_path, 'w') as outfile:
      outfile.write(gs)
      print("\n-> updated gold standard file '{}' ...".format(gs_path),
            file=sys.stderr, end=' ')
  else:
    gs = tests.data_str(gs_path)

  return (gs, tout)


def test_preprocess():
  assert len(set(preprocess_pair('sample-1'))) == 1


def test_preprocess_shim():
  # FLOAT_T is defined in shim header
  assert preprocess.preprocess("""
__kernel void A(__global FLOAT_T* a) { int b; }""", use_shim=True)

  # Preprocess will fail without FLOAT_T defined
  with pytest.raises(errors.BadCodeException):
    preprocess.preprocess("""
__kernel void A(__global FLOAT_T* a) { int b; }""", use_shim=False)


def test_ugly_preprocessed():
  # empty kernel protoype is rejected
  with pytest.raises(errors.NoCodeException):
    preprocess.preprocess("""\
__kernel void A() {
}\
""")
  # kernel containing some code returns the same.
  assert """\
__kernel void A() {
  int a;
}\
""" == preprocess.preprocess("""\
__kernel void A() {
  int a;
}\
""")


def test_preprocess_stable():
  code = """\
__kernel void A(__global float* a) {
  int b;
  float c;
  int d = get_global_id(0);

  a[d] *= 2.0f;
}"""
  # pre-processing is "stable" if the code doesn't change
  out = code
  for _ in range(5):
    out = preprocess.preprocess(out)
    assert out == code


@tests.needs_linux  # FIXME: GPUVerify support on macOS.
def test_gpuverify():
  code = """\
__kernel void A(__global float* a) {
  int b = get_global_id(0);
  a[b] *= 2.0f;
}"""
  assert preprocess.gpuverify(code,
                              ["--local_size=64", "--num_groups=128"]) == code


@tests.needs_linux  # FIXME: GPUVerify support on macOS.
def test_gpuverify_data_race():
  code = """\
__kernel void A(__global float* a) {
  a[0] +=  1.0f;
}"""
  with pytest.raises(preprocess.GPUVerifyException):
    preprocess.gpuverify(code, ["--local_size=64", "--num_groups=128"])


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
