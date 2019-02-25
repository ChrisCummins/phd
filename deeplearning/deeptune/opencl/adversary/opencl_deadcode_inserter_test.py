"""Unit tests for :opencl_deadcode_inserter."""

import numpy as np
import pytest
from absl import flags

from deeplearning.deeptune.opencl.adversary import \
  opencl_deadcode_inserter as dci
from labm8 import test

FLAGS = flags.FLAGS


def test_SetFunctionName_name_not_changed():
  f = dci.OpenClFunction("kernel void A() {}", is_kernel=True)
  f.SetFunctionName("A")
  assert f.src == "kernel void A() {}"


def test_SetFunctionName_name_changed():
  f = dci.OpenClFunction("kernel void A() {}", is_kernel=True)
  f.SetFunctionName("B")
  assert f.src == "kernel void B() {}"


def test_SetFunctionName_multichar_name():
  f = dci.OpenClFunction("kernel void ABCD() {}", is_kernel=True)
  f.SetFunctionName("A")
  assert f.src == "kernel void A() {}"


def test_SetFunctionName_invalid_kernel():
  """Test that error is raised if input is not kernel."""
  with pytest.raises(ValueError):
    # Leading underscores breaks regex match.
    dci.OpenClFunction("__kernel void A() {}").SetFunctionName("B")


def test_KernelToFunctionDeclaration_invalid_kernel():
  """Test that error is raised if input is not kernel."""
  with pytest.raises(ValueError):
    dci.KernelToFunctionDeclaration('Hello, world!')


def test_KernelToFunctionDeclaration_no_args():
  assert dci.KernelToFunctionDeclaration(
      "kernel void B() {}").src == "void B();"


def test_KernelToFunctionDeclaration_args():
  assert (dci.KernelToFunctionDeclaration(
      "kernel void A(global int * a, const int b) {}").src ==
          "void A(global int * a, const int b);")


def test_KernelToDeadCodeBlock_invalid_kernel():
  """Test that error is raised if input is not kernel."""
  with pytest.raises(ValueError):
    dci.KernelToDeadCodeBlock("Hello, world!")


def test_KernelToDeadCodeBlock():
  assert dci.KernelToDeadCodeBlock("""\
kernel void A(global int * a, const int b) {
  if (get_global_id(0) < b) {
    a[get_global_id(0)] *= 2;
  }
}""") == """\
if (0) {
  int * a;
  int b;
  if (get_global_id(0) < b) {
    a[get_global_id(0)] *= 2;
  }
}"""


def test_OpenClDeadcodeInserter_PrependFunctionDefinition():
  """Short summary of test."""
  inserter = dci.OpenClDeadcodeInserter(
      np.random.RandomState(0), "kernel void A(const int b) {}",
      ["kernel void B(global int * a) {}"])
  inserter.PrependUnusedFunctionDeclaration()
  assert inserter.opencl_source == """\
void A(global int * a);

kernel void B(const int b) {}"""


def test_OpenClDeadcodeInserter_AppendFunctionDefinition():
  """Short summary of test."""
  inserter = dci.OpenClDeadcodeInserter(
      np.random.RandomState(0), "kernel void A(const int b) {}",
      ["kernel void B(global int * a) {}"])
  inserter.AppendUnusedFunctionDeclaration()
  assert inserter.opencl_source == """\
kernel void A(const int b) {}

void B(global int * a);"""


if __name__ == '__main__':
  test.Main()
