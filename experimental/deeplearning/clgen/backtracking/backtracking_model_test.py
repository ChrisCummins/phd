"""Unit tests for //experimental/deeplearning/clgen/backtracking:backtracking_model."""
import pytest
from absl import flags

from deeplearning.clgen.corpuses import atomizers
from experimental.deeplearning.clgen.backtracking import backtracking_model
from labm8 import test

FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def atomizer() -> atomizers.AtomizerBase:
  return atomizers.AsciiCharacterAtomizer.FromText("""
kernel void A(global int* a, global int* b, const int c) {
  int d = c;
  if (get_global_id(0) < c) {
    a[get_global_id(0)] = b[get_global_id(0)];
  }
}
""")


def test_Backtracker_TryToCloseProgram_depth1(atomizer: atomizers.AtomizerBase):
  """Depth one test case."""
  backtracker = backtracking_model.Backtracker(atomizer)
  assert backtracker.TryToCloseProgram("""kernel void A() {
  int a = 0;""") == """kernel void A() {
  int a = 0;}"""


def test_Backtracker_TryToCloseProgram_depth2(atomizer: atomizers.AtomizerBase):
  """Depth two test case."""
  backtracker = backtracking_model.Backtracker(atomizer)
  assert backtracker.TryToCloseProgram("""kernel void A() {
  int a = 0;
  if (global_global(0) < 10) {
    int a = 2;""") == """kernel void A() {
  int a = 0;
  if (global_global(0) < 10) {
    int a = 2;}}"""


def test_Backtracker_TryToCloseProgram_invalid(
    atomizer: atomizers.AtomizerBase):
  """For loop cannot be closed, but it is anyway."""
  # TODO(cec): Can this be fixed?
  backtracker = backtracking_model.Backtracker(atomizer)
  assert backtracker.TryToCloseProgram("""kernel void A() {
  for (int a = 0;""") == """kernel void A() {
  for (int a = 0;}"""


def test_Backtracker_TryToCloseProgram_not_end_of_statement(
    atomizer: atomizers.AtomizerBase):
  """Must only be called at end of statement."""
  backtracker = backtracking_model.Backtracker(atomizer)
  with pytest.raises(AssertionError):
    backtracker.TryToCloseProgram("kernel void A(".split())


def test_Backtracker_ShouldProceed_depth1(atomizer: atomizers.AtomizerBase):
  """Depth one test case."""
  backtracker = backtracking_model.Backtracker(atomizer)
  assert backtracker.ShouldProceed("""kernel void A() {
  int a = 0;""")


def test_Backtracker_ShouldProceed_depth2(atomizer: atomizers.AtomizerBase):
  """Depth two test case."""
  backtracker = backtracking_model.Backtracker(atomizer)
  assert backtracker.ShouldProceed("""kernel void A() {
  int a = 0;
  if (global_global(0) < 10) {
    int a = 2;""")


def test_Backtracker_ShouldProceed_invalid(atomizer: atomizers.AtomizerBase):
  """For loop closes to an invalid program."""
  # TODO(cec): Can this be fixed?
  backtracker = backtracking_model.Backtracker(atomizer)
  assert not backtracker.ShouldProceed("""kernel void A() {
  for (int a = 0;""")


if __name__ == '__main__':
  test.Main()
