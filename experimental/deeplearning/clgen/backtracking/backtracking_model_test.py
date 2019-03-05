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


@pytest.fixture(scope='function')
def backtracker(atomizer: atomizers.AtomizerBase
               ) -> backtracking_model.OpenClBacktrackingHelper:
  return backtracking_model.OpenClBacktrackingHelper(
      atomizer, target_features=None)


def test_OpenClBacktrackingHelper_ShouldCheckpoint_no(
    backtracker: backtracking_model.OpenClBacktrackingHelper):
  assert not backtracker.ShouldCheckpoint('int x = 5')


def test_OpenClBacktrackingHelper_ShouldCheckpoint_yes(
    backtracker: backtracking_model.OpenClBacktrackingHelper):
  assert backtracker.ShouldCheckpoint('int x = 5;')


def test_OpenClBacktrackingHelper_ShouldCheckpoint_for_loop_1(
    backtracker: backtracking_model.OpenClBacktrackingHelper):
  assert backtracker.ShouldCheckpoint('for (int i = 0;')


def test_OpenClBacktrackingHelper_ShouldCheckpoint_for_loop_2(
    backtracker: backtracking_model.OpenClBacktrackingHelper):
  assert backtracker.ShouldCheckpoint('for (int i = 0; i < 10;')


def test_OpenClBacktrackingHelper_ShouldCheckpoint_for_loop_3(
    backtracker: backtracking_model.OpenClBacktrackingHelper):
  assert backtracker.ShouldCheckpoint('for (int i = 0; i < 10; ++i) { int x;')


def test_OpenClBacktrackingHelper_TryToCloseProgram_depth1(
    backtracker: backtracking_model.OpenClBacktrackingHelper):
  assert backtracker.TryToCloseProgram("""kernel void A() {
  int a = 0;""") == """kernel void A() {
  int a = 0;}"""


def test_OpenClBacktrackingHelper_TryToCloseProgram_depth2(
    backtracker: backtracking_model.OpenClBacktrackingHelper):
  assert backtracker.TryToCloseProgram("""kernel void A() {
  int a = 0;
  if (global_global(0) < 10) {
    int a = 2;""") == """kernel void A() {
  int a = 0;
  if (global_global(0) < 10) {
    int a = 2;}}"""


def test_OpenClBacktrackingHelper_TryToCloseProgram_loop1(
    backtracker: backtracking_model.OpenClBacktrackingHelper):
  assert backtracker.TryToCloseProgram("""kernel void A() {
  for (int a = 0;""") == """kernel void A() {
  for (int a = 0;;){}}"""


def test_OpenClBacktrackingHelper_TryToCloseProgram_loop2(
    backtracker: backtracking_model.OpenClBacktrackingHelper):
  assert backtracker.TryToCloseProgram("""kernel void A() {
  for (int a = 0; a < 10;""") == """kernel void A() {
  for (int a = 0; a < 10;){}}"""


def test_OpenClBacktrackingHelper_TryToCloseProgram_loop3(
    backtracker: backtracking_model.OpenClBacktrackingHelper):
  assert backtracker.TryToCloseProgram("""kernel void A() {
  for (int a = 0; a < 10; ++a) {
    { int x = 10;""") == """kernel void A() {
  for (int a = 0; a < 10; ++a) {
    { int x = 10;}}}"""


def test_OpenClBacktrackingHelper_TryToCloseProgram_not_end_of_statement(
    backtracker: backtracking_model.OpenClBacktrackingHelper):
  with pytest.raises(AssertionError):
    backtracker.TryToCloseProgram("kernel void A(".split())


def test_OpenClBacktrackingHelper_ShouldProceed_depth1(
    backtracker: backtracking_model.OpenClBacktrackingHelper):
  assert backtracker.ShouldProceed("""kernel void A() {
  int a = 0;""")


def test_OpenClBacktrackingHelper_ShouldProceed_depth2(
    backtracker: backtracking_model.OpenClBacktrackingHelper):
  assert backtracker.ShouldProceed("""kernel void A() {
  int a = 0;
  if (global_global(0) < 10) {
    int a = 2;""")


def test_OpenClBacktrackingHelper_ShouldProceed_loop1(
    backtracker: backtracking_model.OpenClBacktrackingHelper):
  assert backtracker.ShouldProceed("""kernel void A() {
  for (int i = 0;""")


def test_OpenClBacktrackingHelper_ShouldProceed_loop2(
    backtracker: backtracking_model.OpenClBacktrackingHelper):
  assert backtracker.ShouldProceed("""kernel void A() {
  for (int i = 0; i < 10;""")


def test_OpenClBacktrackingHelper_ShouldProceed_loop3(
    backtracker: backtracking_model.OpenClBacktrackingHelper):
  assert backtracker.ShouldProceed("""kernel void A() {
  for (int i = 0; i < 10; ++i) { int x = 10;""")


if __name__ == '__main__':
  test.Main()
