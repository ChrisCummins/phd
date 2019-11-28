# Copyright 2017, 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Unit tests for //research/grewe_2013_cgo:feature_extractor."""
import pathlib

import pytest

from labm8.py import app
from labm8.py import test
from research.grewe_2013_cgo import feature_extractor

FLAGS = app.FLAGS


def test_ExtractFeaturesFromPath_file_not_found(tempdir: pathlib.Path):
  """Error raised when file doesn't exist."""
  with test.Raises(FileNotFoundError):
    feature_extractor.ExtractFeaturesFromPath(tempdir / "notafile")


def test_ExtractFeaturesFromPath_empty_file(tempdir: pathlib.Path):
  """No features returned for an empty file."""
  (tempdir / "empty.cl").touch()
  features = list(
    feature_extractor.ExtractFeaturesFromPath(tempdir / "empty.cl")
  )

  assert not features


def test_ExtractFeaturesFromPath_single_kernel_features_count(
  tempdir: pathlib.Path,
):
  """A single features tuple is returned for a single kernel."""
  with open(tempdir / "kernel.cl", "w") as f:
    f.write(
      """
kernel void foobar(global int* a, const int nelem) {
    int id = get_global_id(0);
    if (id < nelem)
        a[id] = nelem;
}
"""
    )
  features = list(
    feature_extractor.ExtractFeaturesFromPath(tempdir / "kernel.cl")
  )
  assert len(features) == 1


def test_ExtractFeaturesFromPath_integer_index_accces(tempdir: pathlib.Path):
  """An integer-indexed access to global memory is considered coaslesced."""
  with open(tempdir / "foo.cl", "w") as f:
    f.write(
      """
kernel void A(global int* a) {
  a[0] = 0;
}
"""
    )
  features = list(
    feature_extractor.ExtractFeaturesFromPath(tempdir / "foo.cl")
  )[0]
  assert features.coalesced_memory_access_count == 1


def test_ExtractFeaturesFromPath_function_accces(tempdir: pathlib.Path):
  """An function-indexed access to memory is not considered coaslesced.

  NOTE: This is not always correct!
  """
  with open(tempdir / "foo.cl", "w") as f:
    f.write(
      """
kernel void A(global int* a) {
  a[get_global_id(0)] = 0;
}
"""
    )
  features = list(
    feature_extractor.ExtractFeaturesFromPath(tempdir / "foo.cl")
  )[0]
  assert features.coalesced_memory_access_count == 0


def test_ExtractFeaturesFromPath_pointer_offset(tempdir: pathlib.Path):
  """A pointer-arithmetic access to memory is not considered coaslesced.

  NOTE: This is not always correct!
  """
  with open(tempdir / "foo.cl", "w") as f:
    f.write(
      """
kernel void A(global int* a) {
  *(a + 0) = 0;
}
"""
    )
  features = list(
    feature_extractor.ExtractFeaturesFromPath(tempdir / "foo.cl")
  )[0]
  assert features.coalesced_memory_access_count == 0


def test_ExtractFeaturesFromPath_single_kernel_features_values(
  tempdir: pathlib.Path,
):
  """Test feature values returned from a single kernel."""
  with open(tempdir / "kernel.cl", "w") as f:
    f.write(
      """
kernel void foobar(global int* a, const int nelem) {
    int id = get_global_id(0);
    if (id < nelem)
        a[id] = nelem;
}
"""
    )
  features = list(
    feature_extractor.ExtractFeaturesFromPath(tempdir / "kernel.cl")
  )
  assert features[0] == ("kernel.cl", "foobar", 0, 1, 1, 0, 1, 0, 1, 0)


def test_ExtractFeaturesFromPath_two_kernels_features_count(
  tempdir: pathlib.Path,
):
  """Two features tuples are returned from two kernels."""
  with open(tempdir / "kernels.cl", "w") as f:
    f.write(
      """
kernel void A(global int* a) {
  a[get_global_id(0)] = 0;
}

kernel void B(global int* a, const int b) {
  for (int i = 0; i < b; ++i) {
    a[get_global_id(0)] += i;
  }
}
"""
    )
  features = list(
    feature_extractor.ExtractFeaturesFromPath(tempdir / "kernels.cl")
  )
  assert len(features) == 2


def test_ExtractFeaturesFromPath_two_kernels_features_count(
  tempdir: pathlib.Path,
):
  """Test feature values returned from two kernels."""
  with open(tempdir / "kernels.cl", "w") as f:
    f.write(
      """
kernel void A(global int* a) {
  a[get_global_id(0)] = 0;
}

kernel void B(global int* a, const int b) {
  for (int i = 0; i < b; ++i) {
    a[get_global_id(0)] += i;
  }
}
"""
    )
  features = list(
    feature_extractor.ExtractFeaturesFromPath(tempdir / "kernels.cl")
  )

  assert features[0] == ("kernels.cl", "A", 0, 0, 1, 0, 0, 0, 0, 0)
  assert features[1] == ("kernels.cl", "B", 0, 1, 1, 0, 0, 0, 0, 0)


def test_ExtractFeaturesFromPath_syntax_error(tempdir: pathlib.Path):
  """Error is raised if compilation fails."""
  with open(tempdir / "kernel.cl", "w") as f:
    f.write("/@*! Invalid syntax!")
  with test.Raises(feature_extractor.FeatureExtractionError) as e_ctx:
    feature_extractor.ExtractFeaturesFromPath(tempdir / "kernel.cl")

  assert "error: expected identifier or '('" in str(e_ctx.value)


if __name__ == "__main__":
  test.Main()
