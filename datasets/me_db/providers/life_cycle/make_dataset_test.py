# Copyright 2018-2020 Chris Cummins <chrisc.101@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Unit tests for //datasets/me_db/providers/life_cycle:make_dataset."""
import csv
import pathlib
import tempfile
import time
import zipfile

import pytest

from datasets.me_db.providers.life_cycle import make_dataset
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@test.Fixture(scope="function")
def temp_dir() -> pathlib.Path:
  """A test fixture to produce a temporary directory."""
  with tempfile.TemporaryDirectory(prefix="phd_") as d:
    yield pathlib.Path(d)


def test_RandomDatasetGenerator_SampleFile(temp_dir: pathlib.Path):
  """Test SampleFile data."""
  generator = make_dataset.RandomDatasetGenerator(
    start_time_seconds_since_epoch=time.mktime(
      time.strptime("1/1/2018", "%m/%d/%Y")
    ),
    locations=["My House", "The Office", "A Restaurant",],
    names=[
      "Work",
      "Home",
      "Sleep",
      "Fun",
      "Commute to work",
      "Commute to home",
    ],
  )

  generator.SampleFile(temp_dir / "LC_export.csv", 100)

  with open(temp_dir / "LC_export.csv") as f:
    reader = csv.reader(f)
    rows = [row for row in reader]

  # One line for the header.
  assert len(rows) == 102

  # All lines except the second have eight columns.
  assert len(rows[0]) == 8
  for row in rows[2:]:
    assert len(row) == 8


def test_RandomDatasetGenerator_SampleZip(temp_dir: pathlib.Path):
  """Test SampleZip data generated."""
  generator = make_dataset.RandomDatasetGenerator(
    start_time_seconds_since_epoch=time.mktime(
      time.strptime("1/1/2018", "%m/%d/%Y")
    ),
    locations=["My House", "The Office", "A Restaurant",],
    names=[
      "Work",
      "Home",
      "Sleep",
      "Fun",
      "Commute to work",
      "Commute to home",
    ],
  )

  generator.SampleZip(temp_dir / "LC_export.zip", 100)

  with zipfile.ZipFile(temp_dir / "LC_export.zip") as z:
    with z.open("LC_export.csv") as f:
      # Read and decode the compressed CSV into a string.
      string = f.read().decode("utf-8")
      reader = csv.reader(string.split("\n"))
      rows = [row for row in reader]

  # One line for the header.
  assert len(rows) == 103

  # All lines except the second and last have eight columns.
  assert len(rows[0]) == 8
  for row in rows[2:-1]:
    assert len(row) == 8


if __name__ == "__main__":
  test.Main()
