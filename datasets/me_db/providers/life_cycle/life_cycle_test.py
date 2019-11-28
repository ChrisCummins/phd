# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>
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
"""Unit tests for //datasets/me_db/providers/life_cycle."""
import pathlib
import tempfile
import time
from concurrent import futures

import pytest

from datasets.me_db.providers.life_cycle import life_cycle
from datasets.me_db.providers.life_cycle import make_dataset
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@pytest.fixture(scope="function")
def temp_dir() -> pathlib.Path:
  with tempfile.TemporaryDirectory(prefix="phd_") as d:
    yield pathlib.Path(d)


@pytest.fixture(scope="function")
def temp_inbox(temp_dir: pathlib.Path) -> pathlib.Path:
  (temp_dir / "life_cycle").mkdir()
  generator = make_dataset.RandomDatasetGenerator(
    time.mktime(time.strptime("1/1/2018", "%m/%d/%Y")),
    locations=["My House", "The Office",],
    names=["Work", "Home",],
  )
  generator.SampleZip(temp_dir / "life_cycle" / "LC_export.zip", 100)
  yield temp_dir


def ProcessInbox(temp_inbox: pathlib.Path):
  """Test generated series."""
  series_collection = life_cycle.ProcessInbox(temp_inbox)
  series = list(series_collection.series)
  assert len(series) == 2
  series = sorted(series, key=lambda s: s.name)
  assert series[0].name == "HomeTime"
  assert series[0].unit == "milliseconds"
  assert len(series[0].measurement) + len(series[1].measurement) == 100

  for measurement in series[0].measurement:
    assert measurement.group
    assert measurement.source == "LifeCycle"

  assert series[1].name == "WorkTime"
  assert series[1].unit == "milliseconds"
  for measurement in series[1].measurement:
    assert measurement.group
    assert measurement.source == "LifeCycle"


def test_ProcessInbox_future(temp_inbox: pathlib.Path):
  """Test that ProcessInbox() can be called async."""
  with futures.ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(life_cycle.ProcessInbox, temp_inbox)
  series_collection = future.result()
  assert len(series_collection.series) == 2


if __name__ == "__main__":
  test.Main()
