"""Unit tests for //datasets/me_db/providers/life_cycle."""

import pathlib
import tempfile
import time
from concurrent import futures

import pytest
from absl import flags

from datasets.me_db.providers.life_cycle import life_cycle
from datasets.me_db.providers.life_cycle import make_dataset
from labm8 import test

FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def temp_dir() -> pathlib.Path:
  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    yield pathlib.Path(d)


@pytest.fixture(scope='function')
def temp_inbox(temp_dir: pathlib.Path) -> pathlib.Path:
  (temp_dir / 'life_cycle').mkdir()
  generator = make_dataset.RandomDatasetGenerator(
      time.mktime(time.strptime('1/1/2018', '%m/%d/%Y')),
      locations=[
          'My House',
          'The Office',
      ],
      names=[
          'Work',
          'Home',
      ])
  generator.SampleZip(temp_dir / 'life_cycle' / 'LC_export.zip', 100)
  yield temp_dir


def ProcessInbox(temp_inbox: pathlib.Path):
  """Test generated series."""
  series_collection = life_cycle.ProcessInbox(temp_inbox)
  series = list(series_collection.series)
  assert len(series) == 2
  series = sorted(series, key=lambda s: s.name)
  assert series[0].name == 'HomeTime'
  assert series[0].unit == 'milliseconds'
  assert len(series[0].measurement) + len(series[1].measurement) == 100

  for measurement in series[0].measurement:
    assert measurement.group
    assert measurement.source == 'LifeCycle'

  assert series[1].name == 'WorkTime'
  assert series[1].unit == 'milliseconds'
  for measurement in series[1].measurement:
    assert measurement.group
    assert measurement.source == 'LifeCycle'


def test_ProcessInbox_future(temp_inbox: pathlib.Path):
  """Test that ProcessInbox() can be called async."""
  with futures.ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(life_cycle.ProcessInbox, temp_inbox)
  series_collection = future.result()
  assert len(series_collection.series) == 2


if __name__ == '__main__':
  test.Main()
