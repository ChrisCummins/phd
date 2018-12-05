"""Unit tests for //datasets/me_db/life_cycle."""

import pathlib
import pytest
import sys
import tempfile
import time
import typing
from absl import app
from absl import flags
from concurrent import futures

from datasets.me_db.life_cycle import life_cycle
from datasets.me_db.life_cycle import make_dataset


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def temp_dir() -> pathlib.Path:
  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    yield pathlib.Path(d)


@pytest.fixture(scope='function')
def temp_dataset(temp_dir: pathlib.Path) -> pathlib.Path:
  generator = make_dataset.RandomDatasetGenerator(time.mktime(
      time.strptime('1/1/2018', '%m/%d/%Y')), locations=[
    'My House',
    'The Office',
  ], names=[
    'Work',
    'Home',
  ])
  generator.SampleZip(temp_dir / 'LC_export.zip', 100)
  yield temp_dir


def test_ProcessDirectory(temp_dataset: pathlib.Path):
  """Test generated series."""
  protos = list(life_cycle.ProcessDirectory(temp_dataset))
  assert len(protos) == 1
  series_proto = protos[0]
  series = list(series_proto.series)
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


def test_ProcessDirectory_future(temp_dataset: pathlib.Path):
  """Test that ProcessDirectory() can be called async."""
  with futures.ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(life_cycle.ProcessDirectory, temp_dataset)
  protos = list(future.result())
  assert len(protos) == 1
  assert len(protos[0].series) == 2


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
