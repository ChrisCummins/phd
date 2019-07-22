"""Unit tests for //deeplearning/clgen:samples_database."""
import pathlib
import pytest

from deeplearning.clgen import samples_database
from deeplearning.clgen.proto import model_pb2
from labm8 import test

FLAGS = test.FLAGS


@pytest.fixture(scope='function')
def db(tempdir: pathlib.Path) -> samples_database.SamplesDatabase:
  yield samples_database.SamplesDatabase(f'sqlite:///{tempdir}/db')


def test_SamplesDatabaseObserver_add_one(db: samples_database.SamplesDatabase):
  sample_proto = model_pb2.Sample(
      text='Hello, observer',
      num_tokens=10,
      wall_time_ms=5,
      sample_start_epoch_ms_utc=1000,
  )

  with db.Observer() as obs:
    obs.OnSample(sample_proto)

  with db.Session() as s:
    assert s.query(samples_database.Sample).count() == 1
    assert s.query(samples_database.Sample).one().ToProto() == sample_proto


if __name__ == '__main__':
  test.Main()
