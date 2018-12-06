"""Integration tests for HealthKit import to me_db."""
import sys
import typing

import pytest
from absl import app
from absl import flags

from datasets.me_db import me_db


FLAGS = flags.FLAGS


# These tests have been hardcoded to the values in the test_inbox dataset.

def test_measurements_count(db: me_db.Database):
  """Test the number of measurements."""
  with db.Session() as s:
    q = s.query(me_db.Measurement) \
      .filter(me_db.Measurement.source.like('HealthKit:%'))

    assert q.count() == 58


def test_measurements_by_source_count(db: me_db.Database):
  """Test the number of measurements."""
  # Query:
  #   SELECT source,count(*)
  #   FROM measurements
  #   WHERE source like 'HealthKit:%'
  #   GROUP BY source;
  with db.Session() as s:
    assert s.query(me_db.Measurement) \
             .filter(me_db.Measurement.source == 'HealthKit:AppleWatch') \
             .count() == 35

    assert s.query(me_db.Measurement) \
             .filter(me_db.Measurement.source == 'HealthKit:Health') \
             .count() == 5

    assert s.query(me_db.Measurement) \
             .filter(me_db.Measurement.source == 'HealthKit:IPhone') \
             .count() == 6

    assert s.query(me_db.Measurement) \
             .filter(me_db.Measurement.source == 'HealthKit:MiFit') \
             .count() == 4

    assert s.query(me_db.Measurement) \
             .filter(me_db.Measurement.source == 'HealthKit:Workflow') \
             .count() == 5

    assert s.query(me_db.Measurement) \
             .filter(me_db.Measurement.source == 'HealthKit:Health') \
             .count() == 5


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
