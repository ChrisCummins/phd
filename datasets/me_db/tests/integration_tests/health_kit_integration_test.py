"""Integration tests for HealthKit import to me_db."""

from absl import flags

from datasets.me_db import me_db
from labm8 import test

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
             .filter(me_db.Measurement.source == 'HealthKit:Iphone') \
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


if __name__ == '__main__':
  test.Main()
