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
"""Integration tests for HealthKit import to me_db."""

from datasets.me_db import me_db
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS

MODULE_UNDER_TEST = 'datasets.me_db'

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
