"""Unit tests for //datasets/me_db/ynab."""
import pathlib
import pytest
import sys
import tempfile
import time
import typing
from absl import app
from absl import flags

from datasets.me_db.ynab import make_dataset
from datasets.me_db.ynab import ynab


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def temp_dir() -> pathlib.Path:
  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    yield pathlib.Path(d)


def test_ProcessDirectory(temp_dir: pathlib.Path):
  """Test values on fake data."""
  _ = ynab
  generator = make_dataset.RandomDatasetGenerator(
      start_date_seconds_since_epoch=time.mktime(
          time.strptime('1/1/2018', '%m/%d/%Y')),
      categories={
        'Rainy Day': ['Savings', 'Pension'],
        'Everyday Expenses': ['Groceries', 'Clothes'],
      })
  dir = (temp_dir / 'Personal Finances~B0DA25C7.ynab4' / 'data1~8E111055' /
         '12345D63-B6C2-CD11-6666-C7D8733E20AB')
  dir.mkdir(parents=True)
  generator.SampleFile(dir / 'Budget.yfull', 100)

  # One SeriesCollection is generated for each input file.
  series_from_collections = list(ynab.ProcessDirectory(temp_dir))
  assert len(series_from_collections) == 1

  # Two Series are produce for each file.
  series_collection = series_from_collections[0]
  assert len(series_collection.series) == 2

  transactions_series, budget_series = series_collection.series

  # The series name is a CamelCase version of 'Personal Finances' with suffix.
  assert transactions_series.name == 'PersonalFinancesTransactions'
  assert budget_series.name == 'PersonalFinancesBudget'

  assert transactions_series.unit == 'pound_sterling_pence'
  assert budget_series.unit == 'pound_sterling_pence'
  assert transactions_series.family == 'Finances'
  assert budget_series.family == 'Finances'

  # num measurements = num transactions.
  assert len(transactions_series.measurement) == 100
  # num measurements = num categories * num months.
  assert len(budget_series.measurement) == 4 * 12

  for measurement in transactions_series.measurement:
    assert measurement.source == 'YNAB'
    assert measurement.group

  for measurement in budget_series.measurement:
    assert measurement.source == 'YNAB'
    assert measurement.group


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
