"""Unit tests for //alice/ledger."""
import pathlib

import pytest

from alice.ledger import ledger
from labm8 import test


def test_TODO(tempdir: pathlib.Path):
  """Short summary of test."""
  leg = ledger.LedgerService(f'sqlite:///{tempdir}/ledger.db')

  with pytest.raises(ValueError) as e_ctx:
    leg.SelectWorkerIdForRunRequest(None)
  assert str(e_ctx.value) == 'No worker bees available'


if __name__ == '__main__':
  test.Main()
