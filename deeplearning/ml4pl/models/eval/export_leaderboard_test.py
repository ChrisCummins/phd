"""Unit tests for //deeplearning/ml4pl/models/eval:export_leaderboard."""
from labm8.py import test

FLAGS = test.FLAGS

from deeplearning.ml4pl.models.eval import google_sheets

MODULE_UNDER_TEST = None

requires_google_sheets_credentials_file = test.SkipIf(
  not FLAGS.google_sheets_credentials.is_file(),
  reason="google sheets credentials not found",
)


def test_TODO():
  del google_sheets


if __name__ == "__main__":
  test.Main()
