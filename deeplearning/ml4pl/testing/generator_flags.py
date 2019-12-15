"""Flags for random data generators."""
from labm8.py import app


FLAGS = app.FLAGS


app.DEFINE_integer(
  "split_count",
  10,
  "The number of splits for random graphs. If 0, no splits are assigned.",
)
