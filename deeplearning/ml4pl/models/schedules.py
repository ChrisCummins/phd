"""This module defines helper enums for controlling runtime behaviour."""
import enum

from labm8.py import app


FLAGS = app.FLAGS


class SaveOn(enum.Enum):
  """Determine when to save checkpoints."""

  # Never save a checkpoint.
  NONE = 0
  # Make a checkpoint at every epoch.
  EVERY_EPOCH = 1
  # Make a checkpoint when validation accuracy improves.
  VAL_IMPROVED = 2


class KeepCheckpoints(enum.Enum):
  """Determine how many checkpoints to keep."""

  # Keep all checkpoints.
  ALL = 1
  # Keep only the last saved checkpoint.
  LAST = 2


class KeepDetailedBatches(enum.Enum):
  """Determine how many detailed batches to keep."""

  # Keep no detailed batches.
  NONE = 0
  # Keep all detailed batches.
  ALL = 1
  # Keep detailed batches only for the last epoch.
  LAST_EPOCH = 2


class TestOn(enum.Enum):
  """Determine when to run a model on the test set."""

  # Never run on the test set.
  NONE = 0
  # Test at the end of every epoch.
  EVERY = 1
  # Test on validation improvement.
  IMPROVEMENT = 2
