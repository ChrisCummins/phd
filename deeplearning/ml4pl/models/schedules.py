"""This module defines helper enums for controlling runtime behaviour."""
import enum

from labm8.py import app


FLAGS = app.FLAGS


class KeepCheckpoints(enum.Enum):
  """Schedules for keeping checkpoint logs."""

  # Keep no checkpoints.
  NONE = 0
  # Keep all checkpoints.
  ALL = 1
  # Keep the checkpoint for the most recent epoch.
  LAST_EPOCH = 2


class KeepDetailedBatches(enum.Enum):
  """Schedules for keeping detailed batch logs."""

  # Keep no detailed batches.
  NONE = 0
  # Keep all detailed batches.
  ALL = 1
  # Keep detailed batches only for the last epoch.
  LAST_EPOCH = 2
