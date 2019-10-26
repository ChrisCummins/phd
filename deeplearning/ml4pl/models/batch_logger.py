"""This module defines a logger for batch stats."""
import time
import typing

from labm8 import app

FLAGS = app.FLAGS


class InMemoryBatchLogger(object):
  """A named object that stores batch stats and maintains rolling averages."""

  def __init__(self, name: str):
    """Constructor."""
    self.name = name
    self.start_time = time.time()
    self.end_time = None  # Set by StopTheClock().
    self.loss = 0
    self.losses = []
    self.accuracy = 0
    self.accuracies = []
    self.batch_count = 0
    self.instance_count = 0

  def Log(self, batch_size: int, loss: float, accuracy: float) -> str:
    """Log data from a batch.

    Args:
      batch_size: The number of instances in the batch.
      loss: The batch loss.
      accuracy: The batch accuracy.

    Returns:
      String representation.
    """
    self.loss += loss
    self.losses.append(float(loss))
    self.accuracies.append(float(accuracy))
    self.accuracy += accuracy
    self.batch_count += 1
    self.instance_count += batch_size
    return (f"{self.name}, batch {self.batch_count}. "
            f"loss: {loss:.4f} | "
            f"acc: {accuracy:.2%} | "
            f"instances/sec: {self.instances_per_second:.2f}")

  def __repr__(self) -> str:
    """Stringify the aggregate batch stats."""
    return (f"{self.name}. "
            f"Average batch_size={self.average_batch_size:.2f} | "
            f"loss={self.average_loss:.4f} | "
            f"acc={self.average_accuracy:.2%} | "
            f"instances/sec={self.instances_per_second:.2f}")

  def StopTheClock(self):
    """End the batch timer."""
    self.end_time = time.time()

  def ToJson(self) -> typing.Dict[str, float]:
    """JSON-ify the batch aggregates."""
    return {
        "instance_count": self.instance_count,
        "batch_count": self.batch_count,
        "loss": self.average_loss,
        "losses": self.losses,
        "accuracy": self.average_accuracy,
        "accuracies": self.accuracies,
        "instances_per_second": self.instances_per_second,
        "elapsed_seconds": self.elapsed_seconds,
    }

  @property
  def average_batch_size(self) -> float:
    return self.instance_count / max(self.batch_count, 1)

  @property
  def average_loss(self) -> float:
    return self.loss / max(self.batch_count, 1)

  @property
  def average_accuracy(self) -> float:
    return self.accuracy / max(self.batch_count, 1)

  @property
  def elapsed_seconds(self) -> float:
    end_time = self.end_time or time.time()
    return end_time - self.start_time

  @property
  def instances_per_second(self) -> float:
    return self.instance_count / self.elapsed_seconds
