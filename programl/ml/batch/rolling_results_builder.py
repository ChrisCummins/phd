import sys
from queue import Queue
from threading import Thread
from typing import Optional

from programl.ml.batch.batch_data import BatchData
from programl.ml.batch.batch_results import BatchResults
from programl.ml.batch.rolling_results import RollingResults


class RollingResultsBuilder(object):
  """A threaded worker for updating rolling results.

  Using this class aims to decouple the (possible expensive) updating of rolling
  results and logging progress to stderr from the main model loop.

  Use this class as a context manager to build rolling results, then access the
  results once out of the "with" scope. For example:

    with RollingResultsBuilder("my epoch") as builder:
      for batch, batch_results in my_model.Train():
        # Feed the builder with batch results ...
        builder.AddBatch(batch, batch_results)

    # Now you can access builder.results attribute ...
    epoch_results = builder.results
  """

  def __init__(self, log_prefix: str):
    self._q = Queue(maxsize=100)
    self._thread = Thread(target=self.worker)
    self._log_prefix = log_prefix
    self._thread.start()
    self._results = RollingResults()

  def AddBatch(
    self, data: BatchData, results: BatchResults, weight: Optional[float] = None
  ) -> None:
    """Record a new batch of data.

    Arguments are forward to RollingResults.Update().
    """
    self._q.put((data, results, weight), block=True)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self._q.put(None, block=True)
    self._thread.join()

  @property
  def results(self) -> RollingResults:
    """Access the rolling results.

    Results can only be accessed after exiting the "with" scope of an instance.
    """
    if self._thread.is_alive() is None:
      raise TypeError("Cannot access results yet")
    return self._results

  def worker(self):
    """Background thread worker which repeated updates rolling results and logs
    to stderr.
    """
    while True:
      item = self._q.get(block=True)
      # End of epoch results.
      if item is None:
        break

      self._results.Update(*item)
      if not self.results.batch_count % 4:
        print(
          f"\r\033[K{self._log_prefix}: {self._results}",
          end="",
          file=sys.stderr,
        )
    print("", file=sys.stderr)
