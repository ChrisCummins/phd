"""Data flow analyses for testing."""
import random
import time
from typing import Iterable
from typing import Optional

from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow import data_flow_graphs
from labm8.py import app


FLAGS = app.FLAGS


class PassThruAnnotator(data_flow_graphs.DataFlowGraphAnnotator):
  """An analysis that returns its input."""

  def MakeAnnotated(
    self, unlabelled_graph: programl_pb2.ProgramGraph, n: Optional[int] = None
  ) -> Iterable[programl_pb2.ProgramGraph]:
    for _ in range(n or 1):
      yield unlabelled_graph


class FlakyAnnotator(data_flow_graphs.DataFlowGraphAnnotator):
  """An analysis that sometimes fails, and sometimes returns `n` graphs."""

  def MakeAnnotated(
    self, unlabelled_graph: programl_pb2.ProgramGraph, n: Optional[int] = None
  ) -> Iterable[programl_pb2.ProgramGraph]:
    if random.random() < 0.2:
      raise OSError("something went wrong!")
    for _ in range(n or 1):
      yield unlabelled_graph


class TimeoutAnnotator(data_flow_graphs.DataFlowGraphAnnotator):
  """An analysis that times out."""

  def __init__(self, seconds: int = int(1e6)):
    self.seconds = seconds

  def MakeAnnotated(
    self, unlabelled_graph: programl_pb2.ProgramGraph, n: Optional[int] = None
  ) -> Iterable[programl_pb2.ProgramGraph]:
    time.sleep(self.seconds)
    yield programl_pb2.ProgramGraph()


class ErrorAnnotator(data_flow_graphs.DataFlowGraphAnnotator):
  """An analysis that raises an error."""

  def MakeAnnotated(
    self, unlabelled_graph: programl_pb2.ProgramGraph, n: Optional[int] = None
  ) -> Iterable[programl_pb2.ProgramGraph]:
    raise OSError("something went wrong!")
