"""Data flow analyses for testing."""
import time
from typing import Iterable
from typing import Optional

from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow import data_flow_graphs
from labm8.py import app


FLAGS = app.FLAGS


class TimeoutAnnotator(data_flow_graphs.DataFlowGraphAnnotator):
  """An analysis that times out."""

  def __init__(self, seconds: int = int(1e6)):
    self.seconds = seconds

  def MakeAnnotated(
    self, unlabelled_graph: programl_pb2.ProgramGraph, n: Optional[int] = None
  ) -> Iterable[programl_pb2.ProgramGraph]:
    time.sleep(self.seconds)
    yield programl_pb2.ProgramGraph()
