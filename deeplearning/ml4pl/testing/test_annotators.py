"""Data flow analyses for testing."""
import random
import time
from typing import Optional

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs.labelled.dataflow import data_flow_graphs
from labm8.py import app


FLAGS = app.FLAGS


class PassThruAnnotator(data_flow_graphs.DataFlowGraphAnnotator):
  """An analysis that returns its input."""

  def MakeAnnotated(
    self, n: Optional[int] = None
  ) -> data_flow_graphs.NetworkxDataFlowGraphs:
    g = programl.ProgramGraphToNetworkX(self.unlabelled_graph)
    return data_flow_graphs.NetworkxDataFlowGraphs([g for _ in range(n or 1)])


class FlakyAnnotator(data_flow_graphs.DataFlowGraphAnnotator):
  """An analysis that sometimes fails, and sometimes returns `n` graphs."""

  def MakeAnnotated(
    self, n: Optional[int] = None
  ) -> data_flow_graphs.NetworkxDataFlowGraphs:
    if random.random() < 0.2:
      raise OSError("something went wrong!")

    g = programl.ProgramGraphToNetworkX(self.unlabelled_graph)
    return data_flow_graphs.NetworkxDataFlowGraphs([g for _ in range(n or 1)])


class TimeoutAnnotator(data_flow_graphs.DataFlowGraphAnnotator):
  """An analysis that times out."""

  def __init__(self, *args, seconds: int = int(1e6), **kwargs):
    super(TimeoutAnnotator, self).__init__(*args, **kwargs)
    self.seconds = seconds

  def MakeAnnotated(
    self, n: Optional[int] = None
  ) -> data_flow_graphs.NetworkxDataFlowGraphs:
    time.sleep(self.seconds)
    return data_flow_graphs.NetworkxDataFlowGraphs([])


class ErrorAnnotator(data_flow_graphs.DataFlowGraphAnnotator):
  """An analysis that raises an error."""

  def MakeAnnotated(
    self, n: Optional[int] = None
  ) -> data_flow_graphs.NetworkxDataFlowGraphs:
    raise OSError("something went wrong!")


class EmptyAnnotator(data_flow_graphs.DataFlowGraphAnnotator):
  """An analysis which produces no outputs."""

  def MakeAnnotated(
    self, n: int = 0
  ) -> data_flow_graphs.NetworkxDataFlowGraphs:
    return data_flow_graphs.NetworkxDataFlowGraphs([])
