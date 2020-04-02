from typing import List
from typing import NamedTuple

from programl.graph.py.graph_tuple import GraphTuple
from programl.proto import program_graph_pb2


class GgnnBatchData(NamedTuple):
  """The model-specific data generated for a batch."""

  # A combination of one or more graphs into a single disconnected graph.
  disjoint_graph: GraphTuple
  # A list of graphs that were used to construct the disjoint graph.
  graphs: List[program_graph_pb2.ProgramGraph]
