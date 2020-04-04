from typing import List
from typing import NamedTuple
from typing import Optional

import numpy as np

from programl.graph.format.py.graph_tuple import GraphTuple
from programl.proto import program_graph_pb2


class GgnnBatchData(NamedTuple):
  """The model-specific data generated for a batch."""

  # A combination of one or more graphs into a single disconnected graph.
  graph_tuple: GraphTuple
  vocab_ids: np.array
  selector_ids: np.array

  # A list of graphs that were used to construct the disjoint graph.
  # This can be useful for debugging, but is not required by the model.
  graphs: Optional[List[program_graph_pb2.ProgramGraph]] = None

  node_labels: Optional[np.array] = None
  graph_features: Optional[np.array] = None
  graph_labels: Optional[np.array] = None
