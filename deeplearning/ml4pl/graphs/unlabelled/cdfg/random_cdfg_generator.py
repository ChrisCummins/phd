"""A generator for random graphs."""
import pickle
import random

import networkx as nx
import numpy as np
from labm8 import app

from deeplearning.ml4pl.graphs.unlabelled.cdfg import \
  control_and_data_flow_graph as cdfg

FLAGS = app.FLAGS

WEIGHTED_NODE_TYPES = {
    'statement': 1,
    'identifier': .7,
    'immediate': .4,
}

with open(cdfg.DICTIONARY, 'rb') as f:
  DICTIONARY = pickle.load(f)


def FastCreateRandom():
  """Produce a random graph with correct syntax but meaningless semantics.

  This generates a random graph which has the correct attributes for nodes and
  edges, but does not have correct meaning, e.g. there may be data flow edges
  between identifiers, etc. For speed, this generator guarantees only that:
  
    1. There is a 'root' node with outgoing call edges.
    2. Nodes are either statements, identifiers, or immediates.
    3. Nodes have text, original_text, and embedding indices.
    4. Edges are either control, data, or call.
    5. Edges have positions.
    6. The graph is strongly connected.
  """
  num_nodes = random.randint(5, 50)
  g = nx.scale_free_graph(num_nodes)

  node_renamings = {}

  node_types = list(WEIGHTED_NODE_TYPES.keys())
  node_weights = np.array(list(WEIGHTED_NODE_TYPES.values()), dtype=np.float32)
  node_weights /= node_weights.sum()

  edges_to_remove = []
  for i, (node, data) in enumerate(g.nodes(data=True)):
    if not i:
      # Generate the root node, which is the "magic" entry.
      node_renamings[node] = 'root'
      data['type'] = 'magic'
      data['text'] = '!UNK'

      # Root node has no in-edges.
      for src, dst in g.in_edges(node):
        edges_to_remove.append((src, dst))
      # Out-edges from root are call edges.
      for src, dst, edge in g.out_edges(node, data=True):
        edge['flow'] = 'call'
        edge['position'] = 0
    else:
      # Generate a node of random type.
      type_ = np.random.choice(node_types, p=node_weights)
      if type_ == 'statement':
        node_renamings[node] = str(node)
        data['type'] = 'statement'
        data['text'] = np.random.choice(list(DICTIONARY.keys()))
      elif type_ == 'immediate':
        node_renamings[node] = f'{node}_immediate'
        # TODO(github.com/ChrisCummins/ml4pl/issues/6): Update.
        data['type'] = 'identifier'
        data['text'] = '!IDENTIFIER'
        # Immediates have no in-edges.
        for src, dst in g.in_edges(node):
          edges_to_remove.append((src, dst))
      elif type_ == 'identifier':
        node_renamings[node] = f'%{node}'
        data['type'] = 'identifier'
        data['text'] = '!IDENTIFIER'
      else:
        assert False, "Unreachable"

    # Set the embedding indices and "original" text for nodes.
    data['x'] = DICTIONARY[data['text']]
    data['original_text'] = data['text']

  for src, dst in edges_to_remove:
    g.remove_edge(src, dst)

  # Assign position and flow type randomly. This does not produce meaningful
  # graphs, e.g. you can have statements with control edges into identifiers.
  for src, dst, data in g.edges(data=True):
    position = random.randint(0, 2)
    if not src:  # Root node.
      continue
    src_type = g.nodes[src]['type']
    dst_type = g.nodes[dst]['type']
    if src_type == 'identifier':
      data['flow'] = 'data'
    elif src_type == 'immediate':
      data['flow'] = 'data'
    elif src_type == 'statement':
      data['flow'] = np.random.choice(['control', 'call'], p=[.95, .05])

  # Set the new node names.
  nx.relabel_nodes(g, node_renamings, copy=False)

  return g
