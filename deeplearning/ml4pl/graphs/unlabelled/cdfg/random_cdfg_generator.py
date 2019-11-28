"""A generator for random graphs."""
import pickle
import random
import typing

import networkx as nx
import numpy as np

from deeplearning.ml4pl.graphs.unlabelled.cdfg import (
  control_and_data_flow_graph as cdfg,
)
from labm8.py import app

FLAGS = app.FLAGS

with open(cdfg.DICTIONARY, "rb") as f:
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
  adjacency_matrix = np.random.choice(
    [False, True], size=(num_nodes, num_nodes), p=(0.9, 0.1)
  )

  g = nx.MultiDiGraph()

  # Add the edges to the graph, subject to the constraints of CFGs.
  for i, j in np.argwhere(adjacency_matrix):
    # CFG nodes cannot be connected to themselves.
    if i != j:
      g.add_edge(i, j)

  # Make sure that every node has one output. This ensures that the graph is
  # fully connected, but does not ensure that each node has an incoming
  # edge (i.e. is unreachable).
  modified = True
  while modified:
    for node in g.nodes:
      if not g.out_degree(node):
        dst = node
        while dst == node:
          dst = random.randint(0, num_nodes)
        g.add_edge(node, dst)
        break
    else:
      # We iterated through all nodes without making any modifications: we're
      # done.
      modified = False

  node_renamings = {}

  edges_to_remove = []
  for i, (node, data) in enumerate(g.nodes(data=True)):
    if not i:
      # Generate the root node, which is the "magic" entry.
      node_renamings[node] = "root"
      data["type"] = "magic"
      data["text"] = "!UNK"

      # Root node has no in-edges.
      for src, dst in g.in_edges(node):
        edges_to_remove.append((src, dst))
      # Out-edges from root are call edges.
      for src, dst, edge in g.out_edges(node, data=True):
        edge["flow"] = "call"
        edge["position"] = 0
    else:
      # Generate a node of random type.
      type_ = np.random.choice(
        ["statement", "identifier", "immediate"], p=[0.45, 0.3, 0.25]
      )
      if type_ == "statement":
        # node_renamings[node] = str(node)
        data["type"] = "statement"
        data["text"] = "!UNK"
      elif type_ == "immediate":
        node_renamings[node] = f"{node}_immediate"
        # TODO(github.com/ChrisCummins/ml4pl/issues/6): Update.
        data["type"] = "identifier"
        data["text"] = "!IDENTIFIER"
        # Immediates have no in-edges.
        for src, dst in g.in_edges(node):
          edges_to_remove.append((src, dst))
      elif type_ == "identifier":
        # node_renamings[node] = f'%{node}'
        data["type"] = "identifier"
        data["text"] = "!IDENTIFIER"
      else:
        assert False, "Unreachable"

    # Set the embedding indices and "original" text for nodes.
    data["x"] = DICTIONARY[data["text"]]
    data["original_text"] = data["text"]

  for src, dst in edges_to_remove:
    g.remove_edge(src, dst)

  # Assign position and flow type randomly. This does not produce meaningful
  # graphs, e.g. you can have statements with control edges into identifiers.
  for src, dst, data in g.edges(data=True):
    data["position"] = random.randint(0, 2)
    src_type = g.nodes[src]["type"]
    if src_type == "identifier":
      data["flow"] = "data"
    elif src_type == "immediate":
      data["flow"] = "data"
    elif src_type == "magic":
      data["flow"] = "call"
    elif src_type == "statement":
      data["flow"] = np.random.choice(["control", "call"], p=[0.95, 0.05])
    else:
      raise AssertionError(f"unreachable src_type `{src_type}`")

  # Set the new node names.
  nx.relabel_nodes(g, node_renamings, copy=False)

  # Remove any orphans.
  for node in list(nx.isolates(g)):
    g.remove_node(node)

  # Abort, try again.
  if not g.number_of_nodes():
    return FastCreateRandom()

  return g


def AddRandomAnnotations(
  graphs: typing.List[nx.MultiDiGraph],
  auxiliary_node_x_indices_choices=None,
  node_y_choices=None,
  graph_x_choices=None,
  graph_y_choices=None,
):
  """Add random additions to the graphs."""
  if auxiliary_node_x_indices_choices is not None:
    for graph in graphs:
      auxiliary_node_x = [
        [random.choice(x) for x in auxiliary_node_x_indices_choices]
        for _ in range(graph.number_of_nodes())
      ]
      for (_, data), x in zip(graph.nodes(data=True), auxiliary_node_x):
        data["x"] = [data["x"]] + x

  if node_y_choices is not None:
    for graph in graphs:
      node_y = [
        random.choice(node_y_choices) for _ in range(graph.number_of_nodes())
      ]
      for (node, data), y in zip(graph.nodes(data=True), node_y):
        data["y"] = y

  if graph_x_choices is not None:
    graph_x = [random.choice(graph_x_choices) for _ in graphs]
    for graph, x in zip(graphs, graph_x):
      graph.x = x

  if graph_y_choices is not None:
    graph_y = [random.choice(graph_y_choices) for _ in graphs]
    for graph, y in zip(graphs, graph_y):
      graph.y = y
