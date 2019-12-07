"""Migrate networkx graphs to ProGraML protocol buffers.

See <github.com/ChrisCummins/ProGraML/issues/1>.
"""
import networkx as nx

from deeplearning.ml4pl.graphs import programl_pb2
from labm8.py import app


FLAGS = app.FLAGS


def NetworkXGraphToProgramGraphProto(
  g: nx.MultiDiGraph,
) -> programl_pb2.ProgramGraph:
  """Convert a networkx graph constructed using the old control-and-data-flow
  graph builder to a ProGraML graph proto."""
  proto = programl_pb2.ProgramGraph()

  # Create the map from function IDs to function names.
  function_names = list(
    sorted(set([fn for _, fn in g.nodes(data="function") if fn]))
  )
  function_to_idx_map = {fn: i for i, fn in enumerate(function_names)}

  # Create the function list.
  for function_name in function_names:
    function_proto = proto.function.add()
    function_proto.name = function_name

  # Build a translation map from node names to node list indices.
  if "root" not in g.nodes:
    raise ValueError(f"Graph has no root node: {g.nodes}")
  node_to_idx_map = {"root": 0}
  for node in [node for node in g.nodes if node != "root"]:
    node_to_idx_map[node] = len(node_to_idx_map)

  # Create the node list.
  idx_to_node_map = {v: k for k, v in node_to_idx_map.items()}
  for node_idx in range(len(node_to_idx_map)):
    node = g.nodes[idx_to_node_map[node_idx]]
    node_proto = proto.node.add()

    # Translate node attributes.
    node_type = node.get("type")
    if not node_type:
      raise ValueError(f"Node has no type: {node_type}")
    node_proto.type = {
      "statement": programl_pb2.Node.STATEMENT,
      "identifier": programl_pb2.Node.IDENTIFIER,
      "immediate": programl_pb2.Node.IMMEDIATE,
      # We are removing the "magic" node type, replacing them with a regular
      # statement of unknown type.
      "magic": programl_pb2.Node.STATEMENT,
    }[node_type]

    # Get the text of the node.
    if "original_text" in node:
      node_proto.text = node["original_text"]
      node_proto.preprocessed_text = node["text"]
    elif "text" in node:
      node_proto.text = node["text"]
      node_proto.preprocessed_text = node["text"]
    elif "name" in node:
      node_proto.text = node["name"]
      node_proto.preprocessed_text = node["name"]
    else:
      raise ValueError(f"Node has no original_text or name: {node}")

    # Set the encoded representation of the node.
    x = node.get("x", None)
    if x is not None:
      node_proto.x.extend([x])

    # Set the node function.
    function = node.get("function")
    if function:
      node_proto.function = function_to_idx_map[function]

  # Create the edge list.
  for src, dst, data in g.edges(data=True):
    edge = proto.edge.add()
    edge.flow = {
      "call": programl_pb2.Edge.CALL,
      "control": programl_pb2.Edge.CONTROL,
      "data": programl_pb2.Edge.DATA,
    }[data["flow"]]
    edge.source_node = node_to_idx_map[src]
    edge.destination_node = node_to_idx_map[dst]
    edge.position = data.get("position", 0)

  return proto
