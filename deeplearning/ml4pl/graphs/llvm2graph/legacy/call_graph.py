# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LLVM Call Graphs."""
import typing
from typing import List

import networkx as nx
import pydot
import pyparsing

from labm8.py import app
from labm8.py import labtypes


FLAGS = app.FLAGS


def CallGraphFromDotSource(dot_source: str) -> nx.MultiDiGraph:
  """Create a call graph from an LLVM-generated dot file.

  Args:
    dot_source: The dot source generated by the LLVM -dot-callgraph pass.

  Returns:
    A directed multigraph, where each node is a function (or the special
    "external node"), and edges indicate calls between functions.

  Raises:
    pyparsing.ParseException: If dotfile could not be parsed.
    ValueError: If dotfile could not be interpretted / is malformed.
  """
  try:
    parsed_dots = pydot.graph_from_dot_data(dot_source)
  except TypeError as e:
    raise pyparsing.ParseException("Failed to parse dot source") from e

  if len(parsed_dots) != 1:
    raise ValueError(f"Expected 1 Dot in source, found {len(parsed_dots)}")

  dot = parsed_dots[0]

  graph = nx.drawing.nx_pydot.from_pydot(dot)

  # Nodes are given a fairly arbitrary name by pydot, instead, we want to name
  # the nodes by their label, which, for all except the magic "external node"
  # node, is the name of a function.
  node_name_to_label = {}

  nodes_to_delete: List[str] = []

  for node, data in graph.nodes(data=True):
    if "label" not in data:
      nodes_to_delete.append(node)
      continue
    label = data["label"]
    if label and not (label.startswith('"{') and label.endswith('}"')):
      raise ValueError(f"Invalid label: `{label}`")
    label = label[2:-2]
    node_name_to_label[node] = label
    # Remove unneeded data attributes.
    labtypes.DeleteKeys(data, {"shape", "label"})

  # Remove unlabelled nodes.
  for node in nodes_to_delete:
    graph.remove_node(node)

  nx.relabel_nodes(graph, node_name_to_label, copy=False)
  return graph


def CallGraphToFunctionCallCounts(g: nx.MultiDiGraph,) -> typing.Dict[str, int]:
  """Build a table of call counts for each function.

  Args:
    g: A call graph, such as produced by LLVM's -dot-callgraph pass.
      See CallGraphFromDotSource().

  Returns:
    A dictionary where each function in the graph is a key, and the value is the
    number of unique call sites for that function. Note this may be zero, in the
    case of library functions.
  """
  # Initialize the call count table with an entry for each function, except
  # the magic "external node" entry produced by LLVM's -dot-callgraph pass.
  function_names = [n for n in g.nodes if n != "external node"]
  return {
    # Use the indegree, subtracting one for the call from "external node".
    n: g.in_degree(n) - 1
    for n in function_names
  }
