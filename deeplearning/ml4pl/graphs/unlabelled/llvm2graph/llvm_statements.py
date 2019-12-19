# Copyright 2019 the ProGraML authors.
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
"""Utilities for working with LLVM statements."""
import difflib
import re
import typing

import networkx as nx

from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ncc import rgx_utils as rgx
from labm8.py import app


FLAGS = app.FLAGS


def GetAllocationStatementForIdentifier(g: nx.Graph, identifier: str) -> str:
  for node, data in g.nodes(data=True):
    if data["type"] != programl_pb2.Node.STATEMENT:
      continue
    if " = alloca " in data["text"]:
      allocated_identifier = data["text"].split(" =")[0]
      if allocated_identifier == identifier:
        return node
  raise ValueError(
    f"Unable to find `alloca` statement for identifier `{identifier}`"
  )


def StripIdentifiersAndImmediates(stmt: str) -> str:
  """This is a copy of inst2vec_preprocess.PreprocessStatement(), but instead
  of substituting placeholders values, immediates and labels are removed
  entirely from the string.
  """
  # Remove local identifiers
  stmt = re.sub(rgx.local_id, "", stmt)
  # Global identifiers
  stmt = re.sub(rgx.global_id, "", stmt)
  # Remove labels
  if re.match(r"; <label>:\d+:?(\s+; preds = )?", stmt):
    stmt = re.sub(r":\d+", ":", stmt)
  elif re.match(rgx.local_id_no_perc + r":(\s+; preds = )?", stmt):
    stmt = re.sub(rgx.local_id_no_perc + ":", ":", stmt)

  # Remove floating point values
  stmt = re.sub(rgx.immediate_value_float_hexa, "", stmt)
  stmt = re.sub(rgx.immediate_value_float_sci, "", stmt)

  # Remove integer values
  if (
    re.match("<%ID> = extractelement", stmt) is None
    and re.match("<%ID> = extractvalue", stmt) is None
    and re.match("<%ID> = insertelement", stmt) is None
    and re.match("<%ID> = insertvalue", stmt) is None
  ):
    stmt = re.sub(r"(?<!align)(?<!\[) " + rgx.immediate_value_int, " ", stmt)

  # Remove string values
  stmt = re.sub(rgx.immediate_value_string, " ", stmt)

  # Remove index types
  if (
    re.match(" = extractelement", stmt) is not None
    or re.match(" = insertelement", stmt) is not None
  ):
    stmt = re.sub(r"i\d+ ", " ", stmt)

  return stmt


def GetLlvmStatementDefAndUses(
  statement: str, store_destination_is_def: bool = False
) -> typing.Tuple[str, typing.List[str]]:
  """Get the destination identifier for an LLVM statement (if any), and a list
  of operand identifiers (if any).
  """
  # Left hand side.
  destination = ""
  if "=" in statement:
    first_equals = statement.index("=")
    destination = statement[:first_equals]
    statement = statement[first_equals:]

  # Strip the identifiers and immediates from the statement, then use the
  # diff to construct the set of identifiers and immediates that were stripped.
  stripped = StripIdentifiersAndImmediates(statement)
  tokens = []

  last_token = []
  last_index = -1
  for i, diff in enumerate(difflib.ndiff(statement, stripped)):
    if diff[0] == "-":
      if i != last_index + 1 and last_token:
        tokens.append("".join(last_token))
        last_token = []

      last_token.append(diff[-1])
      last_index = i

  if last_token:
    tokens.append("".join(last_token))

  return destination.strip(), tokens


def GetCalledFunctionName(statement) -> typing.Optional[str]:
  """Get the name of a function called in the statement."""
  if "call " not in statement:
    return None
  # Try and resolve the call destination.
  _, m_glob, _, _ = inst2vec_preprocess.get_identifiers_from_line(statement)
  if not m_glob:
    return None
  return m_glob[0][1:]  # strip the leading '@' character


def FindCallSites(graph, source_function, destination_function):
  """Find the statements in function that call another function."""
  call_sites = []
  for node, data in StatementNodeIterator(graph):
    if data["function"] != source_function:
      continue
    statement = data.get("original_text", data["text"])
    called_function = GetCalledFunctionName(statement)
    if not called_function:
      continue
    if called_function == destination_function:
      call_sites.append(node)
  return call_sites
