"""Utilities for working with LLVM statements."""
import copy
import difflib
import itertools
import pickle
import re
import typing

import networkx as nx
import numpy as np

from compilers.llvm import opt_util
from deeplearning.ml4pl.graphs import graph_iterators as iterators
from deeplearning.ml4pl.graphs import graph_query as query
from deeplearning.ml4pl.graphs.unlabelled.cfg import llvm_util
from deeplearning.ml4pl.graphs.unlabelled.cg import call_graph as cg
from deeplearning.ncc import rgx_utils as rgx
from deeplearning.ncc.inst2vec import inst2vec_preprocess
from labm8 import app
from labm8 import bazelutil
from labm8 import decorators

FLAGS = app.FLAGS


def GetAllocationStatementForIdentifier(g: nx.Graph, identifier: str) -> str:
  for node, data in iterators.StatementNodeIterator(g):
    if ' = alloca ' in data['text']:
      allocated_identifier = data['text'].split(' =')[0]
      if allocated_identifier == identifier:
        return node
  raise ValueError(
      f"Unable to find `alloca` statement for identifier `{identifier}`")


def StripIdentifiersAndImmediates(stmt: str) -> str:
  """This is a copy of inst2vec_preprocess.PreprocessStatement(), but instead
  of substituting placeholders values, immediates and labels are removed
  entirely from the string.
  """
  # Remove local identifiers
  stmt = re.sub(rgx.local_id, '', stmt)
  # Global identifiers
  stmt = re.sub(rgx.global_id, '', stmt)
  # Remove labels
  if re.match(r'; <label>:\d+:?(\s+; preds = )?', stmt):
    stmt = re.sub(r":\d+", ":", stmt)
  elif re.match(rgx.local_id_no_perc + r':(\s+; preds = )?', stmt):
    stmt = re.sub(rgx.local_id_no_perc + ':', ":", stmt)

  # Remove floating point values
  stmt = re.sub(rgx.immediate_value_float_hexa, "", stmt)
  stmt = re.sub(rgx.immediate_value_float_sci, "", stmt)

  # Remove integer values
  if (re.match("<%ID> = extractelement", stmt) is None and
      re.match("<%ID> = extractvalue", stmt) is None and
      re.match("<%ID> = insertelement", stmt) is None and
      re.match("<%ID> = insertvalue", stmt) is None):
    stmt = re.sub(r'(?<!align)(?<!\[) ' + rgx.immediate_value_int, " ", stmt)

  # Remove string values
  stmt = re.sub(rgx.immediate_value_string, " ", stmt)

  # Remove index types
  if (re.match(" = extractelement", stmt) is not None or
      re.match(" = insertelement", stmt) is not None):
    stmt = re.sub(r'i\d+ ', ' ', stmt)

  return stmt


def GetLlvmStatementDefAndUses(statement: str,
                               store_destination_is_def: bool = False
                              ) -> typing.Tuple[str, typing.List[str]]:
  """Get the destination identifier for an LLVM statement (if any), and a list
  of operand identifiers (if any).
  """
  # Left hand side.
  destination = ''
  if '=' in statement:
    first_equals = statement.index('=')
    destination = statement[:first_equals]
    statement = statement[first_equals:]

  # Strip the identifiers and immediates from the statement, then use the
  # diff to construct the set of identifiers and immediates that were stripped.
  stripped = StripIdentifiersAndImmediates(statement)
  tokens = []

  last_token = []
  last_index = -1
  for i, diff in enumerate(difflib.ndiff(statement, stripped)):
    if diff[0] == '-':
      if i != last_index + 1 and last_token:
        tokens.append(''.join(last_token))
        last_token = []

      last_token.append(diff[-1])
      last_index = i

  if last_token:
    tokens.append(''.join(last_token))

  return destination.strip(), tokens
