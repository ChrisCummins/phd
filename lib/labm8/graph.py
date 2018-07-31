"""This module defines classes and helper functions for working with graphs."""
import typing
from absl import flags

from lib.labm8 import fmt


FLAGS = flags.FLAGS


class Graph(object):

  def __init__(self, name: str,
               children: typing.Optional[typing.List['Graph']] = None):
    self.name = name
    if children:
      self.children = set(children)
    else:
      self.children = set()

  def _ToDot(self, strings, visited: typing.Set['Graph']) -> str:
    if self in visited:
      return
    visited.add(self)
    for child in self.children:
      strings.append(f'{self.name} -> {child.name}')
    for child in self.children:
      child._ToDot(strings, visited)

  def ToDot(self) -> str:
    strings = []
    self._ToDot(strings, set())
    dot = fmt.IndentList(2, strings)
    return f"digraph graphname {{\n  {dot}\n}}"

  def __eq__(self, other):
    return self.name == other.name

  def __neq__(self, other):
    return not self == other

  def __lt__(self, other):
    return self.name < other.name

  def __le__(self, other):
    return self < other or self == other

  def __hash__(self):
    return hash(self.name)

  def __repr__(self):
    return self.name
