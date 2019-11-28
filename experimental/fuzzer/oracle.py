#!/usr/bin/env python3
"""
The fuzzer "oracle". Exposes an interface for guiding a
grammar-based program generator.
"""
import random
import re
import sys

import rstr  # non-standard package (run `pip3 install rstr`)


class Oracle(object):
  """
  Interface for the oracle.
  """

  def matchesRegex(self, state, regex):
    """
    Generation is in the given state, and needs a token matching the
    regex.

    :param state: the generation state
    :param regex: the regex which must be matched
    :return: token str which matches regex
    """
    raise NotImplementedError("Abstract class")

  def chooseChild(self, state, production):
    """
    Generation is in the given state, and needs to choose a child of
    the current production rule.

    :param state: the generation state
    :param production: the current production rule
    :return: the selected child production rule
    """
    raise NotImplementedError("Abstract class")


class DummyOracle(Oracle):
  """
  Testing oracle.
  """

  def __init__(self, seed=None):
    if seed != None:
      random.seed(seed)
    self._strgen = rstr.Rstr()

  def matchesRegex(self, state, regex):
    return self._strgen.xeger(regex)

  def chooseChild(self, state, production):
    children = production.children()
    idx = random.randint(0, len(children) - 1)
    return children[idx]


class ProductionRule(object):
  """
  TODO: interface undecided
  """

  def children(self):
    """
    :return: a list of ProductionRules which are valid children
    """
    raise NotImplementedError("Abstract class")


def demo(oracle):
  """
  Demo the oracle interface.
  """

  class PlaceholderProductionRule(ProductionRule):
    """
    Placeholder production rule (for demo).
    """

    def __init__(self, name):
      self._name = name

    def children(self):
      return [
        PlaceholderProductionRule(str(self) + "->a"),
        PlaceholderProductionRule(str(self) + "->b"),
        PlaceholderProductionRule(str(self) + "->c"),
      ]

    def __repr__(self):
      return str(self._name)

  # Generate some strs
  print("Oracle.matchesRegex():")
  for i in range(10):
    identifier = oracle.matchesRegex("", re.compile("[abc][def]"))
    print(" " * 3, identifier)

  # Generate some children
  print("\nOracle.chooseChild():")
  production = PlaceholderProductionRule("root")
  for i in range(10):
    production = oracle.chooseChild("", production)
    print(" " * 3, production)


def main(argv):
  oracle = DummyOracle(seed=0xCEC)
  demo(oracle)


if __name__ == "__main__":
  main(sys.argv)
