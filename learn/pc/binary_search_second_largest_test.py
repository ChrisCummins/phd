from typing import Optional
from typing import Union

import pytest

from labm8.py import test

MODULE_UNDER_TEST = None  # No coverage.


class Node(object):
  """Node in a binary search tree."""

  def __init__(self, val: int, left: Optional["Node"], right: Optional["Node"]):
    self.val = val
    self.left = left
    self.right = right


def FindSecondLargestHelper(node: Node, prior: Union[Node, None]) -> int:
  if node.right:
    return FindSecondLargestHelper(node.right, node)
  elif prior:
    # The previously visited node may be either greater or smaller than the
    # current node.
    return min(node.val, prior.val)
  elif node.left:
    return FindSecondLargestHelper(node.left, node)
  else:
    raise ValueError("BST contains insufficient nodes")


# Given the root to a binary search tree, find the second largest node in the
# tree.
def FindSecondLargest(node: Node) -> int:
  return FindSecondLargestHelper(node, None)


def test_FindSecondLargest_tree1():
  # Tree:
  #   5
  #    \
  #     6
  #      \
  #       7
  bst = Node(5, None, Node(6, None, Node(7, None, None)))
  assert FindSecondLargest(bst) == 6


def test_test_FindSecondLargest_tree2():
  # Tree:
  #     6
  #    /
  #   4
  #    \
  #     5
  bst = Node(6, Node(4, None, Node(5, None, None)), None)
  assert FindSecondLargest(bst) == 4


def test_FindSecondLargest_tree3():
  # Tree:
  #     5
  bst = Node(5, None, None)
  with pytest.raises(ValueError):
    FindSecondLargest(bst)


if __name__ == "__main__":
  test.Main()
