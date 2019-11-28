#!/usr/bin/env python3
from collections import deque
from typing import List


class Node(object):
  def __init__(self, data):
    self.data = data
    self.left = None
    self.right = None

  def __lt__(self, rhs: "Node"):
    return self.data < rhs.data

  def __eq__(self, rhs: "Node"):
    return self.data == rhs.data

  def __le__(self, rhs: "Node"):
    return self.__eq__(self, rhs) or self.__lt__(self, rhs)

  def __gt__(self, rhs: "Node"):
    return not self.__le__(self, rhs)

  def __ge__(self, rhs: "Node"):
    return self.__eq__(self, rhs) or self.__gt__(self, rhs)


class Graph(object):
  def __init__(self, root=None):
    self.root = root

  def insert(self, data, root=None):
    if root is None:
      root = self.root

    newnode = Node(data)

    if self.root is None:
      self.root = newnode
    else:
      if data <= root.data:
        if root.left:
          return self.insert(data, root.left)
        else:
          root.left = newnode
      else:
        if root.right:
          return self.insert(data, root.right)
        else:
          root.right = newnode

    return newnode

  @property
  def elements(self) -> List:
    elements = []

    q = deque([])

    if self.root:
      q.append(self.root)

    while len(q):
      node = q.popleft()
      elements.append(node.data)
      if node.left:
        q.append(node.left)
      if node.right:
        q.append(node.right)

    return elements

  @property
  def levels(self) -> List[List]:
    levels = []

    q = deque()
    if self.root:
      q.append((0, self.root))

    while len(q):
      depth, node = q.popleft()
      if len(levels) <= depth:
        levels.append([])
      levels[depth].append(node.data)
      if node.left:
        q.append((depth + 1, node.left))
      if node.right:
        q.append((depth + 1, node.right))

    return levels


if __name__ == "__main__":
  g = Graph()
  assert g.elements == []
  assert g.levels == []

  g.insert(5)
  assert g.elements == [5]
  assert g.levels == [[5]]

  g.insert(4)
  assert g.elements == [5, 4]
  assert g.levels == [[5], [4]]

  g.insert(5)
  assert g.elements == [5, 4, 5]
  assert g.levels == [[5], [4], [5]]

  g.insert(2)
  assert g.root.left.right.data == 5
  assert g.root.left.left.data == 2
  assert g.elements == [5, 4, 2, 5]
  assert g.levels == [[5], [4], [2, 5]]

  g.insert(10)
  g.insert(7)
  g.insert(6)
  assert g.elements == [5, 4, 10, 2, 5, 7, 6]
  assert g.levels == [[5], [4, 10], [2, 5, 7], [6]]
