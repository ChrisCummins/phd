#!/usr/bin/env python3


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

  def isvalid(self, root=None, valmin=None, valmax=None):
    if root is None:
      root = self.root

    if root is not None:
      if valmin is not None and root.data < valmin:
        return False
      if valmax is not None and root.data > valmax:
        return False

      if root.left and not self.isvalid(root=root.left, valmax=root.data):
        return False
      if root.right and not self.isvalid(root=root.right, valmin=root.data):
        return False

    return True


if __name__ == "__main__":
  g = Graph()
  assert g.isvalid()

  g.insert(5)
  g.insert(5)
  g.insert(4)
  g.insert(5)
  g.insert(2)
  g.insert(10)
  g.insert(7)
  g.insert(6)

  assert g.isvalid()
  g.root.left.left.left.left = Node(100)
  assert not g.isvalid()
