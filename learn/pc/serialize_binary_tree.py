# Given the root to a binary tree, implement serialize(root), which serializes
# the tree into a string, and deserialize(s), which deserializes the string
# back into the tree.
import typing


class Node:
  """Example class to serialize."""

  def __init__(self, val, left=None, right=None):
    self.val = val
    self.left = left
    self.right = right


# This implementation is very suboptimal, and overly complex. I was given a hint
# to look at a reverse polish parser and see if that could be applied here.


def set_id(node: Node, counter: int):
  counter += 1
  node.id = counter
  if node.left:
    counter = set_id(node.left, counter)
  if node.right:
    counter = set_id(node.right, counter)
  return counter


def serialize_dfs(node: Node, s: typing.List[str]):
  s.append(' '.join(
      str(x) for x in [
          len(node.val),
          node.val,
          node.left.id if node.left else -1,
          node.right.id if node.right else -1,
      ]))
  if node.left:
    s = serialize_dfs(node.left, s)
  if node.right:
    s = serialize_dfs(node.right, s)
  return s


def read_line(s: str):
  space = s.index(' ')
  val_len = int(s[:space])
  s = s[space + 1:]
  val = s[:val_len]

  s = s[val_len + 1:]
  left = int(s[:s.index(' ')])
  s = s[s.index(' ') + 1:]
  right = int(s[:s.index('\n')])
  s = s[s.index('\n'):]

  return s, val, left, right


# Serialize and deserialize functions.


def serialize(node: Node):
  set_id(node, 0)
  return '\n'.join(serialize_dfs(node, [])) + '\n'


def deserialize(s: str) -> Node:
  nodes = []
  while s.strip():
    s, val, left, right = read_line(s)
    nodes.append(Node(val, left, right))

  for n in nodes:
    if n.left != -1:
      n.left = nodes[n.left - 1]
    if n.right != -1:
      n.right = nodes[n.right - 1]

  return nodes[0]


if __name__ == '__main__':
  node = Node('root', Node('left', Node('left.left')), Node('right'))
  print(deserialize(serialize(node)))
  assert deserialize(serialize(node)).left.left.val == 'left.left'
