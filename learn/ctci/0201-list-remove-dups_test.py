from labm8 import app
from labm8 import test

FLAGS = app.FLAGS

MODULE_UNDER_TEST = None  # No coverage.


## Linked Lists:
#
# Solutions for the code questions posed in Cracking the Code
# Interview, Chapter 2, page 77.
#
class Node:

  def __init__(self, key):
    self.key = key
    self.next = None

  def __str__(self):
    x = self
    s = ""
    while x:
      s += str(x.key) + " "
      x = x.next
    return s[:-1]


# Exercise 2.1:
#
#     Write code to remove duplicates from an unsorted linked list.
#
# The implemented solution uses a runner pointer (x) to traverse every
# node after the current, removing duplicates where found. When this
# is complete, the function recurses to the next node and
# completes. This means it operates in O(n^2) time, although it only
# requires O(1) space.
#
def remove_duplicates_no_buffer(head):
  prev, x = head, head.next

  while x:
    if x.key == head.key:
      prev.next = x.next  # Delete node

    prev, x = x, x.next

  if head.next:
    remove_duplicates_no_buffer(head.next)


#
# A O(1) time solution is to iterate through the list and add each
# node's key into a set. If the key already exists within the set,
# then it is a duplicate and the node may be removed. This operates in
# O(n) space.
#
def remove_duplicates(head):
  values = set()

  prev, x = None, head
  while x:
    if x.key in values:  # Check if node exists in set
      prev.next = x.next  # Delete node
    else:
      values.add(x.key)
    prev, x = x, x.next


def test_main():
  # Exercise 2.1
  a, b, c, d, e = Node(1), Node(2), Node(3), Node(2), Node(4)
  a.next = b
  b.next = c
  c.next = d
  d.next = e

  assert str(a) == "1 2 3 2 4"
  remove_duplicates_no_buffer(a)
  assert str(a) == "1 2 3 4"

  e.next = Node(2)
  assert str(a) == "1 2 3 4 2"
  remove_duplicates(a)
  assert str(a) == "1 2 3 4"


if __name__ == '__main__':
  test.Main()
