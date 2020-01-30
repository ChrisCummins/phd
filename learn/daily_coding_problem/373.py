# This problem was asked by Facebook.
#
# Given a list of integers L, find the maximum length of a sequence of
# consecutive numbers that can be formed using elements from L.
#
# For example, given L = [5, 2, 99, 3, 4, 1, 100], return 5 as we can build a
# sequence [1, 2, 3, 4, 5] which has length 5.
from typing import List

from labm8.py import test


# Time: O(n log n)
# Space: O(1)
def F(L: List[int]) -> int:
  L.sort()

  c = 1
  mc = 0

  i = 0
  for j in range(1, len(L)):
    if L[j] == L[i] + 1:
      i = j
      c += 1
    elif L[j] == L[i]:
      pass  # duplicate value, skip it
    else:
      i = j
      mc = max(mc, c)
      c = 1

  if len(L):
    mc = max(mc, c)

  return mc


def test_empty_list():
  assert F([]) == 0


def test_single_value_list():
  assert F([1]) == 1


def test_ascending_sequence():
  assert F([1, 2, 3]) == 3


def test_asending_sequence_with_tail():
  assert F([1, 2, 3, 1]) == 3


def test_double_sequence():
  assert F([1, 2, 3, 1, 2, 3, 4]) == 4


def test_example_input():
  assert F([5, 2, 99, 3, 4, 1, 100]) == 5


if __name__ == "__main__":
  test.Main()
