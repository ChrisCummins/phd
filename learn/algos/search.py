#!/usr/bin/env python
import sys


def merge(left, right):
  out = []
  while len(left) and len(right):
    if left[0] < right[0]:
      out.append(left.pop(0))
    else:
      out.append(right.pop(0))

  return out + left + right


def mergesort(arr, left=None, right=None):
  if left is None and right is None:
    left = 0
    right = len(arr) - 1

  if left > right:
    return []
  elif left == right:
    return [arr[right]]

  mid = (left + right) // 2
  left = mergesort(arr, left, mid)
  right = mergesort(arr, mid + 1, right)

  return merge(left, right)


def run_tests(search_fn):
  examples = [
    ([], []),
    ([1], [1]),
    ([1, 1], [1, 1]),
    ([1, 2], [1, 2]),
    ([2, 1], [1, 2]),
    ([3, 1, 2], [1, 2, 3]),
    ([7, 9, 1, 2, 5, 4, 3, 7], [1, 2, 3, 4, 5, 7, 7, 9]),
  ]

  failed = False
  for ins, outs in examples:
    print(ins)
    if search_fn(ins) != outs:
      print(ins)
      print(outs),
      print(search_fn(ins))
      print()
      failed = True

  if failed:
    sys.exit(1)


if __name__ == "__main__":
  run_tests(mergesort)
