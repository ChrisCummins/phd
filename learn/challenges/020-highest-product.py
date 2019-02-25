#!/usr/bin/env python3

from collections import deque

from typing import List


def approach1(lst: List[int]) -> int:
  if not isinstance(lst, list):
    raise TypeError
  if len(lst) < 3:
    raise ValueError
  if any(x is None for x in lst):
    raise TypeError

  best = deque(sorted(lst[:3]))

  for x in lst[3:]:
    if x > min(best):
      # TODO: insert into sorted best
      del best[best.index(min(best))]
      best.append(x)

  return best[0] * best[1] * best[2]


if __name__ == "__main__":
  try:
    approach1(2)
    assert False
  except TypeError:
    pass

  try:
    approach1([2, 2])
    assert False
  except ValueError:
    pass

  try:
    approach1([1, 2, 3, None, 2])
    assert False
  except TypeError:
    pass

  examples = [
      ([1, 2, 3], 6),
      ([1, 8, 8], 64),
      ([1, 8, 1, 8], 64),
      ([1, 8, 1, 2, 8], 128),
  ]
  for ins, outs in examples:
    print(ins, outs, approach1(ins))
    assert approach1(ins) == outs
