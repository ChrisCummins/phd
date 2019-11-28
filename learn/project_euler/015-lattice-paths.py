#!/usr/bin/env python3
from collections import deque
from typing import Tuple


def routes_through_grid_bfs(
  start: Tuple[int, int], end: Tuple[int, int]
) -> int:
  """
  breadth first search

  T(n) = O(?)
  S(n) = O(?)
  """
  q = deque([start])
  count = 0
  while len(q):
    n = q.popleft()
    if n == end:
      count += 1
    else:
      if n[0] + 1 <= end[0]:
        q.append((n[0] + 1, n[1]))
      if n[1] + 1 <= end[1]:
        q.append((n[0], n[1] + 1))

  return count


def routes_through_grid_dfs(
  start: Tuple[int, int], end: Tuple[int, int]
) -> int:
  """
  depth first search

  T(n) = O(?)
  S(n) = O(?)
  """
  count = 0

  if start == end:
    count += 1
  else:
    if start[0] + 1 <= end[0]:
      count += routes_through_grid_dfs((start[0] + 1, start[1]), end)
    if start[1] + 1 <= end[1]:
      count += routes_through_grid_dfs((start[0], start[1] + 1), end)

  return count


examples = [
  ((0, 0), (1, 1)),
  ((0, 0), (2, 2)),
  ((0, 0), (3, 3)),
  ((0, 0), (20, 20)),
]

for example in examples:
  print(routes_through_grid_dfs(*example))
