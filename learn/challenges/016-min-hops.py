#!/usr/bin/env python3
#
# Given a file containing words (one line per word). Write a function to find
# the shortest number of hops between a pair of words, where a "hop" is either
# changing, adding, or removing a character.
#
import os
import pickle
import timeit
from collections import deque
from hashlib import sha1
from string import ascii_lowercase
from typing import Iterator
from typing import List
from typing import Set


# Approach 1: BFS, where neighboring nodes are computed on-the-fly.
#
def gen_permutations_in(D: Set[str], s: str) -> List[str]:
  """ generate all permutations of `s` which are in `D` """
  permutations = []
  for i in range(len(s)):
    candidates = (
      [s[:i] + s[i + 1 :]]
      + [s[:i] + c + s[i:] for c in ascii_lowercase]  # remove character
      + [  # add character
        s[:i] + c + s[i + 1 :] for c in ascii_lowercase
      ]  # switch character
    )

    permutations += [p for p in candidates if p in D]

  # append character
  permutations += [s + c for c in ascii_lowercase if s + c in D]

  return permutations


def bfs(start, end, neighbours_fn) -> int:
  """ return distance from start to end, or -1 if no path """
  q = deque([(start, 0)])
  v = set()  # if the size of the graph is large, bit array may be cheaper

  while len(q):
    node, distance = q.popleft()
    if node == end:
      return distance
    for neighbor in neighbours_fn(node):
      if neighbor not in v:
        v.add(neighbor)
        q.append((neighbor, distance + 1))
  return -1


def approach_1(filein: Iterator[str], start: str, end: str) -> int:
  """ generate neighbors on-demand """
  D = set([x.lower() for x in filein])
  start, end = start.lower(), end.lower()
  if start not in D or end not in D:
    raise ValueError

  neighbours = lambda n: gen_permutations_in(D, n)
  return bfs(start, end, neighbours)


# Approach 2: BFS over a graph of nodes
#
def gen_permutations(s: str) -> List[str]:
  """ generate all permutations of `s` which are in `D` """
  permutations = []
  for i in range(len(s)):
    permutations += (
      [s[:i] + s[i + 1 :]]
      + [s[:i] + c + s[i:] for c in ascii_lowercase]  # remove character
      + [  # add character
        s[:i] + c + s[i + 1 :] for c in ascii_lowercase
      ]  # switch character
    )

  permutations += [s + c for c in ascii_lowercase]  # append character

  return permutations


def approach_2(filein: Iterator[str], start: str, end: str) -> int:
  """ generate complete graph """
  D = dict((x.lower(), i) for i, x in enumerate(filein))  # str -> int
  start, end = start.lower(), end.lower()

  # adjacency lists:
  G = [[] for i in range(len(D))]

  # build adjacency lists
  for key, value in D.items():
    for permutation in gen_permutations(key):
      if permutation in D:
        G[value].append(D[permutation])

  neighbours = lambda n: G[n]
  return bfs(D[start], D[end], neighbours)


# Approach 3: BFS over a graph of nodes
#
def approach_3(filein: Iterator[str], start: str, end: str) -> int:
  """ generate complete graph, cache graph """
  start, end = start.lower(), end.lower()

  # checksum input
  m = sha1()
  m.update("\n".join(filein).encode("utf-8"))
  checksum = str(m.hexdigest())

  # if cache file for checksum exists, read from it
  if os.path.exists(f"cache-{checksum}.pkl"):
    with open(f"cache-{checksum}.pkl", "rb") as infile:
      D, G = pickle.load(infile)
  else:
    # adjacency lists:
    D = dict((x.lower(), i) for i, x in enumerate(filein))  # str -> int
    G = [[] for i in range(len(D))]

    # build adjacency lists
    for key, value in D.items():
      for permutation in gen_permutations(key):
        if permutation in D:
          G[value].append(D[permutation])

    with open(f"cache-{checksum}.pkl", "wb") as outfile:
      pickle.dump((D, G), outfile)

  neighbours = lambda n: G[n]
  return bfs(D[start], D[end], neighbours)


### tests
#
def test_algo(fn):
  tests = [
    ((["abc"], "abc", "abc"), 0),
    ((["abc", "abd"], "abc", "abd"), 1),
    ((["abc", "abd"], "abd", "abc"), 1),
    ((["abc", "abde"], "abc", "abde"), -1),  # not found
    ((["ABC", "abcd"], "abc", "abcd"), 1),  # case insensitive
    ((["abc", "abcd", "abcde"], "abc", "abcde"), 2),  # add letters
    ((["abc", "abcd", "abcde"], "abcde", "abc"), 2),  # remove letters
    ((["abcde", "abcee", "abeee"], "abcde", "abeee"), 2),  # remove letters
    (
      (
        ["abcdef", "abcdefg", "abcde", "abcdeg", "abc", "abcd"],
        "abcdef",
        "abc",
      ),
      3,
    ),  # remove letters
  ]
  for ins, outs in tests:
    if fn(*ins) != outs:
      print(*ins)
      print(outs, fn(*ins))
      assert fn(*ins) == outs

  ins, outs = tests[-1]
  nreps = 10000
  runtime = timeit.timeit(lambda: fn(*ins), number=nreps)
  doc = fn.__doc__.strip()
  print(f"{doc:60s} {nreps} reps  {runtime:.4f} s")


if __name__ == "__main__":
  test_algo(approach_1)
  test_algo(approach_2)
  test_algo(approach_3)
