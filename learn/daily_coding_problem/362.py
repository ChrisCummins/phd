# This problem was asked by Twitter.
#
# A strobogrammatic number is a positive number that appears the same after
# being rotated 180 degrees. For example, 16891 is strobogrammatic.
#
# Create a program that finds all strobogrammatic numbers with N digits.
from math import ceil
from typing import Iterable

from labm8.py import test


# Questions:
#
# Q: Integers or strings?
# A: Integers
#
# Q: Numbers starting with zeros?
# A: No


def counters(n, m):
  """iterate over n counters in range [0, m)."""
  c = [0] * n
  c[0] = 1

  while True:
    yield c
    c[-1] += 1
    for i in range(n - 1, -1, -1):
      if c[i] == m:
        if not i:
          return
        c[i] = 0
        c[i - 1] += 1


# Time: O(5 ^ n)
# Space: O(n)
def make_strobogrammatic_numbers(n: int) -> Iterable[int]:
  assert n >= 1, "n must be >= 1"

  lv = [0, 1, 6, 8, 9]
  rv = [0, 1, 9, 8, 6]

  l = ceil(n / 2)
  r = n // 2

  for c in counters(l, len(lv)):
    # Mid values can't be a strobogrammatic number
    if l != r and lv[c[l - 1]] in {6, 9}:
      continue

    x = 0
    for i in range(l):
      L = lv[c[i]] * (10 ** (l + r - i - 1))  # l
      R = 0
      if not (l > r and i == l - 1):
        R = rv[c[i]] * (10 ** i)
      x += L + R

    yield x


def test_strobogrammatic_numbers_1():
  n = list(make_strobogrammatic_numbers(1))
  assert n == [1, 8]


def test_strobogrammatic_numbers_2():
  n = list(make_strobogrammatic_numbers(2))
  assert n == [11, 69, 88, 96]


def test_strobogrammatic_numbers_3():
  n = list(make_strobogrammatic_numbers(3))
  assert n == [101, 111, 181, 609, 619, 689, 808, 818, 888, 906, 916, 986]


def test_strobogrammatic_numbers_4():
  n = list(make_strobogrammatic_numbers(4))
  assert n == [
    1001,
    1111,
    1691,
    1881,
    1961,
    6009,
    6119,
    6699,
    6889,
    6969,
    8008,
    8118,
    8698,
    8888,
    8968,
    9006,
    9116,
    9696,
    9886,
    9966,
  ]


if __name__ == "__main__":
  test.Main()
