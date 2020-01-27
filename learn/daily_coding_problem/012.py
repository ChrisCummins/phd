# This problem was asked by Facebook.
#
# Given a stream of elements too large to store in memory, pick a random
# element from the stream with uniform probability.
from random import random

from labm8.py import test


def f(X):
  e = None
  c = 1
  for i in X:
    if random() < (1 / c):
      e = i
    c += 1
  return e


def it(X):
  yield from X


def test_empty_itetartor():
  assert f([]) is None


def test_single_element():
  assert f([1]) is 1


@test.Flaky(reason="Stochastic")
def test_uniform_distribution():
  x = list(range(10))
  c = [0] * 10

  for _ in range(1000000):
    e = f(x)
    c[e] += 1

  for i, x in enumerate(c):
    x /= 1000000
    assert abs(x - 0.1) < 0.001


if __name__ == "__main__":
  test.Main()
