from labm8.py import test

MODULE_UNDER_TEST = None  # No coverage.

# This problem was asked by Jane Street.
#
# cons(a, b) constructs a pair, and car(pair) and cdr(pair) returns the first
# and last element of that pair. For example, car(cons(3, 4)) returns 3, and
# cdr(cons(3, 4)) returns 4.

# Given this implementation of cons:


def cons(a, b):
  def pair(f):
    return f(a, b)

  return pair


# Implement car and cdr.


def car(pair):
  return pair(lambda a, b: a)


def cdr(pair):
  return pair(lambda a, b: b)


def identity(*args):
  """Helper function which constructs a tuple of args."""
  return args


def test_cons():
  """Test that cons returns a function that takes a pair."""
  assert cons(3, 4)(identity) == (3, 4)


def test_car():
  """Test that car returns the first element of a pair."""
  assert car(cons(3, 4)) == 3


def test_cdr():
  """Test that car returns the last element of a pair."""
  assert cdr(cons(3, 4)) == 4


if __name__ == "__main__":
  test.Main()
