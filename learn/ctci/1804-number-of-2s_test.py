from labm8 import app
from labm8 import test

FLAGS = app.FLAGS

MODULE_UNDER_TEST = None  # No coverage.


# Write a method to count the number of 2s that appear in all the numbers
# between 0 and n (inclusive).
#
# EXAMPLE
# Input: 25
# Output: 9 (2, 12, 20, 21, 22, 23, 24, 25)
#
def count_2s(n):
  count = 0
  for i in range(2, n + 1):
    while i:
      if i % 10 == 2:
        count += 1
      i = i // 10

  return count


def test_main():
  assert 9 == count_2s(25)


if __name__ == '__main__':
  test.Main()
