from labm8 import app
from labm8 import test

FLAGS = app.FLAGS

MODULE_UNDER_TEST = None  # No coverage.


# Exercise 1.1:
#
#     Implement an algorithm to determine if a string has all unique
#     characters. What if you cannot use additional data structures?
#
# This is a variation on the "count the number of times each character
# appears in a string", except that we only need two store two
# possible values: character present, or character not present. On the
# first occurrence of a character recurring, we can return false.
#
# The solution we've implemented operates in O(n) time, with a best
# case time of O(1) (when string length > 256). It operates with O(1)
# space complexity.
#
def characters_are_unique(string):
  # This is a crafty optimisation: since we know the character space
  # is 256 (the number of characters in the ASCII character set),
  # then by definition any string that is longer than this *must*
  # include duplicate characters:
  if len(string) > 256:
    return False

  # We need 256 bytes in order to store our set of character
  # occurrences. If we were interested in bit twiddling, we could
  # reduce the memory footprint by 7/8 by using an array of bytes of
  # using individual bits to represent each character:
  characters = [False] * 256

  for c in string:
    val = ord(c)

    if characters[val] == True:
      return False
    else:
      characters[val] = True

  return True


def test_main():
  assert characters_are_unique("abcdefg") == True
  assert characters_are_unique("abcdefga") == False


if __name__ == "__main__":
  test.Main()
