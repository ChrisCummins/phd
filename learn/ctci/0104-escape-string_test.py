from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


# Exercise 1.4:
#
#     Write a method to replace all spaces in a string with '%20'. You
#     may assume that the string has sufficient space at the end of
#     the string to hold the additional characters, and that you given
#     the "true" length of the string.
#
# First off, let's get the obvious way over and done with:
def escape_spaces_regexp(string, strlen):
  return string.replace(' ', '%20')


# Of course, this misses the purpose of the question by operating on a
# string, not a character array. Implementing a proper character array
# solution requires two passes, and operates in O(n) time, with O(1)
# space complexity (it operates in place):
def escape_spaces(string, strlen):
  # The first pass is to ascertain the number of ' ' characters
  # which need escaping, which can then be used to calculate the new
  # length of the escaped string.
  spaces_count = 0
  for c in list(string[:strlen]):
    if c == ' ':
      spaces_count += 1

  new_strlen = strlen + 2 * spaces_count
  # Now that we know the new string length, we work from front to
  # back, copying original string characters into their new
  # positions. If we come across a ' ' character, it is replaced
  # with the padded equivalent.
  #
  # We can make a cheeky optimisation because we know that if the
  # escaped string length and the original string length are equal,
  # then there are no characters which need escaping, so we don't
  # need to do anything.
  if new_strlen != strlen:
    for i in range(strlen - 1, -1, -1):
      new_strlen -= 1
      if string[i] == ' ':
        string[new_strlen - 2] = '%'
        string[new_strlen - 1] = '2'
        string[new_strlen] = '0'
        new_strlen -= 2
      else:
        string[new_strlen] = string[i]

  return string


def test_main():
  assert escape_spaces_regexp("Hello, the World!",
                              17) == "Hello,%20the%20World!"
  assert (''.join(escape_spaces(list("Hello, the World!        "),
                                17)) == "Hello,%20the%20World!    ")


if __name__ == "__main__":
  test.Main()
