from labm8 import app


# Exercise 1.3:
#
#     Given two strings, write a method to decide if one is a
#     permutation of the other.
#
# A permutation of a string must contain the exact same characters,
# but may contain them in any order. To check whether one string is a
# permutation of another, we can sort the characters within both
# strings in the same way, and check whether these sorted character
# arrays match. The efficiency of this algorithm will depend on the
# efficiency of the sorting algorithm used, as will the memory
# footprint (depends on whether the strings are sorted in place).
#
# An alternative implementation would be to check if the two strings
# have identical character counts, but this requires a priori
# knowledge of the size of the character sets.
def is_permutation(a, b):
  if len(a) != len(b):
    return False

  # Depending on how efficient this comparison is, we may want to
  # skip it.
  if a == b:
    return True

  return sorted(list(a)) == sorted(list(b))


def main(argv):
  del argv
  assert is_permutation("abc", "abc") == True
  assert is_permutation("abc", "abcd") == False
  assert is_permutation("abc", "cab") == True


if __name__ == "__main__":
  app.RunWithArgs(main)
