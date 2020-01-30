"""This problem was asked by Flexport.

Given a string s, rearrange the characters so that any two adjacent characters
are not the same. If this is not possible, return null.

For example, if s = yyz then return yzy. If s = yyy then return null.
"""
from typing import List
from typing import Optional
from typing import Tuple

from labm8.py import test


def F(s: str) -> Optional[str]:
  f = {}
  for c in s:
    f[c] = f.get(c, 0) + 1
  print(f)

  f: List[Tuple[str, int]] = sorted(
    ([k, v] for k, v in f.items()), key=lambda x: -x[1]
  )

  s = []
  i = 0
  while i < len(f):
    s.append(f[i][0])
    f[i][1] -= 1
    if i and f[i - 1][1]:
      i -= 1
    else:
      i += 1

  if f and f[-1][1]:
    print(f, s)
    return None

  return "".join(s)


def test_empty_string():
  assert F("") == ""


def test_single_char():
  assert F("a") == "a"


def test_double_char():
  assert F("ab") == "ab"


def test_double_char_duplicate():
  assert F("aa") is None


@test.XFail(reason="not solved")
def test_triple_char_ok():
  assert F("aab") == "aba"


@test.XFail(reason="not solved")
def test_bigger_string():
  assert F("aabbcc") == "abcabc"


if __name__ == "__main__":
  test.Main()
