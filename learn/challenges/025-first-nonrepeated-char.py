#!/usr/bin/env python3


def first_nonrepeated_char(s: str) -> str:
  d = dict()
  for c in s:
    d[c] = d.get(c, 0) + 1
  for c in s:
    if d[c] == 1:
      d[c] -= 1
      if d[c] == 0:
        return c
  return None


assert first_nonrepeated_char('aabbcddee') == 'c'
assert first_nonrepeated_char('aabbccddee') == None
assert first_nonrepeated_char('') == None
assert first_nonrepeated_char('abcde') == 'a'
assert first_nonrepeated_char('aabcde') == 'b'
assert first_nonrepeated_char('aabcdebcd') == 'e'
