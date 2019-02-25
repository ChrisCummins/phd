#!/usr/bin/env python


def reverse_string(string):
  """
  T(n) = O(n)
  S(n) = O(n)
  """
  out = [None] * len(string)
  for i, c in enumerate(string[::-1]):
    out[i] = c
  return ''.join(out)


if __name__ == "__main__":
  assert reverse_string('') == ''
  assert reverse_string('abc') == 'cba'
  assert reverse_string('f o o') == 'o o f'
