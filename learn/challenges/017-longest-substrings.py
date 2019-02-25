#!/usr/bin/env python3


def approach1(str1: str, str2: str) -> str:
  """
  my first, naive approach

  T(n) = n ** 3
  S(n) = n
  """
  if not isinstance(str1, str) or not isinstance(str2, str):
    raise TypeError

  best = []

  for i in range(len(str1)):
    left_start = i
    right_start = 0
    while right_start < len(str2):
      while right_start < len(str2) and str2[right_start] != str1[left_start]:
        right_start += 1

      buf = []
      for left, right in zip(
          range(left_start, len(str1)), range(right_start, len(str2))):
        if str1[left] == str2[right]:
          buf.append(str1[left])
        else:
          break

      if len(buf) > len(best):
        best = buf

      right_start += 1

  return ''.join(best)


def approach2(str1: str, str2: str) -> str:
  """
  dynamic programming approach

  T(n) = n ** 2
  S(n) = n ** 2
  """
  if not isinstance(str1, str) or not isinstance(str2, str):
    raise TypeError

  num_rows = len(str1) + 1
  num_cols = len(str2) + 1

  dp = [[None] * num_cols for _ in range(num_rows)]

  for j in range(num_rows):
    for i in range(num_cols):
      if not j or not i:
        dp[j][i] = 0
      elif str1[j - 1] == str2[i - 1]:
        dp[j][i] = dp[j - 1][i - 1] + 1
      else:
        dp[j][i] = max(dp[j][i - 1], dp[j - 1][i])

  result = []

  j = num_rows - 1
  i = num_cols - 1
  while dp[j][i]:
    if dp[j][i] == dp[j][i - 1]:
      i -= 1
    elif dp[j][i] == dp[j - 1][i]:
      j -= 1
    elif dp[j][i] == dp[j - 1][i - 1] + 1:
      result.append(str1[j - 1])
      j, i = j - 1, i - 1
    else:
      raise Exception

  return ''.join(result[::-1])


def test_longest_substring(longest_substring) -> None:
  examples = [
      (('', ''), ''),
      (('a', 'a'), 'a'),
      (('abcdefghijk', 'dabc2'), 'abc'),
      (('abc', 'abc'), 'abc'),
      (('abc', 'ab'), 'ab'),
  ]

  try:
    longest_substring('abc', None)
    assert False
  except TypeError:
    pass

  try:
    longest_substring(4.3, 'a')
    assert False
  except TypeError:
    pass

  for ins, outs in examples:
    # print(ins)
    # print(outs)
    # print(longest_substring(*ins))
    # print()
    assert longest_substring(*ins) == outs


if __name__ == "__main__":
  test_longest_substring(approach1)
  test_longest_substring(approach2)
