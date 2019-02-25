#!/usr/bin/env python3


def trinum(n):
  return n * (n + 1) // 2


def finding_missing_number(lst, start_at=0):
  """
  Given an unordered list of n numbers in range 0...n, return the missing
  value.

  Time: O(n)
  Space: O(1)
  """
  expected_sum = trinum(len(lst) + start_at) - trinum(start_at)
  actual_sum = sum(lst)
  return expected_sum - actual_sum + start_at


def test(actual, expected):
  if actual != expected:
    print(f"  actual: {actual}")
    print(f"expected: {expected}")
    print()


if __name__ == "__main__":
  test(finding_missing_number([]), 0)
  test(finding_missing_number([0]), 1)
  test(finding_missing_number([0, 1]), 2)
  test(finding_missing_number([0, 2]), 1)

  test(finding_missing_number([5, 6, 8], start_at=5), 7)
  test(finding_missing_number([6, 7, 8], start_at=5), 5)
  test(finding_missing_number([6, 7, 8], start_at=6), 9)
