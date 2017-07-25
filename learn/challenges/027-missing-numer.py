#!/usr/bin/env python3

def finding_missing_number(lst):
    """
    Given an unordered list of n numbers in range 0...n, return the missing
    value.

    Time: O(n)
    Space: O(1)
    """
    n = len(lst)
    expected_sum = (n * (n + 1)) // 2
    actual_sum = sum(lst)
    return expected_sum - actual_sum


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
