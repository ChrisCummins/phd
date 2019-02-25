#!/usr/bin/env python3


def to_string(i: int) -> str:
  """ stringify a number, without spaces. 1 <= n <= 1000 """
  s = ""

  if i >= 1000:
    s += "onethousand"
    i = i % 1000

  if i >= 100:
    s += [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    ][(i // 100) - 1] + "hundred"
    i = i % 100
    if i:
      s += "and"

  if i >= 20:
    s += [
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    ][(i // 10) - 2]
    i = i % 10
  elif i > 10:
    s += [
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ][(i % 10) - 1]
    i = 0

  if i:
    s += [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
    ][i - 1]

  return s


def test(input, expected):
  actual = to_string(input)
  if actual != expected:
    print(f"failed: to_string({input}) = {actual}, expected {expected}")


test(1, "one")
test(2, "two")
test(10, "ten")
test(25, "twentyfive")
test(16, "sixteen")
test(100, "onehundred")
test(126, "onehundredandtwentysix")
test(113, "onehundredandthirteen")
test(999, "ninehundredandninetynine")
test(1000, "onethousand")

print(sum(len(to_string(i)) for i in range(1, 5 + 1)))
print(sum(len(to_string(i)) for i in range(1, 1000 + 1)))
