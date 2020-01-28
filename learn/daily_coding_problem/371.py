# This problem was asked by Google.
#
# You are given a series of arithmetic equations as a string, such as:
#
# y = x + 1
# 5 = x + 3
# 10 = z + y + 2
#
# The equations use addition only and are separated by newlines. Return a
# mapping of all variables to their values. If it's not possible, then return
# null. In this example, you should return:
#
# {
#   x: 2,
#   y: 3,
#   z: 5
# }
import heapq
from string import ascii_letters
from string import digits
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from labm8.py import test


class Lexer(object):
  def __init__(self, s: str):
    self.s = s
    self.c = 0
    self.equations = [[[], []]]

  def Lex(self) -> List[List[Union[str, int]]]:
    """Lex an input into a list of lists, where each list is a [<lhs>, <rhs>]
    sequence of variables and integers.
    """
    self.lhs()
    if not self.equations[-1][0]:
      del self.equations[-1]
    return self.equations

  def newline(self):
    if self.s[self.c : self.c + 1] == "\n":
      self.c += 1
      self.equations.append([[], []])
    self.lhs()

  def lhs(self):
    if self.s[self.c : self.c + 1] in set(digits):
      self.lhsd()
    elif self.s[self.c : self.c + 1] in set(ascii_letters):
      self.lhsv()
    elif self.s[self.c : self.c + 1] == "\n":
      self.c += 1
      self.equations.append([[], []])
      self.lhs()

  def lhsd(self):
    sc = self.c
    while self.s[self.c : self.c + 1] in set(digits):
      self.c += 1
    self.equations[-1][0].append(int(self.s[sc : self.c]))
    if self.s[self.c : self.c + 3] == " = ":
      self.c += 3
      self.rhs()
    elif self.s[self.c : self.c + 3] == " + ":
      self.c += 3
      self.lhs()
    else:
      self.RaiseParseError("expected ' = '")

  def lhsv(self):
    sc = self.c
    while self.s[self.c : self.c + 1] in set(ascii_letters):
      self.c += 1
    self.equations[-1][0].append(self.s[sc : self.c])
    if self.s[self.c : self.c + 3] == " = ":
      self.c += 3
      self.rhs()
    elif self.s[self.c : self.c + 3] == " + ":
      self.c += 3
      self.lhs()
    else:
      self.RaiseParseError("expected ' = '")

  def rhs(self, eol_ok=False):
    if self.s[self.c : self.c + 1] in set(digits):
      self.rhsd()
    elif self.s[self.c : self.c + 1] in set(ascii_letters):
      self.rhsv()
    elif eol_ok and self.s[self.c : self.c + 1] == "\n":
      self.newline()
    elif eol_ok and self.c >= len(self.s):
      return
    else:
      self.RaiseParseError("expected number or variable")

  def rhsd(self):
    sc = self.c
    while self.s[self.c : self.c + 1] in set(digits):
      self.c += 1
    self.equations[-1][1].append(int(self.s[sc : self.c]))
    if self.s[self.c : self.c + 3] == " + ":
      self.c += 3
      self.rhs()
    elif self.s[self.c : self.c + 1] == "\n":
      self.newline()
    elif self.c >= len(self.s):
      return
    else:
      self.RaiseParseError("expected ' + '")

  def rhsv(self):
    sc = self.c
    while self.s[self.c : self.c + 1] in set(ascii_letters):
      self.c += 1
    self.equations[-1][1].append(self.s[sc : self.c])
    if self.s[self.c : self.c + 3] == " + ":
      self.c += 3
      self.rhs()
    elif self.s[self.c : self.c + 1] == "\n":
      self.newline()
    elif self.c >= len(self.s):
      return
    else:
      self.RaiseParseError("expected ' + '")

  def RaiseParseError(self, msg):
    raise ValueError(f"At character {self.c}: {msg}")


def Lex(s) -> List[List[Union[str, int]]]:
  return Lexer(s).Lex()


# Time: O(n)
# Space: O(n)
def SimplifyEquation(equation: List[List[Union[str, int]]]):
  """Simplify equation:

    * Sum integers in LHS and RHS lists to element 0.
      E.g. ['x', 15, 2, 3] -> [20, 'x'].
    * If LHS and RHS contain a single unknown, move it to LHS.
      E.g. [15, 20], ['x', 2] -> [0, 'x'], [33]
  """

  def Simplify(components):
    t = 0
    v = []
    for c in components:
      if isinstance(c, int):
        t += c
      else:
        v.append(c)
    return [t] + v

  equation[0] = Simplify(equation[0])
  equation[1] = Simplify(equation[1])

  if len(equation[0]) + len(equation[1]) > 3:
    return equation
  elif len(equation[1]) == 2:
    # Transform: 10 = 5 + x -> x = 5
    equation[0][0] -= equation[1][0]
    equation[1][0] = 0
    equation[0], equation[1] = equation[1], equation[0]
  elif len(equation[0]) == 2:
    # Transform: x + 5 = 10 -> x + 0 = 5
    equation[1][0] -= equation[0][0]
    equation[0][0] = 0

  assert len(equation[0]) == 2
  assert equation[0][0] == 0
  assert len(equation[1]) in {1, 2}

  return equation


# Time: O(n ^ 2)
# Space: O(n)
def Solve(s: str) -> Optional[Dict[str, int]]:
  solved = {}
  equations = Lex(s)
  equations = [SimplifyEquation(e) for e in equations]

  # Sort the equations with fewest unknowns first.
  equations = sorted(equations, key=lambda x: len(x[1]))

  # A heap queue
  q = [[(len(x[0]) + len(x[1])), x] for x in equations]
  heapq.heapify(q)

  while q:
    _, (lhs, rhs) = heapq.heappop(q)
    if len(rhs) == 1:
      assert len(lhs) == 2

      solved[lhs[1]] = rhs[0]

      changed = False
      for _, oq in q:
        olhs, orhs = oq
        for k in range(len(orhs) - 1, 0, -1):
          if orhs[k] == lhs[1]:
            orhs[0] += rhs[0]
            del orhs[k]
            changed = True
      if changed:
        for e in q:
          e[1] = SimplifyEquation(e[1])
          e[0] = len(e[1][0]) + len(e[1][1])
        heapq.heapify(q)
    else:
      return None

  return solved


def test_empty_string():
  assert Lex("") == []


def test_left_hand_side_digits_only():
  with test.Raises(ValueError):
    Lex("1234")


def test_left_hand_side_variable_only():
  with test.Raises(ValueError):
    Lex("abc")


def test_invalid_equal_sign():
  with test.Raises(ValueError):
    Lex("abc = ")


def test_var_eq_var():
  assert Lex("abc = cde") == [[["abc"], ["cde"]]]


def test_num_eq_var():
  assert Lex("5 = cde") == [[[5], ["cde"]]]


def test_num_eq_var_plus_num():
  assert Lex("5 = cde + 10") == [[[5], ["cde", 10]]]


def test_multiline_input():
  assert Lex("x = 5 + 10\ny = 5 + 10") == [[["x"], [5, 10]], [["y"], [5, 10]]]


def test_solve_single_equation_with_left_hand_side():
  assert Solve("x = 5 + 10") == {"x": 15}


def test_solve_single_equation_with_right_hand_side():
  assert Solve("5 + 1 + 6 = x") == {"x": 12}


def test_solve_single_equation_with_both_sides():
  assert Solve("10 + x + 5 = 50 + 5") == {"x": 40}


def test_unsolvable_single_equation():
  assert Solve("x = y") is None


def test_solve_two_equations_without_overlap():
  assert Solve("x = 5 + 10\n23 = y + 3") == {"x": 15, "y": 20}


def test_solve_two_equations_with_substitution():
  assert Solve("x = 5 + y\ny = 10 + 15") == {"x": 30, "y": 25}


def test_solve_example_input():
  assert Solve("y = x + 1\n5 = x + 3\n10 = z + y + 2") == {
    "x": 2,
    "y": 3,
    "z": 5,
  }


def test_solve_two_subsitutions():
  assert Solve("y = x + x + 1\n2 = x") == {"x": 2, "y": 5}


def test_solve_messy():
  assert Solve("x = 5\ny = x + x + x\nz = y + 5") == {
    "x": 5,
    "y": 15,
    "z": 20,
  }


if __name__ == "__main__":
  test.Main()
