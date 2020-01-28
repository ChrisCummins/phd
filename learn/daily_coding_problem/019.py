# This problem was asked by Microsoft.
#
# Given a dictionary of words and a string made up of those words (no spaces),
# return the original sentence in a list. If there is more than one possible
# reconstruction, return any of them. If there is no possible reconstruction,
# then return null.
#
# For example, given the set of words 'quick', 'brown', 'the', 'fox', and the
# string "thequickbrownfox", you should return ['the', 'quick', 'brown',
# 'fox'].
#
# Given the set of words 'bed', 'bath', 'bedbath', 'and', 'beyond', and the
# string "bedbathandbeyond", return either ['bed', 'bath', 'and', 'beyond] or
# ['bedbath', 'and', 'beyond'].
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from labm8.py import test


class ParseTree(object):
  """Helper data structure."""

  def __init__(
    self,
    value: Optional[str] = None,
    children: Optional[List["ParseTree"]] = None,
    terminal: bool = False,
  ):
    self.value = value
    self.children: Dict[str, ParseTree] = {}
    self.terminal = terminal
    for child in children or []:
      self.children[child.value] = child

  def __eq__(self, other):
    return (
      self.value == other.value
      and set(self.children.keys()) == set(other.children.keys())
      and self.terminal == other.terminal
    )


# Time: O(n), where n is size of parse tree
# Space: O(n)
def F(s: str, root: ParseTree) -> Optional[List[str]]:
  q: Tuple[int, int, ParseTree, List[str]] = [(0, 0, root, [])]

  while q:
    i, j, n, l = q[-1]
    del q[-1]

    if i >= len(s):
      if n.terminal:
        return l + [s[j:i]]
      elif n == root:
        return l or None
      else:
        return None

    if n.terminal:
      q.append((i, i, root, l + [s[j:i]]))

    for _, child in n.children.items():
      if child.value == s[i]:
        q.append((i + 1, j, child, l))


# Time: O(n * m), where n is the number of words, and m is the length of word
# Space: O(n), where n is the number of sum of word lengths
def MakeParseTree(words: str):
  root = ParseTree()
  for word in words.split():
    s = root
    for char in word:
      if char not in s.children:
        s.children[char] = ParseTree(char)
      s = s.children[char]
    if word:
      s.terminal = True
  return root


def test_MakeParseTree_empty():
  assert MakeParseTree("") == ParseTree()


def test_MakeParseTree_the():
  assert MakeParseTree("the") == ParseTree(
    children=[ParseTree("t", [ParseTree("h", [ParseTree("e", terminal=True)])])]
  )


def test_MakeParseTree_the_them():
  assert MakeParseTree("the them") == ParseTree(
    children=[
      ParseTree(
        "t",
        [
          ParseTree(
            "h",
            [ParseTree("e", [ParseTree("m", terminal=True)], terminal=True)],
          )
        ],
      )
    ]
  )


def test_empty_string():
  assert F("", MakeParseTree("abc")) is None


def test_empty_parse_tree():
  assert F("abc", MakeParseTree("")) is None


def test_the_quick_brown_fox():
  assert F("thequickbrownfox", MakeParseTree("quick brown the fox")) == [
    "the",
    "quick",
    "brown",
    "fox",
  ]


def test_the_quick_brown_fo():
  assert F("thequickbrownfo", MakeParseTree("quick brown the fox")) is None


def test_multi_word_ambiguity():
  assert F("thethethe", MakeParseTree("t th the thet")) == ["the", "the", "the"]


def test_the_meat_locker():
  assert F("themeatlocker", MakeParseTree("the them meat meatlock locker"))


if __name__ == "__main__":
  test.Main()
