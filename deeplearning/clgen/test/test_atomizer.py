#
# Copyright 2016, 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
import pytest

from deeplearning.tmp_clgen import clgen.test as test


def test_CharacterAtomizer_from_text():
  c = clgen.CharacterAtomizer.from_text(clgen.Language.OPENCL, 'abcabc')

  assert c.indices == [0, 1, 2]
  assert c.atoms == ['a', 'b', 'c']
  assert c.vocab_size == 3


def test_CharacterAtomizer_atomize():
  c = clgen.CharacterAtomizer({'a': 1, 'b': 2, 'c': 3})
  assert list(c.atomize('abcabc')) == [1, 2, 3, 1, 2, 3]


def test_CharacterAtomizer_atomize_error():
  c = clgen.CharacterAtomizer({'a': 1, 'b': 2, 'c': 3})
  with pytest.raises(clgen.VocabError):
    c.atomize('abcdeabc')


def test_CharacterAtomizer_deatomize():
  c = clgen.CharacterAtomizer({'a': 1, 'b': 2, 'c': 3})
  assert c.deatomize([1, 2, 3, 1, 2, 3]) == 'abcabc'

  text = """
__kernel void A(__global float* a, const int b, const double c) {
  int d = get_global_id(0);
  if (b < get_global_size(0))
    a[d] *= (float)c;
}
"""
  c = clgen.CharacterAtomizer.from_text(clgen.Language.OPENCL, text)
  assert c.deatomize(c.atomize(text)) == text


# Greedy

def test_CharacterAtomizer_deatomize_error():
  c = clgen.CharacterAtomizer({'a': 1, 'b': 2, 'c': 3})
  with pytest.raises(clgen.VocabError):
    c.deatomize([1, 2, 5, 10, 0])


def test_GreedyAtomizer_tokeize1():
  test_vocab = {'abc': 1, 'a': 2, 'b': 3, 'ab': 4, 'c': 5, 'cab': 6, ' ': 7}
  test_in = 'abcababbaabcabcaabccccabcabccabcccabcabc'
  test_out = ['abc', 'ab', 'ab', 'b', 'a', 'abc', 'abc', 'a', 'abc', 'c', 'c', 'cab', 'cab', 'c',
    'cab', 'c', 'c', 'cab', 'cab', 'c']
  c = clgen.GreedyAtomizer(test_vocab)
  assert c.tokenize(test_in) == test_out


def test_GreedyAtomizer_tokenize2():
  test_vocab = {'volatile': 0, 'voletile': 1, 'vo': 2, ' ': 3, 'l': 4}
  test_in = 'volatile voletile vol '
  test_out = ['volatile', ' ', 'voletile', ' ', 'vo', 'l', ' ']
  c = clgen.GreedyAtomizer(test_vocab)
  assert c.tokenize(test_in) == test_out


def test_GreedyAtomizer_tokenize3():
  test_in = """\
__kernel void A(__global float* a, __global float* b, const int c) {
  int d = get_global_id(0);
  if (d < c) {
    a[d] = b[d] * 10.0f;
  }
}\
"""
  test_out = ['__kernel', ' ', 'void', ' ', 'A', '(', '__global', ' ', 'float', '*', ' ', 'a', ',',
    ' ', '__global', ' ', 'float', '*', ' ', 'b', ',', ' ', 'const', ' ', 'int', ' ', 'c', ')', ' ',
    '{', '\n', '  ', 'int', ' ', 'd', ' ', '=', ' ', 'get_global_id', '(', '0', ')', ';', '\n',
    '  ', 'if', ' ', '(', 'd', ' ', '<', ' ', 'c', ')', ' ', '{', '\n', '  ', '  ', 'a', '[', 'd',
    ']', ' ', '=', ' ', 'b', '[', 'd', ']', ' ', '*', ' ', '1', '0', '.', '0', 'f', ';', '\n', '  ',
    '}', '\n', '}']
  c = clgen.GreedyAtomizer.from_text(clgen.Language.OPENCL, test_in)
  assert c.tokenize(test_in) == test_out


def test_GreedyAtomizer_deatomize():
  test_in = """\
__kernel void A(__global float* a, __global float* b, const int c) {
  int d = get_global_id(0);
  if (d < c) {
    a[d] = b[d] * 10.0f;
  }
}\
"""
  c = clgen.GreedyAtomizer.from_text(clgen.Language.OPENCL, test_in)
  a = c.atomize(test_in)
  assert c.deatomize(a) == test_in


def test_GreedyAtomizer_from_text():
  test_in = """\
__kernel void A(__global float* a, __global float* b, const int c) {
  int d = get_global_id(0);
  if (d < c) {
    a[d] = b[d] * 10.0f;
  }
}\
"""
  tokens = ['__kernel', ' ', 'A', '(', ')', '__global', 'float', '*', 'a', '0', 'b', 'const', 'int',
    'c', '{', '}', '  ', 'd', 'get_global_id', ';', 'if', '<', '[', ']', 'f', '.', '1', '\n', '=',
    ',', 'void']
  c = clgen.GreedyAtomizer.from_text(clgen.Language.OPENCL, test_in)
  assert sorted(c.atoms) == sorted(tokens)
  assert c.vocab_size == len(tokens)
