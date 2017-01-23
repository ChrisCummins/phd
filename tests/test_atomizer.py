#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
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
from unittest import TestCase, main, skip

from clgen import atomizer

class TestCharacterAtomizer(TestCase):
    def test_from_text(self):
        c = atomizer.CharacterAtomizer.from_text('abcabc')

        self.assertCountEqual(c.indices, [0, 1, 2])
        self.assertCountEqual(c.atoms, ['a', 'b', 'c'])
        self.assertEqual(c.vocab_size, 3)

    def test_atomize(self):
        c = atomizer.CharacterAtomizer({'a': 1, 'b': 2, 'c': 3})
        self.assertListEqual([1, 2, 3, 1, 2, 3], list(c.atomize('abcabc')))

    def test_atomize_error(self):
        c = atomizer.CharacterAtomizer({'a': 1, 'b': 2, 'c': 3})
        with self.assertRaises(atomizer.VocabError):
            c.atomize('abcdeabc')

    def test_deatomize(self):
        c = atomizer.CharacterAtomizer({'a': 1, 'b': 2, 'c': 3})
        self.assertEqual('abcabc', c.deatomize([1, 2, 3, 1, 2, 3]))

        text = """
__kernel void A(__global float* a, const int b, const double c) {
  int d = get_global_id(0);
  if (b < get_global_size(0))
    a[d] *= (float)c;
}
"""
        c = atomizer.CharacterAtomizer.from_text(text)
        self.assertEqual(text, c.deatomize(c.atomize(text)))

    def test_deatomize_error(self):
        c = atomizer.CharacterAtomizer({'a': 1, 'b': 2, 'c': 3})
        with self.assertRaises(atomizer.VocabError):
            c.deatomize([1, 2, 5, 10, 0])


class TestGreedyAtomizer(TestCase):
    def test_from_text(self):
        c = atomizer.GreedyAtomizer.from_text(
            '__kernel void A(__global float* a, const int b, const double c) {')
        self.assertEqual(c.vocab_size, 16)

    def test_atomize(self):
        c = atomizer.GreedyAtomizer({'abc': 1, 'a': 2, 'b': 3, 'ab': 4})
        self.assertListEqual([1, 4, 4, 3, 2], list(c.atomize('abcababba')))

    def test_datomize(self):
        c = atomizer.GreedyAtomizer({'abc': 1, 'a': 2, 'b': 3, 'ab': 4})
        self.assertEqual('abcababba', c.deatomize([1, 4, 4, 3, 2]))

        text = """
__kernel void A(__global float* a, const int b, const double c) {
  int d = get_global_id(0);
  if (b < get_global_size(0))
    a[d] *= (float)c;
}
"""
        c = atomizer.GreedyAtomizer.from_text(text)
        self.assertEqual(text, c.deatomize(c.atomize(text)))


if __name__ == "__main__":
    main()
