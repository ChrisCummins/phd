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
from unittest import TestCase

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

    def test_deatomize(self):
        c = atomizer.CharacterAtomizer({'a': 1, 'b': 2, 'c': 3})
        self.assertEqual('abcabc', c.deatomize(c.atomize('abcabc')))
