#
# Copyright 2016 Chris Cummins <chrisc.101@gmail.com>.
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
"""
Converting & encoding text streams into vocabularies for machine learning.
"""
import numpy as np

from collections import Counter

import clgen


class VocabError(clgen.CLgenError):
    """A character sequence is not in the atomizer's vocab"""
    pass


class Atomizer(clgen.CLgenObject):
    """
    Atomizer.
    """
    def __init__(self, vocab: dict):
        """
        Arguments:
            vocab (dict): A dictionary of string -> integer mappings to use for
                atomizing text from atoms into indices.
        """
        assert(isinstance(vocab, dict))

        self.vocab = vocab
        self.decoder = dict((val, key) for key, val in vocab.items())

    @property
    def atoms(self):
        return list(self.vocab.keys())

    @property
    def indices(self):
        return list(self.vocab.values())

    @property
    def vocab_size(self):
        return len(self.vocab)

    def atomize(self, text: str) -> np.array:
        """
        Atomize a text into an array of vocabulary indices.

        Arguments:
            text (str): Input text.

        Returns:
            np.array: Indices into vocabulary for all atoms in text.
        """
        raise NotImplementedError("abstract class")

    def deatomize(self, encoded: np.array) -> str:
        """
        Translate atomized code back into a string.

        Arguments:
            encoded (np.array): Encoded vocabulary indices.

        Returns:
            str: Decoded text.
        """
        raise NotImplementedError("abstact class")

    @staticmethod
    def from_text(text: str):
        """
        Instantiate and specialize an atomizer from a corpus text.

        Arguments:
            text (str): Text corpus

        Returns:
            Atomizer: Specialized atomizer.
        """
        raise NotImplementedError("abstract class")


class CharacterAtomizer(Atomizer):
    """
    An atomizer for character-level syntactic modelling.
    """
    def __init__(self, *args, **kwargs):
        super(CharacterAtomizer, self).__init__(*args, **kwargs)

    def atomize(self, text: str) -> np.array:
        try:
            return np.array(list(map(self.vocab.get, text)))
        except KeyError:
            raise VocabError

    def deatomize(self, encoded: np.array) -> str:
        return ''.join(list(map(self.decoder.get, encoded)))

    @staticmethod
    def from_text(text: str) -> Atomizer:
        counter = Counter(text)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        atoms, _ = zip(*count_pairs)
        vocab = dict(zip(atoms, range(len(atoms))))
        return CharacterAtomizer(vocab)
