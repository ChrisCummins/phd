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
"""
Converting & encoding text streams into vocabularies for machine learning.
"""
import numpy as np
import string

from collections import Counter
from typing import Dict, List

import clgen


# Taken from the C99 spec, OpenCL spec 1.2, and bag-of-words analysis of
# GitHub corpus:
OPENCL_ATOMS = set([
    '  ',
    '__assert',
    '__attribute',
    '__builtin_astype',
    '__clc_fabs',
    '__clc_fma',
    '__constant',
    '__global',
    '__inline',
    '__kernel',
    '__local',
    '__private',
    '__read_only',
    '__read_write',
    '__write_only',
    '*/',
    '/*',
    '//',
    'abs',
    'alignas',
    'alignof',
    'atomic_add',
    'auto',
    'barrier',
    'bool',
    'break',
    'case',
    'char',
    'clamp',
    'complex',
    'const',
    'constant',
    'continue',
    'default',
    'define',
    'defined',
    'do',
    'double',
    'elif',
    'else',
    'endif',
    'enum',
    'error',
    'event_t',
    'extern',
    'fabs',
    'false',
    'float',
    'for',
    'get_global_id',
    'get_global_size',
    'get_local_id',
    'get_local_size',
    'get_num_groups',
    'global',
    'goto',
    'half',
    'if',
    'ifdef',
    'ifndef',
    'image1d_array_t',
    'image1d_buffer_t',
    'image1d_t',
    'image2d_array_t',
    'image2d_t',
    'image3d_t',
    'imaginary',
    'include',
    'inline',
    'int',
    'into',
    'kernel',
    'line',
    'local',
    'long',
    'noreturn',
    'pragma',
    'private',
    'quad',
    'read_only',
    'read_write',
    'register',
    'restrict',
    'return',
    'sampler_t',
    'short',
    'shuffle',
    'signed',
    'size_t',
    'sizeof',
    'sqrt',
    'static',
    'struct',
    'switch',
    'true',
    'typedef',
    'u32',
    'uchar',
    'uint',
    'ulong',
    'undef',
    'union',
    'unsigned',
    'void',
    'volatile',
    'while',
    'wide',
    'write_only',
])


class VocabError(clgen.CLgenError):
    """A character sequence is not in the atomizer's vocab"""
    pass


class InvalidVocab(VocabError):
    """An invalid atomizer vocabulary"""
    pass


class Atomizer(clgen.CLgenObject):
    """
    Abstract base class for atomizers.
    """
    def __init__(self, vocab: Dict[str, int]):
        """
        Arguments:
            vocab (Dict[str, int]): A dictionary of mappings from character
                sequences (atoms) into indices.

        Raises:
            TypeError: If vocab is not a dictionary.
            InvalidVocab: If the dictionary of mappings includes any duplicate
                values.
        """
        self.vocab = vocab
        self._vocab_update()

    @property
    def atoms(self) -> List[str]:
        return list(sorted(self.vocab.keys()))

    @property
    def indices(self) -> List[int]:
        return list(sorted(self.vocab.values()))

    def _vocab_update(self) -> None:
        """ call this when vocab is modified """
        if not isinstance(self.vocab, dict):
            raise TypeError('vocabulary must be a dict')

        # Each atom and index must be unique to ensure deterministic encoding.
        if len(set(self.vocab.keys())) != len(self.vocab):
            raise InvalidVocab('all atoms must be unique')
        if len(set(self.vocab.values())) != len(self.vocab):
            raise InvalidVocab('all indices must be unique')

        self.vocab_size = len(self.vocab)
        self.decoder = dict((val, key) for key, val in self.vocab.items())

    def atomize(self, text: str) -> np.array:
        """
        Atomize a text into an array of vocabulary indices.

        Arguments:
            text (str): Input text.

        Returns:
            np.array: Indices into vocabulary for all atoms in text.
        """
        raise NotImplementedError("abstract class")

    def tokenize(self, text: str) -> List[str]:
        """
        Split the text into atoms, but do not encode to indices.

        Arguments:
            text (str): Input text.

        Returns:
            list of str: Atom strings.
        """
        indices = self.atomize(text)
        return list(map(lambda x: self.decoder[x], indices))

    def deatomize(self, encoded: np.array) -> str:
        """
        Translate atomized code back into a string.

        Arguments:
            encoded (np.array): Encoded vocabulary indices.

        Returns:
            str: Decoded text.
        """
        try:
            return ''.join(list(map(lambda x: self.decoder[x], encoded)))
        except KeyError:
            raise VocabError

    @staticmethod
    def from_text(text: str) -> 'Atomizer':
        """
        Instantiate and specialize an atomizer from a corpus text.

        Arguments:
            text (str): Text corpus

        Returns:
            Atomizer: Specialized atomizer.

        Examples:
            >>> a = CharacterAtomizer.from_text('abcdefg')
            >>> b = a.atomize('abcd')
            >>> len(b)
            4
            >>> a.deatomize(b)
            'abcd'
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
            return np.array(list(map(lambda x: self.vocab[x], text)))
        except KeyError:
            raise VocabError

    def __repr__(self) -> str:
        return "CharacterAtomizer[{n} chars]".format(n=self.vocab_size)

    @staticmethod
    def from_text(text: str) -> 'CharacterAtomizer':
        counter = Counter(text)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        atoms, _ = zip(*count_pairs)
        vocab = dict(zip(atoms, range(len(atoms))))
        return CharacterAtomizer(vocab)


class GreedyAtomizer(Atomizer):
    """
    Greedy encoding for multi-characten modelling.
    """
    def __init__(self, *args, **kwargs):
        self.determine_chars = kwargs.pop("determine_chars", False)
        super(GreedyAtomizer, self).__init__(*args, **kwargs)

        multichars = set(k for k in self.atoms if len(k) > 1)
        first_chars = set(a[0] for a in multichars)
        self.lookup = dict((c, [a for a in multichars if a[0] == c])
                           for c in first_chars)

    def atomize(self, text: str) -> np.array:
        def _add_to_vocab(token: str):
            if self.determine_chars and token not in self.vocab:
                maxind = max(self.vocab.values())
                self.vocab[token] = maxind + 1

            return self.vocab[token]

        indices = []
        i = 0
        j = 2
        try:
            while i < len(text):
                if self.lookup.get(text[i]):
                    if j <= len(text) and any(x.startswith(text[i:j])
                                              for x in self.lookup[text[i]]):
                        j += 1
                    else:
                        while j > i + 1:
                            if any(x == text[i:j]
                                   for x in self.lookup[text[i]]):
                                indices.append(self.vocab[text[i:j]])
                                i = j
                                j = j + 2
                                break
                            else:
                                j -= 1
                        else:
                            indices.append(_add_to_vocab(text[i]))
                            i = i + 1
                            j = j + 2
                else:
                    indices.append(_add_to_vocab(text[i]))
                    i = i + 1
                    j = j + 2
        except KeyError:
            raise VocabError

        if self.determine_chars:
            self._vocab_update()

        return np.array(indices)

    def __repr__(self):
        return "GreedyAtomizer[{n} tokens]".format(n=self.vocab_size)

    @staticmethod
    def from_text(text: str) -> Atomizer:
        opencl_vocab = dict(zip(OPENCL_ATOMS, range(len(OPENCL_ATOMS))))
        c = GreedyAtomizer(opencl_vocab, determine_chars=True)

        tokens = sorted(list(set(c.tokenize(text))))
        vocab = dict(zip(tokens, range(len(tokens))))
        return GreedyAtomizer(vocab)
