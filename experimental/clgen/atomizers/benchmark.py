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
import string
from collections import Counter
from timeit import timeit

import clgen
import numpy as np


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
    assert (isinstance(vocab, dict))

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
    try:
      return ''.join(list(map(lambda x: self.decoder[x], encoded)))
    except KeyError:
      raise VocabError

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
      return np.array(list(map(lambda x: self.vocab[x], text)))
    except KeyError:
      raise VocabError

  @staticmethod
  def from_text(text: str) -> Atomizer:
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
    super(GreedyAtomizer, self).__init__(*args, **kwargs)

  def atomize(self, text: str) -> np.array:
    atoms = set(self.atoms)
    indices = []
    try:
      i, j = 0, 0
      while j < len(text):
        j += 1
        if not any(x.startswith(text[i:j]) for x in atoms):
          indices.append(self.vocab[text[i:j - 1]])
          i = j - 1
          j -= 1
      indices.append(self.vocab[text[i:j]])
      return np.array(indices)
    except KeyError:
      raise VocabError

  @staticmethod
  def from_text(text: str) -> Atomizer:
    available_atoms = set([
        '__assert',
        '__attribute',
        '__builtin_astype',
        '__clc_fabs',
        '__clc_fma',
        '__constant',
        '__global',
        '__inline',
        '__kernel void',
        '__local',
        '__private',
        'abs',
        'atomic_add',
        'barrier',
        'break',
        'case',
        'char',
        'clamp',
        'const',
        'continue',
        'double',
        'else',
        'fabs',
        'false',
        'float',
        'get_global_id',
        'get_global_size',
        'get_local_id',
        'get_local_size',
        'get_num_groups',
        'global',
        'if',
        'inline',
        'int',
        'kernel',
        'local',
        'local',
        'long',
        'private',
        'restrict',
        'return',
        'short',
        'shuffle',
        'size_t',
        'sizeof',
        'sqrt',
        'struct',
        'switch',
        'true',
        'typedef',
        'u32',
        'uchar',
        'uint',
        'ulong',
        'unsigned',
        'void',
        'volatile',
        'while',
        'wide',
    ] + list(string.printable))

    atoms = set()

    i, j = 0, 0
    while j < len(text):
      j += 1
      if not any(x.startswith(text[i:j]) for x in available_atoms):
        atoms.add(text[i:j - 1])
        i = j - 1
        j -= 1
    atoms.add(text[i:j])

    vocab = dict(zip(sorted(atoms), range(len(atoms))))
    return GreedyAtomizer(vocab)


class FastAtomizer(Atomizer):
  """
  Greedy encoding for multi-characten modelling.
  """

  def __init__(self, *args, **kwargs):
    super(FastAtomizer, self).__init__(*args, **kwargs)

  def atomize(self, text: str) -> np.array:
    atoms = set(self.atoms)
    indices = []
    try:
      i, j = 0, 0
      while j < len(text):
        j += 1
        k = j - i
        if k == 1:
          if not any(x[0] == text[j] for x in atoms):
            indices.append(self.vocab[text[j - 1]])
          i = j - 1
          j -= 1
        elif not any(
            len(x) >= k and x[0] == text[i] and x[k - 1] == text[j - 1]
            for x in atoms):
          indices.append(self.vocab[text[i:j - 1]])
          i = j - 1
          j -= 1
      indices.append(self.vocab[text[i:j]])
      return np.array(indices)
    except KeyError:
      raise VocabError

  @staticmethod
  def from_text(text: str) -> Atomizer:
    available_atoms = set([
        '__assert',
        '__attribute',
        '__builtin_astype',
        '__clc_fabs',
        '__clc_fma',
        '__constant',
        '__global',
        '__inline',
        '__kernel void',
        '__local',
        '__private',
        'abs',
        'atomic_add',
        'barrier',
        'break',
        'case',
        'char',
        'clamp',
        'const',
        'continue',
        'double',
        'else',
        'fabs',
        'false',
        'float',
        'get_global_id',
        'get_global_size',
        'get_local_id',
        'get_local_size',
        'get_num_groups',
        'global',
        'if',
        'inline',
        'int',
        'kernel',
        'local',
        'local',
        'long',
        'private',
        'restrict',
        'return',
        'short',
        'shuffle',
        'size_t',
        'sizeof',
        'sqrt',
        'struct',
        'switch',
        'true',
        'typedef',
        'u32',
        'uchar',
        'uint',
        'ulong',
        'unsigned',
        'void',
        'volatile',
        'while',
        'wide',
    ])

    atoms = set()

    i, j = 0, 0
    while j < len(text):
      j += 1
      k = j - i
      if k >= 1 and not any(
          len(x) >= k and x[0] == text[i] and x[k - 1] == text[j - 1]
          for x in available_atoms):
        atoms.add(text[i:j - 1])
        i = j - 1
        j -= 1
    atoms.add(text[i:j])

    vocab = dict(zip(sorted(atoms), range(len(atoms))))
    return GreedyAtomizer(vocab)


with open("corpus.txt") as infile:
  data = infile.read()[:10]

chara = CharacterAtomizer.from_text(data)
fast = FastAtomizer.from_text(data)
greedy = GreedyAtomizer.from_text(data)


def main():
  print("start benchmark")

  print("CharacterAtomizer",
        timeit("chara.atomize(data)", "from __main__ import data, chara"))
  # print("GreedyAtomizer   ", timeit("greedy.atomize(data)",
  #                                   "from __main__ import data, greedy"))
  print("FastAtomizer     ",
        timeit("fast.atomize(data)", "from __main__ import data, fast"))


if __name__ == "__main__":
  main()
