"""This file contains the definition of atomizers.

An atomizer converts a block of text into a sequence of vocbulary tokens.
"""
import typing
from collections import Counter

import humanize
import numpy as np
from absl import logging

from deeplearning.clgen import errors
from lib.labm8 import labdate


class AtomizerBase(object):
  """The base class for implementing atomizers."""

  def __init__(self, vocab: typing.Dict[str, int]):
    """Instantiate an atomizer.

    Args:
      vocab: A dictionary of mappings from character sequences (atoms) into
        indices.

    Raises:
      TypeError: If vocab is not a dictionary.
      InvalidVocab: If the dictionary of mappings includes any duplicate values.
    """
    self.vocab = vocab
    self._UpdateVocabulary()

  @property
  def atoms(self) -> typing.List[str]:
    """A list of atoms in the vocabulary."""
    return list(sorted(self.vocab.keys()))

  @property
  def indices(self) -> typing.List[int]:
    """A list of vocabulary indices."""
    return list(sorted(self.vocab.values()))

  def _UpdateVocabulary(self) -> None:
    """Private method which must be called if vocab is modified."""
    if not isinstance(self.vocab, dict):
      raise TypeError('vocabulary must be a dict')

    # Each atom and index must be unique to ensure deterministic encoding.
    if len(set(self.vocab.keys())) != len(self.vocab):
      raise errors.InvalidVocab('all atoms must be unique')
    if len(set(self.vocab.values())) != len(self.vocab):
      raise errors.InvalidVocab('all indices must be unique')

    self.vocab_size = len(self.vocab)
    self.decoder = dict((val, key) for key, val in self.vocab.items())

  def AtomizeString(self, text: str) -> np.array:
    """Atomize a text into an array of vocabulary indices.

    Args:
      text: Input text.

    Returns:
      An array of indices into vocabulary for all atoms in text.

    Raises:
      VocabError: If the input text contains elements not in the vocabulary.
    """
    raise NotImplementedError("abstract class")

  def TokenizeString(self, text: str) -> typing.List[str]:
    """Split the text into atoms, but do not encode to indices.

    Args:
      text: Input text.

    Returns:
      A list of tokens.
    """
    indices = self.AtomizeString(text)
    return list(map(lambda x: self.decoder[x], indices))

  def DeatomizeIndices(self, encoded: np.array) -> str:
    """Translate atomized code back into a string.

    Args:
      encoded: An nparray of encoded vocabulary indices.

    Returns:
      The decoded text.
    """
    try:
      return ''.join(list(map(lambda x: self.decoder[x], encoded)))
    except KeyError:
      raise errors.VocabError

  @classmethod
  def FromText(cls, text: str) -> 'AtomizerBase':
    """Instantiate and specialize an atomizer from a corpus text.

    Args:
      text: Text corpus

    Returns:
      An atomizer instance.
    """
    raise NotImplementedError("abstract class")


class AsciiCharacterAtomizer(AtomizerBase):
  """An atomizer for character-level syntactic modelling."""

  def AtomizeString(self, text: str) -> np.array:
    """Atomize a text into an array of vocabulary indices.

    Args:
      text: Input text.

    Returns:
      An array of indices into vocabulary for all atoms in text.
    """
    try:
      return np.array(list(map(lambda x: self.vocab[x], text)), dtype=np.int32)
    except KeyError:
      raise errors.VocabError

  def __repr__(self) -> str:
    return f'AsciiCharacterAtomizer[{self.vocab_size} chars]'

  @classmethod
  def FromText(cls, text: str) -> 'AsciiCharacterAtomizer':
    """Instantiate and an atomizer from a corpus text.

    Args:
      text: Text corpus.

    Returns:
      An atomizer instance.
    """
    counter = Counter(text)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    atoms, _ = zip(*count_pairs)
    vocab = dict(zip(atoms, range(len(atoms))))
    return AsciiCharacterAtomizer(vocab)


class GreedyAtomizer(AtomizerBase):
  """A greedy atomizer supports multi-character tokens."""

  def __init__(self, vocab: typing.Dict[str, int], determine_chars=False):
    self.determine_chars = determine_chars
    super(GreedyAtomizer, self).__init__(vocab)

    multichars = set(k for k in self.atoms if len(k) > 1)
    first_chars = set(a[0] for a in multichars)
    self.lookup = dict(
        (c, [a for a in multichars if a[0] == c]) for c in first_chars)

  def AtomizeString(self, text: str) -> np.array:
    """Atomize a text into an array of vocabulary indices.

    Args:
      text: Input text.

    Returns:
      An array of indices into vocabulary for all atoms in text.
    """

    def _AddToVocab(token: str) -> int:
      """Add a token to the vocabulary and return its index."""
      if self.determine_chars and token not in self.vocab:
        max_index = max(self.vocab.values())
        self.vocab[token] = max_index + 1
      return self.vocab[token]

    indices = []
    i = 0
    j = 2
    try:
      while i < len(text):
        if self.lookup.get(text[i]):
          if j <= len(text) and any(
              x.startswith(text[i:j]) for x in self.lookup[text[i]]):
            j += 1
          else:
            while j > i + 1:
              if any(x == text[i:j] for x in self.lookup[text[i]]):
                indices.append(self.vocab[text[i:j]])
                i = j
                j += 2
                break
              else:
                j -= 1
            else:
              indices.append(_AddToVocab(text[i]))
              i += 1
              j += 2
        else:
          indices.append(_AddToVocab(text[i]))
          i += 1
          j += 2
    except KeyError:
      raise errors.VocabError

    if self.determine_chars:
      self._UpdateVocabulary()

    return np.array(indices, dtype=np.int32)

  def __repr__(self) -> str:
    return f'GreedyAtomizer[{self.vocab_size} tokens]'

  @classmethod
  def FromText(cls, text: str, atoms: typing.Set[str]) -> 'GreedyAtomizer':
    """Instantiate and an atomizer from a corpus text.

    Args:
      text: Text corpus
      atoms: A list of multi-character tokens.

    Returns:
      An atomizer instance.
    """
    if not atoms:
      raise errors.UserError('No atoms specified')

    start_time = labdate.MillisecondsTimestamp()
    # Instantiate a greedy atomizer using the full vocabulary.
    full_vocab = dict(zip(atoms, range(len(atoms))))
    c = GreedyAtomizer(full_vocab, determine_chars=True)
    # Derive the subset of the vocabulary required to encode the given text.
    tokens = sorted(list(set(c.TokenizeString(text))))
    vocab_subset = dict(zip(tokens, range(len(tokens))))
    end_time = labdate.MillisecondsTimestamp()
    logging.info('Derived vocabulary of size %s from %s tokens in %s',
                 humanize.intcomma(len(tokens)), humanize.intcomma(len(atoms)),
                 humanize.intcomma(end_time - start_time))
    # Return a new atomizer using the subset vocabulary.
    return GreedyAtomizer(vocab_subset)
