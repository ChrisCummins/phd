"""My (Chris's) API for inst2vec codebase."""
import typing

import numpy as np
from absl import flags

from deeplearning.ncc import vocabulary

FLAGS = flags.FLAGS


def PreprocessLlvmBytecode(bytecode: str) -> str:
  """Pre-process an LLVM bytecode for encoding."""
  raise NotImplementedError


def EncodeLlvmBytecode(bytecode: str,
                       vocab: vocabulary.VocabularyZipFile) -> typing.List[int]:
  """Encode an LLVM bytecode to an array of vocabulary indices."""
  raise NotImplementedError


def EmbedEncoded(encoded: typing.List[int], embedding_matrix) -> np.ndarray:
  """Embed an array of vocabulary indices."""
  raise NotImplementedError


def Inst2Vec(bytecode: str, vocab: vocabulary.VocabularyZipFile,
             embedding) -> np.ndarry:
  """Transform an LLVM bytecode to an array of embeddings.

  Args:
    bytecode: The input bytecode.
    vocab: The vocabulary.
    embedding: The embedding.

  Returns:
    An array of embeddings.
  """
  embed = lambda x: EmbedEncoded(x, embedding)
  encode = lambda x: EncodeLlvmBytecode(x, vocab)
  return embed(encode(PreprocessLlvmBytecode(bytecode)))
