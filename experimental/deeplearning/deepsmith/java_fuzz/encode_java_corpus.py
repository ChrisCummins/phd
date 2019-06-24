"""Preprocess an exported database of Java methods."""
import time

import datetime
import numpy as np
import typing

from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.corpuses import preprocessed
from deeplearning.clgen.proto import internal_pb2
from labm8 import app
from labm8 import bazelutil
from labm8 import humanize
from labm8 import pbutil
from labm8 import sqlutil

FLAGS = app.FLAGS
app.DEFINE_database(
    'input', preprocessed.PreprocessedContentFiles,
    'sqlite:////var/phd/experimental/deeplearning/deepsmith/java_fuzz/preprocessed.db',
    'URL of the database of preprocessed Java methods.')
app.DEFINE_database(
    'output', encoded.EncodedContentFile,
    'sqlite:////var/phd/experimental/deeplearning/deepsmith/java_fuzz/encoded.db',
    'URL of the database to add encoded methods to.')

LEXER_WORKER = bazelutil.DataPath(
    'phd/deeplearning/clgen/corpuses/lexer/lexer_worker')

JAVA_TOKENS = [
    "abstract",
    "assert",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "class",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extends",
    "final",
    "finally",
    "float",
    "for",
    "if",
    "implements",
    "import",
    "instanceof",
    "int",
    "interface",
    "long",
    "native",
    "new",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "short",
    "static",
    "strictfp",
    "super",
    "switch",
    "synchronized",
    "this",
    "throw",
    "throws",
    "transient",
    "try",
    "void",
    "volatile",
    "while",
]


def EncodePreprocessedContentFiles(
    srcs: typing.List[str], vocabulary: typing.Dict[str, int]
) -> typing.Union[typing.List[np.ndarray], typing.Dict[str, int]]:
  message = internal_pb2.LexerBatchJob(
      input=[internal_pb2.LexerJob(string=s) for s in srcs],
      candidate_token=JAVA_TOKENS,
      vocabulary=vocabulary,
  )
  pbutil.RunProcessMessageInPlace([LEXER_WORKER], message, timeout_seconds=3600)
  return ([np.ndarray(j.token) for j in message.input], dict(
      message.vocabulary))


def EmbedVocabInMetaTable(session: sqlutil.Session,
                          vocabulary: typing.Dict[str, int]):
  q = session.query(encoded.Meta).filter(encoded.Meta.key.like('vocab_%'))
  q.delete(synchronize_session=False)

  session.add(encoded.Meta(key='vocab_size', value=str(len(vocabulary))))
  session.add_all(
      [encoded.Meta(key=f'vocab_{v}', value=k) for k, v in vocabulary.items()])


def GetVocabFromMetaTable(session: sqlutil.Session) -> typing.Dict[str, int]:
  q = session.query(encoded.Meta.value).filter(encoded.Meta.key == 'vocab_size')
  if not q.first():
    return {}

  vocab_size = int(q.one()[0])
  q = session.query(encoded.Meta.value)
  return {
      q.filter(encoded.Meta.key == f'vocab_{i}').one()[0]: i
      for i in range(vocab_size)
  }


def EncodePreprocessedFiles(
    cfs: typing.List[preprocessed.PreprocessedContentFile],
    vocab: typing.Dict[str, int]
) -> typing.Union[typing.List[encoded.EncodedContentFile], typing.
                  Dict[str, int]]:
  start_time = time.time()
  encodeds, vocab = EncodePreprocessedContentFiles([cf.text for cf in cfs],
                                                   vocab)
  wall_time_ms = ((time.time() - start_time) * 1000)

  per_item_wall_time_ms = int(wall_time_ms / len(encodeds))
  pp_cfs = [
      encoded.EncodedContentFile(id=cf.id,
                                 data=enc.tostring(),
                                 tokencount=len(enc),
                                 encoding_time_ms=per_item_wall_time_ms,
                                 wall_time_ms=per_item_wall_time_ms,
                                 date_added=datetime.datetime.now())
      for cf, enc in zip(cfs, encodeds)
  ]
  return pp_cfs, vocab


def EncodeFiles(input_session, output_session, batch_size):
  done = output_session.query(encoded.EncodedContentFile.id)
  done = {x[0] for x in done}
  all_files = input_session.query(preprocessed.PreprocessedContentFile)
  to_encode = all_files\
      .filter(preprocessed.PreprocessedContentFile.preprocessing_succeeded == True)\
      .filter(~preprocessed.PreprocessedContentFile.id.in_(done))\
      .limit(batch_size)
  app.Log(1, 'Encoding %s of %s content files',
          humanize.Commas(to_encode.count()), all_files.count())

  vocab = GetVocabFromMetaTable(output_session)
  enc, vocab = EncodePreprocessedFiles(to_encode, vocab)
  output_session.add_all(enc)
  output_session.commit()
  EmbedVocabInMetaTable(output_session, vocab)
  return len(enc)


def main():
  """Main entry point."""
  input_db = FLAGS.input()
  output_db = FLAGS.output()

  with input_db.Session() as input_session:
    with output_db.Session(commit=True) as output_session:
      while EncodeFiles(input_session, output_session, 10000):
        output_session.commit()


if __name__ == '__main__':
  app.Run(main)
