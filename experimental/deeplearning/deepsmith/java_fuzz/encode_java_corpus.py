"""Preprocess an exported database of Java methods."""
import datetime
import time
import typing

import numpy as np

from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.corpuses import preprocessed
from deeplearning.clgen.proto import internal_pb2
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import humanize
from labm8.py import pbutil
from labm8.py import sqlutil

FLAGS = app.FLAGS
app.DEFINE_database(
  "input",
  preprocessed.PreprocessedContentFiles,
  "sqlite:////var/phd/experimental/deeplearning/deepsmith/java_fuzz/preprocessed.db",
  "URL of the database of preprocessed Java methods.",
)
app.DEFINE_database(
  "output",
  encoded.EncodedContentFiles,
  "sqlite:////var/phd/experimental/deeplearning/deepsmith/java_fuzz/encoded.db",
  "URL of the database to add encoded methods to.",
)

LEXER_WORKER = bazelutil.DataPath(
  "phd/deeplearning/clgen/corpuses/lexer/lexer_worker"
)

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


def EmbedVocabInMetaTable(
  session: sqlutil.Session, vocabulary: typing.Dict[str, int]
):
  """Store a vocabulary dictionary in the 'Meta' table of a database."""
  q = session.query(encoded.Meta).filter(encoded.Meta.key.like("vocab_%"))
  q.delete(synchronize_session=False)

  session.add(encoded.Meta(key="vocab_size", value=str(len(vocabulary))))
  session.add_all(
    [encoded.Meta(key=f"vocab_{v}", value=k) for k, v in vocabulary.items()]
  )


def GetVocabFromMetaTable(session: sqlutil.Session) -> typing.Dict[str, int]:
  """Read a vocabulary dictionary from the 'Meta' table of a database."""
  q = session.query(encoded.Meta.value).filter(encoded.Meta.key == "vocab_size")
  if not q.first():
    return {}

  vocab_size = int(q.one()[0])
  q = session.query(encoded.Meta.value)
  return {
    q.filter(encoded.Meta.key == f"vocab_{i}").one()[0]: i
    for i in range(vocab_size)
  }


def EncodePreprocessedContentFiles(
  srcs: typing.List[str], vocabulary: typing.Dict[str, int]
) -> typing.Tuple[typing.List[np.array], typing.Dict[str, int]]:
  message = internal_pb2.LexerBatchJob(
    input=[internal_pb2.LexerJob(string=s) for s in srcs],
    candidate_token=JAVA_TOKENS,
    vocabulary=vocabulary,
  )
  pbutil.RunProcessMessageInPlace([LEXER_WORKER], message, timeout_seconds=3600)
  return ([list(j.token) for j in message.input], dict(message.vocabulary))


def EncodePreprocessedFiles(
  cfs: typing.List[preprocessed.PreprocessedContentFile],
  vocab: typing.Dict[str, int],
) -> typing.Union[
  typing.List[encoded.EncodedContentFile], typing.Dict[str, int]
]:
  start_time = time.time()
  encodeds, vocab = EncodePreprocessedContentFiles(
    [cf.text for cf in cfs], vocab
  )
  wall_time_ms = (time.time() - start_time) * 1000

  per_item_wall_time_ms = int(wall_time_ms / max(len(encodeds), 1))
  pp_cfs = [
    encoded.EncodedContentFile(
      id=cf.id,
      data=".".join(str(x) for x in enc),
      tokencount=len(enc),
      encoding_time_ms=per_item_wall_time_ms,
      wall_time_ms=per_item_wall_time_ms,
      date_added=datetime.datetime.now(),
    )
    for cf, enc in zip(cfs, encodeds)
  ]
  return pp_cfs, vocab


def EncodeFiles(
  input_db: preprocessed.PreprocessedContentFiles,
  output_db: encoded.EncodedContentFiles,
  batch_size: int,
):
  """Encode a batch of preprocessed contentfiles."""
  start_time = time.time()

  with input_db.Session() as input_session, output_db.Session(
    commit=True
  ) as output_session:
    # Process files in order of their numerical ID.
    max_done = (
      output_session.query(encoded.EncodedContentFile.id)
      .order_by(encoded.EncodedContentFile.id.desc())
      .limit(1)
      .first()
    )
    max_done = max_done[0] if max_done else -1

    # Only encode files that were pre-processed successfully.
    all_files = input_session.query(
      preprocessed.PreprocessedContentFile
    ).filter(
      preprocessed.PreprocessedContentFile.preprocessing_succeeded == True
    )
    to_encode = (
      all_files.filter(preprocessed.PreprocessedContentFile.id > max_done)
      .order_by(preprocessed.PreprocessedContentFile.id)
      .limit(batch_size)
    )
    to_encode_count, all_files_count = to_encode.count(), all_files.count()
    done_count = output_session.query(encoded.EncodedContentFile).count()

    vocab = GetVocabFromMetaTable(output_session)

  # This method can take a while to complete, so discard the database sessions
  # and re-open them later to avoid "MySQL server has gone away" errors.
  enc, vocab = EncodePreprocessedFiles(to_encode, vocab)

  sqlutil.ResilientAddManyAndCommit(output_db, enc)
  with output_db.Session(commit=True) as output_session:
    EmbedVocabInMetaTable(output_session, vocab)

  duration = time.time() - start_time
  app.Log(
    1,
    "Encoded %s of %s files (%.2f%%) at a rate of %d ms per file",
    humanize.Commas(to_encode_count),
    humanize.Commas(all_files_count),
    (done_count / max(all_files_count, 1)) * 100,
    (duration / max(to_encode_count, 1)) * 1000,
  )

  return len(enc)


def main():
  """Main entry point."""
  input_db = FLAGS.input()
  output_db = FLAGS.output()

  while EncodeFiles(input_db, output_db, 10000):
    pass
  app.Log(1, "Done!")


if __name__ == "__main__":
  app.Run(main)
