"""Derive a vocabulary from the bytecode database."""
import json

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.models.lstm import bytecode2seq
from labm8 import app
from labm8 import humanize
from labm8 import jsonutil
from labm8 import prof
from labm8 import sqlutil

FLAGS = app.FLAGS

app.DEFINE_database(
    'bytecode_db',
    bytecode_database.Database,
    None,
    'URL of database to read bytecodes from.',
    must_exist=True)
app.DEFINE_output_path(
    'vocabulary', None,
    'The vocabulary file to read and update. If it does not exist, it is '
    'created.')
app.DEFINE_integer('batch_size', 128,
                   'The number of bytecodes to process in a batch')
app.DEFINE_boolean(
    'pact17_opencl_only', False,
    "If true, derive the vocabulary only from the PACT'17 OpenCL sources.")
app.DEFINE_integer(
    "start_at", 0,
    "The row to start at. Use this to resume partially complete jobs.")


def main():
  """Main entry point."""
  bytecode_db = FLAGS.bytecode_db()

  if FLAGS.vocabulary.is_file():
    with open(FLAGS.vocabulary) as f:
      data_to_load = json.load(f)
    vocab = data_to_load['vocab']
    max_encoded_length = data_to_load['max_encoded_length']
  else:
    vocab = {}
    max_encoded_length = 0

  vocab_size = len(vocab)

  app.Log(1, 'Initial vocabulary size %s', humanize.Commas(vocab_size))

  try:
    with bytecode_db.Session() as session:
      query = session.query(bytecode_database.LlvmBytecode.bytecode)
      if FLAGS.pact17_opencl_only:
        query = query.filter(bytecode_database.LlvmBytecode.source_name ==
                             'pact17_opencl_devmap')

      encoded_lengths = []
      for i, chunk in enumerate(
          sqlutil.OffsetLimitBatchedQuery(
              query,
              FLAGS.batch_size,
              start_at=FLAGS.start_at,
              compute_max_rows=True)):
        app.Log(1, "Running batch %s, bytecodes %s of %s", i + 1, chunk.offset,
                chunk.max_rows)

        with prof.Profile(lambda t: (
            f"Encoded {humanize.Commas(FLAGS.batch_size)} bytecodes "
            f"({humanize.Commas(sum(encoded_lengths))} "
            f"tokens, vocab size {len(vocab)}")):
          encoded, vocab = bytecode2seq.Encode([r.bytecode for r in chunk.rows],
                                               vocab)
          encoded_lengths.extend([len(x) for x in encoded])
          if len(vocab) < vocab_size:
            app.FatalWithoutStackTrace("Vocabulary shrunk!?")
          vocab_size = len(vocab)
  finally:
    data_to_save = {
        'vocab': vocab,
        'max_encoded_length': max(encoded_lengths),
        'encoded_lengths': encoded_lengths
    }
    jsonutil.write_file(FLAGS.vocabulary, data_to_save)
    app.Log(1, 'Wrote %s', FLAGS.vocabulary)


if __name__ == '__main__':
  app.Run(main)
