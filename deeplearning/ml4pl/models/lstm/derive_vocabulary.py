"""Derive a vocabulary from the bytecode database."""
import pickle

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.models.lstm import bytecode2seq
from labm8 import app
from labm8 import humanize
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


def main():
  """Main entry point."""
  bytecode_db = FLAGS.bytecode_db()

  if FLAGS.vocabulary.is_file():
    with open(FLAGS.vocabulary, 'rb') as f:
      data_to_load = pickle.load(f)
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
      for i, chunk in enumerate(
          sqlutil.OffsetLimitBatchedQuery(
              query, FLAGS.batch_size, compute_max_rows=True)):
        app.Log(1, "Running batch %s, bytecodes %s of %s", i + 1, chunk.offset,
                chunk.max_rows)

        with prof.Profile(lambda t: (
            f"Encoded {humanize.Commas(FLAGS.batch_size)} bytecodes "
            f"({humanize.Commas(sum(encoded_lengths))} "
            f"tokens, vocab size {len(vocab)}, max encoded "
            f"length {humanize.Commas(max_encoded_length)})")):
          encoded, vocab = bytecode2seq.Encode([r.bytecode for r in chunk.rows],
                                               vocab)
          encoded_lengths = [len(x) for x in encoded]
          max_encoded_length = max(max(encoded_lengths), max_encoded_length)
          if len(vocab) < vocab_size:
            app.FatalWithoutStackTrace("Vocabulary shrunk!?")
          vocab_size = len(vocab)
  finally:
    with open(FLAGS.vocabulary, 'wb') as f:
      data_to_save = {'vocab': vocab, 'max_encoded_length': max_encoded_length}
      pickle.dump(data_to_save, f)
    app.Log(1, 'Wrote %s', FLAGS.vocabulary)


if __name__ == '__main__':
  app.Run(main)