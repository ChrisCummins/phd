# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Derive a vocabulary from the bytecode database."""
import json

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.seq import ir2seq
from labm8.py import app
from labm8.py import humanize
from labm8.py import jsonutil
from labm8.py import prof
from labm8.py import sqlutil


FLAGS = app.FLAGS

app.DEFINE_database(
  "bytecode_db",
  bytecode_database.Database,
  None,
  "URL of database to read bytecodes from.",
  must_exist=True,
)
app.DEFINE_output_path(
  "vocabulary",
  None,
  "The vocabulary file to read and update. If it does not exist, it is "
  "created.",
)
app.DEFINE_integer(
  "batch_size", 128, "The number of bytecodes to process in a batch"
)
app.DEFINE_boolean(
  "pact17_opencl_only",
  False,
  "If true, derive the vocabulary only from the PACT'17 OpenCL sources.",
)
app.DEFINE_integer(
  "start_at",
  0,
  "The row to start at. Use this to resume partially complete jobs.",
)
app.DEFINE_string(
  "language",
  "llvm",
  "The language to derive the vocabulary from. One of {llvm,opencl}.",
)


def main():
  """Main entry point."""
  bytecode_db = FLAGS.bytecode_db()

  if FLAGS.vocabulary.is_file():
    with open(FLAGS.vocabulary) as f:
      data_to_load = json.load(f)
    vocab = data_to_load["vocab"]
    encoded_lengths = data_to_load["encoded_lengths"]
  else:
    vocab = {}
    encoded_lengths = []

  vocab_size = len(vocab)

  app.Log(1, "Initial vocabulary size %s", humanize.Commas(vocab_size))

  try:
    with bytecode_db.Session() as session:
      query = session.query(bytecode_database.LlvmBytecode.bytecode)
      if FLAGS.pact17_opencl_only:
        query = query.filter(
          bytecode_database.LlvmBytecode.source_name == "pact17_opencl_devmap"
        )

      for i, chunk in enumerate(
        sqlutil.OffsetLimitBatchedQuery(
          query,
          FLAGS.batch_size,
          start_at=FLAGS.start_at,
          compute_max_rows=True,
        )
      ):
        app.Log(
          1,
          "Running batch %s, bytecodes %s of %s",
          i + 1,
          chunk.offset,
          chunk.max_rows,
        )

        with prof.Profile(
          lambda t: (
            f"Encoded {humanize.Commas(FLAGS.batch_size)} bytecodes "
            f"({humanize.Commas(sum(encoded_lengths))} "
            f"tokens, vocab size {len(vocab)}"
          )
        ):
          encoded, vocab = ir2seq.Encode(
            [r.bytecode for r in chunk.rows], vocab, language=FLAGS.language
          )
          encoded_lengths.extend([len(x) for x in encoded])
          if len(vocab) < vocab_size:
            app.FatalWithoutStackTrace("Vocabulary shrunk!?")
          vocab_size = len(vocab)
  finally:
    data_to_save = {
      "vocab": vocab,
      "max_encoded_length": max(encoded_lengths),
      "encoded_lengths": encoded_lengths,
    }
    jsonutil.write_file(FLAGS.vocabulary, data_to_save)
    app.Log(1, "Wrote %s", FLAGS.vocabulary)


if __name__ == "__main__":
  app.Run(main)
