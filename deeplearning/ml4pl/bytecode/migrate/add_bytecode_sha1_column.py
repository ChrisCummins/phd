"""Populate the 'bytecode_sha1' column."""
from deeplearning.ml4pl.bytecode import bytecode_database
from labm8.py import app
from labm8.py import crypto
from labm8.py import prof

FLAGS = app.FLAGS

app.DEFINE_database(
  "bytecode_db",
  bytecode_database.Database,
  None,
  "URL of database to read bytecodes from.",
  must_exist=True,
)

app.DEFINE_integer(
  "batch_size", 2048, "The number of bytecodes to process each batch."
)


def main():
  """Main entry point."""
  bytecode_db = FLAGS.bytecode_db()
  batch = 0
  while True:
    batch += 1
    app.Log(1, "Running batch %s", batch)
    with bytecode_db.Session() as session:
      with prof.Profile(f"Read {FLAGS.batch_size} bytecodes"):
        bytecodes_to_process = (
          session.query(bytecode_database.LlvmBytecode)
          .filter(bytecode_database.LlvmBytecode.bytecode_sha1 == "")
          .limit(FLAGS.batch_size)
          .all()
        )

      if not bytecodes_to_process:
        break

      with prof.Profile(f"Computed {FLAGS.batch_size} checksums"):
        for bytecode in bytecodes_to_process:
          bytecode.bytecode_sha1 = crypto.sha1_str(bytecode.bytecode)

      with prof.Profile("Committed changes"):
        session.commit()

  app.Log(1, "Finished after %s batches", batch)


if __name__ == "__main__":
  app.Run(main)
