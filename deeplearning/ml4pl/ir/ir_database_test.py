"""Unit tests for //deeplearning/ml4pl/ir:ir_database."""
from deeplearning.ml4pl.ir import ir_database
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import decorators
from labm8.py import test

FLAGS = test.FLAGS


@test.Fixture(scope="function", params=testing_databases.GetDatabaseUrls())
def db(request) -> ir_database.Database:
  """A test fixture which yields an empty IR database."""
  yield from testing_databases.YieldDatabase(
    ir_database.Database, request.param
  )


# Database stats tests.


@test.Fixture(scope="function", params=(0, 5))
def db_with_empty_ir(request, db: ir_database.Database) -> ir_database.Database:
  empty_ir_count = request.param
  with db.Session(commit=True) as session:
    session.add_all(
      [
        ir_database.IntermediateRepresentation.CreateEmpty(
          source="foo",
          relpath=str(i),
          source_language=ir_database.SourceLanguage.C,
          type=ir_database.IrType.LLVM_6_0,
          cflags="",
        )
        for i in range(empty_ir_count)
      ]
    )
  return db


# Repeat test repeatedly to test memoized property accessor.
@decorators.loop_for(min_iteration_count=3)
def test_fuzz_database_stats_on_empty_db(
  db_with_empty_ir: ir_database.Database,
):
  db = db_with_empty_ir

  assert db.ir_count == 0
  assert db.unique_ir_count == 0
  assert db.ir_data_size == 0
  assert db.char_count == 0
  assert db.line_count == 0


if __name__ == "__main__":
  test.Main()
