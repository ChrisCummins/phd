"""Unit tests for //deeplearning/ml4pl/graphs/unlabelled:split."""
import random
import string

from deeplearning.ml4pl.graphs.unlabelled import split
from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from deeplearning.ml4pl.ir import ir_database
from deeplearning.ml4pl.ir import split as ir_split
from deeplearning.ml4pl.testing import random_programl_generator
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import decorators
from labm8.py import test

FLAGS = test.FLAGS


def CreateRandomString(min_length: int = 1, max_length: int = 1024) -> str:
  """Generate a random string."""
  return "".join(
    random.choice(string.ascii_lowercase)
    for _ in range(random.randint(min_length, max_length))
  )


@test.Fixture(
  scope="session",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("ir_db"),
)
def ir_db(request) -> ir_database.Database:
  """A test fixture which yields an IR database."""
  with testing_databases.DatabaseContext(
    ir_database.Database, request.param
  ) as db:
    rows = []
    for i in range(250):
      ir = ir_database.IntermediateRepresentation.CreateFromText(
        source=random.choice(
          [
            "pact17_opencl_devmap",
            "poj-104:train",
            "poj-104:val",
            "poj-104:test",
          ]
        ),
        relpath=str(i),
        source_language=ir_database.SourceLanguage.OPENCL,
        type=ir_database.IrType.LLVM_6_0,
        cflags="",
        text=CreateRandomString(),
      )
      ir.id = i + 1
      rows.append(ir)

    with db.Session(commit=True) as session:
      session.add_all(rows)

    yield db


@test.Fixture(
  scope="function",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("proto_db"),
)
def proto_db(
  request, ir_db: ir_database.Database
) -> unlabelled_graph_database.Database:
  """A test fixture which yields a graph database with random graph tuples."""
  with ir_db.Session() as session:
    ir_ids = [
      row.id for row in session.query(ir_database.IntermediateRepresentation.id)
    ]

  with testing_databases.DatabaseContext(
    unlabelled_graph_database.Database, request.param
  ) as db:
    with db.Session(commit=True) as session:
      session.add_all(
        [
          unlabelled_graph_database.ProgramGraph.Create(
            proto=random_programl_generator.CreateRandomProto(
              graph_y_dimensionality=2
            ),
            ir_id=ir_id,
          )
          for ir_id in ir_ids
        ]
      )
    yield db


@decorators.loop_for(seconds=5)
@test.Parametrize(
  "splitter_class",
  (
    ir_split.Pact17KFoldSplitter,
    ir_split.TrainValTestSplitter,
    ir_split.Poj104TrainValTestSplitter,
  ),
)
@decorators.loop_for(seconds=5, min_iteration_count=3)
def test_fuzz(
  ir_db: ir_database.IntermediateRepresentation,
  proto_db: unlabelled_graph_database.Database,
  splitter_class,
):
  """Opaque fuzzing of the public method."""
  split.ApplySplit(ir_db, proto_db, splitter_class())


if __name__ == "__main__":
  test.Main()
