"""Module defining smoke test utilities for classifier models."""
import contextlib
import pathlib
import tempfile
import typing

import networkx as nx
import numpy as np

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.unlabelled.cdfg import random_cdfg_generator
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import log_database
from labm8.py import app
from labm8.py import prof

FLAGS = app.FLAGS

app.DEFINE_output_path(
  "smoke_test_working_dir",
  None,
  "A directory to store persistent data files. If not provided, a temporary "
  "directory will be used and deleted upon exit.",
  is_dir=True,
)


def RunSmokeTest(
  model_class,
  node_y_choices: typing.Optional[typing.List[np.array]] = None,
  graph_x_choices: typing.Optional[typing.List[np.array]] = None,
  graph_y_choices: typing.Optional[typing.List[np.array]] = None,
) -> None:
  """Run a simple smoke test on a model.

  The smoke test consists of:
      1. Generate 50 random labelled graphs {30 train, 10 val, 10 test}.
      2. Construct and train a models for 2 epochs.
      3. Check that the model produces logs.

  This high level test is *not* a substitute for proper unit testing of a
  model's components, and does not test the learning power of the model.

  Args:
    model_class: The class under test. Must implement the interface of
      classifier_base.ClassifierBase.
    node_y_choices: A list of options for node labels. Randomly generated graphs
      are randomly assigned labels from this list. If not provided, no node
      labels are used.
    graph_x_choices: A list of options for graph features. Randomly generated
      graphs are randomly assigned labels from this list. If not provided, no
      graph features are used.
    graph_y_choices: A list of options for graph labels. Randomly generated
      graphs are randomly assigned labels from this list. If not provided, no
      graph labels are used.
  """
  with WorkingDirectory() as working_dir:
    graph_db_path = working_dir / "node_classification_graphs.db"
    if graph_db_path.is_file():
      graph_db_path.unlink()

    log_db_path = working_dir / "logs.db"
    if log_db_path.is_file():
      log_db_path.unlink()

    graph_db = graph_database.Database(f"sqlite:///{working_dir}/graphs.db")
    log_db = log_database.Database(f"sqlite:///{working_dir}/logs.db")

    with prof.Profile("Creating random testing graphs"):
      graphs = (
        list(_MakeNRandomGraphs(30, "train"))
        + list(_MakeNRandomGraphs(10, "val"))
        + list(_MakeNRandomGraphs(10, "test"))
      )

    with prof.Profile("Added random graph annotations"):
      # Add node-level labels.
      random_cdfg_generator.AddRandomAnnotations(
        graphs,
        auxiliary_node_x_indices_choices=[[0, 1]],
        node_y_choices=node_y_choices,
        graph_x_choices=graph_x_choices,
        graph_y_choices=graph_y_choices,
      )

    with prof.Profile("Added graphs to database"):
      with graph_db.Session(commit=True) as s:
        s.add_all(
          [graph_database.GraphMeta.CreateFromNetworkX(g) for g in graphs]
        )

    with prof.Profile("Created model"):
      model: classifier_base.ClassifierBase = model_class(graph_db, log_db)

    # Check that model has correct name.
    assert model.ModelFlagsToDict()["model"] == model_class.__name__

    with prof.Profile("Initialized model"):
      model.InitializeModel()

    with prof.Profile("Trained model"):
      model.Train(epoch_count=2)

    # Check a properties of logs.
    with log_db.Session() as session:
      logs = session.query(log_database.BatchLogMeta).all()
      logs = sorted(logs, key=lambda log: log.epoch)

      # There should be at least 5 and no more than 6 logs:
      #
      #   Epoch 1 Training batch 1
      #   Epoch 1 Validation batch 1
      #   Epoch 1 Testing batch 1
      #   Epoch 2 Training batch 1
      #   Epoch 2 Validation batch 1
      #   [Epoch 2 Testing batch 1]  <- only if validation acc improved
      assert len(logs) in {5, 6}

      # Check log properties.
      for log in logs:
        assert isinstance(log.loss, float)
        assert log.type in {"train", "test", "val"}


def _MakeNRandomGraphs(n: int, group: str) -> typing.Iterable[nx.MultiDiGraph]:
  """Private helper to generate random graphs of the given group."""
  for i in range(n):
    g = random_cdfg_generator.FastCreateRandom()
    g.bytecode_id = 0
    g.relpath = str(i)
    g.language = "c"
    g.group = group
    g.source_name = "rand"
    yield g


@contextlib.contextmanager
def WorkingDirectory() -> pathlib.Path:
  """Create and get the model working directory. If --smoke_test_working_dir
  is set, this is used. Else, a random directory is used."""
  with tempfile.TemporaryDirectory() as d:
    working_dir = FLAGS.smoke_test_working_dir or pathlib.Path(d)
    app.Log(1, "Using working directory `%s` for smoke tests", working_dir)
    yield working_dir
