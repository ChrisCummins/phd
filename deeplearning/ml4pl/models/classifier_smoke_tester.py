"""Module defining smoke test utilities for classifier models."""
import contextlib
import pathlib
import tempfile
import typing

import networkx as nx
import numpy as np
from labm8 import app
from labm8 import prof

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.unlabelled.cdfg import random_cdfg_generator
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import log_database

FLAGS = app.FLAGS

app.DEFINE_integer(
    'smoke_test_num_epochs', 2,
    'The number of epochs to run the smoke test for. This overrides the normal '
    '--num_epochs flag.')
app.DEFINE_output_path(
    'smoke_test_working_dir',
    None,
    'A directory to store persistent data files. If not provided, a temporary '
    'directory will be used and deleted upon exit.',
    is_dir=True)


def RunSmokeTest(model_class,
                 node_y_choices: typing.Optional[typing.List[np.array]] = None,
                 graph_x_choices: typing.Optional[typing.List[np.array]] = None,
                 graph_y_choices: typing.Optional[typing.List[np.array]] = None,
                 num_epochs: int = 2):
  """

  :param model_class:
  :param node_y_choices:
  :param graph_x_choices:
  :param graph_y_choices:
  :param num_epochs:
  :return:
  """
  FLAGS.alsologtostderr = True

  with WorkingDirectory() as working_dir:
    graph_db_path = working_dir / 'node_classification_graphs.db'
    if graph_db_path.is_file():
      graph_db_path.unlink()

    log_db_path = working_dir / 'logs.db'
    if log_db_path.is_file():
      log_db_path.unlink()

    graph_db = graph_database.Database(f'sqlite:///{working_dir}/graphs.db')
    log_db = log_database.Database(f'sqlite:///{working_dir}/logs.db')

    with prof.Profile("Creating random testing graphs"):
      graphs = (list(_MakeNGroupGraphs(30, 'train')) +
                list(_MakeNGroupGraphs(10, 'val')) +
                list(_MakeNGroupGraphs(10, 'test')))

    with prof.Profile("Added random graph annotations"):
      # Add node-level labels.
      random_cdfg_generator.AddRandomAnnotations(
          graphs,
          node_y_choices=node_y_choices,
          graph_x_choices=graph_x_choices,
          graph_y_choices=graph_y_choices)

    with prof.Profile("Added graphs to database"):
      with graph_db.Session(commit=True) as s:
        s.add_all(
            [graph_database.GraphMeta.CreateFromNetworkX(g) for g in graphs])

    with prof.Profile("Created model"):
      model: classifier_base.ClassifierBase = model_class(graph_db, log_db)

    with prof.Profile("Trained model"):
      FLAGS.num_epochs = FLAGS.smoke_test_num_epochs
      model.Train(num_epochs=num_epochs)


def _MakeNGroupGraphs(n: int, group: str) -> nx.MultiDiGraph:
  """Private helper to generate random graphs of the given group."""
  for i in range(n):
    g = random_cdfg_generator.FastCreateRandom()
    g.bytecode_id = 0
    g.relpath = str(i)
    g.language = 'c'
    g.group = group
    g.source_name = 'rand'
    yield g


@contextlib.contextmanager
def WorkingDirectory() -> pathlib.Path:
  """Create and get the model working directory. If --smoke_test_working_dir
  is set, this is used. Else, a random directory is used."""
  with tempfile.TemporaryDirectory() as d:
    working_dir = FLAGS.smoke_test_working_dir or pathlib.Path(d)
    app.Log(1, "Using working directory `%s` for smoke tests", working_dir)
    yield working_dir
