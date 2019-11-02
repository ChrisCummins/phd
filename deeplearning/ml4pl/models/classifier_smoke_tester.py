"""Module defining smoke test utilites for classifier models."""
import contextlib
import pathlib
import tempfile

from labm8 import app

from deeplearning.ml4pl.graphs import graph_database
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


@contextlib.contextmanager
def WorkingDirectory() -> pathlib.Path:
  """Create and get the model working directory. If --smoke_test_working_dir
  is set, this is used. Else, a random directory is used."""
  with tempfile.TemporaryDirectory() as d:
    working_dir = FLAGS.smoke_test_working_dir or pathlib.Path(d)
    yield working_dir


class SmokeTesterBase(object):
  """The base class for implementing model smoke tests."""

  def GetModelClass(self):
    """Return the class to be tested."""
    raise NotImplementedError

  def GetAnnotatedGraphDatabase(self) -> graph_database.Database:
    """Generate fake data for the graph database."""

  def Run(self):
    FLAGS.num_epochs = FLAGS.smoke_test_num_epochs

    with WorkingDirectory() as working_dir:
      graph_db_path = working_dir / 'graphs.db'
      if graph_db_path.is_file():
        graph_db_path.unlink()

      log_db_path = working_dir / 'logs.db'
      if log_db_path.is_file():
        log_db_path.unlink()

      graph_db = graph_database.Database(f'sqlite:///{working_dir}/graphs.db')
      log_db = log_database.Database(f'sqlite:///{working_dir}/logs.db')
      self.PopulateDatabase(graph_db)

      model = self.GetModelClass()(graph_db, log_db)
      model.Train()