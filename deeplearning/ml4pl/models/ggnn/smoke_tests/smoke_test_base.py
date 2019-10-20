"""Unit tests for //deeplearning/ml4pl/models/ggnn:ggnn_graph_classifier."""
import functools
import pathlib
import tempfile

from deeplearning.ml4pl.graphs import graph_database
from labm8 import app


FLAGS = app.FLAGS

app.DEFINE_integer(
    'smoke_test_num_epochs', 2,
    '')
app.DEFINE_output_path(
    'working_dir', None,
    'A directory to store persistent data files. If not provided, a temporrary directory will be used and deleted upon exit.',is_dir=True)


@functools.contextmanager
def WorkingDirectory() -> pathlib.Path:
  with tempfile.TemporaryDirectory() as d:
    working_dir = FLAGS.working_dir or pathlib.Path(d)
    yield working_dir

class SmokeTesterBase(object):

  def GetModelClass(self):
    raise NotImplementedError

  def PopulateDatabase(self, db: graph_database.Database):
    raise NotImplementedError

  def Run(self):
    FLAGS.num_epochs = FLAGS.smoke_test_num_epochs
    with WorkingDirectory() as working_dir:
      db_path = working_dir / 'db'
      if db_path.is_file():
        db_path.unlink()

      db = graph_database.Database(f'sqlite:///{tempdir}/db')
      self.PopulateDatabase(db)

      model = self.GetModelClass()(db)
      model.Train()

