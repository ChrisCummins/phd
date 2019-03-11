"""Unit tests for //deeplearning/ncc/inst2vec:inst2vec_datagen."""
import pathlib

from deeplearning.ncc.inst2vec import inst2vec_datagen
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


def test_DownloadDatasets_one_url(tempdir: pathlib.Path):
  """Test downloading one dataset."""
  inst2vec_datagen.DownloadDatasets(
      tempdir, urls=[inst2vec_datagen.DATASETS['BLAS']])
  assert (tempdir / 'BLAS-3.8.0').is_dir()


if __name__ == '__main__':
  test.Main()
