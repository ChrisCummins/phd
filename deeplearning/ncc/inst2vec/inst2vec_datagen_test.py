"""Unit tests for //deeplearning/ncc/inst2vec:inst2vec_datagen."""
import pathlib
import sys
import tempfile
import typing

import pytest
from absl import app
from absl import flags

from deeplearning.ncc.inst2vec import inst2vec_datagen


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def tempdir() -> pathlib.Path:
  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    yield pathlib.Path(d)


def test_DownloadDatasets_one_url(tempdir: pathlib.Path):
  """Test downloading one dataset."""
  inst2vec_datagen.DownloadDatasets(
      tempdir, urls=[inst2vec_datagen.DATASETS['BLAS']])
  assert (tempdir / 'BLAS-3.8.0').is_dir()


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
