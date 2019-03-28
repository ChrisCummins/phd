"""Unit tests for //experimental/deeplearning/clgen/closeness_to_grewe_features/dynamic_features:import_from_legacy_gpgpu."""
import pathlib

import pytest

from experimental.deeplearning.clgen.closeness_to_grewe_features import \
  grewe_features_db
from experimental.deeplearning.clgen.closeness_to_grewe_features.dynamic_features import \
  import_from_legacy_gpgpu
from labm8 import app
from labm8 import bazelutil
from labm8 import test


TEST_LOGS = bazelutil.DataPath('phd/experimental/deeplearning/clgen/closeness_to_grewe_features/dynamic_features/tests/data/legacy_gpgpu_logs.zip')

FLAGS = app.FLAGS

@pytest.fixture(scope='function')
def df(tempdir: pathlib.Path) -> grewe_features_db.Database:
  yield grewe_features_db.Database(f'sqlite:///{tempdir}/db')

def test_ImportFromLegacyGpgpu(db):
  with db.Session(commit=True) as s:
    import_from_legacy_gpgpu.ImportFromLegacyGpgpu(
      s, TEST_LOGS, 'GPU', 'NVIDIA', 'opencl_env', 'hostname')


if __name__ == '__main__':
  test.Main()
