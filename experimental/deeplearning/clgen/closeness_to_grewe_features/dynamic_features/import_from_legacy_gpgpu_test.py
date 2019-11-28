"""Unit tests for //experimental/deeplearning/clgen/closeness_to_grewe_features/dynamic_features:import_from_legacy_gpgpu."""
import hashlib
import pathlib

import pytest

from experimental.deeplearning.clgen.closeness_to_grewe_features import \
  grewe_features_db
from experimental.deeplearning.clgen.closeness_to_grewe_features.dynamic_features import \
  import_from_legacy_gpgpu
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import test

TEST_LOGS = bazelutil.DataPath(
    'phd/experimental/deeplearning/clgen/closeness_to_grewe_features/dynamic_features/tests/data/legacy_gpgpu_logs.zip'
)

FLAGS = app.FLAGS


def _StaticFeatures(origin: str) -> grewe_features_db.StaticFeatures:
  """Create static features instance."""
  src = 'kernel void A() {}'
  return grewe_features_db.StaticFeatures(
      src_sha256=hashlib.sha256(src.encode('utf-8')).hexdigest(),
      origin=origin,
      grewe_compute_operation_count=0,
      grewe_rational_operation_count=0,
      grewe_global_memory_access_count=0,
      grewe_local_memory_access_count=0,
      grewe_coalesced_memory_access_count=0,
      grewe_atomic_operation_count=0,
      src=src,
  )


@pytest.fixture(scope='function')
def db(tempdir: pathlib.Path) -> grewe_features_db.Database:
  db = grewe_features_db.Database(f'sqlite:///{tempdir}/db')
  with db.Session(commit=True) as s:
    s.add_all([
        _StaticFeatures(
            'benchmarks_amd-app-sdk-3.0:FastWalshTransform:fastWalshTransform'),
        _StaticFeatures('benchmarks_npb-3.3:EP:embar'),
        _StaticFeatures(
            'benchmarks_polybench-gpu-1.0:2DConvolution:Convolution2D_kernel'),
        _StaticFeatures('benchmarks_nvidia-4.2:oclBlackScholes:BlackScholes'),
        _StaticFeatures('benchmarks_shoc-1.1.5:BFS:BFS_kernel_warp'),
        _StaticFeatures(
            'benchmarks_rodinia-3.1:backprop:bpnn_layerforward_ocl'),
        _StaticFeatures(
            'benchmarks_rodinia-3.1:backprop:bpnn_adjust_weights_ocl'),
    ])
  yield db


def test_ImportFromLegacyGpgpu(db):
  import_from_legacy_gpgpu.ImportFromLegacyGpgpu(
      db, TEST_LOGS, 'GPU', 'GeForce GTX 1080', 'opencl_env', 'hostname')

  with db.Session() as s:
    assert s.query(grewe_features_db.DynamicFeatures).count() == 7
    for record in s.query(grewe_features_db.DynamicFeatures):
      print(record.id)
      assert record.opencl_env == 'opencl_env'
      assert record.hostname == 'hostname'
      assert record.outcome == 'PASS'
      assert record.gsize > 1
      assert record.wgsize > 1
      assert record.transferred_bytes > 1
      assert record.transfer_time_ns > 1
      assert record.kernel_time_ns > 1
      assert record.run_count >= 1


if __name__ == '__main__':
  test.Main()
