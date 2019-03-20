"""Unit tests for //experimental/deeplearning/clgen/closeness_to_grewe_features/dynamic_features:drive_with_cldrive."""
import hashlib

import pytest

from experimental.deeplearning.clgen.closeness_to_grewe_features import \
  grewe_features_db
from experimental.deeplearning.clgen.closeness_to_grewe_features.dynamic_features import \
  drive_with_cldrive
from gpu.cldrive.legacy import env as cldrive_env
from labm8 import test


def _StaticFeatures(origin: str, src: str) -> grewe_features_db.StaticFeatures:
  """Create static features instance."""
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


def _DynamicFeatures(
    static_features: grewe_features_db.StaticFeatures,
    env: cldrive_env.OpenCLEnvironment,
    outcome: str = 'pass') -> grewe_features_db.DynamicFeatures:
  return grewe_features_db.DynamicFeatures(
      static_features_id=static_features.id,
      opencl_env=env.name,
      hostname='foo',
      outcome=outcome,
      gsize=1,
      wgsize=1,
  )


@pytest.fixture(scope='function')
def db() -> grewe_features_db.Database:
  """A test fixture that yields a database with three static features in it."""
  db_ = grewe_features_db.Database('sqlite:///')
  with db_.Session(commit=True) as s:
    s.add_all([
        _StaticFeatures(
            'origin',
            'kernel void A(global int* a) { a[get_global_id(0)] *= 2; }'),
        _StaticFeatures(
            'github',
            'kernel void B(global int* a) { a[get_global_id(0)] *= 3; }'),
        _StaticFeatures(
            'clgen',
            'kernel void C(global int* a) { a[get_global_id(0)] *= 4; }'),
    ])
  yield db_


@pytest.fixture(scope='env')
def env() -> cldrive_env.OpenCLEnvironment:
  return cldrive_env.OclgrindOpenCLEnvironment()


def test_GetBatchOfKernelsToDrive(db: grewe_features_db.Database,
                                  env: cldrive_env.OpenCLEnvironment):
  batch = drive_with_cldrive.GetBatchOfKernelsToDrive(db, env, 16)
  assert len(batch) == 3
  assert len(set(b.src for b in batch)) == 3


def test_GetBatchOfKernelsToDrive_overlap(db: grewe_features_db.Database,
                                          env: cldrive_env.OpenCLEnvironment):
  # Add a dynamic features to the database.
  with db.Session(commit=True) as s:
    features = s.query(grewe_features_db.StaticFeatures).first()
    s.add(_DynamicFeatures(features, env))

  batch = drive_with_cldrive.GetBatchOfKernelsToDrive(db, env, 16)
  assert len(batch) == 2
  assert len(set(b.src for b in batch)) == 2


if __name__ == '__main__':
  test.Main()
