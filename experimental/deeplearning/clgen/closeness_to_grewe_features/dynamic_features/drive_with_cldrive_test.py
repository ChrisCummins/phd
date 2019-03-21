"""Unit tests for //experimental/deeplearning/clgen/closeness_to_grewe_features/dynamic_features:drive_with_cldrive."""
import hashlib
import pytest
import numpy as np
import sqlalchemy as sql

from experimental.deeplearning.clgen.closeness_to_grewe_features import \
  grewe_features_db
from experimental.deeplearning.clgen.closeness_to_grewe_features.dynamic_features import \
  drive_with_cldrive
from gpu.cldrive.legacy import env as cldrive_env
from labm8 import test
from gpu.cldrive.proto import cldrive_pb2
from labm8 import system


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
  db_ = grewe_features_db.Database('sqlite://')
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


@pytest.fixture(scope='function')
def env() -> cldrive_env.OpenCLEnvironment:
  """Test fixture which yields a functional OpenCL environment."""
  return cldrive_env.OclgrindOpenCLEnvironment()


def test_GetBatchOfKernelsToDrive(db: grewe_features_db.Database,
                                  env: cldrive_env.OpenCLEnvironment):
  with db.Session() as s:
    batch = drive_with_cldrive.GetBatchOfKernelsToDrive(s, env, 16)
  assert len(batch) == 3
  assert len(set(b.src for b in batch)) == 3


def test_GetBatchOfKernelsToDrive_overlap(db: grewe_features_db.Database,
                                          env: cldrive_env.OpenCLEnvironment):
  # Add a dynamic features to the database.
  with db.Session() as s:
    features = s.query(grewe_features_db.StaticFeatures).first()
    s.add(_DynamicFeatures(features, env))
    s.flush()
    batch = drive_with_cldrive.GetBatchOfKernelsToDrive(s, env, 16)
  assert len(batch) == 2
  assert len(set(b.src for b in batch)) == 2


@pytest.mark.parametrize('num_runs', [3, 5, 10])
@pytest.mark.parametrize(
    'dynamic_params',
    [[cldrive_pb2.DynamicParams(global_size_x=1, local_size_x=1)],
     [
         cldrive_pb2.DynamicParams(global_size_x=1, local_size_x=1),
         cldrive_pb2.DynamicParams(global_size_x=2, local_size_x=1),
         cldrive_pb2.DynamicParams(global_size_x=16, local_size_x=4),
     ]])
def test_DriveKernelAndRecordResults(db: grewe_features_db.Database,
                                     env: cldrive_env.OpenCLEnvironment,
                                     dynamic_params, num_runs: int):
  with db.Session(commit=True) as s:
    drive_with_cldrive.DriveKernelAndRecordResults(
        s, 0, "kernel void A(global int* a) { a[get_global_id(0)] *= 2; }", env,
        dynamic_params, num_runs)

    assert s.query(grewe_features_db.DynamicFeatures).count() == (
        len(dynamic_params) * num_runs)
    records = s.query(grewe_features_db.DynamicFeatures).all()

    for i, record in enumerate(records):
      assert record.static_features_id == 0
      assert record.opencl_env == env.name
      assert record.hostname == system.HOSTNAME
      assert record.outcome == 'PASS'

      # FIXME(cec): BYTES !?
      # assert record.gsize == dynamic_params[i // 3].global_size_x
      # assert record.wgsize == dynamic_params[i // 3].local_size_x

      # assert record.work_item_local_mem_size == 0
      # assert record.work_item_private_mem_size == 0
      # assert record.transferred_bytes == (
      #     4 * dynamic_params[i // 3].global_size_x)
      # assert record.transfer_time_ns >= 1000
      # assert record.kernel_time_ns >= 1000


@pytest.mark.parametrize('num_runs', [3, 5, 10])
@pytest.mark.parametrize(
    'dynamic_params',
    [[cldrive_pb2.DynamicParams(global_size_x=1, local_size_x=1)],
     [
         cldrive_pb2.DynamicParams(global_size_x=1, local_size_x=1),
         cldrive_pb2.DynamicParams(global_size_x=2, local_size_x=1),
         cldrive_pb2.DynamicParams(global_size_x=16, local_size_x=4),
     ]])
def test_DriveKernelAndRecordResults_broken_kernel(
    db: grewe_features_db.Database, env: cldrive_env.OpenCLEnvironment,
    dynamic_params, num_runs: int):
  with db.Session() as s:
    drive_with_cldrive.DriveKernelAndRecordResults(s, 10, "invalid kernel", env,
                                                   dynamic_params, num_runs)
    s.commit()

    assert s.query(grewe_features_db.DynamicFeatures).count() == 1

    record = s.query(grewe_features_db.DynamicFeatures).first()
    assert record.static_features_id == 10
    assert record.opencl_env == env.name
    assert record.hostname == system.HOSTNAME
    assert record.outcome == 'PROGRAM_COMPILATION_FAILURE'

    # FIXME(cec): BYTES !?
    # assert record.gsize is None
    # assert record.wgsize is None

    # assert record.work_item_local_mem_size is None
    # assert record.work_item_private_mem_size is None
    # assert record.transferred_bytes is None
    # assert record.transfer_time_ns is None
    # assert record.kernel_time_ns is None


@pytest.mark.parametrize('num_runs', [3, 5, 10])
@pytest.mark.parametrize(
    'dynamic_params',
    [[cldrive_pb2.DynamicParams(global_size_x=1, local_size_x=1)],
     [
         cldrive_pb2.DynamicParams(global_size_x=1, local_size_x=1),
         cldrive_pb2.DynamicParams(global_size_x=2, local_size_x=1),
         cldrive_pb2.DynamicParams(global_size_x=16, local_size_x=4),
     ]])
def test_DriveKernelAndRecordResults_no_output(
    db: grewe_features_db.Database, env: cldrive_env.OpenCLEnvironment,
    dynamic_params, num_runs: int):
  with db.Session(commit=True) as s:
    drive_with_cldrive.DriveKernelAndRecordResults(
        s, 10, "kernel void A(global int* a) {}", env, dynamic_params, num_runs)

  with db.Session() as s:
    assert s.query(
        grewe_features_db.DynamicFeatures).count() == (len(dynamic_params))
    records = s.query(grewe_features_db.DynamicFeatures).all()

    for i, record in enumerate(records):
      assert record.static_features_id == 10
      assert record.opencl_env == env.name
      assert record.hostname == system.HOSTNAME
      assert record.outcome == 'NO_OUTPUT'

      # FIXME(cec): BYTES !?
      # assert record.gsize is None
      # assert record.wgsize is None

      # assert record.work_item_local_mem_size is None
      # assert record.work_item_private_mem_size is None
      # assert record.transferred_bytes is None
      # assert record.transfer_time_ns is None
      # assert record.kernel_time_ns is None


@pytest.mark.parametrize('num_runs', [3, 5, 10])
@pytest.mark.parametrize(
    'dynamic_params',
    [[cldrive_pb2.DynamicParams(global_size_x=1, local_size_x=1)],
     [
         cldrive_pb2.DynamicParams(global_size_x=1, local_size_x=1),
         cldrive_pb2.DynamicParams(global_size_x=2, local_size_x=1),
         cldrive_pb2.DynamicParams(global_size_x=16, local_size_x=4),
     ]])
def test_DriveKernelAndRecordResults_input_insensitive(
    db: grewe_features_db.Database, env: cldrive_env.OpenCLEnvironment,
    dynamic_params, num_runs: int):
  with db.Session() as s:
    drive_with_cldrive.DriveKernelAndRecordResults(
        s, 10, "kernel void A(global int* a) { a[get_global_id(0)] = 0; }", env,
        dynamic_params, num_runs)
    s.commit()

    assert s.query(
        grewe_features_db.DynamicFeatures).count() == (len(dynamic_params))
    records = s.query(grewe_features_db.DynamicFeatures).all()
    for i, record in enumerate(records):
      assert record.static_features_id == 10
      assert record.opencl_env == env.name
      assert record.hostname == system.HOSTNAME
      assert record.outcome == 'INPUT_INSENSITIVE'

      # FIXME(cec): BYTES !?
      # assert record.gsize is None
      # assert record.wgsize is None

      # assert record.work_item_local_mem_size is None
      # assert record.work_item_private_mem_size is None
      # assert record.transferred_bytes is None
      # assert record.transfer_time_ns is None
      # assert record.kernel_time_ns is None


if __name__ == '__main__':
  test.Main()
