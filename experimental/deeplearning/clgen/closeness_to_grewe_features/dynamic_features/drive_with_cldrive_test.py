"""Unit tests for //experimental/deeplearning/clgen/closeness_to_grewe_features/dynamic_features:drive_with_cldrive."""
import pytest

from experimental.deeplearning.clgen.closeness_to_grewe_features import \
  grewe_features_db
from experimental.deeplearning.clgen.closeness_to_grewe_features.dynamic_features import \
  drive_with_cldrive
from gpu.cldrive.legacy import env as cldrive_env
from gpu.cldrive.proto import cldrive_pb2
from labm8 import system
from labm8 import test


def _DynamicFeatures(
    static_features: grewe_features_db.StaticFeatures,
    env: cldrive_env.OpenCLEnvironment,
    outcome: str = 'PASS') -> grewe_features_db.DynamicFeatures:
  return grewe_features_db.DynamicFeatures(
      static_features_id=static_features.id,
      driver=grewe_features_db.DynamicFeaturesDriver.CLDRIVE,
      opencl_env=env.name,
      hostname='foo',
      outcome=outcome,
      gsize=1,
      wgsize=1,
      run_count=30 if outcome == 'PASS' else 0,
  )


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
  drive_with_cldrive.DriveKernelAndRecordResults(
      db, 0, "kernel void A(global int* a) { a[get_global_id(0)] *= 2; }", env,
      dynamic_params, num_runs)

  with db.Session() as s:
    assert s.query(
        grewe_features_db.DynamicFeatures).count() == (len(dynamic_params))
    records = s.query(grewe_features_db.DynamicFeatures).all()

    for i, record in enumerate(records):
      assert record.static_features_id == 0
      assert record.opencl_env == env.name
      assert record.hostname == system.HOSTNAME
      assert record.outcome == 'PASS'
      assert record.run_count == num_runs

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
  drive_with_cldrive.DriveKernelAndRecordResults(db, 10, "invalid kernel", env,
                                                 dynamic_params, num_runs)
  with db.Session() as s:
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
  drive_with_cldrive.DriveKernelAndRecordResults(
      db, 10, "kernel void A(global int* a) {}", env, dynamic_params, num_runs)
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
  drive_with_cldrive.DriveKernelAndRecordResults(
      db, 10, "kernel void A(global int* a) { a[get_global_id(0)] = 0; }", env,
      dynamic_params, num_runs)
  with db.Session() as s:
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
