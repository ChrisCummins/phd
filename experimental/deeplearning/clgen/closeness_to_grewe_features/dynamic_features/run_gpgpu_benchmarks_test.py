"""Unit tests for //experimental/deeplearning/clgen/closeness_to_grewe_features/dynamic_features:run_gpgpu_benchmarks."""
import hashlib

from datasets.benchmarks.gpgpu import gpgpu_pb2
from experimental.deeplearning.clgen.closeness_to_grewe_features import \
  grewe_features_db
from experimental.deeplearning.clgen.closeness_to_grewe_features.dynamic_features import \
  run_gpgpu_benchmarks
from gpu.clinfo.proto import clinfo_pb2
from gpu.libcecl.proto import libcecl_pb2
from labm8.py import test


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


def test_DatabaseObserver(db: grewe_features_db.Database):
  with db.Session(commit=True) as s:
    s.add(
        _StaticFeatures('benchmarks_suite:benchmark:kernel',
                        'kernel void A() {}'))

  observer = run_gpgpu_benchmarks.DatabaseObserver(db)

  observer.OnBenchmarkRun(
      gpgpu_pb2.GpgpuBenchmarkRun(
          benchmark_suite="suite",
          benchmark_name="benchmark",
          dataset_name="dataset",
          hostname="hostname",
          run=libcecl_pb2.LibceclExecutableRun(
              returncode=0,
              device=clinfo_pb2.OpenClDevice(name="opencl_env"),
              kernel_invocation=[
                  libcecl_pb2.OpenClKernelInvocation(
                      kernel_name="kernel",
                      global_size=128,
                      local_size=64,
                      transferred_bytes=123,
                      transfer_time_ns=3000,
                      kernel_time_ns=1000,
                  )
              ])))

  with db.Session() as s:
    assert s.query(grewe_features_db.DynamicFeatures).count() == 0

  observer.CommitRecords()

  with db.Session() as s:
    assert s.query(grewe_features_db.DynamicFeatures).count() == 1
    record = s.query(grewe_features_db.DynamicFeatures).one()
    assert record.driver == grewe_features_db.DynamicFeaturesDriver.LIBCECL
    assert record.static_features_id == s.query(
        grewe_features_db.StaticFeatures.id).one()

    assert record.opencl_env == 'opencl_env'
    assert record.hostname == 'hostname'
    assert record.outcome == 'PASS'

    assert record.gsize == 128
    assert record.wgsize == 64

    assert record.work_item_local_mem_size == None
    assert record.work_item_private_mem_size == None
    assert record.transferred_bytes == 123
    assert record.transfer_time_ns == 3000
    assert record.kerenl_time_ns == 1000


if __name__ == '__main__':
  test.Main()
