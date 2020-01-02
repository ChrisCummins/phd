"""Test fixtures for dynamic features tests."""
import hashlib

from experimental.deeplearning.clgen.closeness_to_grewe_features import (
  grewe_features_db,
)
from gpu.cldrive.legacy import env as cldrive_env
from labm8.py import test


def _StaticFeatures(origin: str, src: str) -> grewe_features_db.StaticFeatures:
  """Create static features instance."""
  return grewe_features_db.StaticFeatures(
    src_sha256=hashlib.sha256(src.encode("utf-8")).hexdigest(),
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
  outcome: str = "PASS",
) -> grewe_features_db.DynamicFeatures:
  return grewe_features_db.DynamicFeatures(
    static_features_id=static_features.id,
    opencl_env=env.name,
    hostname="foo",
    driver=grewe_features_db.DynamicFeaturesDriver.CLDRIVE,
    outcome=outcome,
    gsize=1,
    wgsize=1,
    run_count=30 if outcome == "PASS" else 0,
  )


@test.Fixture(scope="function")
def db() -> grewe_features_db.Database:
  """A test fixture that yields a database with three static features in it."""
  db_ = grewe_features_db.Database("sqlite://")
  with db_.Session(commit=True) as s:
    s.add_all(
      [
        _StaticFeatures(
          "origin", "kernel void A(global int* a) { a[get_global_id(0)] *= 2; }"
        ),
        _StaticFeatures(
          "github", "kernel void B(global int* a) { a[get_global_id(0)] *= 3; }"
        ),
        _StaticFeatures(
          "clgen", "kernel void C(global int* a) { a[get_global_id(0)] *= 4; }"
        ),
      ]
    )
  yield db_


@test.Fixture(scope="function")
def env() -> cldrive_env.OpenCLEnvironment:
  """Test fixture which yields a functional OpenCL environment."""
  return cldrive_env.OclgrindOpenCLEnvironment()
