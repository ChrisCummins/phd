"""A database of OpenCL kernels and their Grewe et al features."""

import hashlib

import sqlalchemy as sql
from absl import flags
from sqlalchemy.dialects import mysql
from sqlalchemy.ext import declarative

from gpu.portable_mapping_of_data_parallel_programs_to_opencl import \
  feature_extractor as grewe_features
from labm8 import sqlutil


FLAGS = flags.FLAGS

Base = declarative.declarative_base()


class OpenCLKernelWithRawGreweFeatures(
    Base, sqlutil.TablenameFromClassNameMixin):
  """A table of OpenCL kernels and their Grewe et. al. feature values."""
  id: int = sql.Column(sql.Integer, primary_key=True)
  # The checksum of the 'src' column.
  src_sha256: str = sql.Column(
    sql.Binary(32).with_variant(mysql.BINARY(32), 'mysql'),
    nullable=False, unique=True)
  # The origin of the opencl kernel, e.g. "clgen" for a clgen-generated
  # benchmark.
  origin: str = sql.Column(sql.String(32), nullable=False)

  # Raw values of Grewe et. al. feature space.
  grewe_compute_operation_count: int = sql.Column(sql.Integer, nullable=False)
  grewe_rational_operation_count: int = sql.Column(sql.Integer, nullable=False)
  grewe_global_memory_access_count: int = sql.Column(
      sql.Integer, nullable=False)
  grewe_local_memory_access_count: int = sql.Column(sql.Integer, nullable=False)
  grewe_coalesced_memory_access_count: int = sql.Column(
      sql.Integer, nullable=False)
  grewe_atomic_operation_count: int = sql.Column(sql.Integer, nullable=False)

  # The kernel source.
  src: str = sql.Column(
      sql.UnicodeText().with_variant(sql.UnicodeText(2 ** 31), 'mysql'),
      nullable=False)

  __table_args__ = (
    # <src,origin> pairs must be unique.
    sql.UniqueConstraint('src_sha256', 'origin', name='unique_src_for_origin'),
  )

  @classmethod
  def FromSrcOriginAndFeatures(
      cls, src: str, origin: str, features: grewe_features.GreweEtAlFeatures
  ) -> 'OpenCLKernelWithRawGreweFeatures':
    """Instantiate a PreprocessedContentFile."""
    return cls(
        src_sha256=hashlib.sha256(src).digest(),
        origin=origin,
        grewe_compute_operation_count=features.compute_operation_count,
        grewe_rational_operation_count=features.rational_operation_count,
        grewe_global_memory_access_count=features.global_memory_access_count,
        grewe_local_memory_access_count=features.local_memory_access_count,
        grewe_coalesced_memory_access_count=features.coalesced_memory_access_count,
        grewe_atomic_operation_count=features.atomic_operation_count,
        src=src,
    )


class Database(sqlutil.Database):

  def __init__(self, url: str):
    super(Database, self).__init__(url, Base)
