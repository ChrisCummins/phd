"""A database of static and dynamic Grewe et al features."""

import hashlib
import multiprocessing
import pathlib
import random
import typing

import numpy as np
import progressbar
import sqlalchemy as sql
from sqlalchemy.ext import declarative

from labm8 import app
from labm8 import humanize
from labm8 import sqlutil
from labm8 import system
from research.grewe_2013_cgo import feature_extractor as grewe_features

FLAGS = app.FLAGS

Base = declarative.declarative_base()


class StaticFeatures(Base, sqlutil.TablenameFromCamelCapsClassNameMixin):
  """A table of OpenCL kernels and their Grewe et. al. static values."""
  id: int = sql.Column(sql.Integer, primary_key=True)
  # The checksum of the 'src' column.
  src_sha256: str = sql.Column(sql.String(64), nullable=False)
  # The origin of the opencl kernel, e.g. "clgen" for a clgen-generated
  # benchmark.
  origin: str = sql.Column(sql.String(128), nullable=False)

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
      sql.UnicodeText().with_variant(sql.UnicodeText(2**31), 'mysql'),
      nullable=False)

  __table_args__ = (
      # <src,origin> pairs must be unique.
      sql.UniqueConstraint(
          'src_sha256', 'origin', name='unique_src_for_origin'),)

  @classmethod
  def FromSrcOriginAndFeatures(
      cls, src: str, origin: str,
      features: grewe_features.GreweEtAlFeatures) -> 'StaticFeatures':
    """Instantiate a PreprocessedContentFile."""
    return cls(
        src_sha256=hashlib.sha256(src.encode('utf-8')).hexdigest(),
        origin=origin,
        grewe_compute_operation_count=features.compute_operation_count,
        grewe_rational_operation_count=features.rational_operation_count,
        grewe_global_memory_access_count=features.global_memory_access_count,
        grewe_local_memory_access_count=features.local_memory_access_count,
        grewe_coalesced_memory_access_count=features.
        coalesced_memory_access_count,
        grewe_atomic_operation_count=features.atomic_operation_count,
        src=src,
    )


def NoneIfNaN(value):
  """Covert NaN numeric value to None."""
  if np.isnan(value):
    return None
  return value


class DynamicFeatures(Base, sqlutil.TablenameFromCamelCapsClassNameMixin):
  id: int = sql.Column(sql.Integer, primary_key=True)
  static_features_id: int = sql.Column(
      sql.Integer, sql.ForeignKey(StaticFeatures.id), nullable=False)
  # The OpenClEnvironment.name of the device.
  opencl_env: str = sql.Column(sql.String(256), nullable=False)
  hostname: str = sql.Column(sql.String(32), nullable=False)
  outcome: str = sql.Column(sql.String(32), nullable=False)

  # Dynamic params that may not be set if outcome != "PASS".
  gsize: int = sql.Column(sql.Integer, nullable=True)
  wgsize: int = sql.Column(sql.Integer, nullable=True)

  # Dynamic features that may not be set if outcome != "PASS".
  work_item_local_mem_size: int = sql.Column(sql.Integer, nullable=True)
  work_item_private_mem_size: int = sql.Column(sql.Integer, nullable=True)
  transferred_bytes: int = sql.Column(sql.Integer, nullable=True)
  kernel_time_ns: int = sql.Column(sql.Integer, nullable=True)
  transfer_time_ns: int = sql.Column(sql.Integer, nullable=True)

  @classmethod
  def FromCldriveDataFrameRecord(cls, record, static_features_id: int):
    features = cls(
        static_features_id=static_features_id,
        opencl_env=record.device,
        hostname=system.HOSTNAME,
        outcome=record.outcome,
        gsize=NoneIfNaN(record.global_size),
        wgsize=NoneIfNaN(record.local_size),
        work_item_local_mem_size=NoneIfNaN(record.work_item_local_mem_size),
        work_item_private_mem_size=NoneIfNaN(record.work_item_private_mem_size),
        transferred_bytes=NoneIfNaN(record.transferred_bytes),
        kernel_time_ns=NoneIfNaN(record.kernel_time_ns),
        transfer_time_ns=NoneIfNaN(record.transfer_time_ns),
    )
    return features


def _DatabaseImporterWorker(
    path: pathlib.Path
) -> typing.Tuple[typing.Optional[str], typing.Optional[grewe_features.
                                                        GreweEtAlFeatures]]:
  """Worker function for multi-processed database import."""
  try:
    features = list(grewe_features.ExtractFeaturesFromPath(path))
  except grewe_features.FeatureExtractionError as e:
    app.Log(2, "Feature extraction failed with message: %s", e)
    return None, None

  if len(features) != 1:
    app.Log(2, "Expected 1 feature vector in %s, found %d", path, len(features))
    return None, None

  try:
    with open(path) as f:
      src = f.read()
  except UnicodeEncodeError:
    app.Log(2, "Failed to encode %s", src)
    return None, None

  return src, features[0]


class Database(sqlutil.Database):
  """Database of kernels."""

  def __init__(self, url: str):
    super(Database, self).__init__(url, Base)

  def ImportStaticFeaturesFromPaths(
      self,
      paths_to_import: typing.Iterable[pathlib.Path],
      origin: str,
      pool: typing.Optional[multiprocessing.Pool] = None
  ) -> typing.Tuple[int, int]:
    """Import a sequence of paths into the database.

    Each path should point to a file containing a single OpenCL kernel.

    Args:
      paths_to_import: The paths to import.
      origin: The origin of the kernels.
      pool: A multiprocessing pool to execute workers in. If not provided,
        workers are processed sequentially.

    Returns:
      The number of samples that were successfully imported.
    """
    success_count = 0
    new_row_count = 0
    paths_to_import = list(paths_to_import)
    random.shuffle(paths_to_import)
    app.Log(1, 'Importing %s files ...', humanize.Commas(len(paths_to_import)))
    bar = progressbar.ProgressBar(
        max_value=len(paths_to_import), redirect_stderr=True)

    # Optionally multiprocess.
    if pool:
      to_import = pool.imap_unordered(_DatabaseImporterWorker, paths_to_import)
    else:
      to_import = (_DatabaseImporterWorker(p) for p in paths_to_import)

    for i, (src, features) in enumerate(to_import):
      bar.update(i)
      # None type return if feature extraction failed.
      if src:
        success_count += 1
        with self.Session(commit=False) as session:
          obj = StaticFeatures.FromSrcOriginAndFeatures(src, origin, features)
          # Check if it already exists in the database.
          exists = session.query(StaticFeatures) \
            .filter_by(src_sha256=obj.src_sha256, origin=origin).first()
          if not exists:
            session.add(obj)
            new_row_count += 1
            try:
              session.commit()
            except (sql.exc.OperationalError, sql.exc.DataError) as e:
              app.Warning('Failed to commit database entry: %s', e)

    return success_count, new_row_count
