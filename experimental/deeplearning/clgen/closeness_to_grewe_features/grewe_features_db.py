"""A database of static and dynamic Grewe et al features."""
import enum
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


class DynamicFeaturesDriver(enum.Enum):
  CLDRIVE = 0  # //gpu/cldrive
  LIBCECL = 1  # //gpu/libcecl


class DynamicFeatures(Base, sqlutil.TablenameFromCamelCapsClassNameMixin):
  """A table of dynamic features."""
  id: int = sql.Column(sql.Integer, primary_key=True)
  static_features_id: int = sql.Column(
      sql.Integer, sql.ForeignKey(StaticFeatures.id), nullable=False)
  driver: DynamicFeaturesDriver = sql.Column(
      sql.Enum(DynamicFeaturesDriver), nullable=False)

  # The OpenClEnvironment.name of the device.
  opencl_env: str = sql.Column(sql.String(256), nullable=False, index=True)
  hostname: str = sql.Column(sql.String(32), nullable=False)
  outcome: str = sql.Column(sql.String(32), nullable=False, index=True)

  # Dynamic params that may not be set if outcome != "PASS".
  gsize: int = sql.Column(sql.Integer, nullable=True, index=True)
  wgsize: int = sql.Column(sql.Integer, nullable=True, index=True)

  # Dynamic features that may not be set if outcome != "PASS".
  work_item_local_mem_size: int = sql.Column(sql.Integer, nullable=True)
  work_item_private_mem_size: int = sql.Column(sql.Integer, nullable=True)
  transferred_bytes: int = sql.Column(sql.BigInteger, nullable=True)
  transfer_time_ns: int = sql.Column(sql.BigInteger, nullable=True)
  kernel_time_ns: int = sql.Column(sql.BigInteger, nullable=True)


class CpuGpuMappingSet(Base, sqlutil.TablenameFromCamelCapsClassNameMixin):
  """A labelled CPU/GPU dataset of dynamic and static feature aggregates.

  The information in this table can be derived solely from StaticFeatures and
  DynamicFeatures tables. It is purely a convenience to cache the results of the
  (expensive) aggregation queries in this table.
  """
  id: int = sql.Column(sql.Integer, primary_key=True)

  # A grouping value.
  cpu_gpu_mapping_set_name: str = sql.Column(
      sql.String(128), nullable=False, index=True)

  static_features_id = sql.Column(
      sql.Integer, sql.ForeignKey(StaticFeatures.id), nullable=False)

  # The origin of the opencl kernel, e.g. "clgen" for a clgen-generated
  # benchmark.
  origin: str = sql.Column(sql.String(128), nullable=False)

  # Dynamic Features values.
  driver: DynamicFeaturesDriver = sql.Column(
      sql.Enum(DynamicFeaturesDriver), nullable=False, index=True)
  gsize: int = sql.Column(sql.Integer, nullable=False, index=True)
  wgsize: int = sql.Column(sql.Integer, nullable=False, index=True)
  transferred_bytes: int = sql.Column(sql.BigInteger, nullable=False)

  # Grewe feature values.
  grewe1: float = sql.Column(sql.Float, nullable=False)
  grewe2: float = sql.Column(sql.Float, nullable=False)
  grewe3: float = sql.Column(sql.Float, nullable=False)
  grewe4: float = sql.Column(sql.Float, nullable=False)

  # Dynamic Features aggregates.

  cpu_opencl_env: str = sql.Column(sql.String(256), nullable=False, index=True)
  gpu_opencl_env: str = sql.Column(sql.String(256), nullable=False, index=True)

  # runtime = transfer + kernel time.
  cpu_runtime_ns: int = sql.Column(sql.BigInteger, nullable=False)
  gpu_runtime_ns: int = sql.Column(sql.BigInteger, nullable=False)

  # One of: {CPU, GPU}
  oracle: str = sql.Column(sql.String(3), nullable=False)
  oracle_runtime_ns: int = sql.Column(sql.BigInteger, nullable=False)
  max_speedup: float = sql.Column(sql.Float, nullable=False)


# TODO(cec): Implement!
# class CpuGpuClassificationResult(
#     Base, sqlutil.TablenameFromCamelCapsClassNameMixin):
#   id: int = sql.Column(sql.Integer, primary_key=True)
#   cpu_gpu_mapping_set_name: str = sql.Column(
#      sql.String(128),
#      sql.ForeignKey(CpuGpuMappingSet.cpu_gpu_mapping_set_name),
#      nullable=False)
#   # A stringifed JSON blob.
#   dataset_params: str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(),
#                                    nullable=False)
#
#   num_training_examples: int = sql.Column(sql.Integer, nullable=False)
#   num_validation_examples: int = sql.Column(sql.Integer, nullable=False)
#   num_test_examples: int = sql.Column(sql.Integer, nullable=False)
#
#   # The ratio of examples that belong to 'GPU' class.
#   training_gpu_ratio: float = sql.Column(sql.Float, nullable=False)
#   training_gpu_ratio: float = sql.Column(sql.Float, nullable=False)
#
#   model_name: str = sql.Column(sql.String(128), nullable=False)
#   # A stringifed JSON blob.
#   model_params: str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(),
#                                  nullable=False)
#
#   # The elapsed time for training and inference.
#   training_time_ms: int = sql.Column(sql.Integer, nullable=False)
#   validation_time_ms: int = sql.Column(sql.Integer, nullable=False)
#   test_time_ms: int = sql.Column(sql.Integer, nullable=False)


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

  @staticmethod
  def AggregateRuntimes(session: sqlutil.Session,
                        opencl_env: typing.Optional[str] = None,
                        min_run_count: int = 30) -> sqlutil.Query:
    """Produce a query which generates a table of runtime aggregates.

    Args:
      session: A database session.
      opencl_env: An optional environment to mask results from. If not provided,
        data is aggregated across all devices.

    Returns:
      A query.
    """
    subquery = session.query(
        # Group-by columns.
        DynamicFeatures.static_features_id,
        DynamicFeatures.driver,
        DynamicFeatures.gsize,
        DynamicFeatures.wgsize,
        DynamicFeatures.transferred_bytes,
        # Aggregated columns.
        sql.func.count(DynamicFeatures.transfer_time_ns).label("run_count"),
        sql.func.min(DynamicFeatures.transfer_time_ns).label("min_transfer_ns"),
        sql.sql.cast(
            sql.func.round(sql.func.avg(DynamicFeatures.transfer_time_ns)),
            sql.BigInteger).label("avg_transfer_ns"),
        sql.func.max(DynamicFeatures.transfer_time_ns).label("max_transfer_ns"),
        sql.func.min(DynamicFeatures.kernel_time_ns).label("min_kernel_ns"),
        sql.sql.cast(
            sql.func.round(sql.func.avg(DynamicFeatures.kernel_time_ns)),
            sql.BigInteger).label("avg_kernel_ns"),
        sql.func.max(DynamicFeatures.kernel_time_ns).label("max_kernel_ns"),
        sql.func.min(DynamicFeatures.transfer_time_ns +
                     DynamicFeatures.kernel_time_ns).label("min_runtime_ns"),
        sql.sql.cast(
            sql.func.round(
                sql.func.avg(DynamicFeatures.transfer_time_ns +
                             DynamicFeatures.kernel_time_ns)),
            sql.BigInteger).label("avg_runtime_ns"),
        sql.func.max(DynamicFeatures.transfer_time_ns +
                     DynamicFeatures.kernel_time_ns).label("max_runtime_ns"),
    ).filter(DynamicFeatures.outcome == 'PASS')

    if opencl_env:
      subquery = subquery.filter(DynamicFeatures.opencl_env == opencl_env)

    subquery = subquery.group_by(
        DynamicFeatures.static_features_id,
        DynamicFeatures.driver,
        DynamicFeatures.gsize,
        DynamicFeatures.wgsize,
        DynamicFeatures.transferred_bytes,
    ).subquery()

    return session.query(
        subquery.c.static_features_id,
        subquery.c.driver,
        subquery.c.gsize,
        subquery.c.wgsize,
        subquery.c.transferred_bytes,
        subquery.c.run_count,
        subquery.c.min_transfer_ns,
        subquery.c.avg_transfer_ns,
        subquery.c.max_transfer_ns,
        subquery.c.min_kernel_ns,
        subquery.c.avg_kernel_ns,
        subquery.c.max_kernel_ns,
        subquery.c.min_runtime_ns,
        subquery.c.avg_runtime_ns,
        subquery.c.max_runtime_ns,
    ).filter(subquery.c.run_count >= min_run_count)

  @classmethod
  def CpuGpuOracleMapping(cls,
                          session: sqlutil.Session,
                          cpu: str,
                          gpu: str,
                          min_run_count: int = 30) -> sqlutil.Query:
    """Produce a query that returns a CPU/GPU oracle mapping table.

    Args:
      session: A session instance.
      cpu: The name of the CPU device.
      gpu: The name of the GPU device.
      min_run_count: The minimum number of runs required on each device.

    Returns:
      A query.
    """
    cpu_q = cls.AggregateRuntimes(session, cpu, min_run_count).subquery()
    gpu_q = cls.AggregateRuntimes(session, gpu, min_run_count).subquery()

    return session.query(
        cpu_q.c.static_features_id,
        cpu_q.c.driver,
        cpu_q.c.gsize,
        cpu_q.c.wgsize,
        cpu_q.c.transferred_bytes,
        cpu_q.c.avg_runtime_ns.label('cpu_runtime_ns'),
        gpu_q.c.avg_runtime_ns.label('gpu_runtime_ns'),
        sql.func.if_(cpu_q.c.avg_runtime_ns < gpu_q.c.avg_runtime_ns,
                     sql.sql.literal('CPU'),
                     sql.sql.literal('GPU')).label("oracle"),
        sql.func.if_(cpu_q.c.avg_runtime_ns < gpu_q.c.avg_runtime_ns,
                     cpu_q.c.avg_runtime_ns,
                     gpu_q.c.avg_runtime_ns).label("oracle_runtime_ns"),
        sql.func.if_(cpu_q.c.avg_runtime_ns < gpu_q.c.avg_runtime_ns,
                     gpu_q.c.avg_runtime_ns / cpu_q.c.avg_runtime_ns,
                     cpu_q.c.avg_runtime_ns /
                     gpu_q.c.avg_runtime_ns).label("max_speedup"),
    ).join(
        gpu_q,
        sql.sql.and_(
            cpu_q.c.static_features_id == gpu_q.c.static_features_id,
            cpu_q.c.driver == gpu_q.c.driver,
            cpu_q.c.gsize == gpu_q.c.gsize,
            cpu_q.c.wgsize == gpu_q.c.wgsize,
            cpu_q.c.transferred_bytes == gpu_q.c.transferred_bytes,
        ))

  @classmethod
  def CreateCpuGpuDataset(cls, session: sqlutil.Session, dataset_name: str,
                          cpu: str, gpu: str,
                          min_run_count: int) -> sqlutil.Query:
    """Create a labelled CPU/GPU dataset.

    Args:
      session: A database session.
      dataset_name: A unique name for the generated dataset.
      cpu: The name of the opencl_env for the CPU.
      gpu: The name of the opencl_env for the GPU.
      min_run_count: The minimum number of runs to include an aggregate in the
        dataset.

    Raises:
      ValueError: If dataset_name already exists in the mapping table.
    """
    if session.query(CpuGpuMappingSet.cpu_gpu_mapping_set_name)\
        .filter(CpuGpuMappingSet.cpu_gpu_mapping_set_name == dataset_name)\
        .first():
      raise ValueError("Dataset name {dataset_name} already exists")

    devmap = cls.CpuGpuOracleMapping(session, cpu, gpu,
                                     min_run_count).subquery()

    grewe1 = (devmap.c.transferred_bytes /
              (StaticFeatures.grewe_compute_operation_count +
               StaticFeatures.grewe_global_memory_access_count))
    grewe2 = (StaticFeatures.grewe_coalesced_memory_access_count /
              StaticFeatures.grewe_global_memory_access_count)
    grewe3 = (
        devmap.c.wgsize * (StaticFeatures.grewe_local_memory_access_count /
                           StaticFeatures.grewe_global_memory_access_count))
    grewe4 = (StaticFeatures.grewe_compute_operation_count /
              StaticFeatures.grewe_global_memory_access_count)

    return session.query(
        devmap.c.gsize,
        devmap.c.wgsize,
        # demap column must appear first to anchor the FROM object in the join.
        sql.sql.literal(dataset_name).label('cpu_gpu_mapping_set_name'),
        StaticFeatures.id,
        StaticFeatures.origin,
        devmap.c.driver,
        devmap.c.transferred_bytes,
        grewe1.label('grewe1'),
        grewe2.label('grewe2'),
        grewe3.label('grewe3'),
        grewe4.label('grewe4'),
        sql.sql.literal(cpu).label('cpu_opencl_env'),
        sql.sql.literal(gpu).label('gpu_opencl_env'),
        devmap.c.cpu_runtime_ns,
        devmap.c.gpu_runtime_ns,
        devmap.c.oracle_runtime_ns,
        devmap.c.oracle,
        devmap.c.max_speedup,
    ).join(StaticFeatures)
