"""A database for incremental-synthesis experimental results."""
import datetime
import typing

import numpy as np
import sqlalchemy as sql
from sqlalchemy.ext import declarative

from labm8 import app
from labm8 import sqlutil
from research.grewe_2013_cgo import feature_extractor as grewe_features

FLAGS = app.FLAGS

Base = declarative.declarative_base()


class FeatureVector(Base, sqlutil.TablenameFromCamelCapsClassNameMixin):
  """A table of feature vectors."""
  id: int = sql.Column(sql.Integer, primary_key=True)

  # Raw values of Grewe et. al. feature space.
  compute_operation_count: int = sql.Column(sql.Integer, nullable=False)
  global_memory_access_count: int = sql.Column(sql.Integer, nullable=False)
  local_memory_access_count: int = sql.Column(sql.Integer, nullable=False)
  coalesced_memory_access_count: int = sql.Column(sql.Integer, nullable=False)

  __table_args__ = (
      # <src,origin> pairs must be unique.
      sql.UniqueConstraint(
          'compute_operation_count',
          'global_memory_access_count',
          'local_memory_access_count',
          'coalesced_memory_access_count',
          name='unique_feature_vector'),)

  @staticmethod
  def FromFeatureTuple(
      features: grewe_features.GreweEtAlFeatures) -> typing.Dict[str, int]:
    """Instantiate a PreprocessedContentFile."""
    return {
        'compute_operation_count': features.compute_operation_count,
        'global_memory_access_count': features.global_memory_access_count,
        'local_memory_access_count': features.local_memory_access_count,
        'coalesced_memory_access_count': features.coalesced_memory_access_count
    }

  @staticmethod
  def FromNumpyArray(features: np.array) -> typing.Dict[str, int]:
    assert features.shape == (4,)
    return {
        'compute_operation_count': features[0],
        'global_memory_access_count': features[1],
        'local_memory_access_count': features[2],
        'coalesced_memory_access_count': features[3]
    }

  def ToNumpyArray(self) -> np.array:
    return np.array([
        self.compute_operation_count,
        self.global_memory_access_count,
        self.local_memory_access_count,
        self.coalesced_memory_access_count,
    ],
                    dtype=int)


class BacktrackingStep(Base, sqlutil.TablenameFromCamelCapsClassNameMixin):
  id: int = sql.Column(sql.Integer, primary_key=True)

  # A unique ID used to group backtracking jobs.
  job_id: int = sql.Column(sql.Integer, nullable=False)

  # The runtime of the job.
  runtime_ms: int = sql.Column(sql.Integer, nullable=False)

  # Features.
  target_features_id: int = sql.Column(
      sql.Integer, sql.ForeignKey(FeatureVector.id), nullable=False)
  features_id: int = sql.Column(
      sql.Integer, sql.ForeignKey(FeatureVector.id), nullable=False)
  feature_distance: float = sql.Column(sql.Float, nullable=False)
  # Feature distance, but normalized to the starting feature distance. In range
  # [0,1] (assuming hill climbing).
  norm_feature_distance: float = sql.Column(sql.Float, nullable=False)

  # The backtracking step number.
  step: int = sql.Column(sql.Integer, nullable=False)
  # The number of rejected attempts before reaching the current step.
  attempt_count: int = sql.Column(sql.Integer, nullable=False)

  # A timestamp.
  date: datetime.datetime = sqlutil.ColumnFactory.MillisecondDatetime()

  # The kernel source.
  token_count: int = sql.Column(sql.Integer, nullable=False)
  src: str = sql.Column(
      sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable=False)

  # Relationship.
  target_features: FeatureVector = sql.orm.relationship(
      'FeatureVector', foreign_keys=[target_features_id])
  features: FeatureVector = sql.orm.relationship(
      'FeatureVector', foreign_keys=[features_id])


class Database(sqlutil.Database):
  """Database of kernels."""

  def __init__(self, url: str):
    super(Database, self).__init__(url, Base)
