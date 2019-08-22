# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""This file defines telemetry data gathers."""
import datetime

import sqlalchemy as sql
from sqlalchemy.dialects import mysql

from labm8 import app
from labm8 import decorators
from labm8 import labdate
from labm8 import sqlutil

FLAGS = app.FLAGS

Base = sqlutil.Base()


class DashboardDatabase(sqlutil.Database):

  def __init__(self, url: str, must_exist: bool):
    super(DashboardDatabase, self).__init__(url, Base, must_exist=must_exist)


app.DEFINE_database('clgen_dashboard_db', DashboardDatabase,
                    'sqlite:////tmp/phd/deeplearning/clgen/dashboard.db',
                    'URL of the dashboard database.')


class Corpus(Base):
  __tablename__ = 'corpuses'

  id: int = sql.Column(sql.Integer, primary_key=True)
  config_proto_sha1: str = sql.Column(sql.String(40), nullable=False)
  config_proto: str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(),
                                 nullable=False)
  preprocessed_url: str = sql.Column(sql.String(256), nullable=False)
  encoded_url: str = sql.Column(sql.String(256), nullable=False)
  summary: str = sql.Column(sql.String(256), nullable=False)

  __table_args__ = (sql.UniqueConstraint('config_proto_sha1',
                                         'preprocessed_url',
                                         'encoded_url',
                                         name='unique_corpus'),)


class Model(Base):
  __tablename__ = 'models'

  id: int = sql.Column(sql.Integer, primary_key=True)
  corpus_id: int = sql.Column(
      sql.Integer,
      sql.ForeignKey('corpuses.id'),
      nullable=False,
  )
  config_proto_sha1: str = sql.Column(sql.String(40), nullable=False)
  config_proto: str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(),
                                 nullable=False)
  cache_path: str = sql.Column(sql.String(256), nullable=False)
  summary: str = sql.Column(sql.String(256), nullable=False)

  corpus: Corpus = sql.orm.relationship('Corpus')
  __table_args__ = (sql.UniqueConstraint('corpus_id',
                                         'config_proto_sha1',
                                         'cache_path',
                                         name='unique_model'),)


class TrainingTelemetry(Base):
  __tablename__ = 'training_telemetry'

  id: int = sql.Column(sql.Integer, primary_key=True)
  model_id: int = sql.Column(
      sql.Integer,
      sql.ForeignKey('models.id'),
      nullable=False,
  )
  timestamp: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), 'mysql'),
      nullable=False,
      default=labdate.GetUtcMillisecondsNow)
  epoch: int = sql.Column(sql.Integer, nullable=False)
  step: int = sql.Column(sql.Integer, nullable=False)
  training_loss: float = sql.Column(sql.Float, nullable=False)
  learning_rate: float = sql.Column(sql.Float, nullable=False)
  ns_per_batch: int = sql.Column(sql.Integer, nullable=False)

  pending: bool = sql.Column(sql.Boolean, nullable=False, default=True)

  model: Model = sql.orm.relationship('Model')
  __table_args__ = (sql.UniqueConstraint('model_id',
                                         'epoch',
                                         'step',
                                         name='unique_telemetry'),)


class TrainingSample(Base):
  __tablename__ = 'training_samples'

  id: int = sql.Column(sql.Integer, primary_key=True)
  model_id: int = sql.Column(
      sql.Integer,
      sql.ForeignKey('models.id'),
      nullable=False,
  )
  epoch: int = sql.Column(sql.Integer, nullable=False)
  step: int = sql.Column(sql.Integer, nullable=False)
  token_count: int = sql.Column(sql.Integer, nullable=False)
  sample_time: int = sql.Column(sql.Integer, nullable=False)
  sample: str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(),
                           nullable=False)

  model: Model = sql.orm.relationship('Model')


@decorators.run_once
def GetDatabase() -> DashboardDatabase:
  db: DashboardDatabase = FLAGS.clgen_dashboard_db()
  app.Log(1, 'Created dashboard database %s', db.url)
  return db
