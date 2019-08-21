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

import sqlalchemy as sql

from labm8 import app
from labm8 import decorators
from labm8 import sqlutil

FLAGS = app.FLAGS

Base = sqlutil.Base()


class Corpus(Base):
  __tablename__ = 'corpuses'

  id: int = sql.Column(sql.Integer, primary_key=True)
  config_proto_sha1: str = sql.Column(sql.String(40), nullable=False)
  config_proto: str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(),
                                 nullable=False)
  preprocessed_url: str = sql.Column(sql.String(256), nullable=False)
  encoded_url: str = sql.Column(sql.String(256), nullable=False)

  __table_args__ = (sql.UniqueConstraint('config_proto_sha1',
                                         'preprocessed_url',
                                         'encoded_url',
                                         name='unique_corpus'),)


class Model(Base):
  __tablename__ = 'models'

  id: int = sql.Column(sql.Integer, primary_key=True)
  corpus_id: int = sql.Column(
      sql.Integer,
      nullable=True,
  )
  config_proto_sha1: str = sql.Column(sql.String(40), nullable=False)
  config_proto: str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(),
                                 nullable=False)
  cache_path: str = sql.Column(sql.String(256), nullable=False)

  __table_args__ = (sql.UniqueConstraint('corpus_id',
                                         'config_proto_sha1',
                                         'cache_path',
                                         name='unique_model'),)


class DashboardDatabase(sqlutil.Database):

  def __init__(self, url: str, must_exist: bool):
    super(DashboardDatabase, self).__init__(url, Base, must_exist=must_exist)


@decorators.run_once
def GetDatabase() -> DashboardDatabase:
  db: DashboardDatabase = FLAGS.clgen_dashboard_db()
  app.Log(1, 'Created dashboard database %s', db.url)
  return db
