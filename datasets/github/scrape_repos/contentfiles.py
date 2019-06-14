# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This file defines a database for importing cloned GitHub repos."""
import binascii
from sqlalchemy import orm

import datetime
import sqlalchemy as sql
import typing
from datasets.github.scrape_repos.proto import scrape_repos_pb2
from labm8 import app
from labm8 import labdate
from labm8 import sqlutil
from sqlalchemy.ext import declarative

FLAGS = app.FLAGS

Base = declarative.declarative_base()


class Meta(Base):
  """The meta table."""
  __tablename__ = 'meta'

  key: str = sql.Column(sql.String(512), primary_key=True)
  value: str = sql.Column(sql.String(1024), nullable=False)


class GitHubRepository(Base):
  """A GitHub repository record."""
  __tablename__ = 'repositories'

  owner: str = sql.Column(sql.String(512), nullable=False)
  name: str = sql.Column(sql.String(512), nullable=False)
  clone_from_url: str = sql.Column(
      sqlutil.ColumnTypes.IndexableString(), primary_key=True)
  num_stars: int = sql.Column(sql.Integer, nullable=False)
  num_forks: int = sql.Column(sql.Integer, nullable=False)
  num_watchers: int = sql.Column(sql.Integer, nullable=False)
  date_scraped: datetime.datetime = sql.Column(sql.DateTime, nullable=False)
  language: str = sql.Column(sql.String(16), nullable=False)

  @staticmethod
  def _GetArgsFromProto(proto: scrape_repos_pb2.GitHubRepoMetadata
                       ) -> typing.Dict[str, typing.Any]:
    date_scraped = labdate.DatetimeFromMillisecondsTimestamp(
        proto.scraped_utc_epoch_ms)
    return {
        "clone_from_url": proto.clone_from_url,
        "owner": proto.owner,
        "name": proto.name,
        "num_stars": proto.num_stars,
        "num_forks": proto.num_forks,
        "num_watchers": proto.num_watchers,
        "date_scraped": date_scraped,
    }

  @classmethod
  def FromProto(cls,
                proto: scrape_repos_pb2.GitHubRepoMetadata) -> \
      'GitHubRepository':
    return cls(**cls._GetArgsFromProto(proto))

  @classmethod
  def GetOrAdd(cls, session: orm.session.Session,
               proto: scrape_repos_pb2.GitHubRepoMetadata) -> \
      'GitHubRepository':
    return sqlutil.GetOrAdd(session, cls, **cls._GetArgsFromProto(proto))

  @classmethod
  def IsInDatabase(cls, session: orm.session.Session,
                   proto: scrape_repos_pb2.GitHubRepoMetadata) -> bool:
    filter_by = cls._GetArgsFromProto(proto)
    # Exclude scraped date from filter because otherwise a repository that is
    # scraped twice will be indexed twice.
    # TODO(cec): At some point it would be nice to have an expiration date for
    # scraped repos, after which the contents are considered "stale" and the
    # repo is re-scraped. This would require extending the behaviour of
    # ShouldImportRepo() to check the expiry date.
    del filter_by['date_scraped']
    instance = session.query(cls).filter_by(**filter_by).first()
    return True if instance else False


class ContentFile(Base):
  """A single content file record."""
  __tablename__ = 'contentfiles'

  id: int = sql.Column(sql.Integer, primary_key=True)
  clone_from_url: str = sql.Column(
      sqlutil.ColumnTypes.IndexableString(),
      sql.ForeignKey('repositories.clone_from_url'))
  # Relative path within the repository. This can be a duplicate.
  relpath: str = sql.Column(sql.String(1024), nullable=False)
  # Index into the content file. Use this to differentiate multiple content
  # files which come from the same source file.
  artifact_index: int = sql.Column(sql.Integer, nullable=False, default=0)
  sha256: str = sql.Column(sql.String(64), nullable=False)
  charcount = sql.Column(sql.Integer, nullable=False)
  linecount = sql.Column(sql.Integer, nullable=False)
  text: str = sql.Column(
      sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable=False)
  date_added: datetime.datetime = sql.Column(
      sql.DateTime, nullable=False, default=datetime.datetime.utcnow)
  __table_args__ = (sql.UniqueConstraint(
      'clone_from_url', 'relpath', 'artifact_index', name='uniq_contentfile'),)

  @staticmethod
  def _GetArgsFromProto(
      proto: scrape_repos_pb2.ContentFile) -> typing.Dict[str, typing.Any]:
    return {
        "clone_from_url": proto.clone_from_url,
        "relpath": proto.relpath,
        'artifact_index': proto.artifact_index,
        'sha256': proto.sha256,
        'charcount': proto.charcount,
        'linecount': proto.linecount,
        'text': proto.text
    }

  @classmethod
  def FromProto(cls, proto: scrape_repos_pb2.ContentFile) -> 'ContentFile':
    """Instantiate a record from a proto buffer.

    Args:
      proto: A ContentFile proto.

    Returns:
      A record instance.
    """
    return cls(**cls._GetArgsFromProto(proto))

  def SetProto(self, proto: scrape_repos_pb2.ContentFile
              ) -> scrape_repos_pb2.ContentFile:
    """Set fields of a protocol buffer representation.

    Returns:
      The proto buffer.
    """
    proto.clone_from_url = self.clone_from_url
    proto.relpath = self.relpath
    proto.artifact_index = self.artifact_index
    proto.sha256 = self.sha256
    proto.charcount = self.charcount
    proto.linecount = self.linecount
    proto.text = self.text
    return proto

  def ToProto(self) -> scrape_repos_pb2.ContentFile:
    """Create protocol buffer representation.

    Returns:
      A ContentFile message.
    """
    proto = scrape_repos_pb2.ContentFile()
    return self.SetProto(proto)


class ContentFiles(sqlutil.Database):
  """A database consisting of a table of ContentFiles and GitHub repos."""

  def __init__(self, url: str, must_exist: bool = False):
    super(ContentFiles, self).__init__(url, Base, must_exist)
