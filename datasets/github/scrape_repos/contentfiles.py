"""This file defines a database for importing cloned GitHub repos."""
import binascii
import datetime
import pathlib
import sqlalchemy as sql
import typing
from absl import flags
from phd.lib.labm8 import labdate
from phd.lib.labm8 import sqlutil
from sqlalchemy import orm
from sqlalchemy.ext import declarative

from datasets.github.scrape_repos.proto import scrape_repos_pb2


FLAGS = flags.FLAGS

Base = declarative.declarative_base()


class Meta(Base):
  """The meta table."""
  __tablename__ = 'meta'

  key: str = sql.Column(sql.String(1024), primary_key=True)
  value: str = sql.Column(sql.String(1024), nullable=False)


class GitHubRepository(Base):
  """A GitHub repository record."""
  __tablename__ = 'repositories'

  owner: str = sql.Column(sql.String(512), nullable=False)
  name: str = sql.Column(sql.String(512), nullable=False)
  clone_from_url: str = sql.Column(sql.String(1024), primary_key=True)
  num_stars: int = sql.Column(sql.Integer, nullable=False)
  num_forks: int = sql.Column(sql.Integer, nullable=False)
  num_watchers: int = sql.Column(sql.Integer, nullable=False)
  date_scraped: datetime.datetime = sql.Column(sql.DateTime, nullable=False)
  language: str = sql.Column(sql.String(16), nullable=False)

  @staticmethod
  def _GetArgsFromProto(proto: scrape_repos_pb2.GitHubRepoMetadata) -> \
      typing.Dict[str, typing.Any]:
    date_scraped = labdate.DatetimeFromMillisecondsTimestamp(
        proto.scraped_utc_epoch_ms)
    return {"clone_from_url": proto.clone_from_url, "owner": proto.owner,
            "name": proto.name, "num_stars": proto.num_stars,
            "num_forks": proto.num_forks, "num_watchers": proto.num_watchers,
            "date_scraped": date_scraped, }

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
    instance = session.query(cls).filter_by(
        **cls._GetArgsFromProto(proto)).first()
    return True if instance else False


class ContentFile(Base):
  """A single content file record."""
  __tablename__ = 'contentfiles'

  id: int = sql.Column(sql.Integer, primary_key=True)
  clone_from_url: str = sql.Column(sql.String(1024), sql.ForeignKey(
      'repositories.clone_from_url'))
  # Relative path within the repository. This can be a duplicate.
  relpath: str = sql.Column(sql.String(1024), nullable=False)
  # Index into the content file. Use this to differentiate multiple content
  # files which come from the same source file.
  artifact_index: int = sql.Column(sql.Integer, nullable=False, default=0)
  sha256: str = sql.Column(sql.Binary(32), nullable=False)
  charcount = sql.Column(sql.Integer, nullable=False)
  linecount = sql.Column(sql.Integer, nullable=False)
  text: str = sql.Column(sql.UnicodeText(), nullable=False)
  date_added: datetime.datetime = sql.Column(sql.DateTime, nullable=False,
                                             default=datetime.datetime.utcnow)
  __table_args__ = (
    sql.UniqueConstraint('clone_from_url', 'relpath', 'artifact_index',
                         name='uniq_contentfile'),)

  @property
  def sha256_hex(self) -> str:
    """Return the 64 character hexadecimal representation of binary sha256."""
    return binascii.hexlify(self.sha256).decode('utf-8')


class ContentFiles(sqlutil.Database):
  """A database consisting of a table of ContentFiles and GitHub repos."""

  def __init__(self, path: pathlib.Path):
    self.database_path = path.absolute()
    self.database_uri = f'sqlite:///{self.database_path}'
    self.engine = sql.create_engine(self.database_uri, encoding='utf-8')
    Base.metadata.create_all(self.engine)
    Base.metadata.bind = self.engine
    self.make_session = orm.sessionmaker(bind=self.engine)
