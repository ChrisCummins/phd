"""A database of compiler intermediate representation files."""
import datetime
import typing

import sqlalchemy as sql
from sqlalchemy.dialects import mysql
from sqlalchemy.ext import declarative

from deeplearning.ml4pl import ml4pl_pb2
from labm8.py import app
from labm8.py import labdate
from labm8.py import sqlutil

FLAGS = app.FLAGS

Base = declarative.declarative_base()


class Meta(Base, sqlutil.TablenameFromClassNameMixin):
  """Key-value database metadata store."""

  key: str = sql.Column(sql.String(64), primary_key=True)
  value: str = sql.Column(sql.String(64), nullable=False)


class IntermediateRepresentationFile(
  Base, sqlutil.TablenameFromCamelCapsClassNameMixin
):
  """A table of compiler intermediate representation files."""

  proto_t = ml4pl_pb2.LlvmBytecode

  id: int = sql.Column(sql.Integer, primary_key=True)

  # The name of the source of the bytecode.
  source: str = sql.Column(sql.String(256), nullable=False, index=True)
  # The relative path of the compiler intermediate representation file.
  relpath: str = sql.Column(sql.String(256), nullable=False)
  language: str = sql.Column(sql.String(16), nullable=False, index=True)
  cflags: str = sql.Column(sql.String(4096), nullable=False)
  charcount: int = sql.Column(sql.Integer, nullable=False)
  linecount: int = sql.Column(sql.Integer, nullable=False)
  bytecode: str = sql.Column(
    sql.UnicodeText().with_variant(sql.UnicodeText(2 ** 31), "mysql"),
    nullable=False,
  )
  clang_returncode: int = sql.Column(sql.Integer, nullable=False)
  error_message: str = sql.Column(
    sql.UnicodeText().with_variant(sql.UnicodeText(2 ** 31), "mysql"),
    nullable=False,
  )

  bytecode_sha1: str = sql.Column(sql.String(40), nullable=False, index=True)

  __table_args__ = (
    sql.UniqueConstraint("source_name", "relpath", name="unique_bytecode"),
  )

  date_added: datetime.datetime = sql.Column(
    sql.DateTime().with_variant(mysql.DATETIME(fsp=3), "mysql"),
    nullable=False,
    default=labdate.GetUtcMillisecondsNow,
  )

  @classmethod
  def FromProto(cls, proto: proto_t) -> typing.Dict[str, typing.Any]:
    """Return a dictionary of instance constructor args from proto."""
    return {
      "source_name": proto.source_name,
      "relpath": proto.relpath,
      "language": proto.lang,
      "cflags": proto.cflags,
      "charcount": len(proto.bytecode),
      "linecount": len(proto.bytecode.split("\n")),
      "bytecode": proto.bytecode,
      "clang_returncode": proto.clang_returncode,
      "error_message": proto.error_message,
    }


class Database(sqlutil.Database):
  def __init__(self, url: str, must_exist: bool = False):
    super(Database, self).__init__(url, Base, must_exist=must_exist)
