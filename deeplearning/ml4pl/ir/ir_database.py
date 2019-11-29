"""A database of compiler intermediate representation files."""
import datetime
import typing
import enum

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
  pickled_value: str = sql.Column(
    sqlutil.ColumnTypes.LargeBinary(), nullable=False
  )
  date_added: datetime.datetime = sql.Column(
    sql.DateTime().with_variant(mysql.DATETIME(fsp=3), "mysql"),
    nullable=False,
    default=labdate.GetUtcMillisecondsNow,
  )

  @property
  def value(self) -> typing.Any:
    return pickle.loads(self.pickled_value)

  @classmethod
  def Create(cls, key: str, value: typing.Any):
    return Meta(key=key, pickled_value=pickle.dumps(value))


class SourceLanguage(enum.Enum):
  """The valid types for parameters."""

  C = 1
  CPP = 2
  OPENCL = 3
  SWIFT = 4
  HASKELL = 5


class IrLanguage(enum.Enum):

  LLVM_6_0 = 1


class IntermediateRepresentationFile(
  Base, sqlutil.TablenameFromCamelCapsClassNameMixin
):
  """A table of compiler intermediate representation files."""
  id: int = sql.Column(sql.Integer, primary_key=True)

  # The following properties uniquely identify an IR file:

  # The name of the source of the intermediate representation.
  source: str = sql.Column(sql.String(256), nullable=False, index=True)
  # The relative path of the compiler intermediate representation file.
  relpath: str = sql.Column(sql.String(256), nullable=False)
  # The name of the language.
  source_language: str = sql.Column(SourceLanguage, nullable=False, index=True)
  ir_language: str = sql.Column(IrLanguage, nullable=False, index=True)

  compiler_arg_group_id: int = sql.Column(sql.Integer, sql.ForeignKey("compiler_args.arg_group_id"), nullable=True)

  # End of unique IR file attributes.

  char_count: int = sql.Column(sql.Integer, nullable=False)
  line_count: int = sql.Column(sql.Integer, nullable=False)

  contents_id: str = sql.Column(sql.Integer, nullable=True)
  error_id: str = sql.Column(sql.Integer, nullable=True)

  date_added: datetime.datetime = sql.Column(
    sql.DateTime().with_variant(mysql.DATETIME(fsp=3), "mysql"),
    nullable=False,
    default=labdate.GetUtcMillisecondsNow,
  )

  @property
  def contents_sha1(self):
    return self.contents.sha1

  @property
  def error_sha1(self):
    return self.error.sha1

  @property
  def compiler_args_as_string(self):
    return ' '.join([row.arg for row in self.compiler_args])

  compiler_args: CompilerArg = sql.orm.relationship(
    "ModelCheckpointMeta", cascade="all, delete-orphan"
  )

  __table_args__ = (
    sql.UniqueConstraint("source_language", "ir_language", "source", "relpath",
                         compiler_arg_group_id, name="unique_ir"),
  )

  @classmethod
  def Create(cls, proto: programl_pb2.IntermediateRepresentationFile) -> 'IntermediateRepresentationFile':
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



class IntermediateRepresentationFileContents(
    Base, sqlutil.TablenameFromCamelCapsClassNameMixin):
  id: int = sql.Column(sql.Integer, sql.ForeignKey("intermediate_representation_files.id") primary_key=True)

  contents: str = sql.Column(
    sqlutil.ColumnTypes.Un  # TODO(github.com/ChrisCummins/ProGraML/issues/6): Implement!
    nullable=False,
  )

class IntermediateRepresentationFileError(
    Base, sqlutil.TablenameFromCamelCapsClassNameMixin):
  id: int = sql.Column(sql.Integer, sql.ForeignKey("intermediate_representation_files.id") primary_key=True)

  error: str = sql.Column(
    sqlutil.ColumnTypes.Un  # TODO(github.com/ChrisCummins/ProGraML/issues/6): Implement!
    nullable=False,
  )

class CompilerArg(Base, sqlutil.TablenameFromCamelCapsClassNameMixin):
  id: int = sql.Column(sql.Integer, primary_key=True)

  arg_group_id: int = sql.Column(sql.Integer, index=True)
  arg: str = sql.Column(sql.String(1024), nullable=False)


class Database(sqlutil.Database):
  """A database of compiler intermediate representations."""

  def __init__(self, url: str, must_exist: bool = False):
    super(Database, self).__init__(url, Base, must_exist=must_exist)
