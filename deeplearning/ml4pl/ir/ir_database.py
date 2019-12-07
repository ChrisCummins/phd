"""A database of compiler intermediate representation files.

When executed as a script, this prints summary statistics of the contents of
the database.
"""
import codecs
import datetime
import enum
import pickle
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import sqlalchemy as sql

from deeplearning.ml4pl import run_id as run_id_lib
from labm8.py import app
from labm8.py import crypto
from labm8.py import humanize
from labm8.py import jsonutil
from labm8.py import progress
from labm8.py import sqlutil

FLAGS = app.FLAGS

Base = sql.ext.declarative.declarative_base()


class Meta(Base, sqlutil.TablenameFromClassNameMixin):
  """Key-value database metadata store."""

  # Unused integer ID for this row.
  id: int = sql.Column(sql.Integer, primary_key=True)

  # The run ID that generated this <key,value> pair.
  run_id: str = run_id_lib.RunId.SqlStringColumn()

  timestamp: datetime.datetime = sqlutil.ColumnFactory.MillisecondDatetime()

  # The <key,value> pair.
  key: str = sql.Column(sql.String(128), index=True)
  pickled_value: bytes = sql.Column(
    sqlutil.ColumnTypes.LargeBinary(), nullable=False
  )

  @property
  def value(self) -> Any:
    """De-pickle the column value."""
    return pickle.loads(self.pickled_value)

  @classmethod
  def Create(cls, key: str, value: Any):
    """Construct a table entry."""
    return Meta(key=key, pickled_value=pickle.dumps(value))


class SourceLanguage(enum.Enum):
  """Source languages."""

  C = 1
  CPP = 2
  OPENCL = 3
  SWIFT = 4
  HASKELL = 5
  FORTRAN = 6


class IrType(enum.Enum):
  """Intermediate representation types."""

  LLVM_3_5 = 1
  LLVM_6_0 = 2


class IntermediateRepresentation(
  Base, sqlutil.TablenameFromCamelCapsClassNameMixin
):
  """A table of compiler intermediate representation files."""

  id: int = sql.Column(sql.Integer, primary_key=True)

  # The name of the source of the intermediate representation.
  source: str = sql.Column(sql.String(256), nullable=False, index=True)

  # The relative path of the compiler intermediate representation file.
  relpath: str = sql.Column(sql.String(256), nullable=False)

  # The source language.
  source_language: SourceLanguage = sql.Column(
    sql.Enum(SourceLanguage), nullable=False, index=True
  )

  # The intermediate representation type.
  type: IrType = sql.Column(sql.Enum(IrType), nullable=False, index=True)

  # The compiler arguments used to generate the intermediate representation.
  cflags_sha1: str = sql.Column(sql.String(40), nullable=False)
  cflags: str = sql.Column(sql.String(4096), nullable=False)

  # A marker to indicate whether compilation of the intermediate representation
  # succeeded.
  compilation_succeeded: bool = sql.Column(sql.Boolean, nullable=False)

  # The size of the
  char_count: int = sql.Column(sql.Integer, nullable=True)
  line_count: int = sql.Column(sql.Integer, nullable=True)

  # The intermediate representation is stored as a binary zlib-compressed blob.
  ir_sha1: str = sql.Column(sql.String(40), nullable=False, index=True)
  # The size of the binary IR blob.
  binary_ir_size: int = sql.Column(sql.Integer, nullable=False)
  binary_ir: bytes = sql.Column(
    sqlutil.ColumnTypes.LargeBinary(), nullable=False
  )

  @property
  def ir(self) -> Any:
    """Return the intermediate representation."""
    return pickle.loads(codecs.decode(self.details.binary_true_y, "zlib"))

  timestamp: datetime.datetime = sqlutil.ColumnFactory.MillisecondDatetime()

  @property
  def compiler_args(self) -> str:
    """Get the compiler args as a string."""
    return self.compiler_args_row.string

  __table_args__ = (
    sql.UniqueConstraint(
      "source",
      "relpath",
      "source_language",
      "type",
      "cflags_sha1",
      name="unique_ir",
    ),
  )

  @classmethod
  def CreateEmpty(
    cls,
    source: str,
    relpath: str,
    source_language: SourceLanguage,
    type: IrType,
    cflags: str,
  ) -> "IntermediateRepresentation":
    """Construct an "empty" intermediate representation, i.e. one where
    compilation failed.

    Args:
      source: The soure name.
      relpath: The relpath of the file.
      source_language: The source language.
      type: The IR type.
      cflags: The compilation flags.

    Returns:
      An IntermediateRepresentation instance.
    """
    return cls(
      source=source,
      relpath=relpath,
      source_language=source_language,
      type=type,
      cflags_sha1=crypto.sha1_str(cflags),
      cflags=cflags,
      compilation_succeeded=False,
      char_count=0,
      line_count=0,
      ir_sha1="",
      binary_ir_size=0,
      binary_ir=codecs.encode(pickle.dumps(""), "zlib"),
    )

  @classmethod
  def CreateFromText(
    cls,
    source: str,
    relpath: str,
    source_language: SourceLanguage,
    type: IrType,
    cflags: str,
    text: str,
  ) -> "IntermediateRepresentation":
    """Construct from textual intermediate representation.

    Args:
      source: The soure name.
      relpath: The relpath of the file.
      source_language: The source language.
      type: The IR type.
      cflags: The compilation flags.
      text: The textual IR.

    Returns:
      An IntermediateRepresentation instance.
    """
    if len(source) > 256:
      raise TypeError("source column is too long")
    if len(relpath) > 256:
      raise TypeError("relpath column is too long")
    if len(cflags) > 4096:
      raise TypeError("cflags column is too long")
    binary_ir = codecs.encode(pickle.dumps(text), "zlib")
    return cls(
      source=source,
      relpath=relpath,
      source_language=source_language,
      type=type,
      cflags_sha1=crypto.sha1_str(cflags),
      cflags=cflags,
      compilation_succeeded=True,
      char_count=len(text),
      line_count=len(text.split("\n")),
      ir_sha1=crypto.sha1(binary_ir),
      binary_ir_size=len(binary_ir),
      binary_ir=binary_ir,
    )


# A registry of database statics, where each entry is a <name, property> tuple.
database_statistics_registry: List[Tuple[str, Callable[["Database"], Any]]] = []


def database_statistic(func):
  """A decorator to mark a method on a Database as a database static.

  Database statistics can be accessed using Database.stats_json property to
  retrieve a <name, vale> dictionary.
  """
  global database_statistics_registry
  database_statistics_registry.append((func.__name__, func))
  return property(func)


class Database(sqlutil.Database):
  """A database of compiler intermediate representations."""

  def __init__(
    self,
    url: str,
    must_exist: bool = False,
    ctx: progress.ProgressContext = progress.NullContext,
  ):
    super(Database, self).__init__(url, Base, must_exist=must_exist)
    self.ctx = ctx

    # Lazily evaluated attributes.
    self._db_stats = None

  @database_statistic
  def ir_count(self) -> int:
    """The number of non-empty IRs in the database."""
    return self.db_stats.ir_count

  @database_statistic
  def unique_ir_count(self) -> int:
    """The number of unique non-empty IRs in the database."""
    return self.db_stats.unique_ir_count

  @database_statistic
  def ir_data_size(self) -> int:
    """The sum of non-empty IR binary data sizes."""
    return self.db_stats.ir_data_size or 0

  @database_statistic
  def char_count(self) -> int:
    """The sum of non-empty IR char counts."""
    return self.db_stats.char_count or 0

  @database_statistic
  def line_count(self) -> int:
    """The sum of non-empty IR line counts."""
    return self.db_stats.line_count or 0

  def RefreshStats(self):
    """Compute the database stats for access via the instance properties.

    Raises:
      ValueError: If the database contains invalid entries, e.g. inconsistent
        vector dimensionalities.
    """
    with self.ctx.Profile(
      2,
      lambda t: (
        "Computed stats over "
        f"{humanize.BinaryPrefix(stats.ir_data_size or 0, 'B')} database "
        f"({humanize.Plural(stats.ir_count, 'intermediate representation')})"
      ),
    ), self.Session() as session:
      query = session.query(
        sql.func.count(IntermediateRepresentation.id).label("ir_count"),
        sql.func.count(
          sql.func.distinct(IntermediateRepresentation.ir_sha1)
        ).label("unique_ir_count"),
        sql.func.sum(IntermediateRepresentation.binary_ir_size).label(
          "ir_data_size"
        ),
        sql.func.sum(IntermediateRepresentation.char_count).label("char_count"),
        sql.func.sum(IntermediateRepresentation.line_count).label("line_count"),
      )

      # Ignore "empty" IRs.
      query = query.filter(
        IntermediateRepresentation.compilation_succeeded == True
      )

      # Compute the stats.
      stats = query.one()

      self._db_stats = stats

  @property
  def db_stats(self):
    """Fetch aggregate database stats, or compute them if not set."""
    if self._db_stats is None:
      self.RefreshStats()
    return self._db_stats

  @property
  def stats_json(self) -> Dict[str, Any]:
    """Fetch the database statics as a JSON dictionary."""
    return {
      name: function(self) for name, function in database_statistics_registry
    }


app.DEFINE_database(
  "ir_db", Database, None, "A database of intermediate representations."
)


def Main():
  """Main entry point."""
  ir_db = FLAGS.ir_db()
  print(jsonutil.format_json(ir_db.stats_json))


if __name__ == "__main__":
  app.Run(Main)
