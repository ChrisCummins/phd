"""Database backend for experimental data."""
import sqlalchemy as sql
from absl import app
from absl import flags
from sqlalchemy.ext import declarative

from experimental.compilers.reachability import reachability_pb2
from labm8 import sqlutil


FLAGS = flags.FLAGS

Base = declarative.declarative_base()


class Bytecode(Base, sqlutil.ProtoBackedMixin,
               sqlutil.TablenameFromClassNameMixin):
  """A table of Llvm bytecodes."""
  proto_t = reachability_pb2.Bytecode

  id: int = sql.Column(sql.Integer, primary_key=True)

  source_name: str = sql.Column(sql.String(256), nullable=False)
  relpath: str = sql.Column(sql.String(256), nullable=False)
  lang: str = sql.Column(sql.String(32), nullable=False)
  cflags: str = sql.Column(sql.String(1024), nullable=False)
  charcount: int = sql.Column(sql.Integer, nullable=False)
  linecount: int = sql.Column(sql.Integer, nullable=False)
  bytecode: str = sql.Column(
      sql.UnicodeText().with_variant(sql.UnicodeText(2 ** 31), 'mysql'),
      nullable=False)
  clang_returncode: int = sql.Column(sql.Integer, nullable=False)
  error_message: str = sql.Column(
      sql.UnicodeText().with_variant(sql.UnicodeText(2 ** 31), 'mysql'))

  def SetProto(self, proto: proto_t) -> None:
    raise NotImplementedError


class Database(sqlutil.Database):
  pass


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))


if __name__ == '__main__':
  app.run(main)
