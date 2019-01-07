"""Database backend for experimental data."""
import typing

import sqlalchemy as sql
from absl import app
from absl import flags
from sqlalchemy.ext import declarative

from experimental.compilers.reachability import reachability_pb2
from labm8 import sqlutil


FLAGS = flags.FLAGS

Base = declarative.declarative_base()


class LlvmBytecode(Base, sqlutil.ProtoBackedMixin,
                   sqlutil.TablenameFromClassNameMixin):
  """A table of Llvm bytecodes."""
  proto_t = reachability_pb2.LlvmBytecode

  id: int = sql.Column(sql.Integer, primary_key=True)

  source_name: str = sql.Column(sql.String(256), nullable=False)
  relpath: str = sql.Column(sql.String(256), nullable=False)
  language: str = sql.Column(sql.String(32), nullable=False)
  cflags: str = sql.Column(sql.String(4096), nullable=False)
  charcount: int = sql.Column(sql.Integer, nullable=False)
  linecount: int = sql.Column(sql.Integer, nullable=False)
  bytecode: str = sql.Column(
      sql.UnicodeText().with_variant(sql.UnicodeText(2 ** 31), 'mysql'),
      nullable=False)
  clang_returncode: int = sql.Column(sql.Integer, nullable=False)
  error_message: str = sql.Column(
      sql.UnicodeText().with_variant(sql.UnicodeText(2 ** 31), 'mysql'),
      nullable=False)

  # TODO(cec): Add unique constraint on source_name and relpath.

  @classmethod
  def FromProto(cls, proto: proto_t) -> typing.Dict[str, typing.Any]:
    """Return a dictionary of instance constructor args from proto."""
    return {
      'source_name': proto.source_name,
      'relpath': proto.relpath,
      'language': proto.lang,
      'cflags': proto.cflags,
      'charcount': len(proto.bytecode),
      'linecount': len(proto.bytecode.split('\n')),
      'bytecode': proto.bytecode,
      'clang_returncode': proto.clang_returncode,
      'error_message': proto.error_message,
    }


class ControlFlowGraphProto(Base, sqlutil.ProtoBackedMixin,
                            sqlutil.TablenameFromClassNameMixin):
  """A table of CFG protos."""

  proto_t = reachability_pb2.ControlFlowGraphFromLlvmBytecode

  bytecode_id: int = sql.Column(
      sql.Integer, sql.ForeignKey(LlvmBytecode.id), nullable=False)
  cfg_id: int = sql.Column(
      sql.Integer, nullable=False)

  # Composite primary key.
  __table_args__ = (
    sql.PrimaryKeyConstraint('bytecode_id', 'cfg_id', name='unique_id'),
  )

  status: int = sql.Column(sql.Integer, nullable=False)
  proto: str = sql.Column(
      sql.UnicodeText().with_variant(sql.UnicodeText(2 ** 31), 'mysql'),
      nullable=False)
  error_message: str = sql.Column(
      sql.UnicodeText().with_variant(sql.UnicodeText(2 ** 31), 'mysql'),
      nullable=False)
  block_count: int = sql.Column(sql.Integer, nullable=False)
  edge_count: int = sql.Column(sql.Integer, nullable=False)
  is_strict_valid: bool = sql.Column(sql.Boolean, nullable=False)

  @classmethod
  def FromProto(cls, proto: proto_t) -> typing.Dict[str, typing.Any]:
    """Return a dictionary of instance constructor args from proto."""
    return {
      'bytecode_id': proto.bytecode_id,
      'cfg_id': proto.cfg_id,
      'proto': proto.control_flow_graph,
      'status': proto.status,
      'error_message': proto.error_message,
      'block_count': proto.block_count,
      'edge_count': proto.edge_count,
      'is_strict_valid': proto.is_strict_valid,
    }


class FullFlowGraphProto(Base, sqlutil.ProtoBackedMixin,
                         sqlutil.TablenameFromClassNameMixin):
  """Table of full flow graphs."""

  proto_t = reachability_pb2.ControlFlowGraphFromLlvmBytecode

  bytecode_id: int = sql.Column(
      sql.Integer, nullable=False)
  cfg_id: int = sql.Column(
      sql.Integer, nullable=False)

  # Composite primary key.
  __table_args__ = (
    sql.ForeignKeyConstraint(
        ['bytecode_id', 'cfg_id'],
        [ControlFlowGraphProto.bytecode_id, ControlFlowGraphProto.cfg_id]),
    sql.PrimaryKeyConstraint('bytecode_id', 'cfg_id', name='unique_id'),
  )

  status: int = sql.Column(sql.Integer, nullable=False)
  serialized_proto: str = sql.Column(
      sql.LargeBinary().with_variant(sql.LargeBinary(2 ** 31), 'mysql'),
      nullable=False)
  error_message: str = sql.Column(
      sql.UnicodeText().with_variant(sql.UnicodeText(2 ** 31), 'mysql'),
      nullable=False)
  block_count: int = sql.Column(sql.Integer, nullable=False)
  edge_count: int = sql.Column(sql.Integer, nullable=False)
  is_strict_valid: bool = sql.Column(sql.Boolean, nullable=False)

  @classmethod
  def FromProto(cls, proto: proto_t) -> typing.Dict[str, typing.Any]:
    """Return a dictionary of instance constructor args from proto."""
    return {
      'bytecode_id': proto.bytecode_id,
      'cfg_id': proto.cfg_id,
      'serialized_proto': proto.control_flow_graph.SerializeToString(),
      'status': proto.status,
      'error_message': proto.error_message,
      'block_count': proto.block_count,
      'edge_count': proto.edge_count,
      'is_strict_valid': proto.is_strict_valid,
    }


class Database(sqlutil.Database):

  def __init__(self, url: str):
    super(Database, self).__init__(url, Base)


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))


if __name__ == '__main__':
  app.run(main)
