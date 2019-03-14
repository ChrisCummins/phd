# Copyright (c) 2017, 2018, 2019 Chris Cummins.
#
# DeepSmith is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepSmith is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepSmith.  If not, see <https://www.gnu.org/licenses/>.
"""This file defines the testcase generator."""
import binascii
import datetime
import hashlib
import typing

import sqlalchemy as sql
from sqlalchemy import orm
from sqlalchemy.dialects import mysql

import labm8.sqlutil
from deeplearning.deepsmith import db
from deeplearning.deepsmith.proto import deepsmith_pb2
from labm8 import labdate

# The index types for tables defined in this file.
_GeneratorId = sql.Integer
_GeneratorOptSetId = sql.Binary(16).with_variant(mysql.BINARY(16), 'mysql')
_GeneratorOptId = sql.Integer
_GeneratorOptNameId = db.StringTable.id_t
_GeneratorOptValueId = db.StringTable.id_t


class Generator(db.Table):
  id_t = _GeneratorId
  __tablename__ = 'generators'

  # Columns.
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), 'mysql'),
      nullable=False,
      default=labdate.GetUtcMillisecondsNow)
  # MySQL maximum key length is 3072, with 3 bytes per character. We must
  # preserve 16 bytes for the unique constraint.
  name: str = sql.Column(
      sql.String(4096).with_variant(sql.String((3072 - 16) // 3), 'mysql'),
      nullable=False)
  optset_id: bytes = sql.Column(_GeneratorOptSetId, nullable=False)

  # Relationships.
  testcases: typing.List['Testcase'] = orm.relationship(
      'Testcase', back_populates='generator')
  optset: typing.List['GeneratorOpt'] = orm.relationship(
      'GeneratorOpt',
      secondary='generator_optsets',
      primaryjoin='GeneratorOptSet.id == Generator.optset_id',
      secondaryjoin='GeneratorOptSet.opt_id == GeneratorOpt.id')

  # Constraints.
  __table_args__ = (sql.UniqueConstraint(
      'name', 'optset_id', name='unique_generator'),)

  @property
  def opts(self) -> typing.Dict[str, str]:
    """Get the generator options.

    Returns:
      A map of generator options.
    """
    return {opt.name.string: opt.value.string for opt in self.optset}

  def SetProto(self, proto: deepsmith_pb2.Generator) -> deepsmith_pb2.Generator:
    """Set a protocol buffer representation.

    Args:
      proto: A protocol buffer message.

    Returns:
      A Generator message.
    """
    proto.name = self.name
    for opt in self.optset:
      proto.opts[opt.name.string] = opt.value.string
    return proto

  def ToProto(self) -> deepsmith_pb2.Generator:
    """Create protocol buffer representation.

    Returns:
      A Generator message.
    """
    proto = deepsmith_pb2.Generator()
    return self.SetProto(proto)

  @classmethod
  def GetOrAdd(cls, session: db.session_t,
               proto: deepsmith_pb2.Generator) -> 'Generator':

    # Build the list of options, and md5sum the key value strings.
    opts = []
    md5 = hashlib.md5()
    for proto_opt_name in sorted(proto.opts):
      proto_opt_value = proto.opts[proto_opt_name]
      md5.update((proto_opt_name + proto_opt_value).encode('utf-8'))
      opt = labm8.sqlutil.GetOrAdd(
          session,
          GeneratorOpt,
          name=GeneratorOptName.GetOrAdd(session, proto_opt_name),
          value=GeneratorOptValue.GetOrAdd(session, proto_opt_value),
      )
      opts.append(opt)

    # Create optset table entries.
    optset_id = md5.digest()
    for opt in opts:
      labm8.sqlutil.GetOrAdd(session, GeneratorOptSet, id=optset_id, opt=opt)

    return labm8.sqlutil.GetOrAdd(
        session,
        cls,
        name=proto.name,
        optset_id=optset_id,
    )


class GeneratorOptSet(db.Table):
  """A set of of generator options.

  An option set groups options for generators.
  """
  __tablename__ = 'generator_optsets'
  id_t = _GeneratorOptSetId

  # Columns.
  id: bytes = sql.Column(id_t, nullable=False)
  opt_id: int = sql.Column(
      _GeneratorOptId, sql.ForeignKey('generator_opts.id'), nullable=False)

  # Relationships.
  generators: typing.List[Generator] = orm.relationship(
      Generator, primaryjoin=id == orm.foreign(Generator.optset_id))
  opt: 'GeneratorOpt' = orm.relationship('GeneratorOpt')

  # Constraints.
  __table_args__ = (sql.PrimaryKeyConstraint(
      'id', 'opt_id', name='unique_generator_optset'),)

  def __repr__(self):
    hex_id = binascii.hexlify(self.id).decode('utf-8')
    return f'{hex_id}: {self.opt_id}={self.opt}'


class GeneratorOpt(db.Table):
  """A generator option consists of a <name, value> pair."""
  id_t = _GeneratorOptId
  __tablename__ = 'generator_opts'

  # Columns.
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), 'mysql'),
      nullable=False,
      default=labdate.GetUtcMillisecondsNow)
  name_id: _GeneratorOptNameId = sql.Column(
      _GeneratorOptNameId,
      sql.ForeignKey('generator_opt_names.id'),
      nullable=False)
  value_id: _GeneratorOptValueId = sql.Column(
      _GeneratorOptValueId,
      sql.ForeignKey('generator_opt_values.id'),
      nullable=False)

  # Relationships.
  name: 'GeneratorOptName' = orm.relationship(
      'GeneratorOptName', back_populates='opts')
  value: 'GeneratorOptValue' = orm.relationship(
      'GeneratorOptValue', back_populates='opts')

  # Constraints.
  __table_args__ = (sql.UniqueConstraint(
      'name_id', 'value_id', name='unique_generator_opt'),)

  def __repr__(self):
    return f'{self.name}: {self.value}'


class GeneratorOptName(db.StringTable):
  """The name of a generator option."""
  id_t = _GeneratorOptNameId
  __tablename__ = 'generator_opt_names'

  # Relationships.
  opts: typing.List[GeneratorOpt] = orm.relationship(
      GeneratorOpt, back_populates='name')


class GeneratorOptValue(db.StringTable):
  """The value of a generator option."""
  id_t = _GeneratorOptValueId
  __tablename__ = 'generator_opt_values'

  # Relationships.
  opts: typing.List[GeneratorOpt] = orm.relationship(
      GeneratorOpt, back_populates='value')
