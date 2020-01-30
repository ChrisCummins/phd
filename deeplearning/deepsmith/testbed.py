# Copyright (c) 2017-2020 Chris Cummins.
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
"""This file implements testbeds."""
import binascii
import datetime
import hashlib
import typing

import sqlalchemy as sql
from sqlalchemy import orm
from sqlalchemy.dialects import mysql

import deeplearning.deepsmith.toolchain
import labm8.py.sqlutil
from deeplearning.deepsmith import db
from deeplearning.deepsmith.proto import deepsmith_pb2
from labm8.py import labdate

# The index types for tables defined in this file.
_TestbedId = sql.Integer
_TestbedOptSetId = sql.Binary(16).with_variant(mysql.BINARY(16), "mysql")
_TestbedOptId = sql.Integer
_TestbedOptNameId = db.StringTable.id_t
_TestbedOptValueId = db.StringTable.id_t


class Testbed(db.Table):
  """A Testbed is a system on which testcases may be run.

  Each testbed is a <toolchain,name,opts> tuple.
  """

  id_t = _TestbedId
  __tablename__ = "testbeds"

  # Columns.
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(
    sql.DateTime().with_variant(mysql.DATETIME(fsp=3), "mysql"),
    nullable=False,
    default=labdate.GetUtcMillisecondsNow,
  )
  toolchain_id: int = sql.Column(
    deeplearning.deepsmith.toolchain.Toolchain.id_t,
    sql.ForeignKey("toolchains.id"),
    nullable=False,
  )
  # MySQL maximum key length is 3072, with 3 bytes per character. We must
  # preserve 16 bytes for the optset_id in the unique constraint and 4 bytes
  # for the toolchain_id.
  name: str = sql.Column(
    sql.String(4096).with_variant(sql.String((3072 - 16 - 6) // 3), "mysql"),
    nullable=False,
  )
  optset_id: bytes = sql.Column(_TestbedOptSetId, nullable=False)

  # Relationships.
  toolchain: deeplearning.deepsmith.toolchain.Toolchain = orm.relationship(
    "Toolchain"
  )
  results: typing.List["Result"] = orm.relationship(
    "Result", back_populates="testbed"
  )
  pending_results: typing.List["PendingResult"] = orm.relationship(
    "PendingResult", back_populates="testbed"
  )
  optset: typing.List["TestbedOpt"] = orm.relationship(
    "TestbedOpt",
    secondary="testbed_optsets",
    primaryjoin="TestbedOptSet.id == Testbed.optset_id",
    secondaryjoin="TestbedOptSet.opt_id == TestbedOpt.id",
  )

  # Constraints.
  __table_args__ = (
    sql.UniqueConstraint(
      "toolchain_id", "name", "optset_id", name="unique_testbed"
    ),
  )

  @property
  def opts(self) -> typing.Dict[str, str]:
    """Get the testbed options.

    Returns:
      A map of testbed options.
    """
    return {opt.name.string: opt.value.string for opt in self.optset}

  def ToProto(self) -> deepsmith_pb2.Testbed:
    """Create protocol buffer representation.

    Returns:
      A Testbed message.
    """
    proto = deepsmith_pb2.Testbed()
    return self.SetProto(proto)

  def SetProto(self, proto: deepsmith_pb2.Testbed) -> deepsmith_pb2.Testbed:
    """Set a protocol buffer representation.

    Args:
      proto: A protocol buffer message.

    Returns:
      A Testbed message.
    """
    proto.toolchain = self.toolchain.string
    proto.name = self.name
    for opt in self.optset:
      proto.opts[opt.name.string] = opt.value.string
    return proto

  @classmethod
  def GetOrAdd(
    cls, session: db.session_t, proto: deepsmith_pb2.Testbed
  ) -> "Testbed":
    """Instantiate a Testbed from a protocol buffer.

    This is the preferred method for creating database-backed Testbed
    instances. If the Testbed does not already exist in the database, it is
    added.

    Args:
      session: A database session.
      proto: A Testbed message.

    Returns:
      A Testbed.
    """
    toolchain = deeplearning.deepsmith.toolchain.Toolchain.GetOrAdd(
      session, proto.toolchain
    )

    # Build the list of options, and md5sum the key value strings.
    opts = []
    md5 = hashlib.md5()
    for proto_opt_name in sorted(proto.opts):
      proto_opt_value = proto.opts[proto_opt_name]
      md5.update((proto_opt_name + proto_opt_value).encode("utf-8"))
      opt = labm8.py.sqlutil.GetOrAdd(
        session,
        TestbedOpt,
        name=TestbedOptName.GetOrAdd(session, proto_opt_name),
        value=TestbedOptValue.GetOrAdd(session, proto_opt_value),
      )
      opts.append(opt)

    # Create optset table entries.
    optset_id = md5.digest()
    for opt in opts:
      db.GetOrAdd(session, TestbedOptSet, id=optset_id, opt=opt)

    return labm8.py.sqlutil.GetOrAdd(
      session, cls, toolchain=toolchain, name=proto.name, optset_id=optset_id,
    )


class TestbedOptSet(db.Table):
  """A set of of testbed options.

  An option set groups options for testbed.
  """

  __tablename__ = "testbed_optsets"
  id_t = _TestbedOptSetId

  # Columns.
  id: bytes = sql.Column(id_t, nullable=False)
  opt_id: int = sql.Column(
    _TestbedOptId, sql.ForeignKey("testbed_opts.id"), nullable=False
  )

  # Relationships.
  testbeds: typing.List[Testbed] = orm.relationship(
    Testbed, primaryjoin=id == orm.foreign(Testbed.optset_id)
  )
  opt: "TestbedOpt" = orm.relationship("TestbedOpt")

  # Constraints.
  __table_args__ = (
    sql.PrimaryKeyConstraint("id", "opt_id", name="unique_testbed_optset"),
  )

  def __repr__(self):
    hex_id = binascii.hexlify(self.id).decode("utf-8")
    return f"{hex_id}: {self.opt_id}={self.opt}"


class TestbedOpt(db.Table):
  """A testbed option consists of a <name, value> pair."""

  id_t = _TestbedOptId
  __tablename__ = "testbed_opts"

  # Columns.
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(
    sql.DateTime().with_variant(mysql.DATETIME(fsp=3), "mysql"),
    nullable=False,
    default=labdate.GetUtcMillisecondsNow,
  )
  name_id: _TestbedOptNameId = sql.Column(
    _TestbedOptNameId, sql.ForeignKey("testbed_opt_names.id"), nullable=False
  )
  value_id: _TestbedOptValueId = sql.Column(
    _TestbedOptValueId, sql.ForeignKey("testbed_opt_values.id"), nullable=False
  )

  # Relationships.
  name: "TestbedOptName" = orm.relationship(
    "TestbedOptName", back_populates="opts"
  )
  value: "TestbedOptValue" = orm.relationship(
    "TestbedOptValue", back_populates="opts"
  )

  # Constraints.
  __table_args__ = (
    sql.UniqueConstraint("name_id", "value_id", name="unique_testbed_opt"),
  )

  def __repr__(self):
    return f"{self.name}: {self.value}"


class TestbedOptName(db.StringTable):
  """The name of a testbed option."""

  id_t = _TestbedOptNameId
  __tablename__ = "testbed_opt_names"

  # Relationships.
  opts: typing.List[TestbedOpt] = orm.relationship(
    TestbedOpt, back_populates="name"
  )


class TestbedOptValue(db.StringTable):
  """The value of a testbed option."""

  id_t = _TestbedOptValueId
  __tablename__ = "testbed_opt_values"

  # Relationships.
  opts: typing.List[TestbedOpt] = orm.relationship(
    TestbedOpt, back_populates="value"
  )
