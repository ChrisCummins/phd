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
"""This file implements the testcase harness."""
import binascii
import datetime
import hashlib
import typing

import sqlalchemy as sql
from sqlalchemy import orm
from sqlalchemy.dialects import mysql

import labm8.py.sqlutil
from deeplearning.deepsmith import db
from deeplearning.deepsmith.proto import deepsmith_pb2
from labm8.py import labdate
from labm8.py import system

# The index types for tables defined in this file.
_HarnessId = sql.Integer
_HarnessOptSetId = sql.Binary(16).with_variant(mysql.BINARY(16), "mysql")
_HarnessOptId = sql.Integer
_HarnessOptNameId = db.StringTable.id_t
_HarnessOptValueId = db.StringTable.id_t


class Harness(db.Table):
  id_t = sql.Integer
  __tablename__ = "harnesses"

  # Columns:
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(
    sql.DateTime().with_variant(mysql.DATETIME(fsp=3), "mysql"),
    nullable=False,
    default=labdate.GetUtcMillisecondsNow,
  )
  # MySQL maximum key length is 3072, with 3 bytes per character. We must
  # preserve 16 bytes for the unique constraint.
  name: str = sql.Column(
    sql.String(4096).with_variant(sql.String((3072 - 16) // 3), "mysql"),
    nullable=False,
  )
  optset_id: bytes = sql.Column(_HarnessOptSetId, nullable=False)

  # Relationships:
  testcases: typing.List["Testcase"] = orm.relationship(
    "Testcase", back_populates="harness"
  )
  optset: typing.List["HarnessOpt"] = orm.relationship(
    "HarnessOpt",
    secondary="harness_optsets",
    primaryjoin="HarnessOptSet.id == Harness.optset_id",
    secondaryjoin="HarnessOptSet.opt_id == HarnessOpt.id",
  )

  # Constraints:
  __table_args__ = (
    sql.UniqueConstraint("name", "optset_id", name="unique_harness"),
  )

  @property
  def opts(self) -> typing.Dict[str, str]:
    """Get the harness options.

    Returns:
      A map of harness options.
    """
    return {opt.name.string: opt.value.string for opt in self.optset}

  def SetProto(self, proto: deepsmith_pb2.Harness) -> deepsmith_pb2.Harness:
    """Set a protocol buffer representation.

    Args:
      proto: A protocol buffer message.

    Returns:
      A Harness message.
    """
    proto.name = self.name
    for opt in self.optset:
      proto.opts[opt.name.string] = opt.value.string
    return proto

  def ToProto(self) -> deepsmith_pb2.Harness:
    """Create protocol buffer representation.

    Returns:
      A Harness message.
    """
    proto = deepsmith_pb2.Harness()
    return self.SetProto(proto)

  @classmethod
  def GetOrAdd(
    cls, session: db.session_t, proto: deepsmith_pb2.Harness
  ) -> "Harness":

    # Build the list of options, and md5sum the key value strings.
    opts = []
    md5 = hashlib.md5()
    for proto_opt_name in sorted(proto.opts):
      proto_opt_value = proto.opts[proto_opt_name]
      md5.update((proto_opt_name + proto_opt_value).encode("utf-8"))
      opt = labm8.py.sqlutil.GetOrAdd(
        session,
        HarnessOpt,
        name=HarnessOptName.GetOrAdd(session, proto_opt_name),
        value=HarnessOptValue.GetOrAdd(session, proto_opt_value),
      )
      opts.append(opt)

    # Create optset table entries.
    optset_id = md5.digest()
    for opt in opts:
      labm8.py.sqlutil.GetOrAdd(session, HarnessOptSet, id=optset_id, opt=opt)

    return labm8.py.sqlutil.GetOrAdd(
      session, cls, name=proto.name, optset_id=optset_id,
    )

  def RunTestcaseOnTestbed(
    self, testcase: deepsmith_pb2.Testcase, testbed: deepsmith_pb2.Testbed
  ) -> deepsmith_pb2.Result:
    start_time = labdate.GetUtcMillisecondsNow()
    # ~~ Begin 'exec' timed region. ~~~
    # TODO: Popen something.
    # ~~~ End 'exec' timed region. ~~~
    end_time = labdate.GetUtcMillisecondsNow()

    # Create the result.
    result = deepsmith_pb2.Result()
    result.testcase = testcase
    result.testbed = testbed
    result.returncode = 0

    # Create profiling events.
    exec_time = result.profiling_events.add()
    exec_time.client = system.HOSTNAME
    exec_time.type = "exec"
    exec_time.duration_ms = end_time - start_time
    exec_time.event_start_epoch_ms = start_time

    return result


class HarnessOptSet(db.Table):
  """A set of of harness options.

  An option set groups options for harnesses.
  """

  __tablename__ = "harness_optsets"
  id_t = _HarnessOptSetId

  # Columns.
  id: bytes = sql.Column(id_t, nullable=False)
  opt_id: int = sql.Column(
    _HarnessOptId, sql.ForeignKey("harness_opts.id"), nullable=False
  )

  # Relationships.
  harnesses: typing.List[Harness] = orm.relationship(
    Harness, primaryjoin=id == orm.foreign(Harness.optset_id)
  )
  opt: "HarnessOpt" = orm.relationship("HarnessOpt")

  # Constraints.
  __table_args__ = (
    sql.PrimaryKeyConstraint("id", "opt_id", name="unique_harness_optset"),
  )

  def __repr__(self):
    hex_id = binascii.hexlify(self.id).decode("utf-8")
    return f"{hex_id}: {self.opt_id}={self.opt}"


class HarnessOpt(db.Table):
  """A harness option consists of a <name, value> pair."""

  id_t = _HarnessOptId
  __tablename__ = "harness_opts"

  # Columns.
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(
    sql.DateTime().with_variant(mysql.DATETIME(fsp=3), "mysql"),
    nullable=False,
    default=labdate.GetUtcMillisecondsNow,
  )
  name_id: _HarnessOptNameId = sql.Column(
    _HarnessOptNameId, sql.ForeignKey("harness_opt_names.id"), nullable=False
  )
  value_id: _HarnessOptValueId = sql.Column(
    _HarnessOptValueId, sql.ForeignKey("harness_opt_values.id"), nullable=False
  )

  # Relationships.
  name: "HarnessOptName" = orm.relationship(
    "HarnessOptName", back_populates="opts"
  )
  value: "HarnessOptValue" = orm.relationship(
    "HarnessOptValue", back_populates="opts"
  )

  # Constraints.
  __table_args__ = (
    sql.UniqueConstraint("name_id", "value_id", name="unique_harness_opt"),
  )

  def __repr__(self):
    return f"{self.name}: {self.value}"


class HarnessOptName(db.StringTable):
  """The name of a harness option."""

  id_t = _HarnessOptNameId
  __tablename__ = "harness_opt_names"

  # Relationships.
  opts: typing.List[HarnessOpt] = orm.relationship(
    HarnessOpt, back_populates="name"
  )


class HarnessOptValue(db.StringTable):
  """The value of a harness option."""

  id_t = _HarnessOptValueId
  __tablename__ = "harness_opt_values"

  # Relationships.
  opts: typing.List[HarnessOpt] = orm.relationship(
    HarnessOpt, back_populates="value"
  )
