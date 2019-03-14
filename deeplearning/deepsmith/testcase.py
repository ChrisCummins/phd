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
"""This file implements testcases."""
import binascii
import datetime
import hashlib
import pathlib
import typing

import sqlalchemy as sql
from sqlalchemy import orm
from sqlalchemy.dialects import mysql

import deeplearning.deepsmith.generator
import deeplearning.deepsmith.harness
import deeplearning.deepsmith.profiling_event
import deeplearning.deepsmith.toolchain
import labm8.sqlutil
from deeplearning.deepsmith import db
from deeplearning.deepsmith.proto import deepsmith_pb2
from labm8 import labdate, pbutil

# The index types for tables defined in this file.
_TestcaseId = sql.Integer
_TestcaseInputSetId = sql.Binary(16).with_variant(mysql.BINARY(16), 'mysql')
_TestcaseInputId = sql.Integer
_TestcaseInputNameId = db.StringTable.id_t
_TestcaseInputValueId = sql.Integer
_TestcaseInvariantOptSetId = sql.Binary(16).with_variant(
    mysql.BINARY(16), 'mysql')
_TestcaseInvariantOptId = sql.Integer
_TestcaseInvariantOptNameId = db.StringTable.id_t
_TestcaseInvariantOptValueId = db.StringTable.id_t


class Testcase(db.Table):
  """A testcase is a set of parameters for a runnable test.

  It is a tuple of <toolchain,generator,harness,inputs,invariant_opts>.
  """
  id_t = _TestcaseId
  __tablename__ = 'testcases'

  # Columns.
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), 'mysql'),
      nullable=False,
      default=labdate.GetUtcMillisecondsNow)
  toolchain_id: int = sql.Column(
      deeplearning.deepsmith.toolchain.Toolchain.id_t,
      sql.ForeignKey('toolchains.id'),
      nullable=False)
  generator_id: int = sql.Column(
      deeplearning.deepsmith.generator.Generator.id_t,
      sql.ForeignKey('generators.id'),
      nullable=False)
  harness_id: int = sql.Column(
      deeplearning.deepsmith.harness.Harness.id_t,
      sql.ForeignKey('harnesses.id'),
      nullable=False)
  inputset_id: bytes = sql.Column(_TestcaseInputSetId, nullable=False)
  invariant_optset_id: bytes = sql.Column(
      _TestcaseInvariantOptSetId, nullable=False)

  # Relationships.
  toolchain: deeplearning.deepsmith.toolchain.Toolchain = orm.relationship(
      'Toolchain')
  generator: deeplearning.deepsmith.generator.Generator = orm.relationship(
      'Generator', back_populates='testcases')
  harness: deeplearning.deepsmith.harness.Harness = orm.relationship(
      'Harness', back_populates='testcases')
  inputset: typing.List['TestcaseInput'] = orm.relationship(
      'TestcaseInput',
      secondary='testcase_inputsets',
      primaryjoin='TestcaseInputSet.id == Testcase.inputset_id',
      secondaryjoin='TestcaseInputSet.input_id == TestcaseInput.id')
  invariant_optset: typing.List['TestcaseInvariantOpt'] = orm.relationship(
      'TestcaseInvariantOpt',
      secondary='testcase_invariant_optsets',
      primaryjoin='TestcaseInvariantOptSet.id == Testcase.invariant_optset_id',
      secondaryjoin='TestcaseInvariantOptSet.invariant_opt_id == '
      'TestcaseInvariantOpt.id')
  profiling_events: typing.List['TestcaseProfilingEvent'] = orm.relationship(
      'TestcaseProfilingEvent', back_populates='testcase')
  results: typing.List['Result'] = orm.relationship(
      'Result', back_populates='testcase')
  pending_results: typing.List['PendingResult'] = orm.relationship(
      'PendingResult', back_populates='testcase')

  @property
  def inputs(self) -> typing.Dict[str, str]:
    """Get the generator inputs.

    Returns:
      A map of generator inputs.
    """
    return {input.name.string: input.value.string for input in self.inputset}

  @property
  def invariant_opts(self) -> typing.Dict[str, str]:
    """Get the generator options.

    Returns:
      A map of generator options.
    """
    return {opt.name.string: opt.value.string for opt in self.invariant_optset}

  def SetProto(self, proto: deepsmith_pb2.Testcase) -> deepsmith_pb2.Testcase:
    """Set a protocol buffer representation.

    Args:
      proto: A protocol buffer message.

    Returns:
      A Testcase message.
    """
    proto.toolchain = self.toolchain.string
    self.generator.SetProto(proto.generator)
    self.harness.SetProto(proto.harness)
    for input_ in self.inputset:
      proto.inputs[input_.name.string] = input_.value.string
    for opt in self.invariant_optset:
      proto.invariant_opts[opt.name.string] = opt.value.string
    for profiling_event in self.profiling_events:
      event = proto.profiling_events.add()
      profiling_event.SetProto(event)
    return proto

  def ToProto(self) -> deepsmith_pb2.Testcase:
    """Create protocol buffer representation.

    Returns:
      A Testcase message.
    """
    proto = deepsmith_pb2.Testcase()
    return self.SetProto(proto)

  @classmethod
  def GetOrAdd(cls, session: db.session_t,
               proto: deepsmith_pb2.Testcase) -> 'Testcase':
    """Instantiate a Testcase from a protocol buffer.

    Args:
      session: A database session.
      proto: A Testcase message.

    Returns:
      A Testcase instance.
    """
    toolchain = deeplearning.deepsmith.toolchain.Toolchain.GetOrAdd(
        session, proto.toolchain)
    generator = deeplearning.deepsmith.generator.Generator.GetOrAdd(
        session, proto.generator)
    harness = deeplearning.deepsmith.harness.Harness.GetOrAdd(
        session, proto.harness)
    # Build the list of inputs, and md5sum the key value strings.
    inputs = []
    md5 = hashlib.md5()
    for proto_input_name in sorted(proto.inputs):
      proto_input_value = proto.inputs[proto_input_name]
      md5.update((proto_input_name + proto_input_value).encode('utf-8'))
      input_ = TestcaseInput.GetOrAdd(session, proto_input_name,
                                      proto_input_value)
      inputs.append(input_)

    # Create invariant optset table entries.
    inputset_id = md5.digest()
    for input in inputs:
      labm8.sqlutil.GetOrAdd(
          session, TestcaseInputSet, id=inputset_id, input=input)

    # Build the list of invariant options, and md5sum the key value strings.
    invariant_opts = []
    md5 = hashlib.md5()
    for proto_invariant_opt_name in sorted(proto.invariant_opts):
      proto_invariant_opt_value = proto.invariant_opts[proto_invariant_opt_name]
      md5.update((
          proto_invariant_opt_name + proto_invariant_opt_value).encode('utf-8'))
      invariant_opt = TestcaseInvariantOpt.GetOrAdd(
          session, proto_invariant_opt_name, proto_invariant_opt_value)
      invariant_opts.append(invariant_opt)

    # Create invariant optset table entries.
    invariant_optset_id = md5.digest()
    for invariant_opt in invariant_opts:
      labm8.sqlutil.GetOrAdd(
          session,
          TestcaseInvariantOptSet,
          id=invariant_optset_id,
          invariant_opt=invariant_opt)

    # Create a new testcase only if everything *except* the profiling events
    # are unique. This means that if a generator produced the same testcase
    # twice (on separate occasions), only the first is added to the datastore.
    testcase = labm8.sqlutil.Get(
        session,
        cls,
        toolchain=toolchain,
        generator=generator,
        harness=harness,
        inputset_id=inputset_id,
        invariant_optset_id=invariant_optset_id)
    if not testcase:
      testcase = cls(
          toolchain=toolchain,
          generator=generator,
          harness=harness,
          inputset_id=inputset_id,
          invariant_optset_id=invariant_optset_id)
      session.add(testcase)
      # Add profiling events.
      for event in proto.profiling_events:
        deeplearning.deepsmith.profiling_event.TestcaseProfilingEvent.GetOrAdd(
            session, event, testcase)

    return testcase

  @classmethod
  def ProtoFromFile(cls, path: pathlib.Path) -> deepsmith_pb2.Testcase:
    """Instantiate a protocol buffer testcase from file.

    Args:
      path: Path to the testcase proto file.

    Returns:
      Testcase message instance.
    """
    return pbutil.FromFile(path, deepsmith_pb2.Testcase())

  @classmethod
  def FromFile(cls, session: db.session_t, path: pathlib.Path) -> 'Testcase':
    """Instantiate a Result from a serialized protocol buffer on file.

    Args:
      session: A database session.
      path: Path to the testcase proto file.

    Returns:
      A Testcase instance.
    """
    return cls.GetOrAdd(session, cls.ProtoFromFile(path))


class TestcaseInputSet(db.Table):
  """A set of testcase inputs.

  An input set groups inputs for testcases.
  """
  __tablename__ = 'testcase_inputsets'
  id_t = _TestcaseInputSetId

  # Columns.
  id: bytes = sql.Column(id_t, nullable=False)
  input_id: int = sql.Column(
      _TestcaseInputId, sql.ForeignKey('testcase_inputs.id'), nullable=False)

  # Relationships.
  testcases: typing.List[Testcase] = orm.relationship(
      Testcase, primaryjoin=id == orm.foreign(Testcase.inputset_id))
  input: 'TestcaseInput' = orm.relationship('TestcaseInput')

  # Constraints.
  __table_args__ = (sql.PrimaryKeyConstraint(
      'id', 'input_id', name='unique_testcase_inputset'),)

  def __repr__(self):
    hex_id = binascii.hexlify(self.id).decode('utf-8')
    return f'{hex_id}: {self.input_id}={self.input}'


class TestcaseInput(db.Table):
  """A testcase input consists of a <name, value> pair."""
  id_t = _TestcaseInputId
  __tablename__ = 'testcase_inputs'

  # Columns.
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), 'mysql'),
      nullable=False,
      default=labdate.GetUtcMillisecondsNow)
  name_id: _TestcaseInputNameId = sql.Column(
      _TestcaseInputNameId,
      sql.ForeignKey('testcase_input_names.id'),
      nullable=False)
  value_id: _TestcaseInputValueId = sql.Column(
      _TestcaseInputValueId,
      sql.ForeignKey('testcase_input_values.id'),
      nullable=False)

  # Relationships.
  name: 'TestcaseInputName' = orm.relationship(
      'TestcaseInputName', back_populates='inputs')
  value: 'TestcaseInputValue' = orm.relationship(
      'TestcaseInputValue', back_populates='inputs')

  # Constraints.
  __table_args__ = (sql.UniqueConstraint(
      'name_id', 'value_id', name='unique_testcase_input'),)

  def __repr__(self):
    return f'{self.name}: {self.value}'

  @classmethod
  def GetOrAdd(cls, session: db.session_t, name: str,
               value: str) -> 'TestcaseInput':
    """Instantiate a TestcaseInput.

    Args:
      session: A database session.
      name: The name of the testcase input.
      value: The value of the testcase input.

    Returns:
      A TestcaseInput instance.
    """
    return labm8.sqlutil.GetOrAdd(
        session,
        TestcaseInput,
        name=TestcaseInputName.GetOrAdd(
            session,
            string=name,
        ),
        value=TestcaseInputValue.GetOrAdd(
            session,
            string=value,
        ),
    )


class TestcaseInputName(db.StringTable):
  """The name of a testcase input."""
  id_t = _TestcaseInputNameId
  __tablename__ = 'testcase_input_names'

  # Relationships.
  inputs: typing.List[TestcaseInput] = orm.relationship(
      TestcaseInput, back_populates='name')


class TestcaseInputValue(db.Table):
  id_t = sql.Integer
  __tablename__ = 'testcase_input_values'

  # Columns.
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), 'mysql'),
      nullable=False,
      default=labdate.GetUtcMillisecondsNow)
  md5: bytes = sql.Column(
      sql.Binary(16).with_variant(mysql.BINARY(16), 'mysql'),
      nullable=False,
      index=True,
      unique=True)
  charcount = sql.Column(sql.Integer, nullable=False)
  linecount = sql.Column(sql.Integer, nullable=False)
  string: str = sql.Column(
      sql.UnicodeText().with_variant(sql.UnicodeText(2**31), 'mysql'),
      nullable=False)

  # Relationships.
  inputs: typing.List[TestcaseInput] = orm.relationship(
      TestcaseInput, back_populates='value')

  @classmethod
  def GetOrAdd(cls, session: db.session_t, string: str) -> 'TestcaseInputValue':
    """Instantiate a TestcaseInputValue entry from a string.

    Args:
      session: A database session.
      string: The string.

    Returns:
      A TestcaseInputValue instance.
    """
    md5 = hashlib.md5()
    md5.update(string.encode('utf-8'))

    return labm8.sqlutil.GetOrAdd(
        session,
        cls,
        md5=md5.digest(),
        charcount=len(string),
        linecount=string.count('\n'),
        string=string,
    )

  def __repr__(self):
    return self.string[:50] or ''


class TestcaseInvariantOptSet(db.Table):
  """A set of of testcase invariant opts.

  An invariant optset groups invariant opts for testcases.
  """
  __tablename__ = 'testcase_invariant_optsets'
  id_t = _TestcaseInvariantOptSetId

  # Columns.
  id: bytes = sql.Column(id_t, nullable=False)
  invariant_opt_id: int = sql.Column(
      _TestcaseInvariantOptId,
      sql.ForeignKey('testcase_invariant_opts.id'),
      nullable=False)

  # Relationships.
  testcases: typing.List[Testcase] = orm.relationship(
      Testcase, primaryjoin=id == orm.foreign(Testcase.invariant_optset_id))
  invariant_opt: 'TestcaseInvariantOpt' = orm.relationship(
      'TestcaseInvariantOpt')

  # Constraints.
  __table_args__ = (sql.PrimaryKeyConstraint(
      'id', 'invariant_opt_id', name='unique_testcase_invariant_optset'),)

  def __repr__(self):
    hex_id = binascii.hexlify(self.id).decode('utf-8')
    return f'{hex_id}: {self.invariant_opt_id}={self.opt}'


class TestcaseInvariantOpt(db.Table):
  """A testcase invariant_opt consists of a <name, value> pair."""
  id_t = _TestcaseInvariantOptId
  __tablename__ = 'testcase_invariant_opts'

  # Columns.
  id: int = sql.Column(id_t, primary_key=True)
  date_added: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), 'mysql'),
      nullable=False,
      default=labdate.GetUtcMillisecondsNow)
  name_id: _TestcaseInvariantOptNameId = sql.Column(
      _TestcaseInvariantOptNameId,
      sql.ForeignKey('testcase_invariant_opt_names.id'),
      nullable=False)
  value_id: _TestcaseInvariantOptValueId = sql.Column(
      _TestcaseInvariantOptValueId,
      sql.ForeignKey('testcase_invariant_opt_values.id'),
      nullable=False)

  # Relationships.
  name: 'TestcaseInvariantOptName' = orm.relationship(
      'TestcaseInvariantOptName', back_populates='invariant_opts')
  value: 'TestcaseInvariantOptValue' = orm.relationship(
      'TestcaseInvariantOptValue', back_populates='invariant_opts')

  # Constraints.
  __table_args__ = (sql.UniqueConstraint(
      'name_id', 'value_id', name='unique_testcase_invariant_opt'),)

  def __repr__(self):
    return f'{self.name}: {self.value}'

  @classmethod
  def GetOrAdd(cls, session: db.session_t, name: str,
               value: str) -> 'TestcaseInvariantOpt':
    """Instantiate a TestcaseInvariantOpt.

    Args:
      session: A database session.
      name: The name of the opt.
      value: The value of the opt.

    Returns:
      A TestcaseInvariantOpt instance.
    """
    return labm8.sqlutil.GetOrAdd(
        session,
        cls,
        name=TestcaseInvariantOptName.GetOrAdd(
            session,
            string=name,
        ),
        value=TestcaseInvariantOptValue.GetOrAdd(
            session,
            string=value,
        ),
    )


class TestcaseInvariantOptName(db.StringTable):
  """The name of a testcase invariant_opt."""
  id_t = _TestcaseInvariantOptNameId
  __tablename__ = 'testcase_invariant_opt_names'

  # Relationships.
  invariant_opts: typing.List[TestcaseInvariantOpt] = orm.relationship(
      TestcaseInvariantOpt, back_populates='name')


class TestcaseInvariantOptValue(db.StringTable):
  """The value of a testcase invariant_opt."""
  id_t = _TestcaseInvariantOptValueId
  __tablename__ = 'testcase_invariant_opt_values'

  # Relationships.
  invariant_opts: typing.List[TestcaseInvariantOpt] = orm.relationship(
      TestcaseInvariantOpt, back_populates='value')
