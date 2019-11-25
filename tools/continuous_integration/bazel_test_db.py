"""A database for storing bazel test results."""
import collections
import datetime
import typing
from xml import etree

import sqlalchemy as sql

from labm8 import app
from labm8 import sqlutil

FLAGS = app.FLAGS

Base = sqlutil.Base()


class TestTargetResult(Base, sqlutil.TablenameFromCamelCapsClassNameMixin):
  """The result of a bazel test invocation on a single test target."""
  id = sql.Column(sql.Integer, primary_key=True)
  # A name to describe the build environment. Use this column to group test
  # results on the same environment.
  host = sql.Column(sql.String(256), nullable=False)
  # The name of the current git branch.
  git_branch = sql.Column(sql.String(128), nullable=False)
  # The hash of the git head. The result of $(git rev-parse HEAD).
  git_commit = sql.Column(sql.String(40), nullable=False)
  # A datetime used to group all test target results from a single bazel test
  # invocation.
  invocation_datetime = sql.Column(sqlutil.ColumnTypes.MillisecondDatetime(),
                                   nullable=False)
  # The name of the bazel target.
  bazel_target = sql.Column(sql.String(256), nullable=False)
  # The number of tests executed.
  test_count = sql.Column(sql.Integer, nullable=False)
  # The number of tests that failed.
  failed_count = sql.Column(sql.Integer, nullable=False)
  # The runtime of the test target, as reported by bazel.
  runtime_ms = sql.Column(sql.Integer, nullable=False)
  # The test target output.
  log = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable=True)

  __table_args__ = (
      # Each target is invoked only once.
      sql.UniqueConstraint('invocation_datetime',
                           'bazel_target',
                           name='unique_target_per_invocation'),)

  @staticmethod
  def FromXml(xml: etree.ElementTree) -> typing.Dict[str, typing.Any]:
    """Get field values from a bazel XML junit file."""
    test_count = 0
    failed_count = 0
    runtime_ms = 0

    for testsuite in xml:
      if testsuite.tag != 'testsuite':
        return ValueError(
            f"Expected tag 'testsuite', found tag '{testsuite.tag}'")

      test_count += int(testsuite.attrib.get('tests', '0'))
      failed_count += int(testsuite.attrib.get('errors', '0'))
      failed_count += int(testsuite.attrib.get('failures', '0'))

      for testcase in testsuite.iter('testcase'):
        runtime_ms += int(round(float(testcase.attrib.get('time', '0')) * 1000))

    return {
        'test_count': test_count,
        'failed_count': failed_count,
        'runtime_ms': runtime_ms,
    }


class TestDelta(typing.NamedTuple):
  broken: int
  fixed: int
  still_broken: int
  still_pass: int


def GetTestDelta(session,
                 invocation_datetime: datetime.datetime,
                 to_return=[TestTargetResult.bazel_target]) -> TestDelta:
  """Get the test delta."""
  failed = session.query(*to_return) \
    .filter(TestTargetResult.invocation_datetime == invocation_datetime) \
    .filter(TestTargetResult.failed_count > 0)
  passed = session.query(*to_return) \
    .filter(TestTargetResult.invocation_datetime == invocation_datetime) \
    .filter(TestTargetResult.failed_count == 0)

  host, git_branch = session.query(
      TestTargetResult.host, TestTargetResult.git_branch)\
    .filter(TestTargetResult.invocation_datetime == invocation_datetime).first()

  # Get the last run.
  previous_invocation = session.query(
      sql.func.max_(TestTargetResult.invocation_datetime)) \
    .filter(TestTargetResult.host == host) \
    .filter(TestTargetResult.git_branch == git_branch) \
    .filter(TestTargetResult.invocation_datetime < invocation_datetime).first()

  if previous_invocation:
    previous_invocation = previous_invocation[0]
    return TestDelta(
        broken=failed.filter(~TestTargetResult.bazel_target.in_(
            session.query(TestTargetResult.bazel_target) \
              .filter(TestTargetResult.invocation_datetime == previous_invocation) \
              .filter(TestTargetResult.failed_count > 0)
        )).all(),
        fixed=passed.filter(~TestTargetResult.bazel_target.in_(
            session.query(TestTargetResult.bazel_target) \
              .filter(TestTargetResult.invocation_datetime == previous_invocation) \
              .filter(TestTargetResult.failed_count == 0)
        )).all(),
        still_broken=failed.filter(TestTargetResult.bazel_target.in_(
            session.query(TestTargetResult.bazel_target) \
              .filter(TestTargetResult.invocation_datetime == previous_invocation) \
              .filter(TestTargetResult.failed_count > 0)
        )).all(),
        still_pass=passed.filter(TestTargetResult.bazel_target.in_(
            session.query(TestTargetResult.bazel_target) \
              .filter(TestTargetResult.invocation_datetime == previous_invocation) \
              .filter(TestTargetResult.failed_count == 0)
        )).all(),
    )
  else:
    return TestDelta(broken=passed.all(),
                     fixed=failed.all(),
                     still_broken=[],
                     still_passing=[])


class Database(sqlutil.Database):
  """A database of test target results."""

  def __init__(self, url: str):
    super(Database, self).__init__(url, Base)
