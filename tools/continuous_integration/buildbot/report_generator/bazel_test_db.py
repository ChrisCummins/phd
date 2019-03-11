"""A database for storing bazel test logs."""
import typing
from xml import etree

import sqlalchemy as sql

from labm8 import app
from labm8 import sqlutil


FLAGS = app.FLAGS

Base = sqlutil.Base()


class TestTargetResult(Base, sqlutil.TablenameFromCamelCapsClassNameMixin):
  id = sql.Column(sql.Integer, primary_key=True)
  # A name to describe the build environment.
  host = sql.Column(sql.String(256), nullable=False)
  git_branch = sql.Column(sql.String(128), nullable=False)
  git_commit = sql.Column(sql.String(40), nullable=False)
  # A datetime used to group all test target results.
  invocation_datetime = sql.Column(
      sqlutil.ColumnTypes.MillisecondDatetime(), nullable=False)
  bazel_target = sql.Column(sql.String(256), nullable=False)
  test_count = sql.Column(sql.Integer, nullable=False)
  failed_count = sql.Column(sql.Integer, nullable=False)
  runtime_ms = sql.Column(sql.Integer, nullable=False)
  system_output = sql.Column(
      sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable=True)

  __table_args__ = (
      # Each target is only invoked once.
      sql.UniqueConstraint(
          'invocation_datetime',
          'bazel_target',
          name='unique_target_per_invocation'),)

  @staticmethod
  def FromXml(xml: etree.ElementTree) -> typing.Dict[str, typing.Any]:
    system_outs = []

    test_count = 0
    failed_count = 0
    runtime_ms = 0

    for testsuite in xml:
      if testsuite.tag != 'testsuite':
        return ValueError(
            f"Expected tag 'testsuite', found tag '{testsuite.tag}'")

      system_out = testsuite.find('system-out')
      if system_out:
        system_outs.append(system_out)

      test_count += int(testsuite.attrib.get('tests', '0'))
      failed_count += int(testsuite.attrib.get('errors', '0'))
      failed_count += int(testsuite.attrib.get('failures', '0'))

      for testcase in testsuite.iter('testcase'):
        runtime_ms += int(round(float(testcase.attrib.get('time', '0')) * 1000))

    return {
        'test_count': test_count,
        'failed_count': failed_count,
        'runtime_ms': runtime_ms,
        'system_output': '\n'.join(system_outs) or None,
    }


class Database(sqlutil.Database):

  def __init__(self, url: str):
    super(Database, self).__init__(url, Base)
