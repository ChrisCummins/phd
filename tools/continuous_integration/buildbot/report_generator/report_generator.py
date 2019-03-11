"""Import bazel test logs and report delta in passes and failures."""
import datetime
import os
import pathlib
import subprocess
import sys
import typing
import xml.etree.ElementTree as ET

import sqlalchemy as sql

from config import getconfig
from labm8 import app
from labm8 import prof
from tools.continuous_integration.buildbot.report_generator import \
  bazel_test_db as db

FLAGS = app.FLAGS
app.DEFINE_string("testlogs", None, "Path to bazel testlogs directory.")
app.DEFINE_string("host", None, "The name of the build host.")
app.DEFINE_string(
    "db", "sqlite:////tmp/phd/tools/continuous_integration/buildbot.db",
    "Path to testlogs summary database.")


def GetBazelTarget(testlogs_root: pathlib.Path, xml_path: pathlib.Path):
  xml_path = xml_path.parent
  if xml_path.parent.name == xml_path.name:
    xml_path = str(xml_path.parent)
  else:
    xml_path = str(f'{xml_path.parent}:{xml_path.name}')

  return '/' + xml_path[len(str(testlogs_root)):]


def GetGitBranchOrDie():
  """Get the name of the current git branch."""
  branches = subprocess.check_output(['git', 'branch'], universal_newlines=True)
  for line in branches.split('\n'):
    if line.startswith('* '):
      return line[2:]
  print("fatal: Unable to determine git branch", file=sys.stderr)
  sys.exit(1)


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  if not FLAGS.testlogs:
    raise app.UsageError("--testlogs must be set")

  testlogs = pathlib.Path(FLAGS.testlogs)
  if not testlogs.is_dir():
    raise FileNotFoundError("--testlogs not a directory: {}".format(
        FLAGS.testlogs))

  database = db.Database(FLAGS.db)

  invocation_datetime = datetime.datetime.now()
  if not FLAGS.host:
    raise app.UsageError("--host must be set")
  host = FLAGS.host

  phd_root = getconfig.GetGlobalConfig().paths.repo_root
  git_branch = GetGitBranchOrDie()
  git_commit = subprocess.check_output(
      ['git', '-C', phd_root, 'rev-parse', 'HEAD'],
      universal_newlines=True).rstrip()

  results = []
  with prof.Profile('Read bazel test XML files'):
    for dir_, _, files in os.walk(testlogs):
      dir_ = pathlib.Path(dir_)
      for file_ in [f for f in files if f.endswith('.xml')]:
        xml_path = dir_ / file_
        xml = ET.parse(xml_path).getroot()

        result = db.TestTargetResult(**db.TestTargetResult.FromXml(xml))
        result.invocation_datetime = invocation_datetime
        result.bazel_target = GetBazelTarget(testlogs, xml_path)
        result.git_branch = git_branch
        result.git_commit = git_commit
        result.host = host
        result.target = GetBazelTarget(testlogs, xml_path)
        result.git_commit = git_commit
        results.append(result)

  with prof.Profile('Import to database'):
    with database.Session(commit=True) as session:
      session.add_all(results)

  with database.Session(commit=False) as session:
    failed = session.query(db.TestTargetResult.bazel_target) \
      .filter(db.TestTargetResult.invocation_datetime == invocation_datetime) \
      .filter(db.TestTargetResult.failed_count > 0)
    passed = session.query(db.TestTargetResult.bazel_target)\
      .filter(db.TestTargetResult.invocation_datetime == invocation_datetime)\
      .filter(db.TestTargetResult.failed_count == 0)

    num_tests, = session.query(sql.func.sum_(db.TestTargetResult.test_count)) \
      .filter(db.TestTargetResult.invocation_datetime == invocation_datetime).one()
    num_failed = failed.count()
    total_runtime_ms, = session.query(sql.func.sum_(db.TestTargetResult.runtime_ms))\
      .filter(db.TestTargetResult.invocation_datetime == invocation_datetime).one()

    # Get the last run.
    previous_invocation = session.query(
        sql.func.max_(db.TestTargetResult.invocation_datetime))\
      .filter(db.TestTargetResult.host == host)\
      .filter(db.TestTargetResult.git_branch == git_branch)\
      .filter(db.TestTargetResult.invocation_datetime < invocation_datetime).first()

    if previous_invocation:
      previous_invocation = previous_invocation[0]
      new_passed = passed.filter(~db.TestTargetResult.bazel_target.in_(
          session.query(db.TestTargetResult.bazel_target) \
            .filter(db.TestTargetResult.invocation_datetime == previous_invocation) \
            .filter(db.TestTargetResult.failed_count == 0)
      ))
      new_failed = failed.filter(~db.TestTargetResult.bazel_target.in_(
          session.query(db.TestTargetResult.bazel_target) \
            .filter(db.TestTargetResult.invocation_datetime == previous_invocation) \
            .filter(db.TestTargetResult.failed_count > 0)
        ))
    else:
      new_passed = passed
      new_failed = failed

    new_passed = [r[0] for r in new_passed]
    new_failed = [r[0] for r in new_failed]

  print(f'{len(results)} targets ({num_tests} tests in '
        f'{total_runtime_ms/1000}s), {num_failed} failed')
  print(f'{len(new_passed)} new passes, {len(new_failed)} new failures')
  for target in new_failed:
    print("NEW FAIL", target)
  if new_failed:
    sys.exit(1)


if __name__ == '__main__':
  app.RunWithArgs(main)
