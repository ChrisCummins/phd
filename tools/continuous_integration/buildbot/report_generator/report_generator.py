"""Import bazel test results and report delta in passes and failures."""
import datetime
import os
import pathlib
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import sqlalchemy as sql

from labm8.py import app
from labm8.py import fs
from labm8.py import humanize
from labm8.py import prof
from tools.continuous_integration import bazel_test_db as db

FLAGS = app.FLAGS
app.DEFINE_string("repo", None, "Path to repo directory.")
app.DEFINE_string("testlogs", None, "Path to bazel testlogs directory.")
app.DEFINE_string("host", None, "The name of the build host.")
app.DEFINE_string(
  "db",
  "sqlite:////tmp/phd/tools/continuous_integration/buildbot.db",
  "Path to testlogs summary database.",
)


def GetBazelTarget(testlogs_root: pathlib.Path, xml_path: pathlib.Path):
  xml_path = xml_path.parent
  if xml_path.parent.name == xml_path.name:
    xml_path = str(xml_path.parent)
  else:
    xml_path = str(f"{xml_path.parent}:{xml_path.name}")

  return "/" + xml_path[len(str(testlogs_root)) :]


def GetGitBranchOrDie(repo_dir: str):
  """Get the name of the current git branch."""
  branches = subprocess.check_output(
    ["git", "-C", repo_dir, "branch"], universal_newlines=True
  )
  for line in branches.split("\n"):
    if line.startswith("* "):
      return line[2:]
  print("fatal: Unable to determine git branch", file=sys.stderr)
  sys.exit(1)


def main():
  """Main entry point."""
  if not FLAGS.testlogs:
    raise app.UsageError("--testlogs must be set")

  testlogs = pathlib.Path(FLAGS.testlogs)
  if not testlogs.is_dir():
    raise FileNotFoundError(f"--testlogs not a directory: {FLAGS.testlogs}")

  database = db.Database(FLAGS.db)

  invocation_datetime = datetime.datetime.now()
  if not FLAGS.host:
    raise app.UsageError("--host must be set")
  host = FLAGS.host

  if not FLAGS.repo:
    raise app.UsageError("--repo must be set")
  repo = pathlib.Path(FLAGS.repo)
  if not repo.is_dir():
    raise FileNotFoundError(f"--repo not a directory: {FLAGS.repo}")

  git_branch = GetGitBranchOrDie(repo)
  git_commit = subprocess.check_output(
    ["git", "-C", repo, "rev-parse", "HEAD"], universal_newlines=True
  ).rstrip()

  with prof.Profile("Import testlogs"), database.Session(
    commit=True
  ) as session:
    for dir_, _, files in os.walk(testlogs):
      dir_ = pathlib.Path(dir_)
      for file_ in [f for f in files if f.endswith(".xml")]:
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

        log_path = dir_ / "test.log"
        assert log_path.is_file()
        log = fs.Read(log_path).rstrip()
        # Strip non-ASCII characters.
        log = log.encode("ascii", "ignore").decode("ascii")
        result.log = log

        # Bazel test runner reports a single test for pytest files, no matter
        # how many tests are actually in the file. Let's report a more account
        # test count by checking for pytest's collector output in the log.
        match = re.search(
          "^collecting ... collected (\d+) items$", result.log, re.MULTILINE
        )
        if match:
          result.test_count = int(match.group(1))

        # Add result to database.
        session.add(result)

  with database.Session(commit=False) as session:
    num_targets = (
      session.query(db.TestTargetResult.id)
      .filter(db.TestTargetResult.invocation_datetime == invocation_datetime)
      .count()
    )
    (num_tests,) = (
      session.query(sql.func.sum_(db.TestTargetResult.test_count))
      .filter(db.TestTargetResult.invocation_datetime == invocation_datetime)
      .one()
    )
    num_failed = (
      session.query(db.TestTargetResult.bazel_target)
      .filter(db.TestTargetResult.invocation_datetime == invocation_datetime)
      .filter(db.TestTargetResult.failed_count > 0)
      .count()
    )
    (total_runtime_ms,) = (
      session.query(sql.func.sum_(db.TestTargetResult.runtime_ms))
      .filter(db.TestTargetResult.invocation_datetime == invocation_datetime)
      .one()
    )

    print(
      f"{humanize.Commas(num_targets)} targets ("
      f"{humanize.Commas(num_tests)} tests in "
      f"{humanize.Duration(total_runtime_ms / 1000)}), {num_failed} failed"
    )

    delta = db.GetTestDelta(session, invocation_datetime)

    print(
      f"{humanize.Commas(len(delta.fixed))} fixed, "
      f"{humanize.Commas(len(delta.broken))} broken."
    )
    for target in delta.broken:
      print("BROKEN", target[0])

    for target in delta.fixed:
      print("FIXED", target[0])


if __name__ == "__main__":
  app.Run(main)
