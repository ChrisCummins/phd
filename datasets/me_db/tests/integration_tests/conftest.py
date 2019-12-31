# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Pytest fixtures for me.db tests."""
import tempfile

from datasets.me_db import me_db
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import test

FLAGS = app.FLAGS
app.DEFINE_string(
  "integration_tests_inbox",
  None,
  "If set, this sets the inbox path to be used by the "
  "integration tests. This overrides the default in "
  "//datasets/me_db/integration_tests/inbox.",
)

TEST_INBOX_PATH = bazelutil.DataPath("phd/datasets/me_db/tests/test_inbox")


@test.Fixture(scope="function")
def mutable_db() -> me_db.Database:
  """Returns a populated database for the scope of the function."""
  with tempfile.TemporaryDirectory(prefix="phd_") as d:
    db = me_db.Database(f"sqlite:///{d}/me.db")
    db.ImportMeasurementsFromInboxImporters(TEST_INBOX_PATH)
    yield db


@test.Fixture(scope="session")
def db() -> me_db.Database:
  """Returns a populated database that is reused for all tests.

  DO NOT MODIFY THE TEST DATABASE. This will break other tests. For a test that
  modifies the database, use the `mutable_db` fixture.
  """
  with tempfile.TemporaryDirectory(prefix="phd_") as d:
    db = me_db.Database(f"sqlite:///{d}/me.db")
    db.ImportMeasurementsFromInboxImporters(TEST_INBOX_PATH)
    yield db
