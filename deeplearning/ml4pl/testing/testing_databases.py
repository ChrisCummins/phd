"""This module handles creation of databases for testing.

This module automatically enumerates a list of database URLs to use for testing,
and provides convenience functions to simplify creating test fixtures.

To create empty databases, use TEST_DB_URLS as parameters to a fixture which
calls YieldDatabase():

    @test.Fixture(scope="function", params=testing_databases.TEST_DB_URLS)
    def db(request) -> MyDatabaseClass:
      yield from testing_databases.YieldDatabase(
          MyDatabaseClass, request.param
      )

To create pre-populated databases, use TEST_DB_URLS as parameters to a fixture
which calls DatabaseContext():

    @test.Fixture(scope="function", params=testing_databases.TEST_DB_URLS)
    def populated_db(request) -> Database:
      with testing_databases.DatabaseContext(Database, request.param) as db:
        with db.Session(commit=True) as session:
          # Go nuts ...
        yield db
"""
import contextlib
import pathlib
from typing import List

from deeplearning.ml4pl import run_id
from labm8.py import app
from labm8.py import sqlutil

FLAGS = app.FLAGS


# A list of database URL files to use for testing. If any of these files exist,
# they will be used as file:// arguments to instantiate testing databases.
TEST_DB_FILES = [
  pathlib.Path("/var/phd/db/testing.mysql"),
]


def EnumerateTestingDatabaseUrls() -> List[str]:
  """Enumerate the list of database URLs to use for tests.

  There is no need to call this, use the TEST_DB_URLS variable instead. See
  module docstring for example usage.

  Returns:
    A list of database URLs.
  """
  # Always test with in-memory SQLite database:
  db_urls = ["sqlite://"]

  for path in TEST_DB_FILES:
    if path.is_file():
      db_urls.append(f"file://{path}?test_{run_id.RUN_ID}")

  return db_urls


TEST_DB_URLS = EnumerateTestingDatabaseUrls()


def YieldDatabase(db_class, db_url: str) -> sqlutil.Database:
  """Create and yield a throwaway database for testing.

  The database is created and immediate destroyed at the end of testing.
  See module docstring for usage.

  Args:
    db_class: A subclass of sqlutil.Database to instantiate.
    db_url: The URL of the database.

  Returns:
    An instance of db_class.
  """
  url = sqlutil.ResolveUrl(db_url)
  db: sqlutil.Database = db_class(url)
  try:
    yield db
  finally:
    db.Drop(are_you_sure_about_this_flag=True)


@contextlib.contextmanager
def DatabaseContext(db_class, db_url: str) -> sqlutil.Database:
  """Create a throwaway database for testing as a context manager.

  The database is created and immediate destroyed at the end of testing.
  See module docstring for usage.

  Args:
    db_class: A subclass of sqlutil.Database to instantiate.
    db_url: The URL of the database.

  Returns:
    An instance of db_class.
  """
  url = sqlutil.ResolveUrl(db_url)
  db: sqlutil.Database = db_class(url)
  try:
    yield db
  finally:
    db.Drop(are_you_sure_about_this_flag=True)
