"""This module handles creation of databases for testing.

This module automatically enumerates a list of database URLs to use for testing,
and provides convenience functions to simplify creating test fixtures.

To create empty databases, use GetDatabaseUrls() as parameters to a
fixture which calls YieldDatabase():

    @test.Fixture(scope="function",
                  params=testing_databases.GetDatabaseUrls(),
                  namer=testing_databases.DatabaseUrlNamer("my_db"))
    def db(request) -> MyDatabaseClass:
      yield from testing_databases.YieldDatabase(
          MyDatabaseClass, request.param
      )

To create pre-populated databases, use GetDatabaseUrls() as parameters to
a fixture which calls DatabaseContext():

    @test.Fixture(scope="function",
                  params=testing_databases.GetDatabaseUrls(),
                  namer=testing_databases.DatabaseUrlNamer("my_db"))
    def populated_db(request) -> Database:
      with testing_databases.DatabaseContext(Database, request.param) as db:
        with db.Session(commit=True) as session:
          # Go nuts ...
        yield db
"""
import contextlib
import pathlib
import random
from typing import Callable
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


def GetDatabaseUrls() -> List[str]:
  """Enumerate a list of database URLs to use for tests.

  The databases are machine-local unique, meaning that multiple fixtures can
  each call this function to generate database URLs that do not conflict.

  Returns:
    A list of database URLs.
  """
  # Generate a unique run ID for this function invocation.
  run_id_ = run_id.RunId.GenerateUnique(
    f"test_{random.randint(0, int(1e8)):08d}"
  )

  # Always test with a file-backed SQLite database. Don't test with in-memory
  # SQLite database as these don't work with multi-threaded code.
  db_urls = [f"sqlite:////tmp/{run_id_}.db"]

  for path in TEST_DB_FILES:
    if path.is_file():
      db_urls.append(f"file://{path}?{run_id_}")

  return db_urls


def DatabaseUrlNamer(db_name: str) -> Callable[[str], str]:
  """Return a @test.Fixture namer callback for the given database.

  This produces a terse name for a database URL parameter by including only
  the specified database name and the URL prefix.

  Args:
    db_name: The "name" for the database, e.g. 'log_db' if enumerating test
      fixtures for logging databases.
  """

  def DatabaseUrlToName(url: str) -> str:
    """Produce a short name for a database URL."""
    # Resolve the actual URL, by expanding file:// URLs.
    full_url = sqlutil.ResolveUrl(url, use_flags=False)

    # Strip everything except the URL prefix, e.g. sqlite:////tmp/foo -> sqlite.
    url_prefix = full_url.split(":")[0]
    return f"{db_name}:{url_prefix}"

  return DatabaseUrlToName


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
