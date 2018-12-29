"""Unit tests for //datasets/github:non_hemetic_credentials_file."""
import pathlib
import sys
import tempfile
import typing

import pytest
from absl import app
from absl import flags

from datasets.github import non_hemetic_credentials_file as cf


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def tempdir() -> pathlib.Path:
  with tempfile.TemporaryDirectory() as d:
    yield pathlib.Path(d)


def test_GitHubCredentialsFromFile_file_not_found(tempdir: pathlib.Path):
  """Test that error raised when file is not found."""
  with pytest.raises(cf.InvalidGitHubCredentialsFile) as e_ctx:
    cf.GitHubCredentialsFromFile(tempdir / 'credentials')
  assert str(e_ctx.value).startswith("File not found: ")


def test_GitHubCredentialsFromFile_values_read(tempdir: pathlib.Path):
  """Test that values are set in tuple."""
  with open(tempdir / 'credentials', 'w') as f:
    f.write("""
[User]
Username = foo
Password = bar
""")

  c = cf.GitHubCredentialsFromFile(tempdir / 'credentials')
  assert c.username == 'foo'
  assert c.password == 'bar'


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
