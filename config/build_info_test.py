"""Unit tests for //config:build_info."""
import re

from config import build_info
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


def test_GetGitRepo():
  assert build_info.GetGitRepo()


def test_GetBuildInfo():
  info = build_info.GetBuildInfo()
  assert re.match(r'[0-9a-f]{40}', info.id)
  assert info.date
  assert re.match(r'.+ <.+@.+>', info.author)
  assert info.branch
  assert info.remote
  assert re.match(r'https://github.com/.+/.+/commit/[0-9a-f]{40}',
                  info.commit_url)


if __name__ == '__main__':
  test.Main()
