"""Unit tests for //tools/git/test:fixtures."""
import pathlib

import git

from labm8.py import test

pytest_plugins = ["tools.git.test.fixtures"]

FLAGS = test.FLAGS


def test_repo_with_history_contents(repo_with_history: git.Repo):
  repo_tree = pathlib.Path(repo_with_history.working_tree_dir)
  assert (repo_tree / "README.txt").is_file()
  assert (repo_tree / "src").is_dir()
  assert (repo_tree / "src" / "main.c").is_file()
  assert not (repo_tree / "src" / "Makefile").is_file()


if __name__ == "__main__":
  test.Main()
