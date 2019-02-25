"""Unit tests for //datasets/github/scrape_repos/github_repo.py."""
import multiprocessing
import pathlib

import pytest
from absl import flags

from datasets.github.scrape_repos import github_repo
from datasets.github.scrape_repos.proto import scrape_repos_pb2
from labm8 import fs
from labm8 import pbutil
from labm8 import test

FLAGS = flags.FLAGS

# Test fixtures.


def _CreateTestRepo(root_dir: pathlib.Path, owner: str,
                    name: str) -> github_repo.GitHubRepo:
  """Create an empty repo for testing indexers."""
  owner_name = f'{owner}_{name}'
  (root_dir / owner_name / '.git').mkdir(parents=True)
  (root_dir / owner_name / 'src').mkdir(parents=True)
  pbutil.ToFile(
      scrape_repos_pb2.GitHubRepoMetadata(owner=owner, name=name),
      root_dir / f'{owner_name}.pbtxt')
  return github_repo.GitHubRepo(root_dir / f'{owner_name}.pbtxt')


@pytest.fixture(scope='function')
def test_repo(tempdir: pathlib.Path) -> github_repo.GitHubRepo:
  (tempdir / 'src').mkdir()
  yield _CreateTestRepo(tempdir / 'src', 'Foo', 'Bar')


# GitHubRepo tests.


def test_GitHubRepo_IsCloned(test_repo: github_repo.GitHubRepo):
  """Test for IsCloned()."""
  assert test_repo.IsCloned()
  fs.rm(test_repo.clone_dir)
  assert not test_repo.IsCloned()


def test_GitHubRepo_IsIndexed(test_repo: github_repo.GitHubRepo):
  """Test for IsIndexed()."""
  assert not test_repo.IsIndexed()
  test_repo.index_dir.mkdir(parents=True)
  (test_repo.index_dir / 'DONE.txt').touch()
  assert test_repo.IsIndexed()


def test_GitHubRepo_Index_not_cloned(test_repo: github_repo.GitHubRepo):
  """Indexing a repo which is not cloned does nothing."""
  fs.rm(test_repo.clone_dir)
  assert not test_repo.IsIndexed()
  test_repo.Index([
      scrape_repos_pb2.ContentFilesImporterConfig(
          source_code_pattern='.*\\.java',
          preprocessor=[
              "datasets.github.scrape_repos.preprocessors."
              "extractors:JavaMethods"
          ]),
  ], multiprocessing.Pool(1))
  assert not test_repo.IsIndexed()


def test_GitHubRepo_Index_Java_repo(test_repo: github_repo.GitHubRepo):
  """An end-to-end test of a Java indexer."""
  (test_repo.clone_dir / 'src').mkdir(exist_ok=True)
  with open(test_repo.clone_dir / 'src' / 'A.java', 'w') as f:
    f.write("""
public class A {
  public static void helloWorld() {
    System.out.println("Hello, world!");
  }
}
""")
  with open(test_repo.clone_dir / 'src' / 'B.java', 'w') as f:
    f.write("""
public class B {
  private static int foo() {return 5;}
}
""")
  with open(test_repo.clone_dir / 'README.txt', 'w') as f:
    f.write('Hello, world!')

  assert not test_repo.index_dir.is_dir()
  assert not list(test_repo.ContentFiles())
  test_repo.Index([
      scrape_repos_pb2.ContentFilesImporterConfig(
          source_code_pattern='.*\\.java',
          preprocessor=[
              "datasets.github.scrape_repos.preprocessors."
              "extractors:JavaMethods"
          ]),
  ], multiprocessing.Pool(1))
  assert test_repo.index_dir.is_dir()

  assert (test_repo.index_dir / 'DONE.txt').is_file()
  assert len(list(test_repo.index_dir.iterdir())) == 3
  contentfiles = list(test_repo.ContentFiles())
  assert len(contentfiles) == 2

  assert set([cf.text for cf in contentfiles]) == {
      ('public static void helloWorld(){\n'
       '  System.out.println("Hello, world!");\n}\n'),
      'private static int foo(){\n  return 5;\n}\n',
  }


def test_GitHubRepo_Index_index_dir_paths(tempdir: pathlib.Path):
  """Test that index directories are produced in the correct location."""
  repo = _CreateTestRepo(tempdir / 'java', 'Foo', 'Bar')
  repo.Index([
      scrape_repos_pb2.ContentFilesImporterConfig(
          source_code_pattern='.*\\.java',
          preprocessor=[
              "datasets.github.scrape_repos.preprocessors."
              "extractors:JavaMethods"
          ]),
  ], multiprocessing.Pool(1))
  assert (tempdir / 'java.index').is_dir()
  assert (tempdir / 'java.index' / 'Foo_Bar').is_dir()


if __name__ == '__main__':
  test.Main()
