"""Unit tests for //datasets/github/scrape_repos/indexer.py."""
import multiprocessing

import pathlib
import pytest
import sys
import tempfile
import typing
from absl import app
from absl import flags
from phd.lib.labm8 import pbutil

from datasets.github.scrape_repos import github_repo
from datasets.github.scrape_repos import indexer
from datasets.github.scrape_repos.proto import scrape_repos_pb2


FLAGS = flags.FLAGS


# Test fixtures.

@pytest.fixture(scope='function')
def tempdir() -> pathlib.Path:
  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    yield pathlib.Path(d)


def _CreateTestRepo(root_dir: pathlib.Path, owner: str, name: str) -> None:
  """Create an empty repo for testing indexers."""
  owner_name = f'{owner}_{name}'
  (root_dir / owner_name / '.git').mkdir(parents=True)
  (root_dir / owner_name / 'src').mkdir(parents=True)
  pbutil.ToFile(scrape_repos_pb2.GitHubRepoMetadata(owner=owner, name=name),
                root_dir / f'{owner_name}.pbtxt')


# ShouldIndexRepo() tests.

def test_ImportFromLanguage_no_importer(tempdir: pathlib.Path):
  """Test that error is raised if no importer specified."""
  language = scrape_repos_pb2.LanguageToClone(
      language='test',
      query=[],
      destination_directory=str(tempdir),
      importer=[])
  with pytest.raises(ValueError):
    indexer.ImportFromLanguage(language, multiprocessing.Pool(1))


def test_ImportFromLanguage_Java_repo(tempdir: pathlib.Path):
  """An end-to-end test of a Java importer."""
  (tempdir / 'src').mkdir()
  (tempdir / 'src' / 'Owner_Name' / '.git').mkdir(parents=True)
  (tempdir / 'src' / 'Owner_Name' / 'src').mkdir(parents=True)

  # A repo will only be imported if there is a repo meta file.
  pbutil.ToFile(scrape_repos_pb2.GitHubRepoMetadata(
      owner='Owner',
      name='Name'),
      tempdir / 'src' / 'Owner_Name.pbtxt')

  # Create some files in our test repo.
  with open(tempdir / 'src' / 'Owner_Name' / 'src' / 'A.java', 'w') as f:
    f.write("""
public class A {
  public static void helloWorld() {
    System.out.println("Hello, world!");
  }
}
""")
  with open(tempdir / 'src' / 'Owner_Name' / 'src' / 'B.java', 'w') as f:
    f.write("""
public class B {
  private static int foo() {return 5;}
}
""")
  with open(tempdir / 'src' / 'Owner_Name' / 'README.txt', 'w') as f:
    f.write('Hello, world!')

  language = scrape_repos_pb2.LanguageToClone(
      language='foolang',
      query=[],
      destination_directory=str(tempdir / 'src'),
      importer=[
        scrape_repos_pb2.ContentFilesImporterConfig(
            source_code_pattern='.*\\.java',
            preprocessor=["datasets.github.scrape_repos.preprocessors."
                          "extractors:JavaMethods"]),
      ]
  )
  indexer.ImportFromLanguage(language, multiprocessing.Pool(1))

  test_repo = github_repo.GitHubRepo(tempdir / 'src' / 'Owner_Name.pbtxt')
  assert (test_repo.index_dir / 'DONE.txt').is_file()
  assert len(list(test_repo.index_dir.iterdir())) == 3
  contentfiles = list(test_repo.ContentFiles())
  assert len(contentfiles) == 2
  assert set([cf.text for cf in contentfiles]) == {
    ('public static void helloWorld(){\n'
     '  System.out.println("Hello, world!");\n}\n'),
    'private static int foo(){\n  return 5;\n}\n',
  }


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
