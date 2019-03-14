# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for //datasets/github/scrape_repos/indexer.py."""
import multiprocessing
import pathlib

import pytest

from datasets.github.scrape_repos import github_repo
from datasets.github.scrape_repos import indexer
from datasets.github.scrape_repos.proto import scrape_repos_pb2
from labm8 import app
from labm8 import pbutil
from labm8 import test

FLAGS = app.FLAGS

# Test fixtures.


def _CreateTestRepo(root_dir: pathlib.Path, owner: str, name: str) -> None:
  """Create an empty repo for testing indexers."""
  owner_name = f'{owner}_{name}'
  (root_dir / owner_name / '.git').mkdir(parents=True)
  (root_dir / owner_name / 'src').mkdir(parents=True)
  pbutil.ToFile(
      scrape_repos_pb2.GitHubRepoMetadata(owner=owner, name=name),
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
  pbutil.ToFile(
      scrape_repos_pb2.GitHubRepoMetadata(owner='Owner', name='Name'),
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
              preprocessor=[
                  "datasets.github.scrape_repos.preprocessors."
                  "extractors:JavaMethods"
              ]),
      ])
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


if __name__ == '__main__':
  test.Main()
