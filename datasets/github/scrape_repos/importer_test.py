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
"""Unit tests for //datasets/github/scrape_repos/importer.py."""
import pathlib

import pytest

from datasets.github.scrape_repos import contentfiles
from datasets.github.scrape_repos import importer
from datasets.github.scrape_repos.proto import scrape_repos_pb2
from labm8.py import app
from labm8.py import pbutil
from labm8.py import test

FLAGS = app.FLAGS


@test.Fixture(scope="function")
def test_db(tempdir) -> contentfiles.ContentFiles:
  yield contentfiles.ContentFiles(f"sqlite:///{tempdir}/test.db")


def test_ImportFromLanguage_no_importer(
  test_db: contentfiles.ContentFiles, tempdir: pathlib.Path
):
  """Test that error is raised if no importer specified."""
  language = scrape_repos_pb2.LanguageToClone(
    language="test", query=[], destination_directory=str(tempdir), importer=[]
  )
  with test.Raises(ValueError):
    importer.ImportFromLanguage(test_db, language)


def test_ImportFromLanguage_Java_repo(
  test_db: contentfiles.ContentFiles, tempdir: pathlib.Path
):
  """An end-to-end test of a Java importer."""
  (tempdir / "Owner_Name" / ".git").mkdir(parents=True)
  (tempdir / "Owner_Name" / "src").mkdir(parents=True)

  # A repo will only be imported if there is a repo meta file.
  pbutil.ToFile(
    scrape_repos_pb2.GitHubRepoMetadata(owner="Owner", name="Name"),
    tempdir / "Owner_Name.pbtxt",
  )

  # Create some files in our test repo.
  with open(tempdir / "Owner_Name" / "src" / "A.java", "w") as f:
    f.write(
      """
public class A {
  public static void helloWorld() {
    System.out.println("Hello, world!");
  }
}
"""
    )
  with open(tempdir / "Owner_Name" / "src" / "B.java", "w") as f:
    f.write(
      """
public class B {
  private static int foo() {return 5;}
}
"""
    )
  with open(tempdir / "Owner_Name" / "README.txt", "w") as f:
    f.write("Hello, world!")

  language = scrape_repos_pb2.LanguageToClone(
    language="foolang",
    query=[],
    destination_directory=str(tempdir),
    importer=[
      scrape_repos_pb2.ContentFilesImporterConfig(
        source_code_pattern=".*\\.java",
        preprocessor=[
          "datasets.github.scrape_repos.preprocessors." "extractors:JavaMethods"
        ],
      ),
    ],
  )
  importer.ImportFromLanguage(test_db, language)
  with test_db.Session() as session:
    query = session.query(contentfiles.ContentFile)
    assert query.count() == 2
    assert set([cf.text for cf in query]) == {
      (
        "public static void helloWorld(){\n"
        '  System.out.println("Hello, world!");\n}\n'
      ),
      "private static int foo(){\n  return 5;\n}\n",
    }


if __name__ == "__main__":
  test.Main()
