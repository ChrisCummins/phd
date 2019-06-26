"""Unit tests for //experimental/deeplearning/deepsmith/java_fuzz:export_java_corpus."""
import datetime
import pathlib
from datasets.github.scrape_repos import contentfiles

import pytest
from experimental.deeplearning.deepsmith.java_fuzz import export_java_corpus
from labm8 import test

FLAGS = test.FLAGS


@pytest.fixture(scope='function')
def db(tempdir: pathlib.Path) -> contentfiles.ContentFiles:
  db_ = contentfiles.ContentFiles(f'sqlite:///{tempdir}/a')
  with db_.Session(commit=True) as session:
    session.add(
        contentfiles.GitHubRepository(
            owner='foo',
            name='bar',
            clone_from_url='abc',
            num_stars=0,
            num_forks=0,
            num_watchers=0,
            active=1,
            exported=0,
            date_scraped=datetime.datetime.utcnow(),
            language='java'))
    session.add(
        contentfiles.ContentFile(
            clone_from_url='abc',
            relpath='foo',
            artifact_index=0,
            sha256='000',
            charcount=100,
            linecount=4,
            active=1,
            text="""
public class HelloWorld {
  private int foo() {
    return 5;
  }

  public static void main(String[] args) {
    System.out.println("Hello, world");
  }
}
"""))
  return db_


@pytest.fixture(scope='function')
def empty_db(tempdir: pathlib.Path) -> contentfiles.ContentFiles:
  return contentfiles.ContentFiles(f'sqlite:///{tempdir}/b')


def test_Exporter(db: contentfiles.ContentFiles,
                  empty_db: contentfiles.ContentFiles):
  """Test that exporter behaves as expected."""
  exporter = export_java_corpus.Exporter(db, empty_db, static_only=True)
  exporter.start()
  exporter.join()

  with empty_db.Session() as s:
    assert s.query(contentfiles.GitHubRepository).count() == 1
    assert s.query(contentfiles.ContentFile).count() == 1

    repo = s.query(contentfiles.GitHubRepository).first()
    assert repo.clone_from_url == 'abc'

    contentfile = s.query(contentfiles.ContentFile).first()
    assert contentfile.sha256 != '000'
    assert contentfile.relpath == 'foo'
    assert contentfile.text == """\
public static void main(String[] args){
  System.out.println("Hello, world");
}
"""
    assert contentfile.charcount == len("""\
public static void main(String[] args){
  System.out.println("Hello, world");
}
""")
    assert contentfile.linecount == 4


def test_Exporter_overloaded_method_extraction(
    db: contentfiles.ContentFiles, empty_db: contentfiles.ContentFiles):
  """Test that exporter behaves as expected."""
  exporter = export_java_corpus.Exporter(db, empty_db, static_only=True)

  with db.Session(commit=True) as s:
    s.add(
        contentfiles.ContentFile(
            clone_from_url='abc',
            relpath='a/file.txt',
            artifact_index=0,
            sha256='000',
            charcount=200,
            linecount=10,
            text="""
public class HelloWorld {
  private static int foo(int a) {
    return 5;
  }

  private static int foo(float a) {
    return 5;
  }

  private static int foo(double a) {
    return 5;
  }
}
"""))

  exporter.start()
  exporter.join()

  with empty_db.Session() as s:
    query = s.query(contentfiles.ContentFile) \
      .filter(contentfiles.ContentFile.relpath == 'a/file.txt')
    assert query.count() == 3
    for cf in query:
      assert 'private static int foo(' in cf.text

    indices = {cf.artifact_index for cf in query}
    assert indices == {0, 1, 2}


if __name__ == '__main__':
  test.Main()
