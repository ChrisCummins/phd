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
        contentfiles.GitHubRepository(owner='foo',
                                      name='bar',
                                      clone_from_url='abc',
                                      num_stars=0,
                                      num_forks=0,
                                      num_watchers=0,
                                      date_scraped=datetime.datetime.utcnow(),
                                      language='java'))
    session.add(
        contentfiles.ContentFile(clone_from_url='abc',
                                 relpath='foo',
                                 artifact_index=0,
                                 sha256='000',
                                 charcount='0',
                                 linecount='0',
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
  exporter = export_java_corpus.Exporter(db, empty_db, [
      'datasets.github.scrape_repos.preprocessors.extractors:JavaStaticMethods',
  ])
  exporter.start()
  exporter.join()

  with empty_db.Session() as s:
    assert s.query(contentfiles.GitHubRepository).count() == 1
    assert s.query(contentfiles.ContentFile).count() == 1


if __name__ == '__main__':
  test.Main()
