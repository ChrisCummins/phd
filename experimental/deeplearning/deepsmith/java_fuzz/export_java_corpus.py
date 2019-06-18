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
"""Export the content files from a database."""
import multiprocessing
import time

import hashlib
import pathlib
import tempfile
import threading
import typing
from concurrent import futures
from datasets.github.scrape_repos import contentfiles

from datasets.github.scrape_repos.preprocessors import preprocessors
from labm8 import app
from labm8 import fs
from labm8 import humanize
from labm8 import sqlutil

FLAGS = app.FLAGS
app.DEFINE_database(
    'input', contentfiles.ContentFiles,
    'sqlite:////var/phd/experimental/deeplearning/deepsmith/java_fuzz/java.db',
    'URL of the database to preprocess content files from.')
app.DEFINE_database(
    'output', contentfiles.ContentFiles,
    'sqlite:////var/phd/experimental/deeplearning/deepsmith/java_fuzz/export.db',
    'URL of the database to export content files to.')
app.DEFINE_list('preprocessors', [], 'The preprocessors to run, in order.')
app.DEFINE_integer('min_line_count', 0,
                   'The minimum number of lines in a contentfile to export.')
app.DEFINE_integer('min_char_count', 0,
                   'The minimum number of chars in a contentfile to export.')


def ImportQueryResults(query, session):
  """Copy results of a query from one session into a new session."""
  # You can't simply use session.add_all() when the objects are already attached
  # to a different session.
  for row in query:
    session.merge(row)


def Preprocess(
    import_root: pathlib.Path, file_relpath: str,
    all_file_relpaths: typing.List[str],
    preprocessor_functions: typing.List[preprocessors.PreprocessorFunction]
) -> typing.List[str]:
  """Preprocess a text using the given preprocessor pipeline.

  If preprocessing succeeds, the preprocessed text is returned. If preprocessing
  fails (in an expected way, for example by trying to compile incorrect code),
  a BadCodeException is raised. Any other error leads to an InternalError.

  Args:
    import_root: The root of the directory to import the file from.
    file_relpath: The path of the file to import, relative to import_root.
    all_file_relpaths: A list of all paths within the current scope, relative to
      import_root.
    preprocessor_functions: The preprocessor functions to run.

  Returns:
    Preprocessed sources.

  Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: If the requested preprocessors cannot be loaded.
    BadCodeException: If one of the preprocessors rejects the input.
    InternalException: In case of some other error.
  """
  path = import_root / file_relpath
  if not path.is_file():
    raise FileNotFoundError(f"File not found: {path}")

  with open(path) as f:
    texts = [f.read()]

  next_texts = []
  for preprocessor in preprocessor_functions:
    for text in texts:
      next_texts += preprocessor(import_root=import_root,
                                 file_relpath=file_relpath,
                                 text=text,
                                 all_file_relpaths=all_file_relpaths)
    texts = next_texts
  return texts


def DoProcessRepo(
    input_session: sqlutil.Session, output_session: sqlutil.Session,
    clone_from_url: str, workding_dir: pathlib.Path,
    preprocessor_functions: typing.List[preprocessors.PreprocessorFunction]
) -> None:
  """Preprocess all content files from a single scraped repo."""
  candidate_contentfiles = input_session.query(
        contentfiles.ContentFile.relpath, contentfiles.ContentFile.text)\
      .filter(contentfiles.ContentFile.clone_from_url == clone_from_url)
  contentfiles_to_export = candidate_contentfiles\
      .filter(contentfiles.ContentFile.linecount >= FLAGS.min_line_count)\
      .filter(contentfiles.ContentFile.charcount >= FLAGS.min_char_count).all()
  app.Log(2, 'Exporting %s of %s content files from %s',
          humanize.Commas(len(contentfiles_to_export)),
          humanize.Commas(candidate_contentfiles.count()), clone_from_url)

  # Create the directory tree first.
  for relpath, text in contentfiles_to_export:
    path = workding_dir / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    fs.Write(path, text.encode("utf-8"), overwrite_existing=False)

  all_files_relpaths = {relpath for relpath, _ in contentfiles_to_export}

  # Run the preprocessors.
  for relpath, text in contentfiles_to_export:
    texts = Preprocess(workding_dir, relpath, all_files_relpaths,
                       preprocessor_functions)
    for i, text in enumerate(texts):
      if (len(text) >= FLAGS.min_char_count and
          len(text.split('\n')) >= FLAGS.min_line_count):

        encoded_text = text.encode('ascii', 'ignore')
        sha256 = hashlib.sha256(encoded_text).hexdigest()
        text = encoded_text.decode('ascii')
        # Add new contentfile.
        output_session.add(
            contentfiles.ContentFile(
                clone_from_url=clone_from_url,
                relpath=relpath,
                artifact_index=i,
                sha256=sha256,
                charcount=len(text),
                linecount=len(text.split('\n')),
                text=text,
            ))

  # Copy repo to output.
  repo = input_session.query(contentfiles.GitHubRepository) \
      .filter(contentfiles.GitHubRepository.clone_from_url == clone_from_url)
  ImportQueryResults(repo, output_session)
  # Mark repo as exported.
  repo.update({"exported": True})


def ProcessRepo(
    input_db: contentfiles.ContentFiles, output_db: contentfiles.ContentFiles,
    clone_from_url: str,
    preprocessor_functions: typing.List[preprocessors.PreprocessorFunction]):
  """Preprocess all content files from a single scraped repo."""
  with input_db.Session(commit=True) as input_session:
    with output_db.Session(commit=True) as output_session:
      with tempfile.TemporaryDirectory(prefix='phd_') as d:
        DoProcessRepo(input_session, output_session, clone_from_url,
                      pathlib.Path(d), preprocessor_functions)


class Exporter(threading.Thread):

  def __init__(self, input_db: contentfiles.ContentFiles,
               output_db: contentfiles.ContentFiles,
               preprocessor_names: typing.List[str]):
    super(Exporter, self).__init__()
    self.input_db = input_db
    self.output_db = output_db
    self.preprocessor_functions = [
        preprocessors.GetPreprocessorFunction(p) for p in preprocessor_names
    ]

  def run(self):
    """Preprocess the content files directory and export to outdir."""
    with self.input_db.Session() as input_session:
      active_repos = input_session.query(
          contentfiles.GitHubRepository.clone_from_url)\
          .filter(contentfiles.GitHubRepository.active == True)
      clone_from_urls = [x[0] for x in active_repos]

    # nproc * 5 is the same as the default used by the standard library.
    max_workers = multiprocessing.cpu_count() * 5
    app.Log(1, "Exporting contentfiles from %s repos in %s worker threads",
            humanize.Commas(len(clone_from_urls)), max_workers)
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
      f = lambda x: ProcessRepo(self.input_db, self.output_db, x, self.
                                preprocessor_functions)
      for _ in executor.map(f, clone_from_urls):
        pass


def main():
  """Main entry point."""
  start_time = time.time()
  exporter = Exporter(FLAGS.input(), FLAGS.output(), FLAGS.preprocessors)
  exporter.start()

  with FLAGS.input().Session() as s:
    repo_count = s.query(contentfiles.GitHubRepository)\
      .filter(contentfiles.GitHubRepository.active == True).count()

  while True:
    time.sleep(15)

    runtime = time.time() - start_time
    with FLAGS.output().Session() as s:
      exported_repo_count = s.query(contentfiles.ContentFile).count()
      exported_contentfile_count = s.query(contentfiles.ContentFile).count()
    print(
        f"Runtime: {humanize.Duration(runtime)}. "
        f"Exported repos: {humanize.Commas(exported_repo_count)} "
        f"of {humanize.Commas(repo_count)} "
        f"({exported_repo_count / repo_count:.2%}), "
        f"exported contentfiles: {humanize.Commas(exported_contentfile_count)}")

    if not exporter.is_alive():
      break
  exporter.join()


if __name__ == '__main__':
  app.Run(main)
