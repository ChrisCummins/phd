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
import sys
import time

import collections
import hashlib
import pathlib
import re
import tempfile
import threading
import typing
from concurrent import futures

from datasets.github.scrape_repos import contentfiles
from datasets.github.scrape_repos.preprocessors import extractors
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
app.DEFINE_boolean('static_only', True, 'Only export static methods.')
app.DEFINE_integer('min_line_count', 3,
                   'The minimum number of lines in a contentfile to export.')
app.DEFINE_integer('min_char_count', 80,
                   'The minimum number of chars in a contentfile to export.')
app.DEFINE_boolean('multithreaded_export', True,
                   'Use multiple threads for export.')
# nproc * 5 is the same as the default used by the standard library.
# You may want to increase --sqlutil_engine_max_overflow to match this value.
app.DEFINE_integer('export_worker_threads',
                   multiprocessing.cpu_count() * 5,
                   "The number of export worker threads.")

# Regex to match a java import.
_JAVA_IMPORT_RE = re.compile(
    r'\s*import\s+(?P<package>[\w\.]+)\.(?P<classname>\w+)\s*;.*')


def ImportQueryResults(query, session):
  """Copy results of a query from one session into a new session."""
  # You can't simply use session.add_all() when the objects are already attached
  # to a different session.
  for row in query:
    session.merge(row)


def MaybeExtractJavaImport(
    line: str) -> typing.Optional[typing.Tuple[str, str]]:
  """Try and extract a java import basename from a given line of Java code.

  E.g. "import java.util.ArrayList;" -> "ArrayList"

  This assumes a fairly sane code style and won't work for some cases such as:
    * multiple imports per line.
    * '.*' globbed imports.

  Args:
    line: A line of Java code.

  Returns:
    A tuple consisting of the full import statement (without trailing ;), and
    the final component of a Java import, e.g. 'ArrayList'. If line does not
    contain an import, returns None.
  """
  match = _JAVA_IMPORT_RE.match(line)
  if match:
    return match.group('package'), match.group('classname')


def GetJavaImports(src: str) -> typing.Set[typing.Tuple[str, str]]:
  """Return the set of import basenames in a Java source.

  Args:
    src: A Java source.

  Returns:
    A (possibly empty) set.
  """
  matches = []
  for line in src.split('\n'):
    match = MaybeExtractJavaImport(line)
    if match:
      matches.append(match)
  return {classname: package for package, classname in set(matches)}


def InsertImportCommentHeader(method: str, imports: typing.Dict[str, str]):
  import_statements = []
  for basename, package in imports.items():
    if basename in method:
      import_statements.append(f'//import {package}.{basename}\n')
  return ''.join(import_statements) + method


def DoProcessRepo(input_session: sqlutil.Session,
                  output_session: sqlutil.Session, clone_from_url: str,
                  workding_dir: pathlib.Path, static_only: bool) -> None:
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
  for relpath, method_text in contentfiles_to_export:
    path = workding_dir / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    fs.Write(path, method_text.encode("utf-8"), overwrite_existing=False)

  # Copy repo to output.
  repo = input_session.query(contentfiles.GitHubRepository) \
      .filter(contentfiles.GitHubRepository.clone_from_url == clone_from_url)
  ImportQueryResults(repo, output_session)

  # Run the preprocessors.
  methods_lists = extractors.BatchedMethodExtractor(
      [text for _, text in contentfiles_to_export])

  relpath_counters = collections.defaultdict(int)

  for (relpath, text), methods in zip(contentfiles_to_export, methods_lists):
    # Attempt to extract all imports for this content file.
    imports = GetJavaImports(text)

    for i, original_method_text in enumerate(methods):
      # Insert "//import ..." comments before each method so that we know which
      # packages must be imported.
      method_text = InsertImportCommentHeader(original_method_text, imports)

      encoded_text = method_text.encode('ascii', 'ignore')
      sha256 = hashlib.sha256(encoded_text).hexdigest()
      method_text = encoded_text.decode('ascii')
      # Add new contentfile.
      output_session.add(
          contentfiles.ContentFile(
              clone_from_url=clone_from_url,
              relpath=relpath,
              artifact_index=relpath_counters[relpath],
              sha256=sha256,
              charcount=len(original_method_text),
              linecount=len(original_method_text.split('\n')),
              text=method_text,
          ))
      relpath_counters[relpath] += 1

  # Mark repo as exported.
  repo.update({"exported": True})


def ProcessRepo(input_db: contentfiles.ContentFiles,
                output_db: contentfiles.ContentFiles, clone_from_url: str,
                static_only: bool):
  """Preprocess all content files from a single scraped repo."""
  with input_db.Session(commit=True) as input_session:
    with output_db.Session(commit=True) as output_session:
      with tempfile.TemporaryDirectory(prefix='phd_') as d:
        DoProcessRepo(input_session, output_session, clone_from_url,
                      pathlib.Path(d), static_only)


class Exporter(threading.Thread):

  def __init__(self, input_db: contentfiles.ContentFiles,
               output_db: contentfiles.ContentFiles, static_only: bool):
    super(Exporter, self).__init__()
    self.input_db = input_db
    self.output_db = output_db
    self.static_only = static_only

  def run(self):
    """Preprocess the content files directory and export to outdir."""
    with self.input_db.Session() as input_session:
      repos_to_export = input_session.query(
          contentfiles.GitHubRepository.clone_from_url)\
          .filter(contentfiles.GitHubRepository.active == True)\
          .filter(contentfiles.GitHubRepository.exported == False)
      clone_from_urls = [x[0] for x in repos_to_export]

    max_workers = FLAGS.export_worker_threads
    app.Log(1, "Exporting contentfiles from %s repos in %s worker threads",
            humanize.Commas(len(clone_from_urls)), max_workers)
    if FLAGS.multithreaded_export:
      with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        f = lambda x: ProcessRepo(self.input_db, self.output_db, x, self.
                                  static_only)
        for _ in executor.map(f, clone_from_urls):
          pass
    else:
      for clone_from_url in clone_from_urls:
        ProcessRepo(self.input_db, self.output_db, clone_from_url,
                    self.preprocessor_functions)


def main():
  """Main entry point."""
  start_time = time.time()
  exporter = Exporter(FLAGS.input(), FLAGS.output(), FLAGS.static_only)
  exporter.start()

  with FLAGS.input().Session() as s:
    repo_count = s.query(contentfiles.GitHubRepository)\
      .filter(contentfiles.GitHubRepository.active == True).count()

  while True:
    runtime = time.time() - start_time
    with FLAGS.output().Session() as s:
      exported_repo_count = s.query(contentfiles.GitHubRepository).count()
      exported_contentfile_count = s.query(contentfiles.ContentFile).count()
    sys.stdout.write(
        f"\rRuntime: {humanize.Duration(runtime)}. "
        f"Exported repos: {humanize.Commas(exported_repo_count)} "
        f"of {humanize.Commas(repo_count)} "
        f"({exported_repo_count / repo_count:.2%}), "
        f"exported contentfiles: {humanize.Commas(exported_contentfile_count)}"
        "    ")
    sys.stdout.flush()

    if not exporter.is_alive():
      break
    time.sleep(1)
  exporter.join()

  sys.stdout.flush()
  sys.stderr.flush()
  print('Done!')


if __name__ == '__main__':
  app.Run(main)
