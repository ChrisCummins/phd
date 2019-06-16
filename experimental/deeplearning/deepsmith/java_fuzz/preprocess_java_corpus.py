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
"""Run preprocessors on content files."""
import hashlib
import pathlib
import tempfile
import typing
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
app.DEFINE_output_path(
    'outdir',
    '/tmp/phd/experimental/deeplearning/deepsmith/java_fuzz/corpus',
    'Directory to export preprocessed content files to.',
    is_dir=True)
app.DEFINE_list('preprocessors', [], 'The preprocessors to run, in order.')


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


def ProcessRepo(
    input_session: sqlutil.Session, outdir: pathlib.Path, clone_from_url: str,
    workding_dir: pathlib.Path,
    preprocessor_functions: typing.List[preprocessors.PreprocessorFunction]):
  """Preprocess all content files from a single scraped repo."""
  contentfiles_to_export = input_session.query(
        contentfiles.ContentFile.relpath, contentfiles.ContentFile.text)\
      .filter(contentfiles.ContentFile.clone_from_url == clone_from_url)
  app.Log(1, 'Exporting %s content files from %s',
          humanize.Commas(contentfiles_to_export.count()), clone_from_url)

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
      encoded_text = text.encode('ascii', 'ignore')
      sha256 = hashlib.sha256(encoded_text).hexdigest()
      text = encoded_text.decode('ascii')
      fs.Write(outdir / (sha256 + '.txt'), text.encode('utf-8'))


def PreprocessDb(input_db: contentfiles.ContentFiles, outdir: pathlib.Path,
                 preprocessor_functions: typing.List[str]):
  """Preprocess the content files directory and export to outdir."""
  outdir.mkdir(parents=True, exist_ok=True)

  with input_db.Session() as input_session:
    clone_from_urls = [
        x[0] for x in input_session.query(
            contentfiles.ContentFile.clone_from_url).distinct()
    ]
    for clone_from_url in clone_from_urls:
      with tempfile.TemporaryDirectory(prefix='phd_') as d:
        ProcessRepo(input_session, outdir, clone_from_url, pathlib.Path(d),
                    preprocessor_functions)


def main():
  """Main entry point."""
  preprocessor_functions = [
      preprocessors.GetPreprocessorFunction(p) for p in FLAGS.preprocessors
  ]

  PreprocessDb(FLAGS.input(), FLAGS.outdir, preprocessor_functions)


if __name__ == '__main__':
  app.Run(main)
