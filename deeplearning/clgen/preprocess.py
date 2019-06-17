# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""Pre-process a corpus to produce a new corpus.

This is a utility script allowing you to invoke preprocessors from the command
line. Preprocessing is performed automatically by CLgen when constructing a
corpus - there is likely never a good reason to use this script. It is intended
only for testing and debugging purposes.
"""
import multiprocessing
import time

import pathlib
import typing
from deeplearning.clgen.proto import internal_pb2
from deeplearning.clgen.preprocessors import preprocessors

import progressbar
from deeplearning.clgen.corpuses import preprocessed
from labm8 import app
from labm8 import humanize

FLAGS = app.FLAGS
app.DEFINE_input_path(
    'contentfiles',
    None,
    'The directory containing content files.',
    is_dir=True)
app.DEFINE_output_path(
    'outdir',
    None,
    'Directory to export preprocessed content files to.',
    is_dir=True)
app.DEFINE_list('preprocessors', [], 'The preprocessors to run, in order.')


def Preprocess(contentfiles: pathlib.Path, outdir: pathlib.Path,
               preprocessor_names):
  # Error early if preprocessors are bad.
  [preprocessors.GetPreprocessorFunction(f) for f in preprocessor_names]

  # This is basically the same code as:
  # deeplearning.clgen.corpuses.preprocessed.PreprocessedContentFiles:Import()
  # Only it's writing the results of preprocessing to files rather than to a
  # database. Consider refactoring.
  relpaths = {f.name for f in contentfiles.iterdir()}
  done = {f.name for f in outdir.iterdir()}
  todo = relpaths - done
  app.Log(1, 'Preprocessing %s of %s content files', humanize.Commas(len(todo)),
          humanize.Commas(len(relpaths)))
  jobs = [
      internal_pb2.PreprocessorWorker(
          contentfile_root=str(contentfiles),
          relpath=t,
          preprocessors=preprocessor_names) for t in todo
  ]
  pool = multiprocessing.Pool()
  bar = progressbar.ProgressBar(max_value=len(jobs))
  wall_time_start = time.time()
  workers = pool.imap_unordered(preprocessed.PreprocessorWorker, jobs)
  succeeded_count = 0
  for preprocessed_cf in bar(workers):
    wall_time_end = time.time()
    preprocessed_cf.wall_time_ms = (int(
        (wall_time_end - wall_time_start) * 1000))
    wall_time_start = wall_time_end
    if preprocessed_cf.preprocessing_succeeded:
      succeeded_count += 1
      with open(outdir / preprocessed_cf.input_relpath, 'w') as f:
        f.write(preprocessed_cf.text)

  app.Log(1, "Successfully preprocessed %s of %s files (%.2f %%)",
          humanize.Commas(succeeded_count), humanize.Commas(len(todo)),
          (succeeded_count / min(len(todo), 1)) * 100)


def main():
  """Main entry point."""
  Preprocess(FLAGS.contentfiles, FLAGS.outdir, FLAGS.preprocessors)


if __name__ == '__main__':
  app.Run(main)
