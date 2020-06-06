# Copyright (c) 2016-2020 Chris Cummins.
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
"""CLgen: a deep learning program generator.

The core operations of CLgen are:

  1. Preprocess and encode a corpus of handwritten example programs.
  2. Define and train a machine learning model on the corpus.
  3. Sample the trained model to generate new programs.

This program automates the execution of all three stages of the pipeline.
The pipeline can be interrupted and resumed at any time. Results are cached
across runs. Please note that many of the steps in the pipeline are extremely
compute intensive and highly parallelized. If configured with CUDA support,
any NVIDIA GPUs will be used to improve performance where possible.

Made with \033[1;31m♥\033[0;0m by Chris Cummins <chrisc.101@gmail.com>.
https://chriscummins.cc/clgen
"""
from pathlib import Path

from tqdm import tqdm

from deeplearning.clgen.corpuses.preprocessed import PreprocessedContentFile
from deeplearning.clgen.corpuses.preprocessed import PreprocessedContentFiles
from labm8.py import app

FLAGS = app.FLAGS

app.DEFINE_input_path(
  "db", None, "Path of a SQLite database of pre-processed contentfiles."
)
app.DEFINE_output_path(
  "path", "/tmp/phd/preprocessed", "The directory to export files to.",
)
app.DEFINE_string("suffix", ".txt", "The filename suffix for exported files.")


def Main():
  """Main entry point."""
  db = PreprocessedContentFiles(FLAGS.db, must_exist=True)
  outpath: Path = FLAGS.path
  outpath.mkdir(parents=True, exist_ok=True)

  with db.Session() as session:
    query = (
      session.query(PreprocessedContentFile)
      .filter(PreprocessedContentFile.preprocessing_succeeded == True)
      .all()
    )
    for cf in tqdm(query):
      with open(outpath / f"{cf.sha256}.txt", "w") as f:
        f.write(cf.text)


if __name__ == "__main__":
  app.Run(Main)
