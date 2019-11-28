# NCC: Neural Code Comprehension
# https://github.com/spcl/ncc
# Copyright 2018 ETH Zurich
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================
"""Main inst2vec and ncc workflow"""
import os
import pathlib
import pickle

from deeplearning.ncc.inst2vec import inst2vec_appflags
from deeplearning.ncc.inst2vec import inst2vec_datagen as i2v_datagen
from deeplearning.ncc.inst2vec import inst2vec_embedding as i2v_emb
from deeplearning.ncc.inst2vec import inst2vec_evaluate as i2v_eval
from deeplearning.ncc.inst2vec import inst2vec_preprocess as i2v_prep
from deeplearning.ncc.inst2vec import inst2vec_vocabulary as i2v_vocab
from labm8.py import app

# Get the app flags from a file.
FLAGS = inst2vec_appflags.FLAGS

# Data set parameters.
app.DEFINE_string(
  "data_folder",
  "/tmp/phd/deeplearning/ncc/inst2vec/data",
  "Dataset folder path.",
)
app.DEFINE_boolean("download_datasets", True, "Whether to use default dataset.")
app.DEFINE_list(
  "dataset_urls",
  [],
  "URLs of datasets to download. If not provided, all datasets will be used.",
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Unrecognized command line flags.")

  data_folder = os.path.join(FLAGS.data_folder)

  # Make the data folder if it does not exist.
  pathlib.Path(data_folder).mkdir(parents=True, exist_ok=True)

  if not os.path.exists(FLAGS.embeddings_file):
    if FLAGS.download_datasets:
      # Generate the data set
      print(
        "Folder", data_folder, "is empty - preparing to download training data"
      )
      i2v_datagen.DownloadDatasets(data_folder, urls=FLAGS.dataset_urls)
    else:
      # Assert the data folder's existence
      assert os.path.exists(data_folder), (
        "Folder " + data_folder + " does not exist"
      )

    # Build XFGs from raw code
    data_folders = i2v_prep.CreateContextualFlowGraphsFromBytecodes(data_folder)

    # Build vocabulary
    i2v_vocab.construct_vocabulary(data_folder, data_folders)

    # Train embeddings
    embedding_matrix, embeddings_file = i2v_emb.train_embeddings(
      data_folder, data_folders
    )

  else:

    print("Loading pre-trained embeddings from", FLAGS.embeddings_file)
    with open(FLAGS.embeddings_file, "rb") as f:
      embedding_matrix = pickle.load(f)
    embeddings_file = FLAGS.embeddings_file

  # Evaluate embeddings (intrinsic evaluation)
  i2v_eval.evaluate_embeddings(data_folder, embedding_matrix, embeddings_file)


if __name__ == "__main__":
  app.RunWithArgs(main)
