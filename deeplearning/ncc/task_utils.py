# NCC: Neural Code Comprehension
# https://github.com/spcl/ncc
# Copyright 2018 ETH Zurich
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the follo
# wing conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================
"""Helper variables and functions for NCC task training"""

import os
import pathlib
import pickle
import re
import struct
import zipfile

import numpy as np
import wget

from deeplearning.ncc import vocabulary
from labm8 import app

# Embedding and vocabulary file paths
app.DEFINE_string('embeddings_file', None, 'Path to the embeddings file')

from labm8 import app
FLAGS = app.FLAGS


########################################################################################################################
# Downloading data sets
########################################################################################################################
def download_and_unzip(url, dataset_name, data_folder):
  """
  Download and unzip data set folder from url
  :param url: from which to download
  :param dataset_name: name of data set (for printing)
  :param data_folder: folder in which to put the downloaded data
  """
  print('Downloading', dataset_name, 'data set...')
  data_zip = wget.download(url, out=data_folder)
  print('\tunzipping...')
  zip_ = zipfile.ZipFile(data_zip, 'r')
  zip_.extractall(data_folder)
  zip_.close()
  print('\tdone')


########################################################################################################################
# Reading, writing and dumping files
########################################################################################################################
def ReadEmbeddingFile(path: pathlib.Path) -> np.ndarray:
  """Load embedding matrix from file"""
  if not path.is_file():
    raise app.UsageError(f"Embedding file not found: '{path}'")
  app.Info('Loading pre-trained embeddings from %s', path)
  with open(path, 'rb') as f:
    embedding_matrix = pickle.load(f)
  vocabulary_size, embedding_dimension = embedding_matrix.shape
  app.Info(
      'Loaded pre-trained embeddings with vocabulary size: %d and '
      'embedding dimension: %d', vocabulary_size, embedding_dimension)
  return embedding_matrix


def ReadEmbeddingFileFromFlags() -> np.ndarray:
  """Load embedding matrix from file"""
  if not FLAGS.embeddings_file:
    raise app.UsageError("--embeddings_file not set")
  return ReadEmbeddingFile(pathlib.Path(FLAGS.embeddings_file))


def read_data_files_from_folder(foldername):
  """
  Read all source files in folder
  Return a list of file contents, whereby each file content is a list of strings, each string representing a line
  :param foldername: name of the folder in which the data files to be read are located
  :return: a list of files where each file is a list of strings
  """
  # Helper variables
  data = list()
  file_names = list()

  files_in_folder = os.listdir(foldername + '/')

  # Loop over files in folder
  for path in files_in_folder:
    if path[0] != '.' and path[-3:] == '.ll':
      # If this isn't a hidden file and it is an LLVM IR file ('.ll' extension),
      # open file and import content
      with open(os.path.join(foldername, path)) as f:
        data.append(f.read())

      # Add file name to dictionary
      file_names.append(path)

  return data, file_names


def CreateSeqDirFromIr(folder_ir: str,
                       vocab: vocabulary.VocabularyZipFile) -> str:
  """Transform a folder of raw IR into trainable data to be used as input data
  in tasks.

  Args:
    folder_ir: The folder of LLVM IR to read. Must end in '_ir'.
    vocab: The vocabulary to use to encode IR.

  Returns:
    The path of the folder of sequences, ending in '_seq'.
  """
  # Setup
  assert folder_ir, "Please specify a folder containing the raw LLVM IR"
  assert os.path.exists(folder_ir), "Folder not found: " + folder_ir
  folder_seq = re.sub('_ir$', '_seq', folder_ir)
  if folder_seq:
    app.Info('Preparing to write LLVM IR index sequences to %s', folder_seq)
    if not os.path.exists(folder_seq):
      os.makedirs(folder_seq)

  # Get sub-folders if there are any
  listing = os.listdir(folder_ir + '/')
  folders_ir = list()
  folders_seq = list()
  found_subfolder = False
  for path in listing:
    if os.path.isdir(os.path.join(folder_ir, path)):
      folders_ir.append(os.path.join(folder_ir, path))
      folders_seq.append(os.path.join(folder_seq, path))
      found_subfolder = True
  if found_subfolder:
    app.Info('Found %d subfolders', len(folders_ir))
  else:
    app.Info('No subfolders found in %s', folder_ir)
    folders_ir = [folder_ir]
    folders_seq = [folder_seq]

  # Loop over sub-folders
  for i, raw_ir_folder in enumerate(folders_ir):

    l = folders_seq[i] + '/'
    if not os.path.exists(l) or not os.listdir(l):
      # Read data from folder
      raw_data, file_names = read_data_files_from_folder(raw_ir_folder)

      # Write indexed sequence of statements
      seq_folder = folders_seq[i]
      if not os.path.exists(seq_folder):
        os.makedirs(seq_folder)

      # Write indexed sequence of statements to files.
      for i, file in enumerate(raw_data):
        result = vocab.EncodeLlvmBytecode(file)

        # Write to csv
        file_name_csv = os.path.join(seq_folder,
                                     file_names[i][:-3] + '_seq.csv')
        file_name_rec = os.path.join(seq_folder,
                                     file_names[i][:-3] + '_seq.rec')
        with open(file_name_csv, 'w') as csv, open(file_name_rec, 'wb') as rec:
          for ind in result.encoded:
            csv.write(str(ind) + '\n')
            rec.write(struct.pack('I', int(ind)))

  return folder_seq
