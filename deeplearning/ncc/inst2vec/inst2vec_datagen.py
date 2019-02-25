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
"""Generate dataset for inst2vec training"""

import os
import pathlib
import shutil
import typing
import zipfile

import wget
from absl import logging

# Datasets and their URLs.
DATASETS = {
    'AMD': 'https://polybox.ethz.ch/index.php/s/SaKQ9L7dGs9zJXK/download',
    'BLAS': 'https://polybox.ethz.ch/index.php/s/5ASMNv6dYsPKjyQ/download',
    'Eigen': 'https://polybox.ethz.ch/index.php/s/52wWiK5fjRGHLJR/download',
    'gemm_synthetic':
    'https://polybox.ethz.ch/index.php/s/Bm6cwAY3eVkR6v3/download',
    'linux': 'https://polybox.ethz.ch/index.php/s/uxAAONROj1Id65y/download',
    'opencv': 'https://polybox.ethz.ch/index.php/s/KnWjolzAL2xxKWN/download',
    'polybenchGPU':
    'https://polybox.ethz.ch/index.php/s/nomO17gdAfHjqFQ/download',
    'rodinia_3': 'https://polybox.ethz.ch/index.php/s/J93jGpevs0lHsHM/download',
    'shoc': 'https://polybox.ethz.ch/index.php/s/7KGEq1Q45Xg0IeL/download',
    'stencil_synthetic':
    'https://polybox.ethz.ch/index.php/s/OOmylxGcBxQM1D3/download',
    'tensorflow':
    'https://polybox.ethz.ch/index.php/s/ojd0RPFOtUTPPRr/download',
}


def DownloadDatasets(data_folder,
                     urls: typing.Optional[typing.List[str]] = None):
  """Download and unzip training data for inst2vec

  Args:
    data_folder: folder in which to put the downloaded data
    urls: An optional list of URLS to download. If not provided,
        DATASETS are used.
  """
  urls = urls or DATASETS.values()

  for url in urls:
    DownloadAndUnzip(url, data_folder)

  # Remove __MACOSX directory resulting from unzipping.
  if os.path.exists(os.path.join(data_folder, '__MACOSX')):
    shutil.rmtree(os.path.join(data_folder, '__MACOSX'))


def DownloadAndUnzip(url, data_folder, delete_after_download: bool = True):
  """Download and unzip data set folder from url.

  Args:
    url: from which to download
    data_folder: folder in which to put the downloaded data
    delete_after_download: If True, delete the file after downloading and
      unzipping.
  """
  logging.info('Downloading dataset from %s', url)
  data_zip = wget.download(url, out=str(data_folder))
  logging.info('Unzipping %s to %s', data_zip, data_folder)
  with zipfile.ZipFile(data_zip, 'r') as f:
    f.extractall(path=data_folder)
  # Delete the zip.
  if delete_after_download:
    pathlib.Path(data_zip).unlink()
