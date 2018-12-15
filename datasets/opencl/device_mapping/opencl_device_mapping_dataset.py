"""A dataset of OpenCL Device Mappings."""

import pandas as pd
from absl import flags

from labm8 import bazelutil
from labm8 import decorators


FLAGS = flags.FLAGS

_AMD_CSV_PATH = bazelutil.DataPath('phd/datasets/opencl/device_mapping/amd.csv')
_NVIDIA_CSV_PATH = bazelutil.DataPath(
    'phd/datasets/opencl/device_mapping/amd.csv')


class OpenClDeviceMappingsDataset(object):
  """A dataset of OpenCL Device Mappings.

  This class provides 'views' of the dataset through properties which return
  Pandas DataFrames.

  The dataset was used in the publication:

      C. Cummins, P. Petoumenos, W. Zang, and H. Leather, “Synthesizing
      Benchmarks for Predictive Modeling,” in CGO, 2017.

  The code, slides, and pre-print of the paper are available at:

      https://chriscummins.cc/cgo17

  When using this dataset, please cite as:

      ﻿@inproceedings{Cummins2017a,
          annote = {NULL},
          author = {Cummins, C. and Petoumenos, P. and Zang, W. and Leather, H.},
          booktitle = {CGO},
          publisher = {IEEE},
          title = {{Synthesizing Benchmarks for Predictive Modeling}},
          year = {2017}
      }
  """

  def __init__(self):
    """Instantiate the dataset. This loads the entire dataset into memory."""
    # Read the CSV files and process them.
    amd_df = pd.read_csv(_AMD_CSV_PATH)
    nvidia_df = pd.read_csv(_NVIDIA_CSV_PATH)

    # Drop the redundant index columns.
    amd_df.drop('Unnamed: 0', axis='columns', inplace=True)
    nvidia_df.drop('Unnamed: 0', axis='columns', inplace=True)

    # Join the AMD and NVIDIA tables.
    self._df = amd_df.join(nvidia_df, lsuffix='_amd', rsuffix='_nvidia')

    # Check that the join has joined the correct benchmarks.
    def JoinedColumnsAreIdentical(column_name: str):
      return all(
          x == y for x, y in self._df[
            [f'{column_name}_amd', f'{column_name}_nvidia']].values)

    assert JoinedColumnsAreIdentical('dataset')
    assert JoinedColumnsAreIdentical('comp')
    assert JoinedColumnsAreIdentical('rational')
    assert JoinedColumnsAreIdentical('mem')
    assert JoinedColumnsAreIdentical('localmem')
    assert JoinedColumnsAreIdentical('coalesced')
    assert JoinedColumnsAreIdentical('atomic')
    assert JoinedColumnsAreIdentical('transfer')
    assert JoinedColumnsAreIdentical('wgsize')
    assert JoinedColumnsAreIdentical('runtime_cpu')

    # Each row in the CSV file has a benchmark name format:
    # '<suite>-<benchmark>-<kernel>.cl'. Extract the
    # '<suite>-<benchmark>-<kernel>' component of each row and assign to new
    # columns.
    self._df['program:benchmark_suite_name'] = [
      '-'.join(b.split('-')[:-2]) for b in self._df['benchmark_amd']]
    self._df['program:benchmark_name'] = [
      b.split('-')[-2] for b in self._df['benchmark_amd']]
    self._df['program:opencl_kernel_name'] = [
      b.split('-')[-1] for b in self._df['benchmark_amd']]

    # Rename columns that we keep unmodified.
    self._df.rename({
      'benchmark_amd': 'benchmark',
      'dataset_amd': 'data:dataset_name',
      'wgsize_amd': 'wgsize',
      'src_amd': 'program:opencl_src',
    }, axis='columns', inplace=True)

    # Drop redundant columns.
    self._df.drop([
      'benchmark_nvidia',
      'dataset_nvidia',
      'wgsize_nvidia',
      'src_nvidia',
    ], axis='columns', inplace=True)

  @decorators.memoized_property
  def programs_df(self) -> pd.DataFrame:
    """Return the a DataFrame view of the programs in the dataset.

    The returned DataFrame has the following schema:

      program:benchmark_suite_name (str): The name of the benchmark suite.
      program:benchmark_name (str): The name of the benchmark program.
      program:opencl_kernel_name (str): The name of the OpenCL kernel.
      program:opencl_src (str): Entire source code of the preprocessed OpenCL
        kernel.
    """
    df = self._df.groupby('benchmark').min()
    df = df[[
      'program:benchmark_suite_name',
      'program:benchmark_name',
      'program:opencl_kernel_name',
      'program:opencl_src',
    ]]
    df.set_index(['program:benchmark_suite_name',
                  'program:benchmark_name',
                  'program:opencl_kernel_name'], inplace=True)
    df.sort_index(inplace=True)
    return df
