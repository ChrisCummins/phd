"""A dataset of OpenCL Device Mappings."""

import functools
import typing

import numpy as np
import pandas as pd
from absl import flags

from deeplearning.deeptune.opencl.adversary import \
  opencl_deadcode_inserter as dci
from labm8 import bazelutil
from labm8 import decorators


FLAGS = flags.FLAGS

_AMD_CSV_PATH = bazelutil.DataPath('phd/datasets/opencl/device_mapping/amd.csv')
_NVIDIA_CSV_PATH = bazelutil.DataPath(
    'phd/datasets/opencl/device_mapping/nvidia.csv')


def ComputeGreweFeaturesForGpu(gpu: str, df: pd.DataFrame) -> pd.DataFrame:
  """Return the Grewe et al. features as a table.

  These are the features used in the publication:

  ﻿    Grewe, D., Wang, Z., & O’Boyle, M. (2013). Portable Mapping of Data
      Parallel Programs to OpenCL for Heterogeneous Systems. In CGO. IEEE.
      https://doi.org/10.1109/CGO.2013.6494993

  Args:
    gpu: The name of the GPU platform to compute the features for: one of
      {amd_tahiti_7970,nvidia_gtx_960}.
    df: The DataFrame.

  Returns:
    A table with the feature values.
  """
  transfer = df[f'feature:{gpu}:transfer'].values
  comp = df['feature:comp'].values
  mem = df['feature:mem'].values
  localmem = df['feature:localmem'].values
  coalesced = df['feature:coalesced'].values
  wgsize = df[f'param:{gpu}:wgsize'].values

  df = pd.DataFrame({
    'feature:grewe1': transfer / (comp + mem),
    'feature:grewe2': coalesced / mem,
    'feature:grewe3': (localmem / mem) * wgsize,
    'feature:grewe4': comp / mem,
  })

  return df


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
    df = amd_df.join(nvidia_df, lsuffix='_amd', rsuffix='_nvidia')

    # Check that the join has joined the correct benchmarks.
    def AssertJoinedColumnsAreIdentical(column_name: str):
      np.testing.assert_array_equal(
          df[f'{column_name}_amd'].values,
          df[f'{column_name}_nvidia'].values)

    AssertJoinedColumnsAreIdentical('dataset')
    AssertJoinedColumnsAreIdentical('comp')
    AssertJoinedColumnsAreIdentical('rational')
    AssertJoinedColumnsAreIdentical('mem')
    AssertJoinedColumnsAreIdentical('localmem')
    AssertJoinedColumnsAreIdentical('coalesced')
    AssertJoinedColumnsAreIdentical('atomic')
    AssertJoinedColumnsAreIdentical('runtime_cpu')

    # Each row in the CSV file has a benchmark name format:
    # '<suite>-<benchmark>-<kernel>.cl'. Extract the
    # '<suite>-<benchmark>-<kernel>' component of each row and assign to new
    # columns.
    df['program:benchmark_suite_name'] = [
      '-'.join(b.split('-')[:-2]) for b in df['benchmark_amd']]
    df['program:benchmark_name'] = [
      b.split('-')[-2] for b in df['benchmark_amd']]
    df['program:opencl_kernel_name'] = [
      b.split('-')[-1] for b in df['benchmark_amd']]

    # Rename columns that we keep unmodified.
    df.rename({
      'benchmark_amd': 'benchmark',
      'dataset_amd': 'data:dataset_name',
      'wgsize_amd': 'wgsize',
      'rational_amd': 'feature:rational',
      'runtime_cpu_amd': 'runtime:intel_core_i7_3820',
      'runtime_gpu_amd': 'runtime:amd_tahiti_7970',
      'runtime_gpu_nvidia': 'runtime:nvidia_gtx_960',
      'src_amd': 'program:opencl_src',
      'wgsize_amd': 'param:amd_tahiti_7970:wgsize',
      'wgsize_nvidia': 'param:nvidia_gtx_960:wgsize',
      'transfer_amd': 'feature:amd_tahiti_7970:transfer',
      'transfer_nvidia': 'feature:nvidia_gtx_960:transfer',
      'comp_amd': 'feature:comp',
      'atomic_amd': 'feature:atomic',
      'mem_amd': 'feature:mem',
      'coalesced_amd': 'feature:coalesced',
      'localmem_amd': 'feature:localmem',
    }, axis='columns', inplace=True)

    # Sort the table values.
    df.sort_values(by=[
      'program:benchmark_suite_name',
      'program:benchmark_name',
      'program:opencl_kernel_name',
      'data:dataset_name',
    ], inplace=True)

    # Reset to default integer index.
    df.reset_index(inplace=True, drop=True)

    # Rearrange a subset of the columns to create the final table.
    self._df = df[[
      'program:benchmark_suite_name',
      'program:benchmark_name',
      'program:opencl_kernel_name',
      'program:opencl_src',
      'data:dataset_name',
      'param:amd_tahiti_7970:wgsize',
      'param:nvidia_gtx_960:wgsize',
      'feature:mem',
      'feature:comp',
      'feature:localmem',
      'feature:coalesced',
      'feature:atomic',
      'feature:rational',
      'feature:amd_tahiti_7970:transfer',
      'feature:nvidia_gtx_960:transfer',
      'runtime:intel_core_i7_3820',
      'runtime:amd_tahiti_7970',
      'runtime:nvidia_gtx_960',
    ]]

  @property
  def df(self) -> pd.DataFrame:
    return self._df

  @functools.lru_cache()
  def ComputeGreweFeaturesForGpu(
      self, gpu: str, df: pd.DataFrame = None) -> pd.DataFrame:
    """Return the Grewe et al. features as a table.

    These are the features used in the publication:

    ﻿    Grewe, D., Wang, Z., & O’Boyle, M. (2013). Portable Mapping of Data
        Parallel Programs to OpenCL for Heterogeneous Systems. In CGO. IEEE.
        https://doi.org/10.1109/CGO.2013.6494993

    Args:
      gpu: The name of the GPU platform to compute the features for: one of
        {amd_tahiti_7970,nvidia_gtx_960}.

    Returns:
      A table with the feature values.
    """
    return ComputeGreweFeaturesForGpu(gpu, self.df if df is None else df)

  @decorators.memoized_property
  def programs_df(self) -> pd.DataFrame:
    """Return a DataFrame containing the unique programs in the dataset.

    The returned DataFrame has the following schema:

      Index:
        program:benchmark_suite_name (str): The name of the benchmark suite.
        program:benchmark_name (str): The name of the benchmark program.
        program:opencl_kernel_name (str): The name of the OpenCL kernel.
      Columns:
        program:opencl_src (str): Entire source code of the preprocessed OpenCL
          kernel.
    """
    df = self._df.groupby([
      'program:benchmark_suite_name',
      'program:benchmark_name',
      'program:opencl_kernel_name',
    ]).min()
    df = df[['program:opencl_src', ]]
    df.sort_index(inplace=True)
    return df

  @functools.lru_cache()
  def AugmentWithDeadcodeMutations(
      self, rand: np.random.RandomState,
      num_permutations_of_kernel: int = 5,
      mutations_per_kernel_min_max: typing.Tuple[int, int] = (1, 5),
      df: pd.DataFrame = None):
    """Return a table with dead code mutations for each kernel.

    This adds num_permutations_of_kernel * len(df) new rows to the data frame,
    where each new row is a duplicate of an existing row, but with dead code
    mutations applied to the OpenCL kernel. An additional column
    `program:is_mutation` marks the new rows.

    Args:
      rand: A random number seed.
      num_permutations_of_kernel: The number of permutations of each kernel to
        produce. The number of rows in the output table is
        num_rows * num_permutations_of_kernel.
      mutations_per_kernel_min_max: The number of mutations to apply to each
        kernel, as a uniform distribution [low,high].
      df: The DataFrame to augment with deadcode mutations.

    Returns:
      The table with num_rows * num_permutations_of_kernel rows, and an
      additional `program:is_mutation` column.
    """
    df = self.df if df is None else df

    new_columns = list(df.columns.values) + ['program:is_mutation']

    # The mutation generator is initialized with the programs it will mutate.
    g = dci.GenerateDeadcodeMutations(
        df['program:opencl_src'].values, rand, num_permutations_of_kernel,
        mutations_per_kernel_min_max)

    # Generate the mutations and update the kernel column in the original table.
    new_rows = []
    for _, row in df.iterrows():
      # Make a copy to allow modifications.
      row = row.copy()
      row['program:is_mutation'] = False
      new_rows.append(row)
      for _ in range(num_permutations_of_kernel):
        row['program:is_mutation'] = True
        row['program:opencl_src'] = next(g)
        new_rows.append(row)

    # Sanity check that the generator has been exhausted.
    try:
      next(g)
      raise ValueError("Ney horsey!")
    except StopIteration:
      pass

    return pd.DataFrame(new_rows, columns=new_columns)
