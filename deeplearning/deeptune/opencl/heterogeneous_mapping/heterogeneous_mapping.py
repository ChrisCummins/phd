# Copyright (c) 2017, 2018, 2019 Chris Cummins.
#
# DeepTune is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepTune is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepTune.  If not, see <https://www.gnu.org/licenses/>.
"""Run the heterogeneous mapping experiment.

This is a port of the Jupyter notebook
"//docs/2017_09_pact/code:Case Study A.ipynb".

Example run:

    $ bazel run //deeplearning/deeptune/opencl/heterogeneous_mapping --
        --cache_directory=/var/phd/shared/deeplearning/deeptune/opencl/heterogeneous_mapping
        --summary_csv_path=$PHD/deeplearning/deeptune/opencl/heterogeneous_mapping/results.csv

"""
import pathlib
import typing

import pandas as pd

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.deeptune.opencl.heterogeneous_mapping import utils
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import models
from labm8.py import app
from labm8.py import decorators

FLAGS = app.FLAGS

app.DEFINE_string(
    'cache_directory',
    '/tmp/phd/deeplearning/deeptune/opencl/heterogeneous_mapping',
    'Path of directory to store cached models and predictions in.')
app.DEFINE_string(
    'summary_csv_path',
    '/tmp/phd/deeplearning/deeptune/opencl/heterogeneous_mapping/results.csv',
    'A path which is used to store a CSV summary of results.')


class HeterogeneousMappingExperiment(object):

  def __init__(self, cache_dir: pathlib.Path):
    self._cache_dir = cache_dir

  @decorators.memoized_property
  def dataset(self):
    return opencl_device_mapping_dataset.OpenClDeviceMappingsDataset()

  @decorators.memoized_property
  def atomizer(self):
    atomizer = utils.GetAtomizerFromOpenClSources(
        self.dataset.programs_df['program:opencl_src'].values)
    return atomizer

  def ResultsDataFrame(self,
                       model_class: typing.Type,
                       df: typing.Optional[pd.DataFrame] = None):
    """Run experiment on model.

    Args:
      model_class: The model class to evaluate.
      df: The dataframe to use for evaluation.

    Returns:
      A DataFrame of evaluation results.
    """
    model = model_class()
    return utils.evaluate(model=model,
                          df=self.dataset.df if df is None else df,
                          atomizer=self.atomizer,
                          workdir=self._cache_dir,
                          seed=0x204)

  @property
  def cache_dir(self) -> pathlib.Path:
    """Return the cache directory."""
    return self._cache_dir

  @staticmethod
  def PrintResultsSummary(df: pd.DataFrame) -> None:
    """Evaluate and print summary of model results.

    Args:
      df: A results table.
    """
    # Get the model name from the "Model" column of the table.
    model_names = set(df['Model'].values)
    if len(model_names) != 1:
      raise ValueError("Results table should contain a single model name. "
                       f"Found: {model_names}")
    model_name = list(model_names)[0]

    print(f'\n=== {model_name} ==========================================')
    print("Results by benchmark suite ...")
    print(
        df.groupby(
            ['Platform',
             'Benchmark Suite'])['Platform', 'Correct?', 'Speedup'].mean())
    print("Results by platform ...")
    print(df.groupby(['Platform'])['Platform', 'Correct?', 'Speedup'].mean())
    print("Results ...")
    print(df[['Platform', 'Correct?', 'Speedup']].mean())
    print(f'=== END {model_name} ======================================\n')


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  # Create the directory to write results summary to.
  summary_csv_path = pathlib.Path(FLAGS.summary_csv_path)
  summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

  experiment = HeterogeneousMappingExperiment(
      pathlib.Path(FLAGS.cache_directory))

  print(experiment.atomizer)

  df = None
  for model in models.ALL_MODELS:
    # TODO(cec): Re-enable LDA once it's implemented.
    if model == models.Lda:
      continue

    # Compute the table of results.
    model_df = experiment.ResultsDataFrame(model)

    # Print a summary of the table of results.
    experiment.PrintResultsSummary(model_df)

    # Concatenate the model's results to the full results table.
    if df is None:
      df = model_df
    else:
      df = pd.concat((df, model_df))

  assert df is not None

  # Write results to file.
  app.Log(1, 'Writing results to %s', summary_csv_path)
  df.sort_values(
      by=['Benchmark Suite', 'Benchmark', 'Dataset', 'Platform', 'Model'],
      inplace=True)
  df['Correct?'] = df['Correct?'].astype(int)
  df.to_csv(str(summary_csv_path), index=False)

  # Print results summary.
  print("\nMODEL                 ACC        SPEEDUP")
  for model in sorted(set(df['Model'])):
    accuracy = df[df['Model'] == model]['Correct?'].mean()
    speedup = df[df['Model'] == model]['Speedup'].mean()
    print(f"{model:20s} {accuracy:8.3%} {speedup:8.3f}x")


if __name__ == '__main__':
  app.RunWithArgs(main)
