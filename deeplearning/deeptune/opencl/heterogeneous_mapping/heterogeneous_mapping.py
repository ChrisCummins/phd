"""Run the heterogeneous mapping experiment.

This is a port of the Jupyter notebook
"//docs/2017_09_pact/code:Case Study A.ipynb".
"""
import pathlib
import typing

import pandas as pd
from absl import app
from absl import flags

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.deeptune.opencl.heterogeneous_mapping import utils
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import models
from labm8 import decorators


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'cache_directory',
    '/tmp/phd/deeplearning/deeptune/opencl/heterogeneous_mapping',
    'Path of directory to store cached models and predictions in.')


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

  def ResultsDataFrame(self, model_class: typing.Type,
                       df: typing.Optional[pd.DataFrame] = None):
    """Run experiment on model.

    Args:
      model_class: The model class to evaluate.
      df: The dataframe to use for evaluation.

    Returns:
      A DataFrame of evaluation results.
    """
    model = model_class()
    return utils.evaluate(
        model=model,
        df=self.dataset.df if df is None else df,
        atomizer=self.atomizer,
        workdir=self._cache_dir,
        seed=0x204)

  def PrintResultsSummary(self, model_class: typing.Type) -> None:
    """Evaluate and print summary of model results.

    Args:
      model_class: The model class to evaluate.
    """
    df = self.ResultsDataFrame(model_class)
    print(f'\n=== {model_class.__name__} =====================================')
    print("Results by benchmark suite ...")
    print(df.groupby(['Platform', 'Benchmark Suite'])[
            'Platform', 'Correct?', 'Speedup'].mean())
    print("Results by platform ...")
    print(df.groupby(['Platform'])[
            'Platform', 'Correct?', 'Speedup'].mean())
    print("Results ...")
    print(df['Platform', 'Correct?', 'Speedup'].mean())
    print(f'=== END {model_class.__name__} =================================\n')


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  experiment = HeterogeneousMappingExperiment(
      pathlib.Path(FLAGS.cache_directory))

  print(experiment.atomizer)

  for model in models.ALL_MODELS:
    experiment.PrintResultsSummary(model)


if __name__ == '__main__':
  app.run(main)
