"""Run the heterogeneous mapping experiment.

This is a port of the Jupyter notebook
"//docs/2017_09_pact/code:Case Study A.ipynb".
"""
import pathlib
import sys
import typing

import pandas as pd
from absl import app
from absl import flags

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.deeptune.opencl.heterogeneous_mapping import models
from deeplearning.deeptune.opencl.heterogeneous_mapping import utils
from labm8 import decorators


FLAGS = flags.FLAGS

flags.DEFINE_string('cache_directory',
                    '/tmp/phd/deeplearning/deeptune/opencl/heterogeneous_mapping',
                    'Path to working dir.')


class ExperimentalResults(object):

  def __init__(self, cache_dir: pathlib.Path):
    self._cache_dir = cache_dir

  def EvaluateModel(self, model: models.HeterogemeousMappingModel,
                    df: typing.Optional[pd.DataFrame] = None):
    """Evaluate a model.

    Args:
      model: The model to evaluate.
      df: The dataframe to use for evaluation.

    Returns:
      A DataFrame of evaluation results.
    """
    return utils.evaluate(
        model=model,
        df=self.dataset.df if df is None else df,
        atomizer=self.atomizer,
        workdir=self._cache_dir,
        seed=0x204)

  @decorators.memoized_property
  def dataset(self):
    return opencl_device_mapping_dataset.OpenClDeviceMappingsDataset()

  @decorators.memoized_property
  def atomizer(self):
    atomizer = utils.GetAtomizerFromOpenClSources(
        self.dataset.programs_df['program:opencl_src'].values)
    return atomizer

  @decorators.memoized_property
  def baseline_model(self):
    return models.StaticMapping()

  @decorators.memoized_property
  def baseline_df(self):
    return self.EvaluateModel(self.baseline_model)

  @decorators.memoized_property
  def grewe_model(self):
    return models.Grewe()

  @decorators.memoized_property
  def grewe_df(self):
    return self.EvaluateModel(self.grewe_model)

  @decorators.memoized_property
  def deeptune_model(self):
    return models.DeepTune()

  @decorators.memoized_property
  def deeptune_df(self):
    return self.EvaluateModel(self.deeptune_model)

  @decorators.memoized_property
  def adversarial_deeptune_df(self):
    adversarial_df = self.dataset.AugmentWithDeadcodeMutations(
        rand=np.random.RandomState(0xCEC),
        num_permutations_of_kernel=5,
        num_mutations_per_kernel=(1, 5))
    return self.EvaluateModel(
        models.DeepTune(),
        df=adversarial_df)


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  results = ExperimentalResults(pathlib.Path(FLAGS.cache_directory))

  print(results.atomizer)

  print("Evaluating static mapping ...", file=sys.stderr)
  print(results.baseline_df.groupby(['Platform', 'Benchmark Suite'])[
          'Platform', 'Correct?', 'Speedup'].mean())
  print(results.baseline_df.groupby(['Platform'])[
          'Platform', 'Correct?', 'Speedup'].mean())

  print("Evaluating Grewe et. al model ...", file=sys.stderr)
  print(results.grewe_df.groupby(['Platform', 'Benchmark Suite'])[
          'Platform', 'Correct?', 'Speedup'].mean())
  print(results.grewe_df.groupby(['Platform'])[
          'Platform', 'Correct?', 'Speedup'].mean())

  print("Evaluating DeepTune model ...", file=sys.stderr)
  print(results.deeptune_df.groupby(['Platform', 'Benchmark Suite'])[
          'Platform', 'Correct?', 'Speedup'].mean())
  print(results.deeptune_df.groupby(['Platform'])[
          'Platform', 'Correct?', 'Speedup'].mean())

  print("Overview of DeepTune model", file=sys.stderr)
  results.deeptune_model.model.summary()


if __name__ == '__main__':
  app.run(main)
