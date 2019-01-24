"""Run heterogeneous mapping experiment with adversarial inputs."""
import pathlib
import typing

import numpy as np
import pandas as pd
from absl import app
from absl import flags
from absl import logging

from deeplearning.deeptune.opencl.adversary import opencl_deadcode_inserter
from deeplearning.deeptune.opencl.heterogeneous_mapping import \
  heterogeneous_mapping
from deeplearning.deeptune.opencl.heterogeneous_mapping import utils
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import models
from labm8 import prof


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'adversary_cache_directory',
    '/tmp/phd/deeplearning/deeptune/opencl/heterogeneous_mapping',
    'Path of directory to store cached models and predictions in.')


class AdversarialDeeptune(models.DeepTune):
  """The original deeptune model, but with """
  __basename__ = 'adversarial_deeptune'
  __name__ = 'Adversarial DeepTune'


def CreateAugmentedDataset(df: pd.DataFrame) -> pd.DataFrame:
  """Augment the dataset with dead code mutations."""
  with prof.Profile('dead code mutations'):
    seed = np.random.RandomState(0xCEC)

    candidate_kernels = df['program:opencl_src'].values.copy()

    new_columns = list(df.columns.values) + ['program:is_mutation']
    new_rows = []

    for _, row in df.iterrows():
      kernel = row['program:opencl_src']

      # Insert the original row.
      row['program:is_mutation'] = False
      new_rows.append(row)

      # Create and insert mutation rows.
      for _ in range(3):
        row = row.copy()
        row['program:is_mutation'] = True
        # Insert a single dead kernel into each original kernel.
        dci = opencl_deadcode_inserter.OpenClDeadcodeInserter(
            seed, kernel, candidate_kernels=candidate_kernels)
        dci.InsertBlockIntoKernel()
        row['program:opencl_src'] = dci.opencl_source
        new_rows.append(row)

  return pd.DataFrame(new_rows, columns=new_columns)


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  assert FLAGS.adversary_cache_directory
  cache_directory = pathlib.Path(FLAGS.adversary_cache_directory)
  cache_directory.mkdir(parents=True, exist_ok=True)

  experiment = heterogeneous_mapping.HeterogeneousMappingExperiment(
      cache_directory)

  augmented_df_path = cache_directory / 'augmented_df.pkl'
  # Create the augmented dataset if required.
  if not augmented_df_path.is_file():
    adversarial_df = CreateAugmentedDataset(experiment.dataset.df)
    adversarial_df.to_pickle(str(augmented_df_path))

  logging.info('Reading %s', augmented_df_path)
  augmented_df = pd.read_pickle(str(augmented_df_path))

  logging.info('Augmented dataframe: %s', augmented_df.shape)
  logging.info('Atomizer: %s', experiment.atomizer)
  longest_seq = max(len(experiment.atomizer.AtomizeString(src))
                    for src in augmented_df['program:opencl_src'])
  logging.info('Longest sequence: %d', longest_seq)

  results_path = cache_directory / 'adversarial_results.pkl'
  if not results_path.is_file():
    model = AdversarialDeeptune(input_shape=(longest_seq,))
    logging.info('Model: %s', model)

    logging.info('Evaluating model ...')
    results = utils.evaluate(model, df=augmented_df,
                             atomizer=experiment.atomizer,
                             workdir=experiment.cache_dir, seed=0x204)

    logging.info('Writing %s', cache_directory / 'adversarial_results.pkl')
    results.to_pickle(str(cache_directory / 'adversarial_results.pkl'))
  else:
    results = pd.read_pickle(str(results_path))

  logging.info('Results: %s', results.shape)

  logging.info('done')


if __name__ == '__main__':
  app.run(main)
