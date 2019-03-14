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
"""Utility code for heterogeneous mapping experiment."""
import collections
import pathlib
import pickle
import typing

import numpy as np
import pandas as pd
from sklearn import model_selection

from deeplearning.clgen.corpuses import atomizers
from labm8 import app

FLAGS = app.FLAGS

# Taken from the C99 spec, OpenCL spec 1.2, and bag-of-words analysis of
# GitHub corpus:
OPENCL_ATOMS = set([
    '  ',
    '__assert',
    '__attribute',
    '__builtin_astype',
    '__clc_fabs',
    '__clc_fma',
    '__constant',
    '__global',
    '__inline',
    '__kernel',
    '__local',
    '__private',
    '__read_only',
    '__read_write',
    '__write_only',
    '*/',
    '/*',
    '//',
    'abs',
    'alignas',
    'alignof',
    'atomic_add',
    'auto',
    'barrier',
    'bool',
    'break',
    'case',
    'char',
    'clamp',
    'complex',
    'const',
    'constant',
    'continue',
    'default',
    'define',
    'defined',
    'do',
    'double',
    'elif',
    'else',
    'endif',
    'enum',
    'error',
    'event_t',
    'extern',
    'fabs',
    'false',
    'float',
    'for',
    'get_global_id',
    'get_global_size',
    'get_local_id',
    'get_local_size',
    'get_num_groups',
    'global',
    'goto',
    'half',
    'if',
    'ifdef',
    'ifndef',
    'image1d_array_t',
    'image1d_buffer_t',
    'image1d_t',
    'image2d_array_t',
    'image2d_t',
    'image3d_t',
    'imaginary',
    'include',
    'inline',
    'int',
    'into',
    'kernel',
    'line',
    'local',
    'long',
    'noreturn',
    'pragma',
    'private',
    'quad',
    'read_only',
    'read_write',
    'register',
    'restrict',
    'return',
    'sampler_t',
    'short',
    'shuffle',
    'signed',
    'size_t',
    'sizeof',
    'sqrt',
    'static',
    'struct',
    'switch',
    'true',
    'typedef',
    'u32',
    'uchar',
    'uint',
    'ulong',
    'undef',
    'union',
    'unsigned',
    'void',
    'volatile',
    'while',
    'wide',
    'write_only',
])


def GetAtomizerFromOpenClSources(
    opencl_srcs: typing.Iterator[str]) -> atomizers.AtomizerBase:
  """Derive a greedy atomizer from a concatenation of OpenCL sources."""
  srcs = '\n'.join(opencl_srcs)
  return atomizers.GreedyAtomizer.FromText(srcs, OPENCL_ATOMS)


def AddClassificationTargetToDataFrame(df: pd.DataFrame,
                                       gpu_name: str) -> pd.DataFrame:
  # Determine the array of optimal mappings 'y'. If y_i is 1, that means that
  # the GPU was faster than the CPU for result i.
  cpu_gpu_runtimes = df[[
      'runtime:intel_core_i7_3820',
      f'runtime:{gpu_name}',
  ]].values
  y = [1 if gpu < cpu else 0 for cpu, gpu in cpu_gpu_runtimes]
  df['target_gpu_name'] = [gpu_name] * len(df)
  df['y'] = y
  # Add a column which contains a [bool,bool] array with a 1-hot encoded
  # optimal value.
  df['y_1hot'] = [np.array([not y_, y_], dtype=np.int32) for y_ in y]
  return df


# A train+test data batch for evaluation.
TrainTestSplit = collections.namedtuple(
    'TrainTestSplit', ['i', 'train_df', 'test_df', 'gpu_name', 'global_step'])

# A train+val+test data batch for evaluation.
TrainValTestSplit = collections.namedtuple('TrainValTestSplit', [
    'gpu_name',
    'train_df',
    'valid_df',
    'test_df',
    'global_step',
])


def TrainTestSplitGenerator(df: pd.DataFrame, seed: int, split_count: int = 10):
  for gpu_name in ["amd_tahiti_7970", "nvidia_gtx_960"]:
    # Add the classification target columns `y` and `y_1hot`.
    df = AddClassificationTargetToDataFrame(df, gpu_name).reset_index()

    # Split into train/test indices for stratified 10-fold cross-validation.
    dataset_splitter = model_selection.StratifiedKFold(
        n_splits=split_count, shuffle=True, random_state=seed)
    dataset_splits = dataset_splitter.split(np.zeros(len(df)), df['y'].values)

    global_step = 0
    for i, (train_index, test_index) in enumerate(dataset_splits):
      global_step += 1
      yield TrainTestSplit(
          i=i + 1,
          train_df=df.iloc[train_index, :].copy(),
          test_df=df.iloc[test_index, :].copy(),
          gpu_name=gpu_name,
          global_step=global_step)


def TrainValidationTestSplits(df: pd.DataFrame,
                              rand: np.random.RandomState,
                              train_val_test_ratios=(0.6, 0.3, 0.1)):
  """Split a dataframe into train/validation/test splits.

  This is stratified across class counts.

  Args:
    df: The dataframe to split.
    rand: A random state.
    train_val_test_ratios: The ratio of train/validation/test data. Must add to
      1.

  Returns:
    An iterator of TrainValTestSplit splits.

  Raises:
    ValueError: If train_val_test_ratios don't sum to 1, or if any of the splits
      contain no elements.
  """
  if abs(sum(train_val_test_ratios) - 1) > 1e-6:
    raise ValueError("Train/validation/test ratios must sum to one")

  for i, gpu_name in enumerate(["amd_tahiti_7970", "nvidia_gtx_960"]):
    # Add the classification target columns `y` and `y_1hot`.
    df = AddClassificationTargetToDataFrame(df, gpu_name).reset_index()

    # Split dataset into two classes.
    gpu_df = df[df['y'] == 1]
    cpu_df = df[df['y'] == 0]

    # Shuffle the separated CPU and GPU data.
    gpu_df = gpu_df.sample(frac=1, random_state=rand)
    cpu_df = cpu_df.sample(frac=1, random_state=rand)

    # Split the two datasets into train / val / test splits.
    gpu_train, gpu_val, gpu_test = np.split(gpu_df, [
        int(train_val_test_ratios[0] * len(gpu_df)),
        int((train_val_test_ratios[0] + train_val_test_ratios[1]) * len(gpu_df))
    ])

    cpu_train, cpu_val, cpu_test = np.split(cpu_df, [
        int(train_val_test_ratios[0] * len(cpu_df)),
        int((train_val_test_ratios[0] + train_val_test_ratios[1]) * len(cpu_df))
    ])

    # Concatenate the CPU and GPU data splits.
    train = pd.concat((gpu_train, cpu_train))
    val = pd.concat((gpu_val, cpu_val))
    test = pd.concat((gpu_test, cpu_test))

    if not len(train) or not len(val) or not len(test):
      # Sanity check that each split contains elements.
      raise ValueError(
          "Datasets splits must each contain one or more elements. Actual "
          f"element: counts train={len(train)}, val={len(validation)}, "
          f"test={len(test)}")

    # Shuffle the concatenated CPU and GPU data.
    train = train.sample(frac=1, random_state=rand)
    val = val.sample(frac=1, random_state=rand)
    test = test.sample(frac=1, random_state=rand)

    yield TrainValTestSplit(
        gpu_name=gpu_name,
        train_df=train,
        valid_df=val,
        test_df=test,
        global_step=i)


def LoadPredictionsFromFile(predictions_path: pathlib.Path):
  with open(predictions_path, 'rb') as f:
    return pickle.load(f)


def SavePredictionsToFile(predictions, predictions_path: pathlib.Path):
  with open(predictions_path, 'wb') as outfile:
    pickle.dump(predictions, outfile)


def evaluate(model: 'HeterogemeousMappingModel', df: pd.DataFrame, atomizer,
             workdir: pathlib.Path, seed: int) -> pd.DataFrame:
  """Evaluate a model.

  Performs 10-fold cross-validation of the model's effectiveness at predicting
  OpenCL device mappings. Results are cached.

  Args:
    model: The predictive model to evaluate.
    df: The dataset to use.
    atomizer: The atomizer to encode source sequences.
    workdir: The path to working directory.
    seed: A random seed value.

  Returns:
    Evaluation results.
  """
  data = []

  for split in TrainTestSplitGenerator(df, seed):
    app.Log(1, 'Evaluating %s on %s, split %d with train=%d/test=%d programs',
            model.__name__, split.gpu_name, split.i, len(split.train_df),
            len(split.test_df))

    # Path of cached model and predictions.
    model_path = (
        workdir /
        f"{model.__basename__}-{split.gpu_name}-{split.i:02d}.trained_model")
    predictions_path = (
        workdir /
        f"{model.__basename__}-{split.gpu_name}-{split.i:02d}.predictions")

    # Create cache directories.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    if predictions_path.is_file():
      # Load predictions from cache, which means we don't need to train a model.
      app.Log(1, 'Loading %s', predictions_path)
      predictions = LoadPredictionsFromFile(predictions_path)
    else:
      if model_path.is_file():
        app.Log(1, 'Loading %s', model_path)
        # Restore trained model from cache.
        model.restore(model_path)
      else:
        # Train and cache a model.
        model.init(seed=seed, atomizer=atomizer)
        model.train(
            df=split.train_df,
            platform_name=split.gpu_name,
            verbose=FLAGS.verbosity)
        model.save(model_path)

      # Test the model.
      app.Log(1, "Predicting %d %s mappings for device %s", len(split.test_df),
              model.__name__, split.gpu_name)
      predictions = model.predict(
          df=split.test_df,
          platform_name=split.gpu_name,
          verbose=FLAGS.verbosity)
      app.Log(1, 'Writing %s', predictions_path)
      SavePredictionsToFile(predictions, predictions_path)

    data += EvaluatePredictions(model, split, predictions)

  return PredictionEvaluationsToTable(data)


def PredictionEvaluationsToTable(
    data: typing.Iterable[typing.Dict[str, typing.Union[str, float, int]]]
) -> pd.DataFrame:
  """Create a table from the results of EvaluatePredictions()."""
  return pd.DataFrame(
      data,
      index=range(1,
                  len(data) + 1),
      columns=[
          "Model", "Platform", "Benchmark Suite", "Benchmark", "Dataset",
          "Oracle Mapping", "Predicted Mapping", "Correct?", "Speedup"
      ])


def EvaluatePredictions(
    model: 'HeterogeneousMappingModel', split: TrainTestSplit,
    predictions: typing.Iterable[int]
) -> typing.List[typing.Dict[str, typing.Union[str, float, int]]]:
  """Get dictionaries of prediction results.

  Args:
    model: The model instance.
    split: The split being evaluated.
    predictions: The predictions that the model produced for the split.test_df.

  Returns:
    A list of dicts of len(predictions), where each element is a dict of stats.
  """
  predictions = list(predictions)

  # oracle device mappings
  oracle_device_mappings = split.test_df['y'].values
  # whether predictions were correct or not
  predicted_is_correct = (predictions == oracle_device_mappings)

  # Runtimes of baseline mapping (CPU on AMD, GPU on NVIDIA).
  if split.gpu_name == "amd_tahiti_7970":
    zero_r_runtime_column = "runtime:intel_core_i7_3820"
  elif split.gpu_name == "nvidia_gtx_960":
    zero_r_runtime_column = "runtime:nvidia_gtx_960"
  else:
    raise ValueError(split.gpu_name)

  zero_r_runtimes = split.test_df[zero_r_runtime_column].values

  # speedups of predictions
  cpu_gpu_runtimes = split.test_df[[
      'runtime:intel_core_i7_3820', f'runtime:{split.gpu_name}'
  ]].values
  p_runtimes = [
      cpu_gpu_runtime_row[prections_row] for prections_row, cpu_gpu_runtime_row
      in zip(predictions, cpu_gpu_runtimes)
  ]
  p_speedup = zero_r_runtimes / p_runtimes

  # sanity check
  assert (len(oracle_device_mappings) == len(p_speedup) ==
          len(predicted_is_correct) == len(predictions))

  # record results
  split_data = []
  for (benchmark, benchmark_suite, dataset, oracle_device_mapping, predicted,
       is_correct, predicted_speedup) in zip(
           split.test_df['program:opencl_kernel_name'].values,
           split.test_df['program:benchmark_suite_name'].values,
           split.test_df['data:dataset_name'].values, oracle_device_mappings,
           predictions, predicted_is_correct, p_speedup):
    split_data.append({
        "Model": model.__name__,
        "Platform": split.gpu_name,
        'Benchmark Suite': benchmark_suite,
        'Benchmark': benchmark,
        'Dataset': dataset,
        "Oracle Mapping": oracle_device_mapping,
        "Predicted Mapping": predicted,
        "Correct?": is_correct,
        "Speedup": predicted_speedup,
    })

  gpu_predicted_count = sum(d['Predicted Mapping'] for d in split_data)
  cpu_predicted_count = len(split_data) - gpu_predicted_count

  app.Log(
      1, 'Results: model=%s, platform=%s, split=%s, n=%d, '
      'predictions=(cpu=%d,gpu=%d) accuracy=%.2f%%, speedup=%.2fx',
      model.__basename__, split.gpu_name, split.global_step, len(split.test_df),
      cpu_predicted_count, gpu_predicted_count,
      np.mean([r['Correct?'] for r in split_data]) * 100,
      np.mean([r['Speedup'] for r in split_data]))
  return split_data
