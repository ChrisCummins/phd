"""Utility code for heterogeneous mapping experiment."""
import collections
import pathlib
import pickle
import typing

import numpy as np
import pandas as pd
from absl import flags
from absl import logging
from sklearn import model_selection

from deeplearning.clgen.corpuses import atomizers


FLAGS = flags.FLAGS

# Taken from the C99 spec, OpenCL spec 1.2, and bag-of-words analysis of
# GitHub corpus:
OPENCL_ATOMS = set(
    ['  ', '__assert', '__attribute', '__builtin_astype', '__clc_fabs',
     '__clc_fma', '__constant', '__global', '__inline', '__kernel', '__local',
     '__private', '__read_only', '__read_write', '__write_only', '*/', '/*',
     '//', 'abs', 'alignas', 'alignof', 'atomic_add', 'auto', 'barrier', 'bool',
     'break', 'case', 'char', 'clamp', 'complex', 'const', 'constant',
     'continue', 'default', 'define', 'defined', 'do', 'double', 'elif', 'else',
     'endif', 'enum', 'error', 'event_t', 'extern', 'fabs', 'false', 'float',
     'for', 'get_global_id', 'get_global_size', 'get_local_id',
     'get_local_size', 'get_num_groups', 'global', 'goto', 'half', 'if',
     'ifdef', 'ifndef', 'image1d_array_t', 'image1d_buffer_t', 'image1d_t',
     'image2d_array_t', 'image2d_t', 'image3d_t', 'imaginary', 'include',
     'inline', 'int', 'into', 'kernel', 'line', 'local', 'long', 'noreturn',
     'pragma', 'private', 'quad', 'read_only', 'read_write', 'register',
     'restrict', 'return', 'sampler_t', 'short', 'shuffle', 'signed', 'size_t',
     'sizeof', 'sqrt', 'static', 'struct', 'switch', 'true', 'typedef', 'u32',
     'uchar', 'uint', 'ulong', 'undef', 'union', 'unsigned', 'void', 'volatile',
     'while', 'wide', 'write_only', ])


def GetAtomizerFromOpenClSources(
    opencl_srcs: typing.Iterator[str]) -> atomizers.AtomizerBase:
  """Derive a greedy atomizer from a concatenation of OpenCL sources."""
  srcs = '\n'.join(opencl_srcs)
  return atomizers.GreedyAtomizer.FromText(srcs, OPENCL_ATOMS)


def AddClassificationTargetToDataFrame(
    df: pd.DataFrame, gpu_name: str) -> pd.DataFrame:
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
    'TrainTestSplit', ['i', 'train_df', 'test_df', 'gpu_name'])


def TrainTestSplitGenerator(df: pd.DataFrame, seed: int, split_count: int = 10):
  for gpu_name in ["amd_tahiti_7970", "nvidia_gtx_960"]:
    # Add the classification target columns `y` and `y_1hot`.
    df = AddClassificationTargetToDataFrame(df, gpu_name).reset_index()

    # Split into train/test indices for stratified 10-fold cross-validation.
    dataset_splitter = model_selection.StratifiedKFold(
        n_splits=split_count, shuffle=True, random_state=seed)
    dataset_splits = dataset_splitter.split(np.zeros(len(df)), df['y'].values)

    for i, (train_index, test_index) in enumerate(dataset_splits):
      yield TrainTestSplit(i=i + 1, train_df=df.iloc[train_index, :].copy(),
                           test_df=df.iloc[test_index, :].copy(),
                           gpu_name=gpu_name)


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
    logging.info(
        'Evaluating %s on %s, split %d with train=%d/test=%d programs',
        model.__name__, split.gpu_name, split.i + 1, len(split.train_df),
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
      # Load predctions from cache, which means we don't need to train a model.
      logging.info('Loading %s', predictions_path)
      predictions = LoadPredictionsFromFile(predictions_path)
    else:
      if model_path.is_file():
        logging.info('Loading %s', model_path)
        # Restore trained model from cache.
        model.restore(model_path)
      else:
        # Train and cache a model.
        model.init(seed=seed, atomizer=atomizer)
        model.train(
            df=split.train_df, platform_name=split.gpu_name, verbose=False)
        model.save(model_path)

      # Test the model.
      logging.info("Predicting %d %s mappings for device %s",
                   len(split.test_df), model.__name__, split.gpu_name)
      predictions = model.predict(
          df=split.test_df, platform_name=split.gpu_name, verbose=False)
      logging.info('Writing %s', predictions_path)
      SavePredictionsToFile(predictions, predictions_path)

    data += EvaluatePredictions(model, split, predictions)

  return PredictionEvaluationsToTable(data)


def PredictionEvaluationsToTable(
    data: typing.Iterable[typing.Dict[str, typing.Union[str, float, int]]]
) -> pd.DataFrame:
  """Create a table from the results of EvaluatePredictions()."""
  return pd.DataFrame(
      data, index=range(1, len(data) + 1), columns=[
        "Model",
        "Platform",
        "Benchmark",
        "Benchmark Suite",
        "Oracle Mapping",
        "Predicted Mapping",
        "Correct?",
        "Speedup"
      ])


def EvaluatePredictions(
    model: 'HeterogeneousMappingModel',
    split: TrainTestSplit,
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

  # benchmarks
  benchmarks = split.test_df['program:opencl_kernel_name'].values
  benchmark_suites = (
    split.test_df['program:benchmark_suite_name'].values)

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
    'runtime:intel_core_i7_3820',
    f'runtime:{split.gpu_name}'
  ]].values
  p_runtimes = [
    cpu_gpu_runtime_row[prections_row]
    for prections_row, cpu_gpu_runtime_row in
    zip(predictions, cpu_gpu_runtimes)]
  p_speedup = zero_r_runtimes / p_runtimes

  # sanity check
  assert (
      len(benchmarks) == len(oracle_device_mappings) == len(p_speedup) ==
      len(predicted_is_correct) == len(predictions))

  # record results
  split_data = []
  for (benchmark, benchmark_suite, oracle_device_mapping, predicted,
       is_correct, predicted_speedup) in zip(
      benchmarks, benchmark_suites, oracle_device_mappings, predictions,
      predicted_is_correct,
      p_speedup):
    split_data.append({
      "Model": model.__name__,
      "Platform": split.gpu_name,
      'Benchmark': benchmark,
      'Benchmark Suite': benchmark_suite,
      "Oracle Mapping": oracle_device_mapping,
      "Predicted Mapping": predicted,
      "Correct?": is_correct,
      "Speedup": predicted_speedup,
    })

  gpu_predicted_count = sum(d['Predicted Mapping'] for d in split_data)
  cpu_predicted_count = len(split_data) - gpu_predicted_count

  logging.info('Results: model=%s, platform=%s, split=%s, n=%d, '
               'predictions=(cpu=%d,gpu=%d) accuracy=%.2f%%, speedup=%.2fx',
               model.__basename__, split.gpu_name, split.i,
               len(split.test_df), cpu_predicted_count, gpu_predicted_count,
               np.mean([r['Correct?'] for r in split_data]) * 100,
               np.mean([r['Speedup'] for r in split_data]))
  return split_data
