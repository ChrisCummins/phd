"""Utility code for heterogeneous mapping experiment."""
import pathlib
import pickle
import typing

import numpy as np
import pandas as pd
import progressbar
from absl import flags
from absl import logging
from sklearn import model_selection

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.clgen.corpuses import atomizers


FLAGS = flags.FLAGS


def platform2str(platform: str) -> str:
  """Get full platform name."""
  if platform == "amd":
    return "AMD Tahiti 7970"
  elif platform == "nvidia":
    return "NVIDIA GTX 970"
  else:
    raise LookupError


def escape_suite_name(g: str) -> str:
  """Format benchmark suite name for display."""
  c = g.split('-')
  if c[0] == "amd" or c[0] == "nvidia":
    return c[0].upper() + " SDK"
  if c[0] == "npb" or c[0] == "shoc":
    return c[0].upper()
  elif c[0] == "parboil" or c[0] == "polybench" or c[0] == "rodinia":
    return c[0].capitalize()
  else:
    raise LookupError


def escape_benchmark_name(g: str) -> str:
  """Escape benchmark name for display."""
  c = g.split('-')
  return escape_suite_name(c[0]).split()[0] + "." + c[-2]


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
  df['y'] = y
  # Add a column which contains a [bool,bool] array with a 1-hot encoded optimal
  # value.
  df['y_1hot'] = [np.array([not y_, y_], dtype=np.int32) for y_ in y]
  return df


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
  bar_tuple = [0, progressbar.ProgressBar(max_value=10 * 2)]

  data = []

  for gpu_name in ["amd_tahiti_7970", "nvidia_gtx_960"]:
    # Add the classification target columns `y` and `y_1hot`.
    df = AddClassificationTargetToDataFrame(df, gpu_name)

    # Values used for training & predictions.
    features = opencl_device_mapping_dataset.ComputeGreweFeaturesForGpu(
        gpu_name, df).values
    aux_in = np.array([
      df[f"feature:{gpu_name}:transfer"].values,
      df[f"param:{gpu_name}:wgsize"].values,
    ]).T

    # Sanity check.
    assert len(features) == len(df)
    assert len(aux_in) == len(df)

    # Determine the array of optimal mappings 'y'. If y_i is 1, that means that
    # the GPU was faster than the CPU for result i.
    cpu_gpu_runtimes = df[[
      'runtime:intel_core_i7_3820',
      f'runtime:{gpu_name}',
    ]].values
    y = np.array([1 if gpu < cpu else 0 for cpu, gpu in cpu_gpu_runtimes])
    y_1hot = encode_1hot(y)

    # 10-fold cross-validation
    kf = model_selection.StratifiedKFold(
        n_splits=10, shuffle=True, random_state=seed)
    for j, (train_index, test_index) in enumerate(kf.split(features, y)):
      # Path of cached model and predictions.
      model_path = workdir / f"{model.__basename__}-{gpu_name}-{j:02d}.model"
      predictions_path = (
          workdir / f"{model.__basename__}-{gpu_name}-{j:02d}.pred")

      # Create cache directories.
      model_path.parent.mkdir(parents=True, exist_ok=True)
      predictions_path.parent.mkdir(parents=True, exist_ok=True)

      if predictions_path.is_file():
        # Load predctions from cache, which means we don't need to train a
        # model.
        logging.info('Loading predictions from cache ...')
        with open(predictions_path, 'rb') as f:
          predictions = pickle.load(f)
      else:
        if model_path.is_file():
          logging.info('Loading trained model ...')
          # Restore trained model from cache.
          model.restore(model_path)
        else:
          # Train and cache a model.

          logging.info('Training new model ...')
          model.init(seed=seed, atomizer=atomizer)
          train_features = features[train_index]
          train_aux_in = aux_in[train_index]
          train_y = y[train_index]
          train_y_1hot = y_1hot[train_index]
          model.train(df=df,
                      features=train_features,
                      aux_in=train_aux_in,
                      srcs=df["program:opencl_src"].values[train_index],
                      y=train_y,
                      y_1hot=train_y_1hot,
                      platform_name=gpu_name,
                      verbose=False)
          model.save(model_path)

        # Test the model.
        predictions = model.predict(
            features=features[test_index],
            aux_in=aux_in[test_index],
            srcs=df["program:opencl_src"].values[test_index],
            y=y[test_index],
            y_1hot=y_1hot[test_index],
            platform_name=gpu_name,
            verbose=False)

        # cache results
        with open(predictions_path, 'wb') as outfile:
          pickle.dump(predictions, outfile)

      # benchmarks
      benchmarks = df['program:opencl_kernel_name'].values[test_index]
      benchmark_suites = (
        df['program:benchmark_suite_name'].values[test_index])

      # oracle device mappings
      oracle_device_mappings = y[test_index]
      # whether predictions were correct or not
      predicted_is_correct = predictions == oracle_device_mappings

      # Runtimes of baseline mapping (CPU on AMD, GPU on NVIDIA).
      if gpu_name == "amd_tahiti_7970":
        zero_r_runtime_column = "runtime:intel_core_i7_3820"
      elif gpu_name == "nvidia_gtx_960":
        zero_r_runtime_column = "runtime:nvidia_gtx_960"
      zero_r_runtimes = df[zero_r_runtime_column][test_index]

      # speedups of predictions
      cpu_gpu_runtimes = df[[
        'runtime:intel_core_i7_3820',
        f'runtime:{gpu_name}'
      ]].values[test_index]
      p_runtimes = [
        cpu_gpu_runtime[predicted_device]
        for predicted_device, cpu_gpu_runtime in
        zip(predictions, cpu_gpu_runtimes)]
      p_speedup = zero_r_runtimes / p_runtimes

      # sanity check
      assert (len(benchmarks) == len(oracle_device_mappings) == len(
          predicted_is_correct) == len(predictions) == len(
          p_speedup))

      # record results
      for (benchmark, benchmark_suite, oracle_device_mapping, predicted,
           is_correct, predicted_speedup) in zip(
          benchmarks, benchmark_suites, oracle_device_mappings, predictions,
          predicted_is_correct,
          p_speedup):
        data.append({
          "Model": model.__name__,
          "Platform": gpu_name,
          'Benchmark': benchmark,
          'Benchmark Suite': benchmark_suite,
          "Oracle Mapping": oracle_device_mapping,
          "Predicted Mapping": predicted,
          "Correct?": is_correct,
          "Speedup": predicted_speedup,
        })

      # update progress bar
      bar_tuple[0] += 1
      bar_tuple[1].update(bar_tuple[0])

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
