import os
import pathlib
import typing

import pandas as pd

from datasets.benchmarks.gpgpu import gpgpu_pb2
from labm8.py import app
from labm8.py import pbutil

FLAGS = app.FLAGS

app.DEFINE_string(
  "cgo17_features_csv",
  "",
  "The csv file containing the features published in cgo17.",
)
app.DEFINE_string(
  "gpgpu_log_dir", "", "The directory containing the libcecl event logs."
)


def print_df(df):
  """Print a dataframe to stdout"""
  with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(df)


def print_df_stats(df):
  """Print statistics of a datafra to stdout"""
  sums = df.sum()
  print("CPU: %f percent of values true" % (sums["CPU"] / len(df) * 100))
  print("GPU: %f percent of values true" % (sums["GPU"] / len(df) * 100))


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(" ".join(argv[1:])))

  # Prepare result df based on cgo17 df
  df = pd.read_csv(FLAGS.cgo17_features_csv)
  df = pd.concat([df["benchmark"], df["dataset"]], axis=1)
  df["CPU"] = False
  df["GPU"] = False

  # Populate with gpgpu data
  for filename in os.listdir(FLAGS.gpgpu_log_dir):
    filename_abs = pathlib.Path(os.path.join(FLAGS.gpgpu_log_dir, filename))

    # Parse protobuf file to object
    benchmark_run = pbutil.FromFile(filename_abs, gpgpu_pb2.GpgpuBenchmarkRun())

    # Parse domain data
    benchmark_suite = benchmark_run.benchmark_suite
    benchmark_name = benchmark_run.benchmark_name
    device_type = benchmark_run.run.device.device_type

    if "." in benchmark_name:
      dataset_name = benchmark_name.split(".")[1]
    else:
      dataset_name = benchmark_run.dataset_name

    kernel_names = set()
    for kernel_invocation in benchmark_run.run.kernel_invocation:
      kernel_names.add(kernel_invocation.kernel_name)

    for kernel_name in kernel_names:
      # Populate dataframe
      df[device_type] = df[device_type] | (
        df["benchmark"].str.contains(benchmark_suite)
        & df["benchmark"].str.contains(kernel_name)
        & (df["dataset"] == dataset_name)
      )

  # Print
  print_df(df)
  print_df_stats(df)


if __name__ == "__main__":
  app.RunWithArgs(main)
