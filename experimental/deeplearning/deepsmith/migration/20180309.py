"""
Export CSV files to Protos.
"""
import os
import typing

import pandas as pd

from absl import app
from absl import flags
from absl import logging

from deeplearning.deepsmith.proto import deepsmith_pb2

FLAGS = flags.FLAGS

flags.DEFINE_string('csv_dir', None, 'Directory to import CSVs from.')
flags.DEFINE_string('proto_dir', None, 'Directory to export protos to.')


OPENCL_CSVS = {
  "assertions.csv",
  "classifications.csv",
  "clsmith_program_metas.csv",
  "clsmith_testcase_metas.csv",
  "dsmith_program_metas.csv",
  "dsmith_testcase_metas.csv",
  "majorities.csv",
  "platforms.csv",
  "programs.csv",
  "reductions.csv",
  "results.csv",
  "results_metas.csv",
  "stackdumps.csv",
  "stderrs.csv",
  "stdouts.csv",
  "testbeds.csv",
  "testcases.csv",
  "threads.csv",
  "unreachables.csv",
}


def _ParseOpenCLTestbeds(path: str) -> pd.DataFrame:
  df = pd.read_csv(path)
  print(df)

def _OpenClCsvsToProtos(csv_dir: str, proto_dir) -> None:
  assert(all(os.path.isfile(os.path.join(csv_dir, x)) for x in os.listdir(csv_dir)))


def _CsvsToProtos() -> None:
  csv_dir = FLAGS.csv_dir
  proto_dir = FLAGS.proto_dir
  logging.info("Converting CSVs in %s to protos in %s", csv_dir)

  _OpenClCsvsToProtos(os.path.join(csv_dir, "opencl_04_opencl"),
                      os.path.join(proto_dir, "opencl"))


def main():  # pylint: disable=missing-docstring
  _CsvsToProtos()


if __name__ == '__main__':
  app.run(main)
