"""Get a baseline reading of CLgen OpenCL models."""
from absl import app
from absl import flags

from deeplearning.clgen import clgen
from labm8 import bazelutil

FLAGS = flags.FLAGS

PROTOS = [
    bazelutil.DataPath('phd/experimental/clgen/keras/opencl_baseline_a.pbtxt'),
    bazelutil.DataPath('phd/experimental/clgen/keras/opencl_baseline_b.pbtxt'),
]


def main(argv):
  del argv
  for proto in PROTOS:
    instance = clgen.Instance.FromFile(proto)
    instance.Sample(min_num_samples=1000)


if __name__ == '__main__':
  app.run(main)
