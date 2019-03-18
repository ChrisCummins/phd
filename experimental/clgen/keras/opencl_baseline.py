"""Get a baseline reading of CLgen OpenCL models."""
from deeplearning.clgen import clgen
from deeplearning.clgen import sample_observers
from labm8 import app
from labm8 import bazelutil

FLAGS = app.FLAGS

PROTOS = [
    bazelutil.DataPath('phd/experimental/clgen/keras/opencl_baseline_a.pbtxt'),
    bazelutil.DataPath('phd/experimental/clgen/keras/opencl_baseline_b.pbtxt'),
]


def main(argv):
  del argv
  for proto in PROTOS:
    instance = clgen.Instance.FromFile(proto)
    instance.Sample(
        sample_observers=[sample_observers.MaxSampleCountObserver(1000)])


if __name__ == '__main__':
  app.RunWithArgs(main)
