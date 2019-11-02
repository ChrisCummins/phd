"""Unit tests for //deeplearning/ml4pl/graphs/unlabelled/cdfg:random_cdfg_generator."""
from labm8 import app
from labm8 import test

from deeplearning.ml4pl.graphs.unlabelled.cdfg import random_cdfg_generator

FLAGS = app.FLAGS


def test_benchmark_FastCreateRandom(benchmark):
  benchmark(random_cdfg_generator.FastCreateRandom)


if __name__ == '__main__':
  test.Main()
