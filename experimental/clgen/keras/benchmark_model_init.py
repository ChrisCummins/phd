"""Profile Keras model instantiate.

Things I learned from this file:
 * Conversion of a model to YAML is free.
 * A one layer LSTM takes twice as long to instantiate than a one layer LSTM.
 * Model compilation is not time consuming.
 * model_from_yaml() takes as long as instantiating a new model.
"""
import sys
import tempfile

import pytest
from absl import app
from absl import flags
from keras import layers
from keras import models


FLAGS = flags.FLAGS


def test_1layer_LSTM_no_compile(benchmark):
  """Benchmark instantiation of a one layer LSTM network without compiling."""

  def Benchmark():
    """Benchmark inner loop."""
    model = models.Sequential()
    model.add(layers.LSTM(512, input_shape=(80, 200)))
    model.add(layers.Dense(200))
    model.add(layers.Activation('softmax'))

  benchmark(Benchmark)


def test_1layer_LSTM(benchmark):
  """Benchmark instantiation of a one layer LSTM network."""

  def Benchmark():
    """Benchmark inner loop."""
    model = models.Sequential()
    model.add(layers.LSTM(512, input_shape=(80, 200)))
    model.add(layers.Dense(200))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

  benchmark(Benchmark)


def test_2layer_LSTM_no_compile(benchmark):
  """Benchmark instantiation of a two layer LSTM network without compiling."""

  def Benchmark():
    """Benchmark inner loop."""
    model = models.Sequential()
    model.add(layers.LSTM(512, input_shape=(80, 200), return_sequences=True))
    model.add(layers.LSTM(512))
    model.add(layers.Dense(200))
    model.add(layers.Activation('softmax'))

  benchmark(Benchmark)


def test_2layer_LSTM(benchmark):
  """Benchmark instantiation of a two layer LSTM network."""

  def Benchmark():
    """Benchmark inner loop."""
    model = models.Sequential()
    model.add(layers.LSTM(512, input_shape=(80, 200), return_sequences=True))
    model.add(layers.LSTM(512))
    model.add(layers.Dense(200))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

  benchmark(Benchmark)


def test_to_yaml(benchmark):
  """Benchmark two layer LSTM to YAML."""

  def Benchmark(model_, path):
    """Benchmark inner loop."""
    with open(path, 'w') as f:
      f.write(model_.to_yaml())

  model = models.Sequential()
  model.add(layers.LSTM(512, input_shape=(80, 200), return_sequences=True))
  model.add(layers.LSTM(512))
  model.add(layers.Dense(200))
  model.add(layers.Activation('softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  with tempfile.NamedTemporaryFile(prefix='clgen_') as f:
    benchmark(Benchmark, model, f.name)


def test_from_yaml(benchmark):
  """Benchmark instantiation of a two layer LSTM from YAML config."""
  model = models.Sequential()
  model.add(layers.LSTM(512, input_shape=(80, 200), return_sequences=True))
  model.add(layers.LSTM(512))
  model.add(layers.Dense(200))
  model.add(layers.Activation('softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  benchmark(models.model_from_yaml, model.to_yaml())


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments '{}'".format(', '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  app.run(main)
