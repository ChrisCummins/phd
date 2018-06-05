"""Train a discriminator."""
import pathlib
import pickle
import typing

import humanize
import numpy as np
from absl import app
from absl import flags
from absl import logging

from deeplearning.clgen.corpuses import atomizers
from experimental.deeplearning.fish.proto import fish_pb2
from lib.labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'export_path', None,
    'Directory to read training data protos from.')
flags.DEFINE_integer(
    'sequence_length', 1024,
    'The length of encoded program source sequences. Sequences shorter than '
    'this value are padded. Sequences longer than this value are truncated.')
flags.DEFINE_integer(
    'lstm_size', 64,
    'The number of neurons in each LSTM layer.')
flags.DEFINE_integer(
    'num_layers', 2,
    'The number of LSTM layers.')
flags.DEFINE_integer(
    'dnn_size', 32,
    'The number of neurons in the DNN layer.')
flags.DEFINE_integer(
    'num_epochs', 50,
    'The number of epochs to train for.')
flags.DEFINE_integer(
    'batch_size', 64,
    'The training batch size.')


def EncodeAndPad(srcs: typing.List[str], padded_length: int,
                 atomizer: atomizers.AtomizerBase) -> np.array:
  """Encode and pad source code strings for training."""
  from keras.preprocessing import sequence

  seqs = [atomizer.AtomizeString(src) for src in srcs]
  pad_val = atomizer.vocab_size
  encoded = np.array(
      sequence.pad_sequences(seqs, maxlen=padded_length, value=pad_val))
  return np.vstack([np.expand_dims(x, axis=0) for x in encoded])


def Encode1HotLabels(y: np.array) -> np.array:
  """1-hot encode labels."""
  labels = np.vstack([np.expand_dims(x, axis=0) for x in y])
  l2 = [x[0] for x in labels]
  l1 = [not x for x in l2]
  return np.array(list(zip(l1, l2)), dtype=np.int32)


def BuildKerasModel(
    sequence_length: int, lstm_size: int, num_layers: int, dnn_size: int,
    atomizer: atomizers.AtomizerBase):
  import keras
  code_in = keras.layers.Input(
      shape=(sequence_length,), dtype='int32', name='code_in')
  x = keras.layers.Embedding(
      # Note the +1 on atomizer.vocab_size to accommodate the padding character.
      input_dim=atomizer.vocab_size + 1, input_length=sequence_length,
      output_dim=lstm_size, name='embedding')(code_in)
  for _ in range(num_layers):
    x = keras.layers.LSTM(
        lstm_size, implementation=1, return_sequences=True)(x)
  x = keras.layers.LSTM(lstm_size, implementation=1)(x)
  x = keras.layers.Dense(dnn_size, activation='relu')(x)
  # There are two output classes.
  out = keras.layers.Dense(2, activation='sigmoid')(x)

  model = keras.models.Model(input=code_in, output=out)
  model.compile(
      optimizer='adam', metrics='accuracy', loss='categorical_crossentropy')
  model.summary()
  return model


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  if not FLAGS.export_path:
    raise app.UsageError('--export_path must be a directory')
  export_path = pathlib.Path(FLAGS.export_path)
  if export_path.is_file():
    raise app.UsageError('--export_path must be a directory')
  export_path.mkdir(parents=True, exist_ok=True)

  positive_protos = [
    pbutil.FromFile(path, fish_pb2.CompilerCrashDiscriminatorTrainingExample())
    for path in sorted(list((export_path / 'build_crash').iterdir()))
  ]
  logging.info('Loaded %s positive data protos',
               humanize.intcomma(len(positive_protos)))
  # Load an equal number of negative protos, sorted by result ID.
  negative_protos = [
    pbutil.FromFile(path, fish_pb2.CompilerCrashDiscriminatorTrainingExample())
    for path in
    sorted(list((export_path / 'pass').iterdir()))[:len(positive_protos)]
  ]
  logging.info('Loaded %s negative training data protos',
               humanize.intcomma(len(negative_protos)))

  training_protos = negative_protos + positive_protos

  logging.info('Number of training examples: %s positive, %s negative',
               humanize.intcomma(len(positive_protos)),
               humanize.intcomma(len(negative_protos)))

  sequence_length = FLAGS.sequence_length
  text = '\n'.join([p.src for p in positive_protos + negative_protos])
  logging.info('Deriving atomizer')
  atomizer = atomizers.AsciiCharacterAtomizer.FromText(text)

  logging.info('Encoding corpus')
  x = EncodeAndPad([p.src for p in positive_protos + negative_protos],
                   sequence_length, atomizer)
  y = np.concatenate((np.ones(len(positive_protos)),
                      np.zeros(len(negative_protos))))
  assert len(x) == len(training_protos)
  assert len(y) == len(training_protos)

  np.random.seed(FLAGS.seed)
  logging.info('Building Keras model')
  model = BuildKerasModel(
      sequence_length=sequence_length, lstm_size=FLAGS.lstm_size,
      num_layers=FLAGS.num_layers, dnn_size=FLAGS.dnn_size, atomizer=atomizer)
  logging.info('Training model')
  model.fit(x, Encode1HotLabels(y), epochs=FLAGS.num_epochs,
            batch_size=FLAGS.batch_size, verbose=True, shuffle=True)


if __name__ == '__main__':
  app.run(main)
