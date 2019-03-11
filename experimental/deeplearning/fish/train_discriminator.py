"""Train a discriminator."""
import collections
import pathlib
import pickle
import typing

import keras
import numpy as np
from keras.preprocessing import sequence

from deeplearning.clgen import telemetry
from deeplearning.clgen.corpuses import atomizers
from experimental.deeplearning.fish.proto import fish_pb2
from labm8 import app
from labm8 import humanize
from labm8 import pbutil

FLAGS = app.FLAGS

app.DEFINE_string(
    'dataset_root', None,
    'Directory to read training data from, as generated by '
    ':prepare_discriminator_dataset.')
app.DEFINE_string('model_path', None, 'Directory to save model files to.')
app.DEFINE_integer(
    'sequence_length', 1024,
    'The length of encoded program source sequences. Sequences shorter than '
    'this value are padded. Sequences longer than this value are truncated.')
app.DEFINE_integer('lstm_size', 64, 'The number of neurons in each LSTM layer.')
app.DEFINE_integer('num_layers', 2, 'The number of LSTM layers.')
app.DEFINE_integer('dnn_size', 32, 'The number of neurons in the DNN layer.')
app.DEFINE_integer('num_epochs', 50, 'The number of epochs to train for.')
app.DEFINE_integer('batch_size', 64, 'The training batch size.')
app.DEFINE_integer('seed', None, 'Random seed value.')

PositiveNegativeDataset = collections.namedtuple('PositiveNegativeDataset',
                                                 ['positive', 'negative'])


def EncodeAndPad(srcs: typing.List[str], padded_length: int,
                 atomizer: atomizers.AtomizerBase) -> np.array:
  """Encode and pad source code strings for training."""
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


def BuildKerasModel(sequence_length: int, lstm_size: int, num_layers: int,
                    dnn_size: int, atomizer: atomizers.AtomizerBase):
  code_in = keras.layers.Input(
      shape=(sequence_length,), dtype='int32', name='code_in')
  x = keras.layers.Embedding(
      # Note the +1 on atomizer.vocab_size to accommodate the padding character.
      input_dim=atomizer.vocab_size + 1,
      input_length=sequence_length,
      output_dim=lstm_size,
      name='embedding')(code_in)
  for i in range(num_layers):
    x = keras.layers.LSTM(
        lstm_size, implementation=1, return_sequences=True,
        go_backwards=not i)(x)
  x = keras.layers.LSTM(lstm_size, implementation=1)(x)
  x = keras.layers.Dense(dnn_size, activation='relu')(x)
  # There are two output classes.
  out = keras.layers.Dense(2, activation='sigmoid')(x)

  model = keras.models.Model(input=code_in, output=out)
  model.compile(
      loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
  model.summary()
  return model


def LoadPositiveNegativeProtos(path: pathlib.Path) -> PositiveNegativeDataset:
  """Load positive and negative training protos from a directory."""
  positive_protos = [
      pbutil.FromFile(p, fish_pb2.CompilerCrashDiscriminatorTrainingExample())
      for p in path.iterdir()
      if p.name.startswith('positive-')
  ]
  app.Log(1, 'Loaded %s positive protos', humanize.Commas(len(positive_protos)))
  negative_protos = [
      pbutil.FromFile(p, fish_pb2.CompilerCrashDiscriminatorTrainingExample())
      for p in path.iterdir()
      if p.name.startswith('negative-')
  ]
  app.Log(1, 'Loaded %s negative protos', humanize.Commas(len(negative_protos)))
  return PositiveNegativeDataset(positive_protos, negative_protos)


def ProtosToModelData(protos: PositiveNegativeDataset, sequence_length: int,
                      atomizer: atomizers.AtomizerBase):
  x = EncodeAndPad([p.src for p in protos.positive + protos.negative],
                   sequence_length, atomizer)
  y = np.concatenate((np.ones(len(protos.positive)),
                      np.zeros(len(protos.negative))))
  assert len(x) == len(protos.positive) + len(protos.negative)
  assert len(y) == len(protos.positive) + len(protos.negative)
  return x, y


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  if not FLAGS.dataset_root:
    raise app.UsageError('--dataset_root must be a directory')
  dataset_root = pathlib.Path(FLAGS.dataset_root)
  assert dataset_root.is_dir()
  assert (dataset_root / 'training').is_dir()
  assert (dataset_root / 'validation').is_dir()
  assert (dataset_root / 'testing').is_dir()

  if not FLAGS.model_path:
    raise app.UsageError('--model_path must be set')
  model_path = pathlib.Path(FLAGS.model_path)
  model_path.mkdir(parents=True, exist_ok=True)

  training_protos = LoadPositiveNegativeProtos(dataset_root / 'training')
  validation_protos = LoadPositiveNegativeProtos(dataset_root / 'validation')
  testing_protos = LoadPositiveNegativeProtos(dataset_root / 'testing')
  app.Log(1, 
      'Number of training examples: %s.',
      humanize.Commas(
          len(training_protos.positive) + len(training_protos.negative)))
  app.Log(1, 
      'Number of validation examples: %s.',
      humanize.Commas(
          len(validation_protos.positive) + len(validation_protos.negative)))
  app.Log(1, 
      'Number of testing examples: %s.',
      humanize.Commas(
          len(testing_protos.positive) + len(testing_protos.negative)))

  sequence_length = FLAGS.sequence_length
  text = '\n\n'.join([
      p.src for p in training_protos.positive + training_protos.negative +
      validation_protos.positive + validation_protos.negative +
      testing_protos.positive + testing_protos.negative
  ])
  app.Log(1, 'Deriving atomizer from %s chars.', humanize.Commas(len(text)))
  atomizer = atomizers.AsciiCharacterAtomizer.FromText(text)
  app.Log(1, 'Vocabulary size: %s.', humanize.Commas(len(atomizer.vocab)))
  app.Log(1, 'Pickled atomizer to %s.', model_path / 'atomizer.pkl')
  with open(model_path / 'atomizer.pkl', 'wb') as f:
    pickle.dump(atomizer, f)

  app.Log(1, 'Encoding training corpus')
  x, y = ProtosToModelData(training_protos, sequence_length, atomizer)

  validation_data = None
  if validation_protos.positive:
    app.Log(1, 'Encoding validation corpus')
    validation_data = ProtosToModelData(validation_protos, sequence_length,
                                        atomizer)

  app.Log(1, 'Encoding test corpus')
  test_x, test_y = ProtosToModelData(testing_protos, sequence_length, atomizer)

  np.random.seed(FLAGS.seed)
  app.Log(1, 'Building Keras model')
  model = BuildKerasModel(
      sequence_length=sequence_length,
      lstm_size=FLAGS.lstm_size,
      num_layers=FLAGS.num_layers,
      dnn_size=FLAGS.dnn_size,
      atomizer=atomizer)
  app.Log(1, 'Training model')

  def OnEpochEnd(epoch, logs):
    """End-of-epoch model evaluate."""
    del logs
    app.Log(1, 'Evaluating model at epoch %d', epoch)
    score, accuracy = model.evaluate(
        test_x,
        Encode1HotLabels(test_y),
        batch_size=FLAGS.batch_size,
        verbose=0)
    app.Log(1, 'Score: %.2f, Accuracy: %.2f', score * 100, accuracy * 100)

  logger = telemetry.TrainingLogger(pathlib.Path(FLAGS.model_path))
  model.fit(
      x,
      Encode1HotLabels(y),
      epochs=FLAGS.num_epochs,
      batch_size=FLAGS.batch_size,
      verbose=True,
      shuffle=True,
      validation_data=validation_data,
      callbacks=[
          keras.callbacks.ModelCheckpoint(
              FLAGS.model_path + '/weights_{epoch:03d}.hdf5',
              verbose=1,
              mode="min",
              save_best_only=False),
          keras.callbacks.LambdaCallback(on_epoch_end=OnEpochEnd),
          logger.KerasCallback(keras),
      ])


if __name__ == '__main__':
  app.RunWithArgs(main)
