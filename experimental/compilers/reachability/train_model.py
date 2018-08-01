"""Train a reachability classifier."""
import humanize
import keras
import numpy as np
import pathlib
import pickle
import random
import typing
from absl import app
from absl import flags
from absl import logging
from keras.preprocessing import sequence
from phd.lib.labm8 import pbutil

from deeplearning.clgen import telemetry
from deeplearning.clgen.corpuses import atomizers
from experimental.compilers.reachability import reachability
from experimental.compilers.reachability.proto import reachability_pb2


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'reachability_num_training_graphs', 10000,
    'The number of training graphs to generate.')
flags.DEFINE_integer(
    'reachability_num_testing_graphs', 1000,
    'The number of testing graphs to generate.')
flags.DEFINE_string(
    'reachability_model_dir',
    '/var/phd/experimental/compilers/reachability/model',
    'Directory to save model files to.')
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
flags.DEFINE_integer(
    'reachability_model_seed', None,
    'Random seed value.')


def MakeReachabilityDataset(
    num_data_points: int) -> reachability_pb2.ReachabilityDataset:
  """Generate a training data proto."""
  data = reachability_pb2.ReachabilityDataset()
  for i in range(num_data_points):
    graph = reachability.ControlFlowGraph.GenerateRandom(
        FLAGS.reachability_num_nodes, seed=i,
        connections_scaling_param=FLAGS.reachability_scaling_param)
    proto = data.entry.add()
    graph.SetCfgWithReachabilityProto(proto)
    proto.seed = i
  return data


def EncodeAndPad(srcs: typing.List[str], padded_length: int,
                 atomizer) -> np.array:
  """Encode and pad source code strings for training."""
  seqs = [atomizer.AtomizeString(src) for src in srcs]
  assert max(len(seq) for seq in seqs) <= padded_length
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
    sequence_length: int, num_classes: int, lstm_size: int, num_layers: int,
    dnn_size: int, atomizer: atomizers.AtomizerBase):
  code_in = keras.layers.Input(
      shape=(sequence_length,), dtype='int32', name='code_in')
  x = keras.layers.Embedding(
      # Note the +1 on atomizer.vocab_size to accommodate the padding character.
      input_dim=atomizer.vocab_size + 1, input_length=sequence_length,
      output_dim=lstm_size, name='embedding')(code_in)
  for i in range(num_layers):
    x = keras.layers.LSTM(
        lstm_size, implementation=1, return_sequences=True,
        go_backwards=not i)(x)
  x = keras.layers.LSTM(lstm_size, implementation=1)(x)
  x = keras.layers.Dense(dnn_size, activation='relu')(x)
  outs = [
    keras.layers.Dense(1, activation='sigmoid',
                       name=reachability.NumberToLetters(i))(x)
    for i in range(num_classes)
  ]

  model = keras.models.Model(input=code_in, outputs=outs)
  model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                optimizer='adam')
  model.summary()
  return model


def ProtosToModelData(data: reachability_pb2.ReachabilityDataset,
                      sequence_length: int,
                      atomizer: atomizers.AtomizerBase
                      ) -> typing.Tuple[np.ndarray, typing.List[np.ndarray]]:
  x = EncodeAndPad(
      [ControlFlowGraphToSequence(entry.graph) for entry in data.entry],
      sequence_length, atomizer).astype(np.int32)
  y = [
    np.array([[entry.reachable[i]] for entry in data.entry]).astype(np.int32)
    for i in range(len(data.entry[0].reachable))
  ]
  return x, y


def ControlFlowGraphToSequence(graph: reachability_pb2.ControlFlowGraph) -> str:
  s = []
  for node in graph.node:
    successors = ' '.join(node.child)
    s.append(f'{node.name}: {successors}\n')
  return ''.join(s)


def FlattenModelOutputs(outs: np.ndarray) -> np.ndarray:
  """Flatten the model output to a 1D vector of predictions."""
  outs = np.array([x[0][0] for x in outs])
  return outs


def FlattenModelData(y, i):
  """Extract labels row 'i' from labels 'y'."""
  outs = np.array([y[j][i][0] for j in range(len(y))])
  return outs


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  model_dir = pathlib.Path(FLAGS.reachability_model_dir)
  model_dir.mkdir(parents=True, exist_ok=True)
  (model_dir / 'logs').mkdir(exist_ok=True)
  (model_dir / 'checkpoints').mkdir(exist_ok=True)

  data = MakeReachabilityDataset(
      FLAGS.reachability_num_training_graphs + FLAGS.reachability_num_testing_graphs)
  training_data = reachability_pb2.ReachabilityDataset()
  training_data.entry.extend(
      data.entry[:FLAGS.reachability_num_training_graphs])
  pbutil.ToFile(training_data, model_dir / 'training_data.pbtxt')
  testing_data = reachability_pb2.ReachabilityDataset()
  testing_data.entry.extend(
      data.entry[FLAGS.reachability_num_training_graphs:])
  pbutil.ToFile(testing_data, model_dir / 'testing_data.pbtxt')

  logging.info('Number of training examples: %s.',
               humanize.intcomma(len(training_data.entry)))
  logging.info('Number of testing examples: %s.',
               humanize.intcomma(len(testing_data.entry)))

  # Maximum sequence length for successor lists is a fully connected graph,
  # e.g. for a three node CFG:
  #     A: B C
  #     B: A C
  #     C: A B
  # Each line is 3 + (n-1) * 2 characters long.
  n = FLAGS.reachability_num_nodes
  sequence_length = n * (3 + (n - 1) * 2)
  logging.info('Using sequence length %s.', humanize.intcomma(sequence_length))
  seqs = [ControlFlowGraphToSequence(entry.graph) for entry in data.entry]
  text = '\n'.join(seqs)
  logging.info('Deriving atomizer from %s chars.', humanize.intcomma(len(text)))
  atomizer = atomizers.AsciiCharacterAtomizer.FromText(text)
  logging.info('Vocabulary size: %s.', humanize.intcomma(len(atomizer.vocab)))
  with open(model_dir / 'atomizer.pkl', 'wb') as f:
    pickle.dump(atomizer, f)
  logging.info('Pickled atomizer to %s.', model_dir / 'atomizer.pkl')

  x, y = ProtosToModelData(training_data, sequence_length, atomizer)
  logging.info('Training data: x %s, y[%s] %s', x.shape, len(y), y[0].shape)

  test_x, test_y = ProtosToModelData(testing_data, sequence_length, atomizer)
  logging.info('Testing data: x %s, y[%s] %s', test_x.shape, len(test_y),
               test_y[0].shape)

  num_uniq_seqs = len(set(seqs))
  logging.info('Unique sequences: %s of %s (%.2f %%)',
               humanize.intcomma(num_uniq_seqs),
               humanize.intcomma(len(seqs)), (num_uniq_seqs / len(seqs)) * 100)
  num_uniq_labels = len(
      set([''.join(str(x) for x in e.reachable) for e in data.entry]))
  logging.info('Unique labels: %s of %s (%.2f %%)',
               humanize.intcomma(num_uniq_labels),
               humanize.intcomma(len(seqs)),
               (num_uniq_labels / len(seqs)) * 100)

  np.random.seed(FLAGS.reachability_model_seed)
  random.seed(FLAGS.reachability_model_seed)
  logging.info('Building Keras model ...')
  model = BuildKerasModel(
      sequence_length=sequence_length, num_classes=n,
      lstm_size=FLAGS.lstm_size, num_layers=FLAGS.num_layers,
      dnn_size=FLAGS.dnn_size, atomizer=atomizer)

  logging.info('Training model ...')

  def OnEpochEnd(epoch, logs):
    """End-of-epoch model evaluate."""
    del logs
    logging.info('Evaluating model at epoch %d', epoch)
    # score, accuracy
    row = model.evaluate(
        test_x, test_y, batch_size=FLAGS.batch_size, verbose=0)
    overall_loss, losses, accuracies = row[0], row[1:1 + n], row[n + 1:]
    logging.info('Accuracy (excluding first class): %.2f %%',
                 (sum(accuracies[1:]) / len(accuracies[1:])) * 100)

  logger = telemetry.TrainingLogger(logdir=model_dir / 'logs')
  model.fit(x, y, epochs=FLAGS.num_epochs,
            batch_size=FLAGS.batch_size, verbose=True, shuffle=True,
            callbacks=[
              keras.callbacks.ModelCheckpoint(
                  str(model_dir / 'checkpoints') + '/weights_{epoch:03d}.hdf5',
                  verbose=1, mode="min", save_best_only=False),
              keras.callbacks.LambdaCallback(on_epoch_end=OnEpochEnd),
              logger.KerasCallback(keras),
            ])

  for i in range(5):
    outs = FlattenModelOutputs(model.predict(np.array([x[i]])))
    logging.info('outs:    %s', outs)
    logging.info('clamped: %s', np.rint(outs).astype(np.int32))
    logging.info('true:    %s', FlattenModelData(y, i))
    logging.info('')
  logging.info('done')


if __name__ == '__main__':
  app.run(main)
