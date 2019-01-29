"""Implementation of LSTM networks for compilers."""
import pathlib
import pickle
import random
import typing

import humanize
import keras
import numpy as np
import pandas as pd
from absl import app
from absl import flags
from absl import logging
from keras.preprocessing import sequence

from deeplearning.clgen import telemetry
from deeplearning.clgen.corpuses import atomizers
from experimental.compilers.reachability import control_flow_graph
from labm8 import prof


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'df', '/tmp/phd/docs/wip_graph/lda_opencl_device_mapping_dataset.pkl',
    'Path of the dataframe to load')
flags.DEFINE_string(
    'outdir', '/tmp/phd/docs/wip_graph/model_files',
    'Path of directory to generate files')
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
  """Instantiate reachability classifier model."""
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
                       name=control_flow_graph.NumberToLetters(i))(x)
    for i in range(num_classes)
  ]

  model = keras.models.Model(input=code_in, outputs=outs)
  model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                optimizer='adam')
  model.summary()
  return model


def GraphsAndSrcsToModelData(
    graphs_and_src: typing.List[
      typing.Tuple[control_flow_graph.ControlFlowGraph, int]],
    sequence_length: int,
    atomizer: atomizers.AtomizerBase
) -> typing.Tuple[np.ndarray, typing.List[np.ndarray]]:
  """Convert proto dataset into x,y data for model.

  Args:
    graphs_and_src: A list of <graph,node> tuples, where the node is a src node
      for computing reachabilities of nodes in graph.
    sequence_length: The length of encoded sequences.
    atomizer: The encoding atomizer.

  Returns:
    x,y data for feeding into keras model.
  """
  x = EncodeAndPad([g[0].ToSuccessorsString() for g in graphs_and_src],
                   sequence_length, atomizer).astype(np.int32)
  y = np.array([g[0].Reachables(g[1]) for g in graphs_and_src])
  return x, y


def FlattenModelOutputs(outs: np.ndarray) -> np.ndarray:
  """Flatten the model output to a 1D vector of predictions."""
  outs = np.array([x[0][0] for x in outs])
  return outs


def FlattenModelData(y, i):
  """Extract labels row 'i' from labels 'y'."""
  outs = np.array([y[j][i][0] for j in range(len(y))])
  return outs


def DataFrameToModelData(
    df: pd.DataFrame, sequence_length: int, atomizer: atomizers.AtomizerBase
) -> typing.List[typing.Tuple[control_flow_graph.ControlFlowGraph]]:
  graphs_and_srcs = list(zip(
      df['cfg:graph'].values,
      [row['reachability:target_node_index'] for _, row in df.iterrows()]))
  return GraphsAndSrcsToModelData(graphs_and_srcs, sequence_length, atomizer)


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  logging.info('Starting evaluating LSTM model')

  # Load graphs from file.
  df_path = pathlib.Path(FLAGS.df)
  assert df_path.is_file()

  assert FLAGS.outdir
  outdir = pathlib.Path(FLAGS.outdir)
  # Make the output directories.
  outdir.mkdir(parents=True, exist_ok=True)
  (outdir / 'logs').mkdir(exist_ok=True)
  (outdir / 'checkpoints').mkdir(exist_ok=True)

  with prof.Profile('load dataframe'):
    df = pd.read_pickle(df_path)
  logging.info('Loaded %s dataframe from %s', df.shape, df_path)

  seqs = np.array(
      [row['cfg:graph'].ToSuccessorList() for _, row in df.iterrows()])
  text = '\n'.join(seqs)

  sequence_length = max(len(s) for s in seqs)
  logging.info("Sequence length: %d", sequence_length)

  # Create the model.
  if (outdir / 'atomizer.pkl').is_file():
    with open(outdir / 'atomizer.pkl', 'rb') as f:
      atomizer = pickle.load(f)
  else:
    logging.info('Deriving atomizer from %s chars.',
                 humanize.intcomma(len(text)))
    atomizer = atomizers.AsciiCharacterAtomizer.FromText(text)
    logging.info('Vocabulary size: %s.', humanize.intcomma(len(atomizer.vocab)))
    with open(outdir / 'atomizer.pkl', 'wb') as f:
      pickle.dump(atomizer, f)
    logging.info('Pickled atomizer to %s.', outdir / 'atomizer.pkl')

  train_df = df[df['split:type'] == 'training']
  x, y = DataFrameToModelData(train_df, sequence_length, atomizer)
  logging.info('Training data: x %s, y[%s] %s', x.shape, len(y), y[0].shape)

  valid_df = df[df['split:type'] == 'training']
  valid_x, valid_y = DataFrameToModelData(valid_df, sequence_length, atomizer)
  logging.info('Validation data: x %s, y[%s] %s', valid_x.shape, len(valid_y),
               valid_y[0].shape)

  test_df = df[df['split:type'] == 'training']
  test_x, test_y = DataFrameToModelData(test_df, sequence_length, atomizer)
  logging.info('Testing data: x %s, y[%s] %s', test_x.shape, len(test_y),
               test_y[0].shape)

  num_uniq_seqs = len(set(seqs))
  logging.info('Unique sequences: %s of %s (%.2f %%)',
               humanize.intcomma(num_uniq_seqs),
               humanize.intcomma(len(seqs)), (num_uniq_seqs / len(seqs)) * 100)

  return
  n = 10  # TODO

  np.random.seed(FLAGS.reachability_model_seed)
  random.seed(FLAGS.reachability_model_seed)
  logging.info('Building Keras model ...')
  model = BuildKerasModel(
      sequence_length=sequence_length, num_classes=n,
      lstm_size=FLAGS.lstm_size, num_layers=FLAGS.num_layers,
      dnn_size=FLAGS.dnn_size, atomizer=atomizer)

  model_json = model.to_json()
  with open(outdir / 'model.json', 'w') as f:
    f.write(model_json)
  logging.info('Wrote model to %s', outdir / 'model.json')

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

  logger = telemetry.TrainingLogger(logdir=outdir / 'logs')
  model.fit(x, y, epochs=FLAGS.num_epochs,
            batch_size=FLAGS.batch_size, verbose=True, shuffle=True,
            callbacks=[
              keras.callbacks.ModelCheckpoint(
                  str(outdir / 'checkpoints') + '/weights_{epoch:03d}.hdf5',
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
