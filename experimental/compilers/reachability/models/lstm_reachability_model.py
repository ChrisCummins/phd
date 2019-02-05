"""Implementation of LSTM networks for compilers."""
import json
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
from keras import layers
from keras.preprocessing import sequence

from deeplearning.clgen.corpuses import atomizers
from experimental.compilers.reachability import control_flow_graph
from experimental.compilers.reachability import \
  control_flow_graph_generator as cfg_generator
from experimental.compilers.reachability.datasets import \
  make_reachability_dataset as dataset
from labm8 import labdate
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
    'num_classes', 15,
    'The number of classes')
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
    'reachability_model_seed', 0,
    'Random seed for the model.')
flags.DEFINE_boolean(
    'neighbors_only', False,
    'If true, generate lists of only immediate neighbors for each node, not '
    'all successor nodes.')
flags.DEFINE_boolean(
    'zero_r', False,
    'If true, train and evaluate a ZeroR model, not an LSTM model.')


def EncodeAndPad(srcs: typing.List[str], padded_length: int,
                 atomizer) -> np.array:
  """Encode and pad source code strings for training."""
  seqs = [atomizer.AtomizeString(src) for src in srcs]
  assert max(len(seq) for seq in seqs) <= padded_length
  pad_val = atomizer.vocab_size
  encoded = np.array(
      sequence.pad_sequences(seqs, maxlen=padded_length, value=pad_val))
  return np.vstack([np.expand_dims(x, axis=0) for x in encoded])


def BuildKerasModel(
    sequence_length: int, num_classes: int, lstm_size: int,
    dnn_size: int, atomizer: atomizers.AtomizerBase) -> keras.models.Model:
  """Instantiate reachability classifier model."""
  code_in = layers.Input(
      shape=(sequence_length,), dtype='int32', name='code_in')
  language_model = layers.Embedding(
      # Note the +1 on atomizer.vocab_size to accommodate the padding character.
      input_dim=atomizer.vocab_size + 1, input_length=sequence_length,
      output_dim=lstm_size, name='embedding')(code_in)

  # LSTM model.
  language_model = layers.LSTM(
      lstm_size, implementation=1, return_sequences=True)(language_model)
  language_model = layers.LSTM(lstm_size, implementation=1)(language_model)

  # Node selector, a 1-hot vector which selects a node in the graph to predict
  # reachability for.
  node_selector = layers.Input(shape=(num_classes,), name="node_selector")

  # Heuristic model. Takes as inputs a concatenation of the language model
  # and auxiliary inputs, outputs 1-hot encoded device mapping.
  heuristic_model = layers.Concatenate()([node_selector, language_model])
  heuristic_model = layers.BatchNormalization()(heuristic_model)

  heuristic_model = layers.Dense(dnn_size, activation='relu')(
      heuristic_model)
  heuristic_model_outputs = [
    layers.Dense(2, activation="sigmoid", name=f"node_{i}")(heuristic_model)
    for i in range(num_classes)
  ]

  model = keras.models.Model(
      inputs=[node_selector, code_in],
      outputs=heuristic_model_outputs)
  model.compile(optimizer="adam", metrics=["accuracy"],
                loss=["categorical_crossentropy"] * num_classes)
  return model


def DataFrameToModelData(
    df: pd.DataFrame, sequence_length: int, atomizer: atomizers.AtomizerBase
) -> typing.List[typing.Tuple[control_flow_graph.ControlFlowGraph]]:
  sequences = EncodeAndPad(
      df['text:successors'], sequence_length, atomizer).astype(np.int32)
  # The `reachability:node_selector` column contains a one-hot encoded index
  # of the source node for reachability. Each array has num_class elements.
  node_selectors = np.vstack(df['reachability:node_selector'])
  x = [node_selectors, sequences]
  # The `reachability:reachble_nodes` column contains a bool array of size
  # num_class, where each element is a one-hot encoding of whether a node is
  # reachable from the source node indicated by `reachability:node_selector`.
  outputs = []
  # Slice the data into vertical columns, so that we have one output for each
  # node.
  for col in range(len(df['reachability:reachable_nodes'].iloc[0])):
    column = np.vstack([r[col] for r in df['reachability:reachable_nodes']])
    outputs.append(column)
  # A list of numpy arrays, with shape (num_classes,len(df)).
  y = outputs
  return x, y


def OneHot(i: typing.Union[int, bool], n: int = 2):
  """One-hot encode value 'i' with dimensionality 'n'."""
  out = np.zeros(n, dtype=np.int32)
  out[int(i)] = 1
  return out


def ExpandToClassificationDataset(df: pd.DataFrame, num_classes: int):
  """Produce classification dataset for reachability.

  The returned table has len(df) * num_classes rows.
  """
  rows = []
  for _, row in df.iterrows():
    for i in range(num_classes):
      row = row.copy()
      graph: control_flow_graph.ControlFlowGraph = row['cfg:graph']
      row['reachability:node_selector'] = OneHot(i, num_classes)
      # Array of shape: (num_classes, 2).
      row['reachability:reachable_nodes'] = np.array([
        OneHot(graph.IsReachable(i, j)) for j in range(num_classes)
      ], dtype=np.int32)
      rows.append(row)
  return pd.DataFrame(rows)


class LstmReachabilityModel(object):

  def __init__(self, df: pd.DataFrame, outdir: pathlib.Path, num_classes: int,
               lstm_size: int = None, dnn_size: int = None):
    self.lstm_size = lstm_size or FLAGS.lstm_size
    self.dnn_size = dnn_size or FLAGS.dnn_size

    if set(df['cfg:block_count']) != {num_classes}:
      raise ValueError(f"All graphs must have {num_classes} blocks")

    # Make the output directories.
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'telemetry').mkdir(exist_ok=True)
    (outdir / 'checkpoints').mkdir(exist_ok=True)

    self.num_classes = num_classes
    self.outdir = outdir

    # Derive atomizer and sequence length.

    # If --neighbours_only, train on list of immediate neighbor nodes. Else,
    # train on a list of all successors. Immediate neighbors is a much more
    # difficult problem.
    if FLAGS.neighbors_only:
      df['text:successors'] = [
        row['cfg:graph'].ToNeighborsString() for _, row in df.iterrows()
      ]
    else:
      df['text:successors'] = [
        row['cfg:graph'].ToSuccessorsString() for _, row in df.iterrows()
      ]

    text = '\n'.join(df['text:successors'])

    sequence_length = max(len(s) for s in df['text:successors'])
    logging.info("Sequence length: %d", sequence_length)

    df = ExpandToClassificationDataset(df, self.num_classes)
    if not len(df):
      raise ValueError("Empty dataframe!")
    logging.info('Expanded dataframe to %s classification data points', len(df))

    # Create the model.
    if (outdir / 'atomizer.pkl').is_file():
      with open(outdir / 'atomizer.pkl', 'rb') as f:
        atomizer = pickle.load(f)
    else:
      logging.info('Deriving atomizer from %s charss',
                   humanize.intcomma(len(text)))
      atomizer = atomizers.AsciiCharacterAtomizer.FromText(text)
      logging.info('Vocabulary size: %s',
                   humanize.intcomma(len(atomizer.vocab)))
      with open(outdir / 'atomizer.pkl', 'wb') as f:
        pickle.dump(atomizer, f)
      logging.info('Pickled atomizer to %s', outdir / 'atomizer.pkl')

    train_df = df[df['split:type'] == 'training']
    if not len(train_df):
      raise ValueError("No training graphs!")
    self.train_x, self.train_y = DataFrameToModelData(
        train_df, sequence_length, atomizer)
    logging.info('Training data: x=[%s, %s], y=[15x %s]',
                 self.train_x[0].shape, self.train_x[1].shape,
                 self.train_y[0].shape)

    valid_df = df[df['split:type'] == 'validation']
    if not len(valid_df):
      raise ValueError("No validation graphs!")
    self.valid_x, self.valid_y = DataFrameToModelData(
        valid_df, sequence_length, atomizer)
    logging.info('Validation data: x=[%s, %s], y=[15x %s]',
                 self.valid_x[0].shape, self.valid_x[1].shape,
                 self.valid_y[0].shape)

    test_df = df[df['split:type'] == 'test']
    if not len(test_df):
      raise ValueError("No test graphs!")
    self.test_x, self.test_y = DataFrameToModelData(
        test_df, sequence_length, atomizer)
    logging.info('Testing data: x=[%s, %s], y=[15x %s]',
                 self.test_x[0].shape, self.test_x[1].shape,
                 self.test_y[0].shape)

    num_uniq_seqs = len(set(df['text:successors']))
    logging.info('Unique sequences: %s of %s (%.2f %%)',
                 humanize.intcomma(num_uniq_seqs),
                 humanize.intcomma(len(df)),
                 (num_uniq_seqs / len(df)) * 100)

    np.random.seed(FLAGS.reachability_model_seed)
    random.seed(FLAGS.reachability_model_seed)
    logging.info('Building Keras model ...')
    self.model = BuildKerasModel(
        sequence_length=sequence_length, num_classes=num_classes,
        lstm_size=self.lstm_size, dnn_size=self.dnn_size, atomizer=atomizer)

    self.num_classes = num_classes

    with open(outdir / 'model.json', 'w') as f:
      f.write(self.model.to_json())
    logging.info('Wrote model to %s', outdir / 'model.json')

  def TrainAndEvaluate(self, num_epochs: int):
    def OnEpochEnd(epoch, logs):
      """End-of-epoch model evaluate."""
      logging.info('Evaluating model at epoch %d', epoch + 1)

      with prof.Profile('test set'):
        test_logs = dict(zip(
            self.model.metrics_names,
            self.model.evaluate(
                self.test_x, self.test_y, batch_size=FLAGS.batch_size,
                verbose=0)))
        outputs = self.model.predict(
            self.test_x, batch_size=FLAGS.batch_size)

        accuracies = []
        solutions = []
        for col in range(self.num_classes):
          predictions = outputs[col]
          ground_truth = self.test_y[col]

          # Flatten 1-hot encoding to lists.
          predictions = np.array([np.argmax(x) for x in predictions])
          ground_truth = np.array([np.argmax(x) for x in ground_truth])

          # Compare lists.
          correct = ground_truth == predictions
          solutions.append(correct)
          accuracies.append(correct.mean())

        accuracy = np.mean(accuracies)

        # Transpose to make a (num_rows, num_classes) shape tensor.
        solutions = np.array(solutions).T
        solved = np.mean([row.all() for row in solutions])

      logging.info("Evaluated test set at epoch %s, accuracy: %.2f%%, "
                   "solved: %.2f%%", epoch + 1, accuracy * 100, solved * 100)

      validation_accuracy = np.mean([
        logs[f'val_node_{i}_acc'] for i in range(self.num_classes)
      ])

      telemetry = {
        'epoch': epoch + 1,
        'lstm_size': self.lstm_size,
        'dnn_size': self.dnn_size,
        'num_classes': self.num_classes,
        'batch_size': FLAGS.batch_size,
        # Dataset attributes. These are constant across epochs.
        'training_graph_count': len(self.train_x[0]),
        'validation_graph_count': len(self.valid_x[0]),
        'test_graph_count': len(self.test_x[0]),
        'training_loss': logs['loss'],
        'validation_loss': logs['val_loss'],
        'validation_accuracy': validation_accuracy,
        'test_loss': test_logs['loss'],
        'test_accuracy': accuracy,
        'test_solved': solved,
      }
      with open(f'{self.outdir}/telemetry/epoch_{epoch+1:03d}.'
                f'T{labdate.MillisecondsTimestamp()}.json', 'w') as f:
        json.dump(telemetry, f)

    self.model.fit(
        self.train_x, self.train_y, epochs=num_epochs,
        validation_data=(self.valid_x, self.valid_y),
        batch_size=FLAGS.batch_size, verbose=True, shuffle=True,
        callbacks=[
          keras.callbacks.ModelCheckpoint(
              # The {epoch} placeholder in the string below will be substituted
              # by Keras.
              str(self.outdir) + '/checkpoints/weights_{epoch:03d}.hdf5',
              verbose=True, mode="min", save_best_only=False),
          keras.callbacks.LambdaCallback(on_epoch_end=OnEpochEnd),
        ])


class ZeroRReachabilityModel(LstmReachabilityModel):

  def __init__(self, df: pd.DataFrame, outdir: pathlib.Path, num_classes: int,
               lstm_size: int = None, dnn_size: int = None):
    if set(df['cfg:block_count']) != {num_classes}:
      raise ValueError(f"All graphs must have {num_classes} blocks")

    # Make the output directories.
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'telemetry').mkdir(exist_ok=True)

    self.num_classes = num_classes
    self.outdir = outdir

    # Derive atomizer and sequence length.

    df['text:successors'] = [
      row['cfg:graph'].ToNeighborsString() for _, row in df.iterrows()
    ]

    text = '\n'.join(df['text:successors'])

    sequence_length = max(len(s) for s in df['text:successors'])
    logging.info("Sequence length: %d", sequence_length)

    df = ExpandToClassificationDataset(df, self.num_classes)
    if not len(df):
      raise ValueError("Empty dataframe!")
    logging.info('Expanded dataframe to %s classification data points', len(df))

    # Create the model.
    if (outdir / 'atomizer.pkl').is_file():
      with open(outdir / 'atomizer.pkl', 'rb') as f:
        atomizer = pickle.load(f)
    else:
      logging.info('Deriving atomizer from %s charss',
                   humanize.intcomma(len(text)))
      atomizer = atomizers.AsciiCharacterAtomizer.FromText(text)
      logging.info('Vocabulary size: %s',
                   humanize.intcomma(len(atomizer.vocab)))
      with open(outdir / 'atomizer.pkl', 'wb') as f:
        pickle.dump(atomizer, f)
      logging.info('Pickled atomizer to %s', outdir / 'atomizer.pkl')

    train_df = df[df['split:type'] == 'training']
    if not len(train_df):
      raise ValueError("No training graphs!")
    self.train_x, self.train_y = DataFrameToModelData(
        train_df, sequence_length, atomizer)
    logging.info('Training data: x=[%s, %s], y=[15x %s]',
                 self.train_x[0].shape, self.train_x[1].shape,
                 self.train_y[0].shape)

    test_df = df[df['split:type'] == 'test']
    if not len(test_df):
      raise ValueError("No test graphs!")
    self.test_x, self.test_y = DataFrameToModelData(
        test_df, sequence_length, atomizer)
    logging.info('Testing data: x=[%s, %s], y=[15x %s]',
                 self.test_x[0].shape, self.test_x[1].shape,
                 self.test_y[0].shape)

  def TrainAndEvaluate(self, num_epochs: int):
    del num_epochs

    avg = np.array(self.train_y).mean()
    rule = np.ones if avg > .5 else np.zeros
    y = rule(np.array(self.test_y).shape)

    correct = y == self.test_y
    accuracy = correct.mean()
    solved = np.mean([row.all() for row in correct])
    logging.info('Rule: %s, Accuracy: %s, Solved: %s', avg, accuracy, solved)

    return accuracy, solved


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

  with prof.Profile('load dataframe'):
    df = pd.read_pickle(df_path)
  logging.info('Loaded %s dataframe from %s', df.shape, df_path)
  # Keep only the test set.
  test_df = df[df['split:type'] == 'test']
  old_len = len(test_df)
  # Filter the test set to match the num classes we want.
  test_df = test_df[test_df['cfg:block_count'] == FLAGS.num_classes]
  logging.info('Filtered test set from %d to %d graphs with %d blocks in each',
               old_len, len(test_df), FLAGS.num_classes)
  del df

  # Replace the training and validation data in the loaded dataset with our own
  # that we generate now.
  random_state = np.random.RandomState(FLAGS.synthetic_generator_seed)

  with prof.Profile('synthetic training graphs'):
    training_graph_generator = dataset.SpecGenerator(
        cfg_generator.ControlFlowGraphGenerator(
            random_state, (FLAGS.num_classes, FLAGS.num_classes),
            dataset.edge_density_tr, strict=False))
    train_df = dataset.SpecsToDataFrame(
        training_graph_generator.Generate(FLAGS.num_synthetic_training_graphs),
        'training')

  with prof.Profile('synthetic validation graphs'):
    validation_graph_generator = dataset.SpecGenerator(
        cfg_generator.ControlFlowGraphGenerator(
            random_state, (FLAGS.num_classes, FLAGS.num_classes),
            dataset.edge_density_ge, strict=False))
    valid_df = dataset.SpecsToDataFrame(
        validation_graph_generator.Generate(
            FLAGS.num_synthetic_validation_graphs),
        'validation')

  # Assemble the new dataset.
  df = pd.concat((train_df, valid_df, test_df))

  if FLAGS.zero_r:
    model = ZeroRReachabilityModel(df, outdir, FLAGS.num_classes)
  else:
    model = LstmReachabilityModel(df, outdir, FLAGS.num_classes)
  model.TrainAndEvaluate(num_epochs=FLAGS.num_epochs)


if __name__ == '__main__':
  app.run(main)
