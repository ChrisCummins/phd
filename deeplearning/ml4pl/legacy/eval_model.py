"""Evaluate a reachability classifier."""
import keras
import pathlib
import pickle
from deeplearning.ml4pl.proto import ml4pl_pb2

from deeplearning.clgen import telemetry
from deeplearning.clgen.corpuses import atomizers
from deeplearning.ml4pl import train_model
from labm8 import app
from labm8 import humanize
from labm8 import pbutil

FLAGS = app.FLAGS


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  model_dir = pathlib.Path(FLAGS.reachability_model_dir)
  if not model_dir.is_dir():
    raise app.UsageError('--reachability_model_dir is not a directory.')

  logger = telemetry.TrainingLogger(logdir=model_dir / 'logs')
  telemetry_ = logger.EpochTelemetry()
  num_epochs = len(telemetry_)
  training_time_ms = sum(t.epoch_wall_time_ms for t in telemetry_)
  training_time_natural = humanize.Duration(training_time_ms / 1000)
  time_per_epoch_natural = humanize.Duration(
      training_time_ms / num_epochs / 1000)
  losses = [round(t.loss, 2) for t in telemetry_]

  with open(model_dir / 'model.json') as f:
    model: keras.models.Model = keras.models.model_from_json(f.read())

  model.compile(loss='binary_crossentropy',
                metrics=['accuracy'],
                optimizer='adam')
  model.summary()
  print(f'Total training time: {training_time_natural} '
        f'({time_per_epoch_natural} per epoch).')
  print(f'Number of epochs: {num_epochs}.')
  print(f'Training losses: {losses}.')

  training_data = pbutil.FromFile(model_dir / 'training_data.pbtxt',
                                  ml4pl_pb2.ReachabilityDataset())
  testing_data = pbutil.FromFile(model_dir / 'testing_data.pbtxt',
                                 ml4pl_pb2.ReachabilityDataset())
  data = ml4pl_pb2.ReachabilityDataset()
  data.entry.extend(training_data.entry)
  data.entry.extend(testing_data.entry)

  num_nodes = len(training_data.entry[0].graph.node)
  num_nodes_natural = humanize.Commas(num_nodes)
  num_training_graphs_natural = humanize.Commas(len(training_data.entry))
  num_testing_graphs_natural = humanize.Commas(len(testing_data.entry))
  print(f'Training data: {num_training_graphs_natural} graphs of '
        f'{num_nodes_natural} nodes each.')
  print(f'Testing data: {num_testing_graphs_natural} graphs of '
        f'{num_nodes_natural} nodes each.')

  num_connections_training = sum(
      sum(len(n.child)
          for n in entry.graph.node)
      for entry in training_data.entry)
  num_connections_testing = sum(
      sum(len(n.child)
          for n in entry.graph.node)
      for entry in testing_data.entry)

  print('Average graph connections: {:.1f} training ({:.1f} per node), '
        '{:.1f} testing ({:.1f} per node).'.format(
            num_connections_training / len(training_data.entry),
            num_connections_training / (len(training_data.entry) * num_nodes),
            num_connections_testing / len(testing_data.entry),
            num_connections_testing / (len(testing_data.entry) * num_nodes)))

  sequence_length = train_model.GetSequenceLength(
      len(training_data.entry[0].graph.node))
  print('Sequence length:', sequence_length)

  with open(model_dir / 'atomizer.pkl', 'rb') as f:
    atomizer: atomizers.AtomizerBase = pickle.load(f)

  print('Vocabulary size:', atomizer.vocab_size)

  seqs = [
      train_model.ControlFlowGraphToSequence(entry.graph)
      for entry in data.entry
  ]
  num_uniq_seqs = len(set(seqs))
  print('Unique sequences: {} of {} ({:.2f}%)'.format(
      humanize.Commas(num_uniq_seqs), humanize.Commas(len(seqs)),
      (num_uniq_seqs / len(seqs)) * 100))
  num_uniq_labels = len(
      set([''.join(str(x) for x in e.reachable) for e in data.entry]))
  print('Unique labels: {} of {} ({:.2f}%)'.format(
      humanize.Commas(num_uniq_labels), humanize.Commas(len(seqs)),
      (num_uniq_labels / len(seqs)) * 100))

  test_x, test_y = train_model.ProtosToModelData(testing_data, sequence_length,
                                                 atomizer)

  zero_r_acc = sum(sum(x) for x in test_y) / len(testing_data.entry) / num_nodes
  zero_r_acc = max(zero_r_acc[0], 1 - zero_r_acc[0])
  print('Zero-R accuracy: {:.2%}'.format(zero_r_acc))

  row = model.evaluate(test_x, test_y, batch_size=FLAGS.batch_size, verbose=0)
  overall_loss, losses, accuracies = (row[0], row[1:1 + num_nodes],
                                      row[num_nodes + 1:])
  print('Accuracy: {:.2%}'.format(sum(accuracies) / len(accuracies)))
  print('Accuracy (excluding first class): {:.2%}'.format(
      sum(accuracies[1:]) / len(accuracies[1:])))
  print('done.')


if __name__ == '__main__':
  app.RunWithArgs(main)
