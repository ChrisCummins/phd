"""Script to automate execution of devmap experiment over k-fold splits."""
import subprocess

from labm8 import app
from labm8 import bazelutil

app.DEFINE_string('nvidia_graph_db', None, 'Path of the NVIDIA dataset graphs')
app.DEFINE_string('amd_graph_db', None, 'Path of the AMD dataset graphs')
app.DEFINE_string('log_db', None, 'Path of the log database')
app.DEFINE_string('bytecode_db', None, 'Path of the bytecode database')
app.DEFINE_string('model', None, 'Name of the model to run. One of '
                  '{zero_r,lstm,ggnn')
app.DEFINE_string('working_dir', '/var/phd/ml4pl/models',
                  'Path of the working directory')
app.DEFINE_list('groups', [str(x) for x in range(10)],
                'The test groups to use.')

FLAGS = app.FLAGS

# Paths of the model binaries.
ZERO_R = bazelutil.DataPath('phd/deeplearning/ml4pl/models/zero_r/zero_r')
LSTM = bazelutil.DataPath(
    'phd/deeplearning/ml4pl/models/lstm/lstm_graph_classifier')
GGNN = bazelutil.DataPath('phd/deeplearning/ml4pl/models/ggnn/ggnn')


def GetModelCommandFromFlagsOrDie(graph_db: str, val_group: str,
                                  test_group: str):
  if not FLAGS.log_db:
    app.FatalWithoutStackTrace("--log_db must be set")
  if not FLAGS.working_dir:
    app.FatalWithoutStackTrace("--working_dir must be set")

  base_flags = [
      '--log_db',
      FLAGS.log_db,
      '--graph_db',
      graph_db,
      '--working_dir',
      FLAGS.working_dir,
      '--batch_scores_averaging_method',
      'binary',
      '--test_group',
      test_group,
      '--val_group',
      val_group,
  ]

  if FLAGS.model == 'zero_r':
    return [
        ZERO_R,
        '--num_epochs',
        '1',
        '--batch_size',
        '100000',
    ] + base_flags
  elif FLAGS.model == 'lstm':
    if not FLAGS.bytecode_db:
      app.FatalWithoutStackTrace("--bytecode_db must be set")
    return [
        LSTM,
        '--num_epochs',
        '100',
        '--bytecode_db',
        FLAGS.bytecode_db,
        '--input_sequence_len',
        '10000',
        '--hidden_size',
        '64',
    ]
  elif FLAGS.model == 'ggnn':
    return [
        GGNN,
        '--num_epochs',
        '100',
        '--graph_state_dropout_keep_prob',
        '.5',
        '--output_layer_dropout_keep_prob',
        '.5',
        '--edge_weight_dropout_keep_prob',
        '.5',
    ]
  else:
    app.FatalWithoutStackTrace('Unknown model name `%s`', FLAGS.model)


def RunKFoldOnGraphsOrDie(graph_db: str):
  for test_group in FLAGS.groups:
    test_group_as_num = int(test_group)
    assert 10 > test_group_as_num >= 0
    val_group = str((test_group_as_num + 1) % 10)
    cmd = GetModelCommandFromFlagsOrDie(graph_db, val_group, test_group)
    app.Log(1, '$ %s', ' '.join(cmd))
    subprocess.check_call(cmd)


def main():
  """Main entry point."""
  if FLAGS.nvidia_graph_db:
    RunKFoldOnGraphsOrDie(FLAGS.nvidia_graph_db)
  if FLAGS.amd_graph_db:
    RunKFoldOnGraphsOrDie(FLAGS.amd_graph_db)


if __name__ == '__main__':
  app.Run(main)
