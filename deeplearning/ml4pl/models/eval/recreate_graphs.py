"""Recreate the input/output graphs for an epoch."""
import pathlib
import pickle

import sqlalchemy as sql

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.models import log_analysis
from deeplearning.ml4pl.models import log_database
from labm8 import app
from labm8 import prof

app.DEFINE_database('log_db',
                    log_database.Database,
                    None,
                    'The path of the log database.',
                    must_exist=True)
app.DEFINE_database('graph_db',
                    graph_database.Database,
                    None,
                    'The path of the graph database.',
                    must_exist=True)
app.DEFINE_output_path('outpath', '/tmp/phd/ml4pl/graphs.pickle',
                       'The destination file to write.')
app.DEFINE_string('run_id', None, 'The run ID')
app.DEFINE_string('epoch_type', 'test',
                  'The epoch type to reproduce the graphs of.')
app.DEFINE_integer('epoch_num', None,
                   'The epoch number to reproduce the graphs of.')
app.DEFINE_integer('max_logs', None,
                   'Limit the maximum number of logs to recreate.')
app.DEFINE_string('weighting', 'weighted',
                  'The {binary,weighted} scores weighting to use.')

FLAGS = app.FLAGS


def RecreateInputOutputGraphs(run: log_analysis.RunLogAnalyzer, epoch_num: int,
                              epoch_type: str, outpath: pathlib.Path,
                              weighting: str):
  batches = run.batch_logs[(run.batch_logs['epoch'] == epoch_num) &
                           (run.batch_logs['type'] == epoch_type)]

  with prof.Profile(lambda t: f"Read {len(logs)} logs"):
    with run.log_db.Session() as session:
      query = session.query(log_database.BatchLogMeta)
      query = query.filter(log_database.BatchLogMeta.run_id == run.run_id)
      query = query.filter(
          log_database.BatchLogMeta.global_step.in_(batches.global_step))
      query = query.options(
          sql.orm.joinedload(log_database.BatchLogMeta.batch_log))

      if FLAGS.max_logs:
        query = query.limit(FLAGS.max_logs)

      logs = query.all()

  if not logs:
    raise OSError("No logs found.")

  input_output_graphs = []
  with prof.Profile(
      lambda t: f"Reconstructed {len(input_output_graphs)} graphs"):
    for log in logs:
      input_output_graphs += run.GetInputOutputGraphsFromLog(log)

  with prof.Profile("Annotate graphs"):
    for input_graph, output_graph in input_output_graphs:
      log_analysis.ComputeGraphAccuracy(input_graph, output_graph, weighting)

  with prof.Profile(f"Wrote {outpath}"):
    with open(outpath, 'wb') as f:
      pickle.dump(input_output_graphs, f)


def main():
  """Main entry point."""
  if not FLAGS.log_db:
    raise app.UsageError("--log_db not set")
  if not FLAGS.graph_db:
    raise app.UsageError("--graph_db not set")
  if not FLAGS.run_id:
    raise app.UsageError("--run_id not set")
  if not FLAGS.epoch_num:
    raise app.UsageError("--epoch_num not set")

  FLAGS.outpath.parent.mkdir(exist_ok=True, parents=True)

  run = log_analysis.RunLogAnalyzer(FLAGS.graph_db(), FLAGS.log_db(),
                                    FLAGS.run_id)

  RecreateInputOutputGraphs(run, FLAGS.epoch_num, FLAGS.epoch_type,
                            FLAGS.outpath, FLAGS.weighting)


if __name__ == '__main__':
  app.Run(main)
