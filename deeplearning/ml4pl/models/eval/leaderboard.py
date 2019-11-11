"""Print a summary table of model results."""
import pickle
import re

import pandas as pd
import sqlalchemy as sql

from deeplearning.ml4pl.models import log_database
from labm8 import app
from labm8 import humanize
from labm8 import pdutil

app.DEFINE_database('log_db',
                    log_database.Database,
                    None,
                    'The input log database.',
                    must_exist=True)
FLAGS = app.FLAGS


def GetProblemFromPickledGraphDbUrl(pickled_column_value: bytes):
  db_url = pickle.loads(pickled_column_value)
  if 'reachability' in db_url:
    return 'reachability'
  elif 'domtree' in db_url:
    return 'domtree'
  elif 'datadep' in db_url:
    return 'datadep'
  elif 'liveness' in db_url:
    return 'liveness'
  elif 'subexpressions' in db_url:
    return 'subexpressions'
  elif 'alias_set' in db_url:
    return 'alias_sets'
  elif 'devmap_amd' in db_url:
    return 'devmap_amd'
  elif 'devmap_nvidia' in db_url:
    return 'devmap_nvidia'
  else:
    raise ValueError(f"Could not interpret database URL '{db_url}'")


def GetBestEpochStats(session, df):
  index = []
  rows = []
  for run_id, row in df.iterrows():
    query = session.query(log_database.ModelCheckpointMeta.epoch)
    query = query.filter(log_database.ModelCheckpointMeta.run_id == run_id)
    query = query.filter(
        log_database.ModelCheckpointMeta.validation_accuracy == row['val_acc'])
    best_epoch = query.one().epoch

    query = session.query(
        sql.func.avg(log_database.BatchLogMeta.accuracy).label('test_acc'),
        sql.func.avg(log_database.BatchLogMeta.precision).label('precision'),
        sql.func.avg(log_database.BatchLogMeta.recall).label('recall'))
    query = query.filter(log_database.BatchLogMeta.run_id == run_id)
    query = query.filter(log_database.BatchLogMeta.epoch == best_epoch)
    test_acc = query.one().test_acc

    index.append(run_id)
    rows.append([best_epoch, test_acc])

  return pd.DataFrame(rows, index=index, columns=['best_epoch', 'test_acc'])


def GetLeaderboard(log_db: log_database.Database,
                   human_readable: bool = False) -> pd.DataFrame:
  """Compute a leaderboard."""
  with log_db.Session() as session:
    # Create a table with batch log stats.
    query = session.query(
        log_database.BatchLogMeta.run_id,
        sql.func.sum(
            log_database.BatchLogMeta.elapsed_time_seconds).label('runtime'),
        sql.func.max(log_database.BatchLogMeta.date_added).label('last_log'),
        sql.func.count(log_database.BatchLogMeta.run_id).label("batches"))
    query = query.group_by(log_database.BatchLogMeta.run_id)
    batch_df = pdutil.QueryToDataFrame(session, query)
    batch_df.set_index('run_id', inplace=True)

    # Create a table with model checkpoint stats.
    query = session.query(
        log_database.ModelCheckpointMeta.run_id,
        sql.func.count(
            log_database.ModelCheckpointMeta.run_id).label("checkpoints"),
        sql.func.max(
            log_database.ModelCheckpointMeta.validation_accuracy).label(
                "val_acc"))
    query = query.group_by(log_database.ModelCheckpointMeta.run_id)
    checkpoint_df = pdutil.QueryToDataFrame(session, query)
    checkpoint_df.set_index('run_id', inplace=True)

    # Create a table with the names of the graph databases.
    query = session.query(log_database.Parameter.run_id,
                          log_database.Parameter.pickled_value.label('problem'))
    query = query.filter(
        log_database.Parameter.type == log_database.ParameterType.FLAG)
    query = query.filter(log_database.Parameter.parameter ==
                         'deeplearning.ml4pl.models.classifier_base.graph_db')
    graph_df = pdutil.QueryToDataFrame(session, query)
    # Un-pickle the parameter values and extract the database names from between
    # the `?` delimiters.
    graph_df['problem'] = [
        GetProblemFromPickledGraphDbUrl(x) for x in graph_df['problem']
    ]
    graph_df.set_index('run_id', inplace=True)

    # Create a table with the names of the test groups.
    query = session.query(log_database.Parameter.run_id,
                          log_database.Parameter.pickled_value.label('test'))
    query = query.filter(
        log_database.Parameter.type == log_database.ParameterType.FLAG)
    query = query.filter(log_database.Parameter.parameter ==
                         'deeplearning.ml4pl.models.classifier_base.test_group')
    test_group_df = pdutil.QueryToDataFrame(session, query)
    test_group_df['test'] = [pickle.loads(x) for x in test_group_df['test']]
    test_group_df.set_index('run_id', inplace=True)

    # Create a table with the names of the val groups.
    query = session.query(log_database.Parameter.run_id,
                          log_database.Parameter.pickled_value.label('val'))
    query = query.filter(
        log_database.Parameter.type == log_database.ParameterType.FLAG)
    query = query.filter(log_database.Parameter.parameter ==
                         'deeplearning.ml4pl.models.classifier_base.val_group')
    val_group_df = pdutil.QueryToDataFrame(session, query)
    val_group_df['val'] = [pickle.loads(x) for x in val_group_df['val']]
    val_group_df.set_index('run_id', inplace=True)

    # Create a table with the names of the models.
    query = session.query(log_database.Parameter.run_id,
                          log_database.Parameter.pickled_value.label('model'))
    query = query.filter(
        log_database.Parameter.type == log_database.ParameterType.MODEL_FLAG)
    query = query.filter(log_database.Parameter.parameter == 'model')
    model_df = pdutil.QueryToDataFrame(session, query)
    # Un-pickle the parameter values and extract the database names from between
    # the `?` delimiters.
    model_df['model'] = [pickle.loads(x) for x in model_df['model']]
    model_df.set_index('run_id', inplace=True)

    df = graph_df.join(
        [model_df, test_group_df, val_group_df, batch_df, checkpoint_df])

    best_epoch_df = GetBestEpochStats(session, df)
    df = df.join(best_epoch_df)

    # Strip redundant suffix from model names.
    pdutil.RewriteColumn(df, 'model',
                         lambda x: re.sub(r'(Classifier|Model)$', '', x))

    # Rewrite columns to be more user friendly.
    pdutil.RewriteColumn(df, 'last_log', humanize.Time)
    if human_readable:
      pdutil.RewriteColumn(df, 'runtime', humanize.Duration)
      pdutil.RewriteColumn(df, 'val_acc', lambda x: f'{x:.2%}')
      pdutil.RewriteColumn(df, 'accuracy', lambda x: f'{x:.2%}')

    return df


def main():
  """Main entry point."""
  df = GetLeaderboard(FLAGS.log_db(), human_readable=True)
  print(pdutil.FormatDataFrameAsAsciiTable(df))


if __name__ == '__main__':
  app.Run(main)
