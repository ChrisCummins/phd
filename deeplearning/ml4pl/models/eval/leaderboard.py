"""Print a summary table of model results."""
import io
import pickle
import re
import typing

import numpy as np
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
app.DEFINE_string('format', 'txt',
                  'The format to print the result table. One of {txt,csv}')
app.DEFINE_boolean('human_readable', True,
                   'Format the column data in a human-readable format.')
app.DEFINE_list('extra_model_flags', [], 'Additional model flags to print.')
app.DEFINE_list('extra_flags', [], 'Additional flags to print.')
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


def GetBestEpochStats(session, df) -> typing.Optional[pd.DataFrame]:
  """Fetch the acc/prec/recall stats for the best epoch."""
  index = []
  rows = []
  for run_id, row in df.iterrows():
    # Skip runs for which there is no checkpoint data.
    if np.isnan(row['val_acc']):
      continue

    # Find the "best" epoch - the one which produced the best validation
    # accuracy.
    query = session.query(log_database.ModelCheckpointMeta.epoch)
    query = query.filter(log_database.ModelCheckpointMeta.run_id == run_id)
    # If multiple epochs produced the same validation accuracy, select the
    # first.
    query = query.order_by(log_database.ModelCheckpointMeta.epoch)
    best_epoch = query.first().epoch

    # Aggregate performance on the test set at the best epoch.
    query = session.query(
        log_database.BatchLogMeta.type,
        sql.func.avg(log_database.BatchLogMeta.accuracy).label('accuracy'),
        sql.func.avg(log_database.BatchLogMeta.precision).label('precision'),
        sql.func.avg(log_database.BatchLogMeta.recall).label('recall'))
    query = query.filter(log_database.BatchLogMeta.run_id == run_id,
                         log_database.BatchLogMeta.epoch == best_epoch)
    query = query.group_by(log_database.BatchLogMeta.run_id,
                           log_database.BatchLogMeta.epoch,
                           log_database.BatchLogMeta.type)
    results = {row[0]: row[1:] for row in query}

    index.append(run_id)
    column_names = ['best_epoch']
    column_values = [best_epoch]
    for type_ in ['train', 'val', 'test']:
      column_names.extend([f'{type_}_acc', f'{type_}_prec', f'{type_}_rec'])
      column_values.extend(results.get(type_, ['-', '-', '-']))
    rows.append(column_values)

  if not rows:
    return None

  # Now that we are done we can drop the duplicate validation accuracy.
  df.drop(columns=['val_acc'], inplace=True)

  return pd.DataFrame(rows, index=index, columns=column_names)


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
        sql.func.count(log_database.BatchLogMeta.run_id).label("batches"),
        sql.func.max(log_database.BatchLogMeta.epoch).label("epoch_count"))
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

    # Add extra model flags.
    for flag in FLAGS.extra_model_flags:
      query = session.query(log_database.Parameter.run_id,
                            log_database.Parameter.pickled_value.label(flag))
      query = query.filter(
          sql.func.lower(log_database.Parameter.type) == 'model_flag')
      query = query.filter(log_database.Parameter.parameter == flag)
      flag_df = pdutil.QueryToDataFrame(session, query)
      # Un-pickle flag value.
      pdutil.RewriteColumn(flag_df, flag, lambda x: pickle.loads(x))
      flag_df.set_index('run_id', inplace=True)
      graph_df = graph_df.join(flag_df)

    # Add extra flags.
    for flag in FLAGS.extra_flags:
      # Strip the fully qualified flag name, e.g. "foo.bar.flag" -> "flag".
      flag_name = flag.split('.')[-1]
      query = session.query(
          log_database.Parameter.run_id,
          log_database.Parameter.pickled_value.label(flag_name))
      query = query.filter(
          log_database.Parameter.type == log_database.ParameterType.FLAG)
      query = query.filter(log_database.Parameter.parameter == flag)
      flag_df = pdutil.QueryToDataFrame(session, query)
      # Un-pickle flag values.
      pdutil.RewriteColumn(flag_df, flag_name, lambda x: pickle.loads(x))
      flag_df.set_index('run_id', inplace=True)
      graph_df = graph_df.join(flag_df)

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
    if best_epoch_df is not None:
      df = df.join(best_epoch_df)

    # Strip redundant suffix from model names.
    pdutil.RewriteColumn(
        df, 'model',
        lambda s: re.sub(r'(Classifier|ClassifierModel|Model)$', '', s))

    def Time(x):
      """Humanize or default to '-' on failure."""
      try:
        return humanize.Time(x)
      except:
        return '-'

    def Duration(x):
      """Humanize or default to '-' on failure."""
      try:
        return humanize.Duration(x)
      except:
        return '-'

    def Percent(x):
      try:
        return f'{x:.2%}'
      except:
        return '-'

    def Float(x):
      try:
        return f'{x:.3f}'
      except:
        return '-'

    df.fillna('-', inplace=True)

    # Rewrite columns to be more user friendly.
    if human_readable:
      pdutil.RewriteColumn(df, 'last_log', Time)
      pdutil.RewriteColumn(df, 'runtime', Duration)
      pdutil.RewriteColumn(df, 'train_acc', Percent)
      pdutil.RewriteColumn(df, 'val_acc', Percent)
      pdutil.RewriteColumn(df, 'test_acc', Percent)
      pdutil.RewriteColumn(df, 'train_prec', Percent)
      pdutil.RewriteColumn(df, 'val_prec', Percent)
      pdutil.RewriteColumn(df, 'test_prec', Percent)
      pdutil.RewriteColumn(df, 'train_rec', Percent)
      pdutil.RewriteColumn(df, 'val_rec', Percent)
      pdutil.RewriteColumn(df, 'test_rec', Percent)

    return df


def main():
  """Main entry point."""
  df = GetLeaderboard(FLAGS.log_db(), human_readable=FLAGS.human_readable)
  if FLAGS.format == 'csv':
    buf = io.StringIO()
    df.to_csv(buf)
    print(buf.getvalue())
  elif FLAGS.format == 'txt':
    print(pdutil.FormatDataFrameAsAsciiTable(df))
  else:
    raise app.UsageError(f"Unknown --format='{FLAGS.format}'")


if __name__ == '__main__':
  app.Run(main)
