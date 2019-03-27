"""Run kernels in features database using CGO'17 driver and settings."""
import collections
import typing

import numpy as np
import sqlalchemy as sql

from experimental.deeplearning.clgen.closeness_to_grewe_features import \
  grewe_features_db as db
from gpu.cldrive import api as cldrive
from gpu.cldrive.legacy import env as cldrive_env
from gpu.cldrive.proto import cldrive_pb2
from labm8 import app
from labm8 import pbutil
from labm8 import prof
from labm8 import sqlutil
from labm8 import system
from research.cummins_2017_cgo import opencl_kernel_driver

FLAGS = app.FLAGS

app.DEFINE_string(
    'db',
    'sqlite:///tmp/phd/experimental/deplearning/clgen/closeness_to_grewe_features/db',
    'URL of the database to load static features from, and store dynamic '
    'features to.')
app.DEFINE_string(
    'env', 'Emulator|Oclgrind|Oclgrind_Simulator|Oclgrind_18.3|1.2',
    'The OpenCL environment to execute benchmark suites on. To list the '
    'available environments, run `bazel run //gpu/clinfo`.')
app.DEFINE_integer('num_runs', 30, 'The number of runs for each benchmark.')
app.DEFINE_integer('cldrive_timeout_seconds', 60,
                   'The number of seconds to allow cldrive to run for.')
app.DEFINE_integer(
    'batch_size', 2048,
    'The number of kernels to process at a time. A larger batch_size reduces '
    'the frequency of full table queries, which can be expensive for large '
    'datasets. However, if running multiple concurrent workers with random '
    'orders, a larger batch_size increases the chances of workers performing '
    'redundant work.')
app.DEFINE_boolean(
    'random_order', True,
    'Select kernels to run in a random order. Randomizing the order can be '
    'slow for large databases, but is useful when you have multiple '
    'concurrent workers to prevent races.')
app.DEFINE_boolean(
    'reverse_order', False,
    'Select kernels to run in reverse order of their static features ID. If '
    '--norandom_order, results are ordered by their static features ID in '
    'ascending values. This flag reverses that order.')

KernelToDrive = collections.namedtuple('KernelToDrive', ['id', 'src'])

# Use the same combinations of local and global sizes as in the CGO'17 paper.
LSIZE_GSIZE_PROTO_PAIRS = [
    cldrive_pb2.DynamicParams(global_size_x=y, local_size_x=x)
    for x, y in opencl_kernel_driver.LSIZE_GSIZE_PAIRS
]


def GetBatchOfKernelsToDrive(session: sqlutil.Session,
                             env: cldrive_env.OpenCLEnvironment,
                             batch_size: int):
  """Get a batch of kernels to run."""
  already_done = session.query(db.DynamicFeatures.static_features_id) \
    .filter(db.DynamicFeatures.opencl_env == env.name,
            db.DynamicFeatures.driver == db.DynamicFeaturesDriver.CLDRIVE)
  q = session.query(
      db.StaticFeatures.id, db.StaticFeatures.src) \
    .filter(~db.StaticFeatures.id.in_(already_done))

  if FLAGS.random_order:
    q = q.order_by(sql.func.random())
  elif FLAGS.reverse_order:
    q = q.order_by(db.DynamicFeatures.static_features_id.desc())

  q = q.limit(batch_size)
  return [KernelToDrive(*row) for row in q]


def DriveKernelAndRecordResults(
    database: db.Database, static_features_id: int, src: str,
    env: cldrive_env.OpenCLEnvironment,
    dynamic_params: typing.List[cldrive_pb2.DynamicParams],
    num_runs: int) -> None:
  """Drive a single kernel and record results."""

  def ErrorFeatures(outcome: str) -> db.DynamicFeatures:
    return db.DynamicFeatures(
        static_features_id=static_features_id,
        driver=db.DynamicFeaturesDriver.CLDRIVE,
        opencl_env=env.name,
        hostname=system.HOSTNAME,
        outcome=outcome,
        run_count=0,
    )

  try:
    df = cldrive.DriveToDataFrame(
        cldrive_pb2.CldriveInstances(instance=[
            cldrive_pb2.CldriveInstance(
                device=env.proto,
                opencl_src=src,
                dynamic_params=dynamic_params,
                min_runs_per_kernel=num_runs,
            )
        ]),
        timeout_seconds=FLAGS.cldrive_timeout_seconds)

    # Record programs which contain no kernels.
    if not len(df):
      with database.Session(commit=True) as session:
        session.add(ErrorFeatures('NO_KERNELS'))
      return

    # Remove the columns which are not exported to the database:
    # 'instance' is not used since we only drive a single instance at a time.
    # 'build_opts' is never changed. 'kernel' is not needed because each static
    # features entry is a single kernel.
    df.drop(columns=['instance', 'build_opts', 'kernel'], inplace=True)

    # Fix the naming differences between cldrive and the database.
    df.rename(
        columns={
            'device': 'opencl_env',
            'global_size': 'gsize',
            'local_size': 'wgsize'
        },
        inplace=True)

    # NaN values are excluded in groupby statements, and we need to groupby
    # columns that may be NaN (gsize and wgsize). Replace NaN with -1 since all
    # integer column values are >= 0, so this value will never occur normally.
    # See: https://github.com/pandas-dev/pandas/issues/3729
    nan_placeholder = -1
    df[['gsize', 'wgsize']] = df[['gsize', 'wgsize']].fillna(nan_placeholder)

    # Aggregate runtimes and append run_count.
    groupby_columns = ['opencl_env', 'gsize', 'wgsize', 'outcome']
    run_counts = df.groupby(groupby_columns).count()['kernel_time_ns']
    df = df.groupby(groupby_columns).mean()
    df['run_count'] = run_counts
    df.reset_index(inplace=True)

    # Now that we have done the groupby, replace the NaN placeholder values
    # with true NaN.
    df[['gsize', 'wgsize']] = df[['gsize', 'wgsize']].replace(
        nan_placeholder, np.nan)

    # Add missing columns.
    df['static_features_id'] = static_features_id
    df['driver'] = db.DynamicFeaturesDriver.CLDRIVE
    df['hostname'] = system.HOSTNAME

    # Import the dataframe into the SQL table.
    df.to_sql(
        db.DynamicFeatures.__tablename__,
        con=database.engine,
        if_exists='append',
        index=False,
        dtype={'driver': sql.Enum(db.DynamicFeaturesDriver)})
    app.Log(1, 'Imported %d dynamic features', len(df))
  except cldrive.CldriveCrash:
    with database.Session(commit=True) as session:
      session.add(ErrorFeatures('DRIVER_CRASH'))
    app.Log(1, 'Driver crashed')
  except pbutil.ProtoWorkerTimeoutError:
    with database.Session(commit=True) as session:
      session.add(ErrorFeatures('DRIVER_TIMEOUT'))
    app.Log(1, 'Driver timed out')


def DriveBatchAndRecordResults(database: db.Database,
                               batch: typing.List[KernelToDrive],
                               env: cldrive_env.OpenCLEnvironment) -> None:
  """Drive a batch of kernels and record dynamic features."""
  # Irrespective of batch size we still run each program in the batch as
  # separate cldrive instance.
  for static_features_id, src in batch:
    with prof.Profile(f'Run static features ID {static_features_id}'):
      DriveKernelAndRecordResults(database, static_features_id, src, env,
                                  LSIZE_GSIZE_PROTO_PAIRS, FLAGS.num_runs)


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  database = db.Database(FLAGS.db)
  env = cldrive_env.OpenCLEnvironment.FromName(FLAGS.env)

  batch_num = 0
  while True:
    batch_num += 1
    with database.Session() as session, prof.Profile(f'Batch {batch_num}'):
      with prof.Profile(f'Get batch of {FLAGS.batch_size} kernels'):
        batch = GetBatchOfKernelsToDrive(session, env, FLAGS.batch_size)
    if not batch:
      app.Log(1, 'Done. Nothing more to run!')
      return

    DriveBatchAndRecordResults(database, batch, env)


if __name__ == '__main__':
  app.RunWithArgs(main)
