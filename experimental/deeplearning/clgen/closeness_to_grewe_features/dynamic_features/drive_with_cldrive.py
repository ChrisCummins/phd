"""Run kernels in features database using CGO'17 driver and settings."""
import collections
import typing

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
app.DEFINE_integer('batch_size', 1024,
                   'The number of kernels to process at a time.')
app.DEFINE_boolean(
    'random_order', True,
    'Select kernels to run in a random order. Randomizing the order can be '
    'slow for large databases, but is useful when you have multiple '
    'concurrent workers to prevent races.')

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

  q = q.limit(batch_size)
  return [KernelToDrive(*row) for row in q]


def DriveKernelAndRecordResults(
    database: db.Database, static_features_id: int, src: str,
    env: cldrive_env.OpenCLEnvironment,
    dynamic_params: typing.List[cldrive_pb2.DynamicParams],
    num_runs: int) -> None:
  """Drive a single kernel and record results."""
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
        session.add(
            db.DynamicFeatures(
                static_features_id=static_features_id,
                driver=db.DynamicFeaturesDriver.CLDRIVE,
                opencl_env=env.name,
                hostname=system.HOSTNAME,
                outcome='NO_KERNELS',
                run_count=0,
            ))
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

    # Aggregate runtimes and append run_count.
    group_by_columns = ['opencl_env', 'gsize', 'wgsize']
    run_counts = df.group_by(group_by_columns).count()['kernel_time_ns']
    df = df.group_by(group_by_columns).mean()
    df['run_count'] = run_counts

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
      session.add(
          db.DynamicFeatures(
              static_features_id=static_features_id,
              driver=db.DynamicFeaturesDriver.CLDRIVE,
              opencl_env=env.name,
              hostname=system.HOSTNAME,
              outcome='DRIVER_CRASH',
          ))
    app.Log(1, 'Driver crashed')
  except pbutil.ProtoWorkerTimeoutError:
    with database.Session(commit=True) as session:
      session.add(
          db.DynamicFeatures(
              static_features_id=static_features_id,
              driver=db.DynamicFeaturesDriver.CLDRIVE,
              opencl_env=env.name,
              hostname=system.HOSTNAME,
              outcome='DRIVER_TIMEOUT',
          ))
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
