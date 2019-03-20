"""Run kernels in features database using CGO'17 driver and settings."""
import collections
import typing
import subprocess

from experimental.deeplearning.clgen.closeness_to_grewe_features import \
  grewe_features_db as db
from gpu.cldrive import api as cldrive
from gpu.cldrive.legacy import env as cldrive_env
from gpu.cldrive.proto import cldrive_pb2
from labm8 import app
from labm8 import pbutil
from labm8 import sqlutil
from labm8 import system
from labm8 import prof
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
app.DEFINE_integer('batch_size', 16,
                   'The number of kernels to process at a time.')

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
    .filter(db.DynamicFeatures.opencl_env == env.name)
  q = session.query(
      db.StaticFeatures.id, db.StaticFeatures.src) \
    .filter(~db.StaticFeatures.id.in_(already_done)) \
    .limit(batch_size)
  return [KernelToDrive(*row) for row in q]


def DriveKernelAndRecordResults(
    session: sqlutil.Session, static_features_id: int, src: str,
    env: cldrive_env.OpenCLEnvironment, num_runs: int) -> None:
  """Drive a single kernel and record results."""
  try:
    df = cldrive.DriveToDataFrame(
        cldrive_pb2.CldriveInstances(instance=[
            cldrive_pb2.CldriveInstance(
                device=env.proto,
                opencl_src=src,
                dynamic_params=LSIZE_GSIZE_PROTO_PAIRS,
                min_runs_per_kernel=num_runs,
            )
        ]))
    dynamic_features = [
        db.DynamicFeatures.FromCldriveDataFrameRecord(df.iloc[i],
                                                      static_features_id)
        for i in range(len(df))
    ]
    session.add_all(dynamic_features)
    app.Log(1, 'Imported %d dynamic features', len(dynamic_features))
  except subprocess.CalledProcessError:
    session.add(
        db.DynamicFeatures(
            static_features_id=static_features_id,
            opencl_env=env.name,
            hostname=system.HOSTNAME,
            outcome='DRIVER_CRASH',
        ))
    app.Log(1, 'Driver crashed')
  except pbutil.ProtoWorkerTimeoutError:
    session.add(
        db.DynamicFeatures(
            static_features_id=static_features_id,
            opencl_env=env.name,
            hostname=system.HOSTNAME,
            outcome='DRIVER_TIMEOUT',
        ))
    app.Log(1, 'Driver timed out')
  finally:
    session.commit()


def DriveBatchAndRecordResults(session: sqlutil.Session,
                               batch: typing.List[KernelToDrive],
                               env: cldrive_env.OpenCLEnvironment) -> None:
  """Drive a batch of kernels and record dynamic features."""
  # Irrespective of batch size we still run each program in the batch as
  # separate cldrive instance.
  for static_features_id, src in batch:
    with prof.Profile('Runs statif features ID {static_features_id}}'):
      DriveKernelAndRecordResults(session, static_features_id, src, env,
                                  FLAGS.num_runs)


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
      batch = GetBatchOfKernelsToDrive(session, env, FLAGS.batch_size)
      if not batch:
        app.Log(1, 'Done. Nothing more to run!')
        return

      DriveBatchAndRecordResults(session, batch, env)


if __name__ == '__main__':
  app.RunWithArgs(main)
