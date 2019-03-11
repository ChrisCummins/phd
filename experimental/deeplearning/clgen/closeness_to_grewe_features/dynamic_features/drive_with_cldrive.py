"""Run kernels in features database using CGO'17 driver and settings."""
import collections
import typing

from experimental.deeplearning.clgen.closeness_to_grewe_features import \
  grewe_features_db
from gpu.cldrive import api as cldrive
from gpu.cldrive.legacy import env as cldrive_env
from gpu.cldrive.proto import cldrive_pb2
from labm8 import app
from labm8 import pbutil
from labm8 import system

FLAGS = app.FLAGS

app.DEFINE_string(
    'db',
    'sqlite:///tmp/phd/experimental/deplearning/clgen/closeness_to_grewe_features/db.db',
    'URL of the database to load and store results to.')
app.DEFINE_string(
    'env', 'Emulator|Oclgrind|Oclgrind_Simulator|Oclgrind_18.3|1.2',
    'The OpenCL environment to execute benchmark suites on. To list the '
    'available environments, run `bazel run //gpu/clinfo`.')
app.DEFINE_integer('num_runs', 30, 'The number of runs for each benchmark.')
app.DEFINE_integer('batch_size', 16,
                   'The number of kernels to process at a time.')

KernelToDrive = collections.namedtuple('KernelToDrive', ['id', 'src'])

# All the combinations of local and global sizes used for synthetic kernels in
# the CGO'17 experiments. These are the first dimension values, the other two
# dimensions are ones. E.g. the tuple (64, 128) means a local (workgroup) size
# of (64, 1, 1), and a global size of (128, 1, 1).
LSIZE_GSIZE_PAIRS = [
    (64, 64),
    (128, 128),
    (256, 256),
    (256, 512),
    (256, 1024),
    (256, 2048),
    (256, 4096),
    (256, 8192),
    (256, 16384),
    (256, 65536),
    (256, 131072),
    (256, 262144),
    (256, 524288),
    (256, 1048576),
    (256, 2097152),
    (256, 4194304),
]

LSIZE_GSIZE_PROTO_PAIRS = [
    cldrive_pb2.DynamicParams(global_size_x=y, local_size_x=x)
    for x, y in LSIZE_GSIZE_PAIRS
]


def GetBatchOfKernelsToDrive(db: grewe_features_db.Database,
                             env: cldrive_env.OpenCLEnvironment):
  with db.Session(commit=False) as session:
    for lsize, gsize in LSIZE_GSIZE_PAIRS:
      already_done_ids = session.query(
          grewe_features_db.DriverResult.static_features_id) \
        .filter(grewe_features_db.DriverResult.opencl_env == env.name)

      # TODO(cec): Why exclude benchmarks from cldrive?
      q = session.query(grewe_features_db.StaticFeatures.id,
                        grewe_features_db.StaticFeatures.src) \
        .filter(~grewe_features_db.StaticFeatures.id.in_(already_done_ids)) \
        .filter(grewe_features_db.StaticFeatures.origin != 'benchmarks') \
        .limit(FLAGS.batch_size)

      if q.count():
        return [KernelToDrive(*row) for row in q]


def DriveBatchAndRecordResults(db: grewe_features_db.Database,
                               batch: typing.List[KernelToDrive],
                               env: cldrive_env.OpenCLEnvironment) -> None:
  """Drive a batch of kernels and record dynamic features."""
  try:
    instances = cldrive.DriveToDataFrame(
        cldrive_pb2.CldriveInstances(instance=[
            cldrive_pb2.CldriveInstance(
                device=env.proto,
                opencl_src=src,
                min_runs_per_kernel=FLAGS.num_runs,
                dynamic_params=LSIZE_GSIZE_PROTO_PAIRS,
            ) for _, src in batch
        ]))
    print(instances)
    import sys
    sys.exit(0)
    if len(instances.instance) != len(batch):
      raise OSError(f"Number of instances ({len(instances.instance)}) != "
                    f"batch size ({len(batch)})")

    for (static_features_id, _), instance in zip(batch, instances.instance):
      with db.Session(commit=True) as session:
        if len(instance.kernel) < 1:
          session.add(
              grewe_features_db.DriverResult(
                  static_features_id=static_features_id,
                  opencl_env=env.name,
                  hostname=system.HOSTNAME,
                  result=cldrive_pb2.CldriveInstance.InstanceOutcome.Name(
                      instance.outcome),
              ))
        else:
          if len(instance.kernel) != 1:
            raise OSError(f"{instance.kernel} kernels found!")

          result = cldrive_pb2.CldriveInstance.InstanceOutcome.Name(
              instance.outcome)
          if result == 'PASS':
            result = cldrive_pb2.CldriveKernelInstance.KernelInstanceOutcome.Name(
                instance.kernel[0].outcome)

          session.add(
              grewe_features_db.DriverResult(
                  static_features_id=static_features_id,
                  opencl_env=env.name,
                  hostname=system.HOSTNAME,
                  result=result,
              ))

          for run in instance.kernel[0].run:
            session.add_all([
                grewe_features_db.DynamicFeatures(
                    static_features_id=static_features_id,
                    opencl_env=env.name,
                    hostname=system.HOSTNAME,
                    dataset=f'{log.global_size},{log.local_size}',
                    gsize=log.global_size,
                    wgsize=log.local_size,
                    transferred_bytes=log.transferred_bytes,
                    runtime_ms=log.runtime_ms,
                ) for log in run.log
            ])
  except pbutil.ProtoWorkerTimeoutError:
    with db.Session(commit=True) as session:
      session.add_all([
          grewe_features_db.DriverResult(
              static_features_id=static_features_id,
              opencl_env=env.name,
              hostname=system.HOSTNAME,
              result='DRIVER_TIMEOUT',
          ) for (static_features_id, _) in batch
      ])


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  db = grewe_features_db.Database(FLAGS.db)
  env = cldrive_env.OpenCLEnvironment.FromName(FLAGS.env)

  batch_num = 0
  while True:
    batch_num += 1
    app.Info('Batch %d', batch_num)
    batch = GetBatchOfKernelsToDrive(db, env)
    if not batch:
      app.Info('Done. Nothing more to run!')
      return

    DriveBatchAndRecordResults(db, batch, env)


if __name__ == '__main__':
  app.RunWithArgs(main)
