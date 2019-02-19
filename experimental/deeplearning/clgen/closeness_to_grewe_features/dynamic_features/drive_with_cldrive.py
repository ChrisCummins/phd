"""Run kernels in features database using CGO'17 driver and settings."""
import collections
import typing

from absl import app
from absl import flags
from absl import logging

from experimental.deeplearning.clgen.closeness_to_grewe_features import \
  grewe_features_db
from gpu.cldrive import env as cldrive_env
from labm8 import system
from labm8 import text
from research.cummins_2017_cgo import opencl_kernel_driver


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'db',
    'sqlite:///tmp/phd/experimental/deplearning/clgen/closeness_to_grewe_features/db.db',
    'URL of the database to load and store results to.')
flags.DEFINE_string(
    'env', 'Emulator|Oclgrind|Oclgrind_Simulator|Oclgrind_18.3|1.2',
    'The OpenCL environment to execute benchmark suites on. To list the '
    'available environments, run `bazel run //gpu/clinfo`.')
flags.DEFINE_integer('num_runs', 30, 'The number of runs for each benchmark.')
flags.DEFINE_integer('batch_size', 16,
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


def GetBatchOfKernelsToDrive(db: grewe_features_db.Database,
                             env: cldrive_env.OpenCLEnvironment):
  with db.Session(commit=False) as session:
    for lsize, gsize in LSIZE_GSIZE_PAIRS:
      dataset = f'{lsize},{gsize}'
      already_done_ids = session.query(
          grewe_features_db.DriverResult.static_features_id) \
        .filter(grewe_features_db.DriverResult.opencl_env == env.name) \
        .filter(grewe_features_db.DriverResult.dataset == dataset)

      # TODO(cec): Why exclude benchmarks from cldrive?
      q = session.query(grewe_features_db.StaticFeatures.id,
                        grewe_features_db.StaticFeatures.src) \
        .filter(~grewe_features_db.StaticFeatures.id.in_(already_done_ids)) \
        .filter(grewe_features_db.StaticFeatures.origin != 'benchmarks') \
        .limit(FLAGS.batch_size)

      if q.count():
        return [KernelToDrive(*row) for row in q]


def DriveBatchAndRecordResults(
    db: grewe_features_db.Database,
    batch: typing.List[KernelToDrive],
    env: cldrive_env.OpenCLEnvironment) -> None:
  for static_features_id, opencl_kernel in batch:
    for lsize, gsize in LSIZE_GSIZE_PAIRS:
      dataset = f'{lsize},{gsize}'

      try:
        logs = opencl_kernel_driver.Drive(
            opencl_kernel, lsize_x=lsize, gsize_x=gsize, opencl_env=env,
            num_runs=FLAGS.num_runs)
        logging.info('%d:%s PASS', static_features_id, dataset)
        assert len(logs) == FLAGS.num_runs
        with db.Session(commit=True) as session:
          session.add(grewe_features_db.DriverResult(
              static_features_id=static_features_id,
              opencl_env=env.name,
              hostname=system.HOSTNAME,
              dataset=dataset,
              result='PASS',
          ))
          dynamic_features = [
            grewe_features_db.DynamicFeatures(
                static_features_id=static_features_id,
                opencl_env=env.name,
                hostname=system.HOSTNAME,
                dataset=dataset,
                gsize=log.kernel_invocation[0].global_size,
                wgsize=log.kernel_invocation[0].local_size,
                transferred_bytes=log.kernel_invocation[0].transferred_bytes,
                runtime_ms=log.kernel_invocation[0].runtime_ms,
            ) for log in logs
          ]
          session.add_all(dynamic_features)
      except opencl_kernel_driver.DriverFailure as e:
        with db.Session(commit=True) as session:
          failure_name = text.CamelCapsToUnderscoreSeparated(type(e).__name__)
          logging.info('%d:%s %s', static_features_id, dataset, failure_name)
          session.add(grewe_features_db.DriverResult(
              static_features_id=static_features_id,
              opencl_env=env.name,
              hostname=system.HOSTNAME,
              dataset=dataset,
              result=failure_name.upper(),
          ))


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  db = grewe_features_db.Database(FLAGS.db)
  env = cldrive_env.OpenCLEnvironment.FromName(FLAGS.env)

  batch_num = 0
  while True:
    batch_num += 1
    logging.info('Batch %d', batch_num)
    batch = GetBatchOfKernelsToDrive(db, env)
    if not batch:
      logging.info('Done. Nothing more to run!')
      return

    DriveBatchAndRecordResults(db, batch, env)


if __name__ == '__main__':
  app.run(main)
