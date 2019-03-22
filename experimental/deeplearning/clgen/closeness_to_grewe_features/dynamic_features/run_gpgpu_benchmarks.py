"""Run kernels in features database using CGO'17 driver and settings."""
import typing

from datasets.benchmarks.gpgpu import gpgpu
from datasets.benchmarks.gpgpu import gpgpu_pb2
from experimental.deeplearning.clgen.closeness_to_grewe_features import \
  grewe_features_db as db
from gpu.cldrive.legacy import env as cldrive_env
from labm8 import app
from labm8 import sqlutil

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


def GetBenchmarkSuiteToRun(
    session: sqlutil.Session,
    env: cldrive_env.OpenCLEnvironment) -> typing.Optional[str]:
  """Get the name of a benchmark suite to run."""
  already_done = session.query(db.DynamicFeatures.static_features_id) \
    .filter(db.DynamicFeatures.opencl_env == env.name,
            db.DynamicFeatures.driver == db.DynamicFeaturesDriver.LIBCECL)
  origin = session.query(db.StaticFeatures.origin) \
    .filter(db.StaticFeatures.origin.like('benchmarks_%'))\
    .filter(~db.StaticFeatures.id.in_(already_done)).first()

  if origin:
    benchmark_suite = origin[0][len('benchmarks_'):].split(':')[0]
    app.Log(1, 'Found benchmark suite to run: %s', benchmark_suite)
    return benchmark_suite


class DatabaseObserver(gpgpu.BenchmarkRunObserver):
  """Converter from GPGPU protos to DynamicFeatures database entries."""

  def __init__(self, database: db.Database):
    self.session = database.MakeSession()
    self.records = []
    self.origin_to_features_id_map = {}

  def OnBenchmarkRun(self, log: gpgpu_pb2.GpgpuBenchmarkRun):
    for kernel_invocation in log.run.kernel_invocation:
      # Re-construct the benchmark origin name. This is the inverse of the
      # origin field creation in
      # //experimental/deeplearning/clgen/closeness_to_grewe_features/static_features:import_from_gpgpu_benchmarks.
      origin = (f'benchmarks_{log.benchmark_suite}:{log.benchmark_name}:'
                f'{log.dataset_name}')
      app.Log(1, 'Looking up static features id for origin `%s`', origin)
      if origin in self.origin_to_features_id_map:
        static_features_id - self.origin_to_features_id_map[origin]
      else:
        static_features_id = self.session.query(db.StaticFeatures.static_features_id) \
          .filter(db.StaticFeatures.origin == origin).one()
        origin_to_features_id_map[origin] = static_features_id

      self.records.append(
          db.DynamicFeatures(
              driver=db.DynamicFeaturesDriver.LIBCECL,
              static_features_id=self.origin_to_features_id_map[origin],
              opencl_env=log.run.device.name,
              hostname=log.hostname,
              outcome='PASS',
              gsize=kernel_invocation.global_size,
              wgsize=kernel_invocation.local_size,
              transferred_bytes=kernel_invocation.transferred_bytes,
              transfer_time_ns=kernel_invocation.transfer_time_ns,
              kernel_time_ns=kernel_invocation.kernel_time_ns,
          ))

    return True

  def CommitRecords(self):
    app.Log(1, 'Commiting %d benchmark records', len(self.records))
    self.session.add_all(self.records)
    self.session.commit()
    self.session.close()
    self.records = []


def DriveBenchmarkSuiteAndRecordResults(
    database: db.Database, benchmark_suite_name: str,
    env: cldrive_env.OpenCLEnvironment, num_runs: int) -> None:
  """Drive a single kernel and record results."""
  benchmark_suite_class = gpgpu.ResolveBenchmarkSuiteClassesFromNames(
      [benchmark_suite_name])[0]
  observer = DatabaseObserver(database)
  with benchmark_suite_class() as benchmark_suite:
    app.Log(1, 'Building and running %s', benchmark_suite.name)
    benchmark_suite.ForceOpenCLEnvironment(env)
    for i in range(num_runs):
      app.Log(1, 'Starting run %d of %s', i + 1, benchmark_suite.name)
      benchmark_suite.Run([observer])
  observer.CommitRecords()


def main():
  """Main entry point."""
  database = db.Database(FLAGS.db)
  env = cldrive_env.OpenCLEnvironment.FromName(FLAGS.env)

  while True:
    with database.Session() as session:
      benchmark_suite = GetBenchmarkSuiteToRun(session, env)
    if not benchmark_suite:
      app.Log(1, 'Done. Nothing more to run!')
      return
    DriveBenchmarkSuiteAndRecordResults(database, benchmark_suite, env,
                                        FLAGS.num_runs)


if __name__ == '__main__':
  app.Run(main)
