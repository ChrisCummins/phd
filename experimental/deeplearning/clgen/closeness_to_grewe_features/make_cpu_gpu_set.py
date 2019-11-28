"""Script to produce a labelled CPU/GPU mapping dataset."""
import sqlalchemy as sql

from experimental.deeplearning.clgen.closeness_to_grewe_features import \
  grewe_features_db
from labm8.py import app
from labm8.py import prof

FLAGS = app.FLAGS

app.DEFINE_string(
    'db',
    'sqlite:////tmp/phd/experimental/deplearning/clgen/closeness_to_grewe_features/db',
    'URL of the database to load static features from, and store dynamic '
    'features to.')
app.DEFINE_string(
    'name', None,
    'The name of the CPU/GPU dataset. This name is used to group.')
app.DEFINE_string(
    'cpu', None,
    'The opencl_env name of the CPU device. To see the list of available '
    'environments, run `SELECT DISTINCT(opencl_env) FROM dynamic_features`.')
app.DEFINE_string(
    'gpu', None,
    'The opencl_env name of the GPU device. To see the list of available '
    'environments, run `SELECT DISTINCT(opencl_env) FROM dynamic_features`.')
app.DEFINE_integer(
    'min_run_count', 30,
    'The minimum number of runs in order to include an aggregate in the data '
    'set')


def main():
  """Main entry point."""
  min_run_count = FLAGS.min_run_count
  cpu = FLAGS.cpu
  gpu = FLAGS.gpu
  dataset_name = FLAGS.name
  db = grewe_features_db.Database(FLAGS.db)

  with prof.Profile('query database'), db.Session(commit=True) as session:
    # The query that constructs the labelled dataset.
    query = db.CreateCpuGpuDataset(session, dataset_name, cpu, gpu,
                                   min_run_count)

    # Insert the results of the query into a table.
    insert = sql.insert(grewe_features_db.CpuGpuMappingSet).from_select(
        [column['name'] for column in query.column_descriptions], query)

    # Run the query.
    session.execute(insert)


if __name__ == '__main__':
  app.Run(main)
