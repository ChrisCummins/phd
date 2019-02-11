"""Import kernels from GPGPU benchmark suites."""
import typing

from absl import app
from absl import flags

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import ncc
from experimental.deeplearning.clgen.closeness_to_grewe_features import \
  grewe_features_db


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'db',
    'sqlite:///tmp/phd/experimental/deplearning/clgen/closeness_to_grewe_features/db.db',
    'URL of the database to import OpenCL kernels to.')


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  db = grewe_features_db.Database(FLAGS.db)

  df = opencl_device_mapping_dataset.OpenClDeviceMappingsDataset()
  with ncc.DEEPTUNE_INST2VEC_DATA_ARCHIVE as datafolder:
    src_paths = list(set(
        ncc.DataFrameRowToKernelSrcPath(row, datafolder) for _, row in
        df.iterrows()))

    db.ImportFromPaths(src_paths, 'benchmarks')


if __name__ == '__main__':
  app.run(main)
