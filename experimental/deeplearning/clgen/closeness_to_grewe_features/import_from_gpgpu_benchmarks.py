"""Import kernels from GPGPU benchmark suites."""
import hashlib
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


def RowToOpenCLKernelWithRawGreweFeatures(
    row: typing.Dict[str, typing.Any], datafolder
) -> grewe_features_db.OpenCLKernelWithRawGreweFeatures:
  path = ncc.DataFrameRowToKernelSrcPath(row, datafolder)
  with open(path, 'rb') as f:
    src = f.read().decode('unicode_escape')
  src = src.encode('ascii', 'ignore').decode('ascii')

  return grewe_features_db.OpenCLKernelWithRawGreweFeatures(
      src_sha256=hashlib.sha256(src).hexdigest(),
      origin=f'benchmarks_{row["program:benchmark_suite_name"]}',
      grewe_compute_operation_count=row["feature:comp"],
      grewe_rational_operation_count=row["feature:rational"],
      grewe_global_memory_access_count=row["feature:mem"],
      grewe_local_memory_access_count=row["feature:localmem"],
      grewe_coalesced_memory_access_count=row["feature:coalesced"],
      grewe_atomic_operation_count=row["feature:atomic"],
      src=src,
  )


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  db = grewe_features_db.Database(FLAGS.db)
  df = opencl_device_mapping_dataset.OpenClDeviceMappingsDataset().df

  with ncc.DEEPTUNE_INST2VEC_DATA_ARCHIVE as datafolder:
    with db.Session(commit=True) as sess:
      for _, row in df.iterrows():
        sess.GetOrAdd(RowToOpenCLKernelWithRawGreweFeatures(row, datafolder))


if __name__ == '__main__':
  app.run(main)
