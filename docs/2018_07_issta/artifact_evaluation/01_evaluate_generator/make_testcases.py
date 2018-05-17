"""Generate testcases from CLgen kernels."""
import pathlib
import typing

from absl import app
from absl import flags

from deeplearning.deepsmith.proto import deepsmith_pb2
from lib.labm8 import fs
from lib.labm8 import jsonutil
from lib.labm8 import pbutil

FLAGS = flags.FLAGS

flags.DEFINE_string('testcase_dir',
                    './01_evaluate_generator/output/generated_testcases',
                    'Directory to put generated testcases in.')
flags.DEFINE_string('kernels_dir',
                    './01_evaluate_generator/output/generated_kernels',
                    'Directory to read kernels from.')
flags.DEFINE_string('clgen_model_json',
                    './01_evaluate_generator/data/model.json',
                    'Path to CLgen model JSON.')


def MakeGeneratorFromClgenModelJson(
    model_path: pathlib.Path) -> deepsmith_pb2.Generator:
  """Convert a CLgen model JSON into a deepsmith Generator."""
  generator = deepsmith_pb2.Generator()
  config = jsonutil.read_file(model_path)
  generator.name = 'clgen'
  generator.opts['epochs'] = str(config['train_opts'].get('epochs', 10))
  generator.opts['model_type'] = str(config['architecture'].get('model_type', 'lstm'))
  generator.opts['rnn_size'] = str(config['architecture'].get('rnn_size', 256))
  generator.opts['num_layers'] = str(config['architecture'].get('num_layers', 2))
  return generator


def MakeTestcaseFromKernel(
    generator: deepsmith_pb2.Generator,
    harness: deepsmith_pb2.Harness,
    kernel_src: str, gsize: typing.List[int],
    lsize: typing.List[int]) -> deepsmith_pb2.Testcase():
  """Make a testcase from a generated kernel source."""
  t = deepsmith_pb2.Testcase()
  t.generator.CopyFrom(generator)
  t.harness.CopyFrom(harness)
  t.toolchain = 'opencl'
  t.inputs['src'] = kernel_src
  t.inputs['gsize'] = ','.join(str(x) for x in gsize)
  t.inputs['lsize'] = ','.join(str(x) for x in lsize)
  return t


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    unknown_args = ', '.join(argv[1:])
    raise app.UsageError(f"Unknown arguments '{unknown_args}'")

  # Read kernels from...
  indir = pathlib.Path(FLAGS.kernels_dir)
  assert indir.is_dir()

  # Write testcases to...
  outdir = pathlib.Path(FLAGS.testcase_dir)
  outdir.mkdir(exist_ok=True)

  generator = MakeGeneratorFromClgenModelJson(FLAGS.clgen_model_json)
  harness = deepsmith_pb2.Harness()
  harness.name = 'cldrive'
  harness.opts['timeout_seconds'] = '60'

  for kernel_path in fs.ls(indir, abspaths=True):
    kernel_path = pathlib.Path(kernel_path)
    with open(kernel_path) as f:
      kernel = f.read()

    t1 = MakeTestcaseFromKernel(
      generator, harness, kernel, [1, 1, 1], [1, 1, 1])
    pbutil.ToFile(t1, pathlib.Path(outdir / (kernel_path.name + '-1.pbtxt')))
    t2 = MakeTestcaseFromKernel(
      generator, harness, kernel, [128, 16, 1], [32, 1, 1])
    pbutil.ToFile(t2, pathlib.Path(outdir / (kernel_path.name + '-2.pbtxt')))


if __name__ == '__main__':
  app.run(main)
