"""Train and sample a deep learning model to generate OpenCL testcases."""
import pathlib

from deeplearning.deepsmith.generators import clgen
from deeplearning.deepsmith.proto import generator_pb2
from labm8 import app
from labm8 import bazelutil
from labm8 import crypto
from labm8 import pbutil

FLAGS = app.FLAGS

app.DEFINE_string(
    'generator',
    str(
        bazelutil.DataPath('phd/docs/2018_07_issta/artifact_evaluation/'
                           'data/clgen.pbtxt')),
    'The path of the generator config proto.')
app.DEFINE_integer('num_testcases', 1024,
                   'The number of testcases to generate.')
app.DEFINE_string(
    'output_directory', '/tmp/phd/docs/2018_07_issta/artifact_evaluation',
    'The directory to write generated programs and testcases to.')


def GenerateTestcases(generator_config: generator_pb2.ClgenGenerator,
                      output_directory: pathlib.Path,
                      num_testcases: int) -> None:
  app.Log(1, 'Writing output to %s', output_directory)
  (output_directory / 'generated_kernels').mkdir(parents=True, exist_ok=True)
  (output_directory / 'generated_testcases').mkdir(parents=True, exist_ok=True)

  app.Log(1, 'Preparing test case generator.')
  generator = clgen.ClgenGenerator(generator_config)

  # Generate testcases.
  app.Log(1, 'Generating %d testcases ...', num_testcases)
  req = generator_pb2.GenerateTestcasesRequest()
  req.num_testcases = num_testcases
  res = generator.GenerateTestcases(req, None)

  for testcase in res.testcases:
    # Write kernel to file.
    kernel = testcase.inputs['src']
    kernel_id = crypto.md5_str(kernel)
    with open(output_directory / 'generated_kernels' / f'{kernel_id}.cl',
              'w') as f:
      f.write(kernel)

    # Write testcase to file.
    testcase_id = crypto.md5_str(str(testcase))
    pbutil.ToFile(
        testcase,
        output_directory / 'generated_testcases' / f'{testcase_id}.pbtxt')

  app.Log(1, '%d testcases written to %s', num_testcases,
          output_directory / 'generated_testcases')
  generation_times = [
      testcase.profiling_events[0].duration_ms for testcase in res.testcases
  ]
  app.Log(1, 'Average time to generate testcase: %.2f ms',
          sum(generation_times) / len(generation_times))


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  config = pathlib.Path(FLAGS.generator)
  if not pbutil.ProtoIsReadable(config, generator_pb2.ClgenGenerator()):
    raise app.UsageError('--generator is not a deepsmith.ClgenGenerator proto')
  generator_config = pbutil.FromFile(config, generator_pb2.ClgenGenerator())
  output_directory = pathlib.Path(FLAGS.output_directory)
  GenerateTestcases(generator_config, output_directory, FLAGS.num_testcases)


if __name__ == '__main__':
  app.RunWithArgs(main)
