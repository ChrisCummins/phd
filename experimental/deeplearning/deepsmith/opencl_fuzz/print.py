import pathlib

from deeplearning.deepsmith.proto import deepsmith_pb2
from labm8 import app
from labm8 import fmt
from labm8 import pbutil

FLAGS = app.FLAGS

app.DEFINE_string('interesting_results_dir', '/tmp/',
                  'Directory to write interesting results to.')


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized arguments')

  # Parse flags and instantiate testing objects.
  if not FLAGS.interesting_results_dir:
    raise app.UsageError('--interesting_results_dir must be set')
  interesting_results_dir = pathlib.Path(FLAGS.interesting_results_dir)
  if interesting_results_dir.exists() and not interesting_results_dir.is_dir():
    raise app.UsageError('--interesting_results_dir must be a directory')
  app.Info('Recording interesting results in %s.', interesting_results_dir)

  for path in interesting_results_dir.iterdir():
    result = pbutil.FromFile(path, deepsmith_pb2.Result())
    print(f'=== BEGIN INTERESTING RESULT {path.stem} ===')
    print('Outcome:', deepsmith_pb2.Result.Outcome.Name(result.outcome))
    print()
    print('OpenCL kernel')
    print('-------------')
    print(fmt.Indent(2, result.testcase.inputs['src']))
    print()
    print('Stdout')
    print('------')
    print(fmt.Indent(2, result.outputs['stderr']))
    print()


if __name__ == '__main__':
  app.RunWithArgs(main)
