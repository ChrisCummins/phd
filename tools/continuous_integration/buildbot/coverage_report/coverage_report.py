"""Create python test coverage report."""
import pathlib

from coverage import cmdline as coverage_cli

from labm8.py import app
from labm8.py import fs
from labm8.py import prof

FLAGS = app.FLAGS

app.DEFINE_string('coverage_data_dir', '/coverage/data',
                  'Path to directory containing coverage.py data files.')
app.DEFINE_string('coverage_html_dir', '/coverage/html',
                  'Path to directory to put HTML report in.')


def main():
  """Main entry point."""
  coverage_data_dir = pathlib.Path(FLAGS.coverage_data_dir)
  coverage_html_dir = pathlib.Path(FLAGS.coverage_html_dir)

  if not coverage_data_dir.is_dir():
    raise app.UsageError(f"--coverage_data_dir not found: {coverage_data_dir}")
  coverage_html_dir.mkdir(parents=True, exist_ok=True)

  with fs.chdir(coverage_data_dir):
    with prof.Profile('Combine coverage.py data files'):
      coverage_cli.main(['combine'])
    with prof.Profile('Generate HTML report'):
      coverage_cli.main(['html', '-d', str(coverage_html_dir)])


if __name__ == '__main__':
  app.Run(main)
