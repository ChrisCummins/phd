"""Gather and serve tensorboard logs."""
import pathlib
import tempfile
import typing

import fs

from labm8 import app
from labm8 import system

FLAGS = app.FLAGS

app.DEFINE_input_path(
    'logdir', None, 'Path to store CLgen cache files.', is_dir=True)
app.DEFINE_list('hosts', list(
    sorted({'cc1', 'cc2', 'cc3'} - {system.HOSTNAME})),
                'List of hosts to gather data from')


def OpenFsFromPaths(paths: typing.List[str]):
  return [fs.open_fs(path) for path in paths]


def GatherFromPaths(paths: typing.List[str], dst: pathlib.Path):
  logdirs = OpenFsFromPaths(paths)
  dst_fs = fs.open_fs(str(dst))
  for path, logdir in zip(paths, logdirs):
    app.Log(1, "Gathering tensorboard events from: %s", path)
    fs.copy.copy_fs(logdir, dst_fs)


def ParseTensorBoardPath(path: pathlib.Path) -> typing.Tuple[str, str]:
  if not path.name.startswith('events.out.tfevents.'):
    raise ValueError(f'Invalid log path: `{path}`')
  name = path.name[len('events.out.tfevents.'):]
  components = name.split('.')
  timestamp, host = components[0], '.'.join(components[1:])
  return host, timestamp


def ArrangeFiles(indir: pathlib.Path, outdir: pathlib.Path):
  for path in indir.iterdir():
    host, timestamp = ParseTensorBoardPath(path)
    run_name = f'{host}_{timestamp}'
    dst = outdir / run_name / path.name
    dst.parent.mkdir()
    app.Log(1, "mv '%s' '%s'", path, dst)
    path.rename(dst)


def GatherAndServe(logdir: pathlib.Path, hosts: typing.List[str]):
  paths = [str(logdir)] + [f'ssh://{host}{logdir}' for host in FLAGS.hosts]

  with tempfile.TemporaryDirectory(prefix='phd_java_fuzz_') as d:
    working_dir = pathlib.Path(d)
    (working_dir / 'in').mkdir()
    (working_dir / 'out').mkdir()

    GatherFromPaths(paths, working_dir / 'in')
    ArrangeFiles(working_dir / 'in', working_dir / 'out')
    raise NotImplementedError("TODO: Server tensorboard from working dir and "
                              "perioridcally update files")


def main():
  """Main entry point."""
  GatherAndServe(FLAGS.logdir, FLAGS.hosts)


if __name__ == '__main__':
  app.Run(main)
